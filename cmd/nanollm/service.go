package main

import (
	"bytes"
	"fmt"
	"path/filepath"
	"strings"
	"text/template"
)

// How the service identifies itself to the platform's service manager (SPEC
// §9). The name is what sc.exe, Get-Service and systemctl take; the label is
// launchd's. The display name and the description are what an operator reads
// in services.msc, and they are the only explanation of this process anyone
// will go looking for.
const (
	serviceName        = "nanollm"
	serviceWindowsName = "NanoLLM"
	serviceLabel       = "com.github.usunrise88.nanollm"
	serviceDisplayName = "NanoLLM local LLM server"
	serviceDescription = "Offline LLM inference. Serves the OpenAI-compatible and native HTTP APIs and the web UI on one port."
)

// serviceVerbs are the subcommands of `nanollm service`, in the order the help
// lists them. `unit` prints what install would write, without writing it.
var serviceVerbs = []string{"install", "uninstall", "start", "stop", "restart", "status"}

// listVerbs renders the accepted subcommands for an error message.
func listVerbs() string {
	return strings.Join(serviceVerbs[:len(serviceVerbs)-1], ", ") + " or " + serviceVerbs[len(serviceVerbs)-1]
}

// serviceFlag maps the flag spelling of a service verb onto the subcommand, so
// that `nanollm --install` and `nanollm service install` are one command.
//
// Both spellings exist because both are expected: a Windows program that
// registers itself is invoked as `program.exe --install`, and every other
// command this binary has is a bare verb.
func serviceFlag(arg string) (string, bool) {
	verb := strings.TrimLeft(arg, "-")
	if verb == arg {
		return "", false // not a flag at all
	}
	for _, v := range serviceVerbs {
		if verb == v {
			return v, true
		}
	}
	// "remove" is not offered in the help, but sc.exe spells it that way and
	// an operator who types it means uninstall.
	if verb == "remove" {
		return "uninstall", true
	}
	return "", false
}

// splitVerb takes the subcommand off the front of the arguments: `service`
// alone is `service status`.
func splitVerb(args []string) (string, []string) {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return "status", args
	}
	verb := args[0]
	if verb == "remove" {
		verb = "uninstall"
	}
	return verb, args[1:]
}

// unitParams is everything a unit file says about one installation.
type unitParams struct {
	Exe     string // the binary, absolute
	Config  string // nanollm.yaml, absolute
	DataDir string // weights, the job database, logs
	User    string // systemd: the account it runs as
	Group   string
	LogFile string // launchd: the rolling log -log-file writes
	Label   string // launchd
	System  bool   // launchd: a LaunchDaemon rather than a LaunchAgent
}

// systemdTemplate is the unit for Linux. deploy/nanollm.service is this file
// rendered with the defaults of install.sh, kept in step by a test.
//
// LimitMEMLOCK=infinity: range-lock pins weight groups in RAM with mlock
// (SPEC §4, M2–M8), and the default limit of 8 MiB would make every locked
// mode fall back. No PrivateDevices and no MemoryDenyWriteExecute: Vulkan and
// CUDA need /dev/dri and /dev/nvidia*, and GPU drivers compile shaders into
// executable memory.
const systemdTemplate = `[Unit]
Description={{.Description}}
Documentation=https://github.com/usunrise88/NanoLLM
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
User={{.User}}
Group={{.Group}}
WorkingDirectory={{.DataDir}}
ExecStart={{.ExecStart}}
Restart=on-failure
RestartSec=5s
# In-flight requests get 30 s after SIGTERM (serve's shutdown grace).
TimeoutStopSec=45s
KillSignal=SIGTERM
# Range-lock pins weights in RAM with mlock (SPEC §4); the default 8 MiB
# limit would make every locked memory mode fall back.
LimitMEMLOCK=infinity
LimitNOFILE=65536

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={{.DataDir}}
ProtectKernelTunables=true
ProtectControlGroups=true
RestrictSUIDSGID=true
LockPersonality=true

[Install]
WantedBy=multi-user.target
`

// launchdTemplate is the property list for macOS, an agent in
// ~/Library/LaunchAgents by default and a daemon in /Library/LaunchDaemons
// with -system. deploy/nanollm.plist is the daemon rendered with defaults.
//
// KeepAlive restarts it after a crash but not after `service stop`; the log
// goes to a rolling file because launchd's StandardErrorPath grows forever,
// which is kept only for what is written before the logger exists.
const launchdTemplate = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>{{.Label}}</string>
	<key>ProgramArguments</key>
	<array>
{{- range .Args}}
		<string>{{.}}</string>
{{- end}}
	</array>
	<key>WorkingDirectory</key>
	<string>{{.DataDir}}</string>
	<key>RunAtLoad</key>
	<true/>
	<key>KeepAlive</key>
	<dict>
		<key>SuccessfulExit</key>
		<false/>
	</dict>
	<key>ThrottleInterval</key>
	<integer>5</integer>
	<key>ExitTimeOut</key>
	<integer>45</integer>
	<key>StandardErrorPath</key>
	<string>{{.Stderr}}</string>
{{- if .System}}
	<key>UserName</key>
	<string>{{.User}}</string>
{{- end}}
</dict>
</plist>
`

// renderSystemd is the unit file for p.
func renderSystemd(p unitParams) (string, error) {
	group := p.Group
	if group == "" {
		group = p.User
	}
	return render(systemdTemplate, map[string]any{
		"Description": serviceDisplayName,
		"User":        p.User,
		"Group":       group,
		"DataDir":     systemdPath(p.DataDir),
		"ExecStart":   systemdCommand(p.Exe, "serve", "-config", p.Config),
	})
}

// renderLaunchd is the property list for p.
func renderLaunchd(p unitParams) (string, error) {
	label := p.Label
	if label == "" {
		label = serviceLabel
	}
	args := []string{p.Exe, "serve", "-config", p.Config}
	if p.LogFile != "" {
		args = append(args, "-log-file", p.LogFile)
	}
	stderr := filepath.Join(filepath.Dir(p.LogFile), "nanollm.stderr.log")
	if p.LogFile == "" {
		stderr = filepath.Join(p.DataDir, "logs", "nanollm.stderr.log")
	}
	esc := make([]string, len(args))
	for i, a := range args {
		esc[i] = xmlEscape(a)
	}
	return render(launchdTemplate, map[string]any{
		"Label":   xmlEscape(label),
		"Args":    esc,
		"DataDir": xmlEscape(p.DataDir),
		"Stderr":  xmlEscape(stderr),
		"System":  p.System,
		"User":    xmlEscape(p.User),
	})
}

func render(text string, data any) (string, error) {
	var b bytes.Buffer
	if err := template.Must(template.New("unit").Parse(text)).Execute(&b, data); err != nil {
		return "", err
	}
	return b.String(), nil
}

// systemdCommand quotes a command line the way systemd.service(5) parses
// ExecStart: double quotes around a word with a space, backslash before a
// quote or a backslash, and %% for a literal percent sign, which would
// otherwise be read as a specifier.
func systemdCommand(words ...string) string {
	out := make([]string, len(words))
	for i, w := range words {
		out[i] = systemdWord(w)
	}
	return strings.Join(out, " ")
}

func systemdWord(w string) string {
	w = strings.ReplaceAll(w, "%", "%%")
	if w != "" && !strings.ContainsAny(w, " \t\"'\\;$") {
		return w
	}
	r := strings.NewReplacer(`\`, `\\`, `"`, `\"`, `$`, `$$`)
	return `"` + r.Replace(w) + `"`
}

// systemdPath escapes a path for a setting that takes one path, where only %
// is special.
func systemdPath(p string) string { return strings.ReplaceAll(p, "%", "%%") }

func xmlEscape(s string) string {
	var b bytes.Buffer
	for _, r := range s {
		switch r {
		case '&':
			b.WriteString("&amp;")
		case '<':
			b.WriteString("&lt;")
		case '>':
			b.WriteString("&gt;")
		case '"':
			b.WriteString("&quot;")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// defaultUnitParams are the paths install.sh uses on each platform (SPEC §10:
// /opt/nanollm + /var/lib/nanollm on Linux, /usr/local/nanollm on macOS). The
// configuration lives with the data, so that an uninstall without --purge
// keeps the keys along with the weights.
func defaultUnitParams(goos string) unitParams {
	switch goos {
	case "darwin":
		data := "/Library/Application Support/NanoLLM"
		return unitParams{
			Exe:     "/usr/local/nanollm/nanollm",
			Config:  data + "/nanollm.yaml",
			DataDir: data,
			LogFile: "/Library/Logs/NanoLLM/nanollm.log",
			Label:   serviceLabel,
			System:  true,
			User:    "root",
		}
	default:
		return unitParams{
			Exe:     "/opt/nanollm/nanollm",
			Config:  "/var/lib/nanollm/nanollm.yaml",
			DataDir: "/var/lib/nanollm",
			User:    "nanollm",
			Group:   "nanollm",
		}
	}
}

// unitFor renders the unit file for goos.
func unitFor(goos string, p unitParams) (string, error) {
	switch goos {
	case "linux":
		return renderSystemd(p)
	case "darwin":
		return renderLaunchd(p)
	}
	return "", fmt.Errorf("no unit file for %s; the Windows service is registered with `nanollm service install`", goos)
}
