package main

import (
	_ "embed"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"text/template"

	"github.com/usunrise88/nanollm/internal/config"
	"github.com/usunrise88/nanollm/internal/registry"
)

//go:embed init.yaml
var initTemplate string

// initCommand writes a working configuration with two fresh keys (SPEC §9).
//
// It ends by downloading the default model and its drafter, verified, unless
// -no-download; a failed download leaves the configuration in place and says
// how to resume.
func initCommand(args []string) error {
	fs := flag.NewFlagSet("init", flag.ExitOnError)
	cfgPath := fs.String("config", "nanollm.yaml", "configuration file to write")
	dataDir := fs.String("data-dir", config.DefaultDataDir(), "where weights and the job database live")
	addr := fs.String("addr", "127.0.0.1:8080", "listen address")
	model := fs.String("model", config.DefaultModel, "default model (a catalog id)")
	noDownload := fs.Bool("no-download", false, "write the configuration but download nothing")
	force := fs.Bool("force", false, "overwrite an existing configuration file")
	if _, err := parseFlags(fs, args); err != nil {
		return err
	}

	if _, err := os.Stat(*cfgPath); err == nil && !*force {
		return fmt.Errorf("%s already exists; pass -force to overwrite it", *cfgPath)
	}
	catalog, err := registry.Builtin()
	if err != nil {
		return err
	}
	if _, ok := registry.Find(catalog, *model); !ok {
		return fmt.Errorf("-model %q is not in the catalog; `nanollm models catalog` lists it", *model)
	}

	adminKey, err := config.NewKeySecret()
	if err != nil {
		return err
	}
	userKey, err := config.NewKeySecret()
	if err != nil {
		return err
	}
	host, _ := os.Hostname()

	var b strings.Builder
	tmpl := template.Must(template.New("init").Parse(initTemplate))
	if err := tmpl.Execute(&b, map[string]string{
		"Host": host, "Addr": yamlString(*addr), "DataDir": yamlString(*dataDir),
		"Model": *model, "AdminKey": adminKey, "UserKey": userKey,
	}); err != nil {
		return err
	}

	if err := writeValidated(*cfgPath, []byte(b.String())); err != nil {
		return err
	}

	fmt.Printf("wrote %s\n\n", *cfgPath)
	fmt.Printf("  admin key  %s\n", adminKey)
	fmt.Printf("  user key   %s\n\n", userKey)
	fmt.Println("The keys are in the file and shown only this once here.")
	if !*noDownload {
		fmt.Printf("\nDownloading %s; Ctrl-C stops, and the same command resumes.\n", *model)
		if err := modelsPull([]string{"-config", *cfgPath, *model}); err != nil {
			return fmt.Errorf("%w\nthe configuration is written; resume the download with: nanollm models pull -config %s %s", err, *cfgPath, *model)
		}
	}
	fmt.Printf("Start the server with: nanollm serve -config %s\n", *cfgPath)
	return nil
}

// writeValidated writes the configuration atomically, and only once it loads
// (config.WriteValidated). The file holds secrets.
func writeValidated(path string, data []byte) error { return config.WriteValidated(path, data, 0o600) }

// yamlString makes a value safe inside a double-quoted YAML scalar. Windows
// data directories are full of backslashes, which YAML would read as escapes.
func yamlString(s string) string {
	q := strconv.Quote(s)
	return q[1 : len(q)-1]
}
