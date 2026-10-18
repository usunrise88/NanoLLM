// Command nanollm is the NanoLLM server and its command-line tools (SPEC §9).
//
//	nanollm init      write a config, issue keys, fetch the default model
//	nanollm serve     run the HTTP server
//	nanollm run       interactive terminal chat without the server
//	nanollm devices   list backends, devices and memory
//	nanollm plan      print a placement plan without loading
//	nanollm bench     prefill/decode speed per model × preset × mode
//	nanollm eval      the quality suite from SPEC §13
//	nanollm models    catalog | list | pull | rm | inspect
//	nanollm key       list | issue | remove
//	nanollm service   install | uninstall | start | stop | restart | status
//	nanollm version   binary version and the pinned llama.cpp build
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	_ "github.com/usunrise88/nanollm/internal/api/native"
	_ "github.com/usunrise88/nanollm/internal/api/openai"
	_ "github.com/usunrise88/nanollm/internal/backend/fake"
	_ "github.com/usunrise88/nanollm/internal/backend/llamacpp"
)

// version is stamped by the linker: -ldflags "-X main.version=$(git describe)".
var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch verb, args := os.Args[1], os.Args[2:]; verb {
	case "init":
		err = initCommand(args)
	case "serve":
		err = serve(args)
	case "run":
		err = runCommand(args)
	case "devices":
		err = devicesCommand(args)
	case "plan":
		err = planCommand(args, os.Stdout)
	case "bench":
		err = notYet("bench", "v0.1")
	case "eval":
		err = evalCommand(args)
	case "models":
		err = modelsCommand(args)
	case "key", "keys":
		err = keyCommand(args)
	case "service":
		err = notYet("service", "v1.0")
	case "version":
		printVersion()
	case "-h", "--help", "help":
		usage()
	default:
		// `nanollm --install` is how a Windows program that registers itself
		// is invoked, so the service verbs answer to that spelling too.
		if _, ok := serviceFlag(verb); ok {
			err = notYet("service", "v1.0")
			break
		}
		fmt.Fprintf(os.Stderr, "nanollm: unknown command %q\n\n", verb)
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "nanollm:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `nanollm — offline LLM inference server

  nanollm init     [-config FILE] [-data-dir DIR] [-addr ADDR] [-model ID] [-no-download] [-force]
  nanollm serve    [-config FILE] [-addr ADDR]
  nanollm run      --model FILE --prompt TEXT [--device D] [-n N] [--temp T]
  nanollm devices  [-config FILE] [-lib-dir DIR]
  nanollm plan     MODEL --preset P [--device D] [--device-memory SIZE] [--mode M] [--kv T] [--json]
  nanollm bench    MODEL
  nanollm eval     [MODEL] --suite NAME|FILE [--thinking off|on|both] | rate REPORT | compare A B | suites
  nanollm models   catalog | list | pull ID... | rm ID | inspect FILE|ID [--tensors] [--json] [--drafter FILE]
  nanollm key      list | issue NAME [-admin] [-interactive] [-rps N] [-hash] | remove NAME
  nanollm service  install | uninstall | start | stop | restart | status
  nanollm version

Start here: "nanollm init" writes a configuration, issues an admin and a user
key, and names the default model, gemma-4-e4b-qat.

Configuration precedence: flags > NANOLLM_* env > config file > defaults.
`)
}

// errNotYet marks a command that is part of the CLI but not of this build.
var errNotYet = errors.New("not implemented yet")

func notYet(verb, milestone string) error {
	return fmt.Errorf("%s: %w; it lands in %s (see SPEC §14)", verb, errNotYet, milestone)
}

// serviceFlag maps the Windows-style spelling of a service verb onto the verb.
func serviceFlag(arg string) (string, bool) {
	switch arg {
	case "--install", "--uninstall", "--start", "--stop", "--restart", "--status":
		return strings.TrimPrefix(arg, "--"), true
	}
	return "", false
}

// parseFlags parses fs from args given in any order and returns the positional
// arguments.
//
// The standard flag package stops at the first positional argument, so
// "run gemma-4-e4b-qat -preset ram-4g" would leave -preset unparsed and run with
// the default. Rather than teach every user that flags come first, the
// arguments are partitioned before parsing. Whether a flag consumes the next
// token is asked of the flag set, so a boolean flag does not swallow a
// positional argument that follows it.
func parseFlags(fs *flag.FlagSet, args []string) ([]string, error) {
	var flags, positional []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			positional = append(positional, args[i+1:]...)
			break
		}
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positional = append(positional, arg)
			continue
		}
		flags = append(flags, arg)
		name := strings.TrimLeft(arg, "-")
		if strings.Contains(name, "=") {
			continue // the value travels with the flag
		}
		if takesValue(fs, name) && i+1 < len(args) {
			i++
			flags = append(flags, args[i])
		}
	}
	if err := fs.Parse(flags); err != nil {
		return nil, err
	}
	return positional, nil
}

func takesValue(fs *flag.FlagSet, name string) bool {
	f := fs.Lookup(name)
	if f == nil {
		return false // unknown: let flag.Parse produce the real complaint
	}
	b, ok := f.Value.(interface{ IsBoolFlag() bool })
	return !ok || !b.IsBoolFlag()
}

// configFlag is the -config flag every command that reads the configuration
// takes, defaulting to $NANOLLM_CONFIG.
func configFlag(fs *flag.FlagSet) *string {
	return fs.String("config", os.Getenv("NANOLLM_CONFIG"), "path to nanollm.yaml")
}
