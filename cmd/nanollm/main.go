// Command nanollm is the NanoLLM server and its command-line tools (SPEC §9).
//
//	nanollm init      write a config, issue keys, fetch the default model
//	nanollm serve     run the HTTP server
//	nanollm managed   run as the background process of an application
//	nanollm run       interactive terminal chat without the server
//	nanollm devices   list backends, devices and memory
//	nanollm plan      print a placement plan without loading
//	nanollm bench     prefill/decode speed per model × preset × mode
//	nanollm optimize  pick the setup that suits the machine and a mode
//	nanollm eval      the quality suite from SPEC §13
//	nanollm models    catalog | list | pull | rm | inspect | relayout
//	nanollm key       list | issue | remove
//	nanollm service   install | uninstall | start | stop | restart | status
//	nanollm version   binary version and the pinned llama.cpp build
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	_ "github.com/usunrise88/nanollm/internal/api/native"
	_ "github.com/usunrise88/nanollm/internal/api/ollama"
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
	// Asked before the arguments are looked at: the Windows service control
	// manager starts the registered command line, and a process that does not
	// report in over its pipe is killed after thirty seconds.
	if handled, err := runAsService(); handled {
		if err != nil {
			fmt.Fprintln(os.Stderr, "nanollm:", err)
			os.Exit(1)
		}
		return
	}
	// A worker of `nanollm bench` is placed before it reads its spec: the
	// placement survives the re-execution, a pipe does not.
	placeFromEnv()
	var err error
	switch verb, args := os.Args[1], os.Args[2:]; verb {
	case "init":
		err = initCommand(args)
	case "serve":
		err = serve(args)
	case "managed":
		err = managedCommand(context.Background(), args, os.Stdin, os.Stdout)
	case "run":
		err = runCommand(args)
	case "devices":
		err = devicesCommand(args)
	case "plan":
		err = planCommand(args, os.Stdout)
	case "bench":
		err = benchCommand(args)
	case "optimize":
		err = optimizeCommand(args)
	case "eval":
		err = evalCommand(args)
	case "models":
		err = modelsCommand(args)
	case "key", "keys":
		err = keyCommand(args)
	case "service":
		err = serviceCommand(args)
	case "version":
		printVersion()
	case "-h", "--help", "help":
		usage()
	default:
		// `nanollm --install` is how a Windows program that registers itself
		// is invoked, so the service verbs answer to that spelling too.
		if v, ok := serviceFlag(verb); ok {
			err = serviceCommand(append([]string{v}, args...))
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
	fmt.Fprint(os.Stderr, `nanollm — decent LLM for any device

  nanollm init     [-config FILE] [-data-dir DIR] [-addr ADDR] [-model ID] [-no-download] [-force]
  nanollm serve    [-config FILE] [-addr ADDR] [-log-file FILE]
  nanollm managed  [-data-dir DIR] [-parent-pid PID] [-watch-stdin=false] [-low-priority=false] [-ui]
  nanollm run      [MODEL] [--prompt TEXT] [--preset P] [--device D] [--think] [--system TEXT] [-n N] [--temp T]
  nanollm run      --model FILE.gguf --prompt TEXT [--drafter FILE] [--device D] [-n N] [--temp T]
  nanollm devices  [-config FILE] [-lib-dir DIR]
  nanollm plan     MODEL --preset P [--device D] [--device-memory SIZE] [--mode M] [--kv T] [--layout nano] [--json]
  nanollm bench    [MODEL] [--suite standard|dev] [--full] [--preset P] [--submit] | submit FILE | withdraw ID
  nanollm optimize [--mode silent|optimal|powerful] [--yes] [--dry-run] [--json] [--model ID]
  nanollm eval     [MODEL] --suite NAME|FILE [--thinking off|on|both] | rate REPORT | compare A B | suites
  nanollm models   catalog | list | pull ID... | rm ID | inspect FILE|ID [--tensors] [--json] [--drafter FILE]
  nanollm models   relayout ID [--align 4k] [--force] [--verify] [--dry-run]
  nanollm key      list | issue NAME [-admin] [-interactive] [-rps N] [-hash] | remove NAME
  nanollm service  install | uninstall | start | stop | restart | status | unit
                   (also nanollm --install, --uninstall, --start, --stop, --restart, --status)
  nanollm version

Start here: "nanollm init" writes a configuration, issues an admin and a user
key, and names the default model, gemma-4-e4b-qat.

Configuration precedence: flags > NANOLLM_* env > config file > defaults.
`)
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
