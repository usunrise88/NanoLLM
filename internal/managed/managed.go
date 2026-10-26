// Package managed runs NanoLLM as the background process of another
// application: started by it, owned by it, and gone when it is.
//
// The application starts `nanollm managed` as a child process with pipes on
// its standard streams. The child takes a free port on the loopback interface,
// makes up a token, and says both on its standard output as one line of JSON;
// everything after that is the ordinary HTTP API with that token. It stops when
// its standard input closes — which the operating system does for it when the
// application exits, however it exits — or when the process it was told to
// watch is gone.
//
// Nothing here needs administrative rights: the data lives in a directory the
// user owns, the port is above 1024 on the loopback interface, and the process
// only ever lowers its own priority.
//
// What the application chose — the model, the autotuner's mode — is kept
// beside the data in settings.json. The first start has none and says so; the
// application chooses through the API, and every later start comes up with
// what it chose. Unlike a service, a managed instance tunes itself: when the
// mode has no current profile for the model, it measures as soon as the model
// is on disk and applies the result (docs/OPTIMIZE.md, §3.7).
package managed

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
)

// APIVersion is the version of this protocol: the events, the settings and
// the managed endpoints. It changes only when an application written against
// an older one would misread a newer one.
const APIVersion = 1

// Paths is where a managed instance keeps everything it writes.
type Paths struct {
	Data     string
	Settings string
	Lock     string
	Models   string
	Tune     string
	Log      string
}

// PathsFor lays out a data directory. The job database, the prompt cache and
// the profiles go where the server's configuration puts them under its
// data_dir, which is this directory.
func PathsFor(data string) Paths {
	return Paths{
		Data:     data,
		Settings: filepath.Join(data, "settings.json"),
		Lock:     filepath.Join(data, "nanollm.lock"),
		Models:   filepath.Join(data, "models"),
		Tune:     filepath.Join(data, "tune"),
		Log:      filepath.Join(data, "logs", "nanollm.log"),
	}
}

// Create makes the directories a managed instance writes into.
func (p Paths) Create() error {
	for _, dir := range []string{p.Data, p.Models, p.Tune, filepath.Dir(p.Log)} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return err
		}
	}
	return nil
}

// DefaultDataDir is the per-user directory used when the application names
// none. It is local rather than roaming on Windows: models are gigabytes and
// must not travel with a profile. On macOS the service already keeps its data
// in Application Support/NanoLLM, so a managed instance takes a directory of
// its own under it.
//
//	Windows  %LOCALAPPDATA%\NanoLLM
//	macOS    ~/Library/Application Support/NanoLLM/managed
//	Linux    $XDG_DATA_HOME/nanollm, else ~/.local/share/nanollm
func DefaultDataDir() (string, error) {
	switch runtime.GOOS {
	case "windows":
		if dir := os.Getenv("LOCALAPPDATA"); dir != "" {
			return filepath.Join(dir, "NanoLLM"), nil
		}
		dir, err := os.UserCacheDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(dir, "NanoLLM"), nil
	case "darwin":
		dir, err := os.UserConfigDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(dir, "NanoLLM", "managed"), nil
	default:
		if dir := os.Getenv("XDG_DATA_HOME"); filepath.IsAbs(dir) {
			return filepath.Join(dir, "nanollm"), nil
		}
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		if home == "" {
			return "", errors.New("no home directory to keep the data in; pass -data-dir")
		}
		return filepath.Join(home, ".local", "share", "nanollm"), nil
	}
}
