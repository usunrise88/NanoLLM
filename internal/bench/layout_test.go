package bench

import (
	"errors"
	"runtime"
	"strings"
	"testing"

	"github.com/usunrise88/nanollm/internal/backend"
	"github.com/usunrise88/nanollm/internal/gguf"
	"github.com/usunrise88/nanollm/internal/plan"
	"github.com/usunrise88/nanollm/internal/registry"
)

func TestVariantLayoutKey(t *testing.T) {
	vs, err := ParseSets([]string{"layout=original,nano"})
	if err != nil || len(vs) != 2 || vs[0].String() != "layout=original" || vs[1].String() != "layout=nano" {
		t.Fatalf("%v %v", vs, err)
	}
	if _, err := ParseSets([]string{"layout=sorted"}); err == nil {
		t.Error("layout=sorted accepted")
	}
	if !strings.Contains(VariantKeys(), "layout") {
		t.Error("layout is not in the help text of --set")
	}
	// It is a loader-level choice, not a planner request: the planner gets
	// the other inventory instead.
	req := plan.Request{}
	vs[1].Plan(&req)
	if req.Mode != "" || req.NoRepack {
		t.Errorf("layout changed the request: %+v", req)
	}
}

// OPT-7 in a sweep: layout=nano plans from the relaid copy's inventory and
// loads its path, and a result says which layout ran.
func TestPlanRunLayout(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("M8 is not proven on macOS (range-lock), so the planner refuses ram-2g there")
	}
	cat, err := registry.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	m, ok := registry.Find(cat, "gemma-4-e4b-qat")
	if !ok {
		t.Fatal("no E4B in the catalog")
	}
	model, err := plan.ModelFrom(m.ID, m.Inventory)
	if err != nil {
		t.Fatal(err)
	}
	l, err := gguf.PlanInventory(model.Target, gguf.DefaultRelayoutAlignment)
	if err != nil {
		t.Fatal(err)
	}
	inv := l.Inventory(model.Target)
	inv.Role = registry.RoleModel
	rl := registry.Relaid{Path: "/models/model.nano.gguf", Inventory: inv}
	cpu := &backend.Device{Backend: "llamacpp", ID: "cpu", Type: backend.CPU, Name: "CPU"}
	o := Options{Manifest: m, ModelPath: "/models/model.gguf", Advisory: true,
		Relaid: func() (registry.Relaid, error) { return rl, nil }}

	var vs []Variant
	if vs, err = ParseSets([]string{"mode=M8", "layout=original,nano"}); err != nil {
		t.Fatal(err)
	}
	a, wa, refused := planRun(o, model, "ram-2g", vs[0], cpu, nil)
	if refused {
		t.Fatal(a.Reason)
	}
	b, wb, refused := planRun(o, model, "ram-2g", vs[1], cpu, nil)
	if refused {
		t.Fatal(b.Reason)
	}
	if a.Plan.Layout != "original" || b.Plan.Layout != "nano" {
		t.Errorf("plan layouts %q and %q", a.Plan.Layout, b.Plan.Layout)
	}
	if wa.Load.Path != o.ModelPath || wb.Load.Path != rl.Path || wa.Layout != "" || wb.Layout != "nano" {
		t.Errorf("paths %s and %s, worker layouts %q and %q", wa.Load.Path, wb.Load.Path, wa.Layout, wb.Layout)
	}
	// Two layouts are two measurements, never one skipped as "the same plan".
	if a.PlanHash == b.PlanHash || a.PlanHash == "" {
		t.Errorf("plan hashes %q and %q", a.PlanHash, b.PlanHash)
	}
	// The same plan otherwise, with the copy's ranges.
	if a.Plan.Mode != b.Plan.Mode || a.Plan.NCtx != b.Plan.NCtx || a.Plan.Repack != b.Plan.Repack {
		t.Errorf("plans differ beyond the layout: %+v, %+v", a.Plan, b.Plan)
	}
	if got, want := len(wb.Load.Stream.Ranges), 42; got != want || len(wa.Load.Stream.Ranges) != 126 {
		t.Errorf("stream ranges: %d, then %d (want %d)", len(wa.Load.Stream.Ranges), got, want)
	}
	if len(wb.Locks) != 1 || len(wa.Locks) <= 1 {
		t.Errorf("locks: %d, then %d", len(wa.Locks), len(wb.Locks))
	}

	// A sweep without the key is the original, and its hash is what it was.
	c, wc, _ := planRun(o, model, "ram-2g", Variant{"mode": "M8"}, cpu, nil)
	if c.Plan.Layout != "original" || wc.Layout != "" || c.PlanHash != a.PlanHash {
		t.Errorf("no key: layout %q, hash %s against %s", c.Plan.Layout, c.PlanHash, a.PlanHash)
	}

	// Without a relaid copy the run is refused with the reason, not failed.
	for name, relaid := range map[string]func() (registry.Relaid, error){
		"none": nil,
		"error": func() (registry.Relaid, error) {
			return registry.Relaid{}, errors.New("run `nanollm models relayout x`")
		},
	} {
		o2 := o
		o2.Relaid = relaid
		r, _, refused := planRun(o2, model, "ram-2g", vs[1], cpu, nil)
		if !refused || r.Status != "refused" || !strings.Contains(r.Reason, "layout=nano") {
			t.Errorf("%s: refused %v, %q, %q", name, refused, r.Status, r.Reason)
		}
	}
}
