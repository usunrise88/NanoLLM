import sys, collections, coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
path = sys.argv[1]
units = {"ane": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL, "gpu": ct.ComputeUnit.CPU_AND_GPU}[sys.argv[2]]
m = ct.models.MLModel(path, compute_units=units)
plan = MLComputePlan.load_from_path(path=m.get_compiled_model_path(), compute_units=units)
ops = plan.model_structure.program.functions["main"].block.operations
by = collections.defaultdict(lambda: collections.Counter())
sup = {}
for op in ops:
    if op.operator_name.startswith("const"): continue
    u = plan.get_compute_device_usage_for_mlprogram_operation(op)
    pref = type(u.preferred_compute_device).__name__.replace("ComputeDevice","").replace("ML","") if u else "none"
    by[op.operator_name][pref] += 1
    if u and op.operator_name not in sup:
        sup[op.operator_name] = [type(d).__name__.replace("ComputeDevice","").replace("ML","") for d in u.supported_compute_devices]
for k, v in sorted(by.items()):
    print(f"{k:28s} {dict(v)}  supported={sup.get(k)}")
