// predict_bench times Core ML predictions without a scripting layer in the way:
//
//   clang -fobjc-arc -O2 -framework CoreML -framework Foundation -o predict_bench predict_bench.m
//   predict_bench MODEL[,MODEL...] ane|gpu|cpu|all [ITERATIONS] [FUNCTION]
//
// MODEL is an .mlpackage or an .mlmodelc. Several models, comma separated, are loaded together and
// called one after the other in every iteration: the chunks of one token. Every input is allocated
// once as a float16 array (zeros for names starting with "m", the masks; 0.1 elsewhere) and handed
// to every call, so the numbers are the models', not the marshalling of numpy arrays.
//
// Prints one JSON line: the load seconds (the first load compiles an .mlpackage for the device),
// the median, p90 and minimum of an iteration (all models) and the median of each model in
// milliseconds, the phys_footprint of the process, and what the load and the run took from the
// system in MiB (host_statistics64: free, wired, compressed, anonymous and file-backed pages).
// Stage 1a of docs/research/coreml-ane.md; the bridge of stage 2 starts from this file.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <time.h>

static double now_ms(void) { return (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }

static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmp);
  return v[n / 2];
}

static double footprint_mib(void) {
  task_vm_info_data_t info;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &n) != KERN_SUCCESS) return -1;
  return (double)info.phys_footprint / (1 << 20);
}

typedef struct { double free, wired, compressed, internal, external; } sysmem;

static sysmem sys_mem(void) {
  vm_statistics64_data_t v;
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n);
  double mib = (double)vm_page_size / (1 << 20);
  sysmem s = {v.free_count * mib, v.wire_count * mib, v.compressor_page_count * mib, v.internal_page_count * mib, v.external_page_count * mib};
  return s;
}

static MLDictionaryFeatureProvider *inputs_for(MLModel *model) {
  NSError *err = nil;
  NSMutableDictionary<NSString *, MLFeatureValue *> *in = [NSMutableDictionary new];
  for (NSString *name in model.modelDescription.inputDescriptionsByName) {
    MLFeatureDescription *d = model.modelDescription.inputDescriptionsByName[name];
    if (d.type != MLFeatureTypeMultiArray) { fprintf(stderr, "input %s is not an array\n", name.UTF8String); return nil; }
    MLMultiArray *a = [[MLMultiArray alloc] initWithShape:d.multiArrayConstraint.shape dataType:d.multiArrayConstraint.dataType error:&err];
    if (!a) { fprintf(stderr, "array %s: %s\n", name.UTF8String, [[err description] UTF8String]); return nil; }
    BOOL mask = [name hasPrefix:@"m"];
    if (d.multiArrayConstraint.dataType == MLMultiArrayDataTypeFloat16) {
      __fp16 v = mask ? 0.0f : 0.1f;
      __fp16 *p = (__fp16 *)a.dataPointer;
      for (NSInteger i = 0; i < a.count; i++) p[i] = v;
    } else {
      memset(a.dataPointer, 0, a.count * 4);
    }
    in[name] = [MLFeatureValue featureValueWithMultiArray:a];
  }
  MLDictionaryFeatureProvider *fp = [[MLDictionaryFeatureProvider alloc] initWithDictionary:in error:&err];
  if (!fp) fprintf(stderr, "features: %s\n", [[err description] UTF8String]);
  return fp;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr, "usage: predict_bench MODEL[,MODEL...] ane|gpu|cpu|all [ITERATIONS] [FUNCTION]\n");
      return 2;
    }
    // An 8 GiB Mac that runs out of memory and swap stops answering its watchdog and panics (2026-10-22):
    // refuse to load models into a system that has little left to give.
    int level = 100;
    size_t levelSize = sizeof(level);
    if (sysctlbyname("kern.memorystatus_level", &level, &levelSize, NULL, 0) == 0 && level < 35) {
      fprintf(stderr, "only %d%% of memory is available, 35%% needed: close something first\n", level);
      return 3;
    }
    NSArray<NSString *> *paths = [[NSString stringWithUTF8String:argv[1]] componentsSeparatedByString:@","];
    NSString *units = [NSString stringWithUTF8String:argv[2]];
    int iters = argc > 3 ? atoi(argv[3]) : 50;
    NSError *err = nil;
    MLModelConfiguration *cfg = [MLModelConfiguration new];
    if ([units isEqualToString:@"ane"]) cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    else if ([units isEqualToString:@"gpu"]) cfg.computeUnits = MLComputeUnitsCPUAndGPU;
    else if ([units isEqualToString:@"cpu"]) cfg.computeUnits = MLComputeUnitsCPUOnly;
    else cfg.computeUnits = MLComputeUnitsAll;
    if (argc > 4) cfg.functionName = [NSString stringWithUTF8String:argv[4]];

    sysmem m0 = sys_mem();
    double t0 = now_ms();
    NSMutableArray<MLModel *> *models = [NSMutableArray new];
    NSMutableArray<MLDictionaryFeatureProvider *> *feats = [NSMutableArray new];
    for (NSString *path in paths) {
      NSURL *url = [NSURL fileURLWithPath:path];
      if ([[url pathExtension] isEqualToString:@"mlpackage"]) {
        url = [MLModel compileModelAtURL:url error:&err];
        if (!url) { fprintf(stderr, "compile %s: %s\n", path.UTF8String, [[err description] UTF8String]); return 1; }
      }
      MLModel *model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
      if (!model) { fprintf(stderr, "load %s: %s\n", path.UTF8String, [[err description] UTF8String]); return 1; }
      MLDictionaryFeatureProvider *fp = inputs_for(model);
      if (!fp) return 1;
      [models addObject:model];
      [feats addObject:fp];
    }
    double load_s = (now_ms() - t0) / 1e3;

    NSUInteger nm = models.count;
    for (int i = 0; i < 3; i++) {  // warm-up: every model once per round, as a token does
      for (NSUInteger j = 0; j < nm; j++) {
        @autoreleasepool { if (![models[j] predictionFromFeatures:feats[j] error:&err]) { fprintf(stderr, "predict: %s\n", [[err description] UTF8String]); return 1; } }
      }
    }
    sysmem m1 = sys_mem();
    const char *ready = getenv("READY_FILE");  // tells tools/coreml/energy.sh the models are loaded and warm
    if (ready) [[NSData data] writeToFile:[NSString stringWithUTF8String:ready] atomically:YES];
    double *total = calloc(iters, sizeof(double));
    double *each = calloc((size_t)iters * nm, sizeof(double));
    for (int i = 0; i < iters; i++) {
      double ti = now_ms();
      for (NSUInteger j = 0; j < nm; j++) {
        @autoreleasepool {
          double t = now_ms();
          if (![models[j] predictionFromFeatures:feats[j] error:&err]) { fprintf(stderr, "predict: %s\n", [[err description] UTF8String]); return 1; }
          each[(size_t)j * iters + i] = now_ms() - t;
        }
      }
      total[i] = now_ms() - ti;
    }
    sysmem m2 = sys_mem();
    printf("{\"units\":\"%s\",\"models\":%lu,\"load_s\":%.2f,\"iters\":%d", argv[2], (unsigned long)nm, load_s, iters);
    printf(",\"each_ms_median\":[");
    for (NSUInteger j = 0; j < nm; j++) printf("%s%.2f", j ? "," : "", median(each + (size_t)j * iters, iters));
    printf("]");
    double min = total[0];
    for (int i = 1; i < iters; i++) if (total[i] < min) min = total[i];
    double p90 = ({ qsort(total, iters, sizeof(double), cmp); total[(int)(iters * 0.9)]; });
    printf(",\"step_ms_median\":%.2f,\"step_ms_p90\":%.2f,\"step_ms_min\":%.2f,\"footprint_mib\":%.0f", total[iters / 2], p90, min, footprint_mib());
    printf(",\"load_took_mib\":{\"free\":%.0f,\"wired\":%.0f,\"compressed\":%.0f,\"anonymous\":%.0f,\"file\":%.0f}",
           m0.free - m1.free, m1.wired - m0.wired, m1.compressed - m0.compressed, m1.internal - m0.internal, m1.external - m0.external);
    printf(",\"run_took_mib\":{\"free\":%.0f,\"wired\":%.0f}}\n", m1.free - m2.free, m2.wired - m1.wired);
    free(total);
    free(each);
  }
  return 0;
}
