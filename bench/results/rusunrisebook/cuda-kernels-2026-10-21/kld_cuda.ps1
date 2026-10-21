# KLD of our ggml-cuda kernels against upstream's on the RTX: the reference logits come from the build with every
# GGML_CUDA_NANO_* switch off (upstream's kernels), then the same model with the defaults, with each inexact change
# alone, and the noise of the other direction (the CPU). llama-perplexity -c 2048 on a local text corpus; reports go to
# raw\kld-*.txt. The quality gate of docs/AUTOTUNE.md (mean KLD <= 4.0e-3, p99 <= 3.5e-2, top-1 >= 95.3 %) was calibrated
# on the CPU kernels; for the GPU the reference is the noise that two correct CUDA kernels already differ by.
param([string]$Corpus = "$env:TEMP\claude\C--Projects-nanollm\9ae507c4-1490-4f40-9237-6c55572cd24b\scratchpad\kld-docs.txt",
      [int]$Ctx = 2048, [int]$Chunks = 6)
$T = 'C:\Projects\nanollm\.cache\llama.cpp\out-windows_amd64-cuda\tools'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$base = 'C:\Projects\runtimes\kld-cuda-off-base.bin'
$raw = Join-Path $PSScriptRoot 'raw'
$names = 'GRAPH_SHAPES', 'MMVQ_NWARPS', 'MMVQ_NWARPS_N', 'FUSE_BATCH', 'Q8_CACHE', 'NORM_LEAN', 'GELU_MUL', 'FA_VEC', 'NORM_Q8'
$common = @('-m', $m, '-f', $Corpus, '-c', "$Ctx", '--chunks', "$Chunks", '-b', '2048', '-ub', '512', '-ngl', '99')
function Run($name, [string[]]$extra, [string[]]$env = @()) {
    foreach ($e in $env) { $k, $v = $e -split '=', 2; [Environment]::SetEnvironmentVariable("GGML_CUDA_NANO_$k", $v) }
    $out = Join-Path $raw "kld-$name.txt"
    & "$T\llama-perplexity.exe" @common @extra 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $out
    foreach ($e in $env) { $k, $v = $e -split '=', 2; [Environment]::SetEnvironmentVariable("GGML_CUDA_NANO_$k", $null) }
    "$name : " + ((Get-Content $out | Select-String -Pattern 'Final estimate|Mean    KLD|Maximum KLD|99.9%   KLD|99.0%   KLD|Same top p|RMS Δp|Mean    Δp' | ForEach-Object { ($_ -replace '\s+', ' ').Trim() }) -join ' | ')
}
$off = $names | ForEach-Object { "$_=0" }
Run 'cuda-off-base' @('--kl-divergence-base', $base) $off
Run 'cuda-defaults' @('--kl-divergence-base', $base, '--kl-divergence')
Run 'cuda-off-again' @('--kl-divergence-base', $base, '--kl-divergence') $off
foreach ($one in 'MMVQ_NWARPS', 'MMVQ_NWARPS_N', 'NORM_LEAN') {
    # only this inexact change on, the others off
    $e = $names | Where-Object { $_ -notin @($one, 'MMVQ_NWARPS_N') -or ($one -eq 'MMVQ_NWARPS_N' -and $_ -eq 'MMVQ_NWARPS_N') } | ForEach-Object { "$_=0" }
    $e = $names | Where-Object { $_ -ne $one } | ForEach-Object { "$_=0" }
    Run "cuda-only-$one" @('--kl-divergence-base', $base, '--kl-divergence') $e
}
'DONE'
