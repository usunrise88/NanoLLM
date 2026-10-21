# KLD of the grouped-query CPU flash attention (patch 0008 under test) on the decode shapes: llama-perplexity on the
# CPU with ubatches of 1 and 3 tokens (the decode and MTP-verification paths; the kernel takes over from 512 cells of
# context), -c 1024, 2 chunks of a local ru/en text (docs/*.md). Reference: the n6 kernel, ubatch 1. Noise: the n6
# kernel with flash attention off. Reports in raw\kld-cpu-*.txt.
param([string]$New = 'C:\Projects\nanollm\.cache\llama.cpp\out-fagqa\lib\ggml-cpu-alderlake.dll',
      [string]$Old = 'C:\Projects\nanollm\dist\windows_amd64\lib\ggml-cpu-alderlake.dll',
      [string]$Only = '',
      [string]$Corpus = "$env:TEMP\claude\C--Projects-nanollm\9ae507c4-1490-4f40-9237-6c55572cd24b\scratchpad\kld-docs.txt")
$T = 'C:\Projects\runtimes\tools-cpu-kld'
if (-not (Test-Path $T)) {
    Copy-Item -Recurse 'C:\Projects\runtimes\tools-0007' $T
    Remove-Item "$T\ggml-vulkan.dll"
}
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$base = 'C:\Projects\runtimes\kld-cpu-fa-base.bin'
$raw = Join-Path $PSScriptRoot 'raw'
$common = @('-m', $m, '-f', $Corpus, '-c', '1024', '--chunks', '2', '-b', '1024', '-t', '6', '-ngl', '0')
function Run($name, $dll, [string[]]$extra) {
    if ($Only -and ($Only -split ',') -notcontains $name) { return }
    Copy-Item -Force $dll "$T\ggml-cpu-alderlake.dll"
    $out = Join-Path $raw "kld-cpu-$name.txt"
    $p = Start-Process -FilePath "$T\llama-perplexity.exe" -ArgumentList ($common + $extra) -WorkingDirectory $T -PassThru -NoNewWindow `
        -RedirectStandardOutput "$out.out" -RedirectStandardError $out
    try { $p.ProcessorAffinity = [IntPtr]0x555 } catch {}
    $p.WaitForExit()
    "$name : " + ((Get-Content "$out.out", $out | Select-String -Pattern 'Final estimate|Mean    KLD|Median  KLD|99.0%   KLD|99.9%   KLD|Same top p' | ForEach-Object { ($_ -replace '\s+', ' ').Trim() }) -join ' | ')
}
Run 'base-ub1' $Old @('-ub', '1', '-fa', 'on', '--kl-divergence-base', $base)
Run 'new-ub1' $New @('-ub', '1', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
Run 'new-ub3' $New @('-ub', '3', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
Run 'old-ub3' $Old @('-ub', '3', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
Run 'faoff-ub1' $Old @('-ub', '1', '-fa', 'off', '--kl-divergence-base', $base, '--kl-divergence')
'DONE'
