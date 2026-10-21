# KLD of the attention paths on the Iris Xe (patch 0007): reference logits from Vulkan with flash attention
# everywhere, then small_batch (matmul attention for long batches on the full-attention layers), flash attention
# off (matmul everywhere, the noise of another exact algorithm) and the CPU with flash attention (cross-backend
# noise), llama-perplexity -c 2048 on a local text corpus (docs/*.md). Reports go to raw\kld-*.txt.
param([string]$Corpus = "$env:TEMP\claude\C--Projects-nanollm\9ae507c4-1490-4f40-9237-6c55572cd24b\scratchpad\kld-docs.txt",
      [int]$Ctx = 2048, [int]$Chunks = 6)
$T = 'C:\Projects\runtimes\tools-0007'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$base = 'C:\Projects\runtimes\kld-vk-fa-base.bin'
$raw = Join-Path $PSScriptRoot 'raw'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$common = @('-m', $m, '-f', $Corpus, '-c', "$Ctx", '--chunks', "$Chunks", '-b', '2048', '-ub', '512')
function Run($name, [string[]]$extra) {
    $out = Join-Path $raw "kld-$name.txt"
    & "$T\llama-perplexity.exe" @common @extra 2>&1 | ForEach-Object { "$_" } | Out-File -Encoding utf8 $out
    "$name : " + ((Get-Content $out | Select-String -Pattern 'Final estimate|Mean    KLD|Maximum KLD|99.9%   KLD|Same top p|RMS Δp|Mean    Δp' | ForEach-Object { ($_ -replace '\s+', ' ').Trim() }) -join ' | ')
}
Run 'vk-fa-base' @('-ngl', '99', '-fa', 'on', '--kl-divergence-base', $base)
Run 'vk-small' @('-ngl', '99', '-fa', 'small_batch', '--kl-divergence-base', $base, '--kl-divergence')
Run 'vk-faoff' @('-ngl', '99', '-fa', 'off', '--kl-divergence-base', $base, '--kl-divergence')
Run 'cpu-fa' @('-ngl', '0', '-t', '20', '-fa', 'on', '--kl-divergence-base', $base, '--kl-divergence')
'DONE'
