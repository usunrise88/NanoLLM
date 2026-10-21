# Flash attention against small_batch (patch 0007) at small depths, where the matmul attention may lose:
# llama-bench pp512 at depths 0-2048, the two modes in alternating invocations.
param([int]$Rounds = 2, [string]$Depths = '0,512,1024,2048', [string]$Out = 'lb-0007-low.txt')
$T = 'C:\Projects\runtimes\tools-0007'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$env:GGML_VK_VISIBLE_DEVICES = '1'
$outf = Join-Path $PSScriptRoot "raw\$Out"
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($fa in @('1', 'small_batch')) {
        $o = & "$T\llama-bench.exe" -m $m -ngl 99 -fa $fa -p 512 -n 0 -d $Depths -r 2 -o md 2>&1 | ForEach-Object { "$_" }
        $rows = $o | Where-Object { $_ -match '\|\s*pp\d+' } | ForEach-Object { $f = ($_ -split '\|'); '{0}={1}' -f ($f[-3].Trim() -replace ' ', ''), ($f[-2].Trim() -split ' ')[0] }
        $line = 'round {0} {1,-12} {2}' -f $i, $fa, ($rows -join '  ')
        $line; Add-Content $outf $line
        Start-Sleep -Seconds 5
    }
}
