# Where the CPU spends a decode step with flash attention on and off: llama-bench on a dev ggml-cpu-alderlake.dll
# with a per-op wall-time counter (thread 0, op + barrier; GGML_OP_TIMING=<file>), 6 threads on the P-cores, at
# 4096 tokens of context: tg32 (one token per graph) and pp3 (three tokens: the MTP verification shape).
param([string]$Tag = '', [string]$Dll = 'C:\Projects\nanollm\.cache\llama.cpp\out-prof\lib\ggml-cpu-alderlake.dll', [string]$Depth = '4096')
$R = 'C:\Projects\runtimes'
$dir = "$R\bench-cpu-prof"
if (-not (Test-Path $dir)) { Copy-Item -Recurse "$R\bench-cpu-n6" $dir }
Copy-Item -Force $Dll "$dir\ggml-cpu-alderlake.dll"
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$raw = Join-Path $PSScriptRoot 'raw'
foreach ($case in @(@('tg', @('-p', '0', '-n', '32')), @('pp3', @('-p', '3', '-n', '0')))) {
    foreach ($fa in @('1', '0')) {
        $name = "cpu-prof$Tag-$($case[0])-fa$fa-d$Depth"
        $env:GGML_OP_TIMING = Join-Path $raw "$name.ops.txt"
        $a = @('-m', $m, '-ngl', '0', '-t', '6', '-fa', $fa, '-d', $Depth, '-r', '2', '-o', 'md') + $case[1]
        $p = Start-Process -FilePath "$dir\llama-bench.exe" -ArgumentList $a -WorkingDirectory $dir -PassThru -NoNewWindow `
            -RedirectStandardOutput (Join-Path $raw "$name.md") -RedirectStandardError (Join-Path $raw "$name.err")
        try { $p.ProcessorAffinity = [IntPtr]0x555 } catch {}
        $p.WaitForExit()
        Remove-Item Env:GGML_OP_TIMING
        $row = Get-Content (Join-Path $raw "$name.md") | Where-Object { $_ -match '\|\s*(pp|tg)\d+' } | Select-Object -Last 1
        "$name : $row"
        Start-Sleep -Seconds 5
    }
}
