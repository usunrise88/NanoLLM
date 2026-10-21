# Runs llama-bench of ik_llama.cpp under different CPU affinity masks and
# thread counts on the i7-12700H (logical CPUs 0-11 are the six P-cores with
# two threads each, 12-19 are the eight E-cores).
param(
    [string]$Bench = 'C:\Projects\runtimes\ik_llama.cpp\build\bin\llama-bench.exe',
    [string]$Model = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf',
    [string]$Out = 'C:\Projects\runtimes\results\ik-affinity-sweep.txt',
    [string]$Extra = ''
)
$env:PATH = 'C:\msys64\mingw64\bin;' + $env:PATH
New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
# name, mask, threads
$cases = @(
    @('P-phys 6T',        0x555,   6),
    @('P-phys 5T',        0x155,   5),
    @('P-phys 4T',        0x055,   4),
    @('P-HT 12T',         0xFFF,   12),
    @('P-phys+E 14T',     0xFF555, 14),
    @('E-only 8T',        0xFF000, 8),
    @('P-phys+4E 10T',    0xF0555, 10),
    @('all 20T',          0xFFFFF, 20)
)
"" | Set-Content $Out
foreach ($c in $cases) {
    $name, $mask, $t = $c
    $args = @('-m', $Model, '-p', '512', '-n', '128', '-t', "$t", '-r', '3', '-o', 'md') + ($Extra -split ' ' | Where-Object { $_ })
    $p = Start-Process -FilePath $Bench -ArgumentList $args -PassThru -NoNewWindow -RedirectStandardOutput "$Out.tmp"
    $p.ProcessorAffinity = [IntPtr]$mask
    $p.WaitForExit()
    $rows = Get-Content "$Out.tmp" | Where-Object { $_ -match 'pp512|tg128' } | ForEach-Object {
        $f = $_ -split '\|'; ('{0} {1}' -f $f[-3].Trim(), $f[-2].Trim())
    }
    $line = '{0,-16} mask 0x{1:X} t={2,-2}: {3}' -f $name, $mask, $t, ($rows -join '   ')
    $line
    Add-Content $Out $line
}
