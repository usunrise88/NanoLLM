# The same llama-bench (upstream b11600) against different backend libraries,
# one run per configuration: pp512 and tg128 on the Gemma 4 E4B Q4_0 GGUF.
param([string[]]$Cases = @('vulkan-igpu', 'sycl-igpu', 'cpu-upstream-p6', 'cpu-patched-p6', 'cpu-patched-t14'), [int]$Reps = 1, [string]$OutFile = 'bench-matrix.txt')
$R = 'C:\Projects\runtimes'
$m = 'C:\ProgramData\NanoLLM\models\gemma-4-e4b-qat\8c5a9e4fd5482e2be20fe0bf013b4c262a8f4265\gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
$out = "$R\results\$OutFile"
# dir, extra args, affinity mask (0 = none)
$cfg = @{
    'vulkan-igpu'     = @("$R\bench-vulkan", @('-ngl', '99'), 0)
    'sycl-igpu'       = @("$R\llamacpp-b11600\sycl", @('-ngl', '99'), 0)
    'cpu-upstream-p6' = @("$R\bench-upstream", @('-ngl', '0', '-t', '6'), 0x555)
    'cpu-patched-p6'  = @("$R\bench-patched", @('-ngl', '0', '-t', '6'), 0x555)
    'cpu-patched-t14' = @("$R\bench-patched", @('-ngl', '0', '-t', '14'), 0)
    'cpu-upstream-t14' = @("$R\bench-upstream", @('-ngl', '0', '-t', '14'), 0)
}
if (-not (Test-Path "$R\bench-vulkan")) {
    Copy-Item -Recurse "$R\bench-upstream" "$R\bench-vulkan"
    Copy-Item 'C:\Projects\nanollm\dist\windows_amd64\lib\ggml-vulkan.dll' "$R\bench-vulkan\"
}
$env:GGML_VK_VISIBLE_DEVICES = '1'
Remove-Item -ErrorAction SilentlyContinue Env:GGML_OPENVINO_DEVICE
"" | Set-Content $out
foreach ($c in $Cases) {
    $dir, $extra, $mask = $cfg[$c]
    $tmp = "$R\results\bm-$c-r$Reps.txt"
    $a = @('-m', $m, '-p', '512', '-n', '128', '-r', "$Reps", '-o', 'md') + $extra
    $p = Start-Process -FilePath "$dir\llama-bench.exe" -ArgumentList $a -WorkingDirectory $dir -PassThru -NoNewWindow -RedirectStandardOutput $tmp -RedirectStandardError "$tmp.err"
    if ($mask) { try { $p.ProcessorAffinity = [IntPtr]$mask } catch {} }
    if (-not $p.WaitForExit(900000)) { $p.Kill(); Add-Content $out "$c : TIMEOUT"; continue }
    $rows = Get-Content $tmp | Where-Object { $_ -match 'pp512|tg128' } | ForEach-Object {
        $f = ($_ -replace '[^\x20-\x7E]', '') -split '\|'; ('{0} {1}' -f $f[-3].Trim(), $f[-2].Trim())
    }
    $line = '{0,-18} {1}' -f $c, ($rows -join '   ')
    $line; Add-Content $out $line
}
'ALL DONE'
