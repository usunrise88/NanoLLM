$here = $PSScriptRoot
foreach ($rep in 1, 2) {
    & (Join-Path $here 'run_variants.ps1') -Variants mtp3, mtp2 -Tag "-base$rep" -intelLuid 'luid_0x00000000_0x0C4E07D4'
}
'ALL DONE'
