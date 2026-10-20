# patches/proposed: revision 2 of the next series (written, not built or measured as a series yet)

0001 (released) + 0002 revision 2 + 0003 (rebased on it).

- 0002 r2 = patches/next/0002 + GEMV prefetch 8 blocks ahead (e2e +4.2 % strong, +5.7 % weak decode, AUTOTUNE OPT-16 table) + the feature
  REPACK_Q4_0_ROWS in the CPU backend's feature list (llama_print_system_info) for the MTP default on CPU.
- series commit (checked by a clean `git am --committer-date-is-author-date`): 5d7f1b89e09ea7b96f2f166b1881ecc0bee4be69

To take it in, after the long checks of the current series (stage2-final-checks.sh) are done:
  cp patches/proposed/000[23]-*.patch patches/next/ ; scripts/build-libs.sh --next --cpus 8 --jobs 8 --out dist/linux_amd64-next-v2
then A/B against dist/linux_amd64-next (scripts/tools-ab.sh), golden, KLD (scripts/kld-cand.sh), native tests, then swap.
