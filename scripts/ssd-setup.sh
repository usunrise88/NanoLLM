#!/usr/bin/env bash
# Prepare a data directory on a fast disk for the SSD run of the night queue
# (docs/STAGE2.md): the model files are copied (not linked: the point is where
# they are read from) and the configs written next to them, pointing at the
# same libraries.
#
#   scripts/ssd-setup.sh /mnt/ssd/nanollm-data
#   SSD=1 DATA=/mnt/ssd/nanollm-data scripts/hyper-exclusive.sh -- scripts/night-hdd.sh ssd
set -euo pipefail
cd "$(dirname "$0")/.."
DST=${1:?usage: scripts/ssd-setup.sh DATA_DIR_ON_THE_SSD}
SRC=${NANOLLM_DATA:-$HOME/nanollm-data}
mkdir -p "$DST/models"
for d in "$SRC"/models/gemma-4-e4b-qat; do
	rm -rf "$DST/models/$(basename "$d")" 2>/dev/null || true
	cp -r "$d" "$DST/models/"
done
for f in base ref ref-ngram next; do
	[[ -f $SRC/$f.yaml ]] && sed "s#data_dir: .*#data_dir: $DST#" "$SRC/$f.yaml" >"$DST/$f.yaml"
done
echo "ssd-setup: $(du -sh "$DST/models" | cut -f1) copied to $DST; configs: $(ls "$DST"/*.yaml | xargs -n1 basename | tr '\n' ' ')"
findmnt -T "$DST" -no SOURCE,FSTYPE | sed 's/^/ssd-setup: on /'
lsblk -dno NAME,ROTA,SIZE,MODEL "$(findmnt -T "$DST" -no SOURCE | sed 's#/dev/##; s/[0-9]*$//')" 2>/dev/null | sed 's/^/ssd-setup: device /' || true
