#!/usr/bin/env bash
# Download the DFL / Sportec open dataset (Bassek et al., Scientific Data 2025, CC-BY 4.0)
# from its Hugging Face mirror into external/dfl/: match information, human-annotated events and
# tracking. ~2.6 GB. Idempotent: skips finished files.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/dfl"
BASE="https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
mkdir -p "$DEST"
for match in J03WMX:000001 J03WN1:000001 J03WOH:000002 J03WOY:000002 J03WPY:000002 J03WQQ:000002 J03WR9:000002; do
  id="${match%%:*}"; com="${match##*:}"
  for kind in "02_01_matchinformation" "03_02_events_raw" "04_03_positions_raw_observed"; do
    name="DFL_${kind}_DFL-COM-${com}_DFL-MAT-${id}.xml"
    [ -s "$DEST/$name" ] && continue
    curl -sSfL -o "$DEST/$name.part" "$BASE/$name"
    mv "$DEST/$name.part" "$DEST/$name"
  done
done
echo "done: $(ls "$DEST"/*.xml | wc -l) files in $DEST"
