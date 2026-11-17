"""Download the seven DFL / Sportec open-data matches (CC-BY 4.0) into external/dfl.

The same files as scripts/fetch_dfl.sh, with the standard library only, for Windows. Skips files
already present.

    python scripts/fetch_dfl.py
"""

import shutil
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
MATCHES = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
           "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
KINDS = ("02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed")

if __name__ == "__main__":
    dest = Path(__file__).resolve().parent.parent / "external" / "dfl"
    dest.mkdir(parents=True, exist_ok=True)
    for match, com in MATCHES.items():
        for kind in KINDS:
            name = "DFL_{}_DFL-COM-{}_DFL-MAT-{}.xml".format(kind, com, match)
            target = dest / name
            if target.is_file() and target.stat().st_size > 0:
                continue
            part = target.with_suffix(".xml.part")
            print("downloading", name, flush=True)
            with urllib.request.urlopen("{}/{}".format(BASE, name)) as response, open(part, "wb") as out:
                shutil.copyfileobj(response, out)
            part.replace(target)
    print("done: {} files in {}".format(len(list(dest.glob("*.xml"))), dest))
