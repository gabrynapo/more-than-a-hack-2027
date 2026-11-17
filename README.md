# The DataFellas — ball possession, More Than A Hack 2027 (Phase 1)

Frame-level ball possession (`H` home, `A` away, `D` dead ball, `X` not graded) for the 199,319
frames of the tactical video, from the automatic tracking. Submission: **`dfl_sup_v11`**.
How it works, which assets it uses, the possession rules and the limitations: [REPORT.md](REPORT.md).

The match data is licensed for the competition only and is **not** in this repository.

## Setup

Python 3.11 and the pinned environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-pipeline.txt     # Windows: .venv\Scripts\python.exe
```

Unpack the competition bundle into `data/` as described in [data/README.md](data/README.md)
(`data/tactical/BAR-ATH_tactical_atd.txt`, `..._atd.xml`, `..._smart_tagging.xml`, and the
`data/broadcast/` files), or point `MTAH_DATA_DIR` at the folder that holds it.

## Reproduce the submission

```bash
bash scripts/run_local_v11.sh          # Linux/macOS
scripts\run_local_v11.bat              # Windows
```

It downloads the public DFL / Sportec open data (CC-BY 4.0) into `external/dfl`, trains every model,
writes `evaluation/results/dfl_sup_v11.csv` twice, checks that the two files are identical (and
prints the sha256), validates the file and scores it on the public five minutes. The single command
behind it:

```bash
python scripts/make_submission.py --name dfl_sup_v11 --networks light_symsw --evidence-weight 0.25 --extra-features
```

The models it trains (from the DFL data only, no competition data) are also committed in `models/`.
To load them instead of training — minutes instead of hours, no DFL download:

```bash
scripts\run_local_v11_fast.bat
python scripts/make_submission.py --name dfl_sup_v11 --networks light_symsw --evidence-weight 0.25 --extra-features --model-dir models
```

Training is single-threaded with fixed seeds, so two runs give the same file. `make_submission.py
--threads N` trains faster but not bit for bit: never for the submission. `scripts/train_models.py
--extra-features` retrains the files in `models/`.

Expected sha256 of `dfl_sup_v11.csv`: `8bca9a4bf0ca846775818fc8d707723f82a9e00893184fa26674e85a8d727744`.

## Check and score

```bash
python evaluation/validate_submission.py evaluation/results/dfl_sup_v11.csv
python evaluation/run_evaluation.py         # scores everything in evaluation/results/ on the public 5 minutes
python -m pytest tests                      # unit tests, synthetic data only
```

## Layout

| path | content |
|---|---|
| `possession/` | the pipeline: loading, features, DFL data and noise, labels, classifiers, networks, decoder |
| `scripts/` | `make_submission.py`, `build_features.py`, `train_models.py`, `fetch_dfl.*`, the `run_local_v11*` scripts |
| `models/` | the DFL-trained models of `v11` |
| `evaluation/` | the official scorer, validator and public ground truth (unchanged) |
| `starter/`, `games.py` | official loaders used by the pipeline (unchanged) |
| `tests/` | unit tests on synthetic data |

Some docstrings mention development scripts and notes (`scripts/proxy_*.py`,
`docs/MIGLIORAMENTI_SPELLF1.md`) that stay in our development repository: they hold the validation
experiments behind each choice and are not needed to regenerate the file.

## Licence and attribution

`evaluation/`, `starter/`, `games.py` and `data/README.md` come unchanged from
[metrica-sports/more-than-a-hack-2027](https://github.com/metrica-sports/more-than-a-hack-2027),
© Metrica Sports, MIT licence (see [LICENSE](LICENSE), which also states that the match data is not
covered). The DFL / Sportec open data used for training is CC-BY 4.0 (Bassek et al., Scientific Data
12, 195, 2025) and is downloaded, not redistributed.
