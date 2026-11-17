"""Optional on-disk store of the trained models, so a submission can be regenerated without training.

Off by default: `DIRECTORY` is None and `get` just trains. With a directory set
(make_submission.py --model-dir models), `get(name, train)` loads `<directory>/<name>.pkl` when it
exists and otherwise trains, saves and returns the model. Every stored model is trained on the
public DFL data only (no competition data), so the files can be shared. Pickles are tied to the
pinned library versions of requirements-pipeline.txt.
"""

from __future__ import annotations

import os
import pickle

DIRECTORY: str | None = None


def get(name: str, train):
    """The model called `name`: loaded from DIRECTORY, or `train()`ed (and saved there if set)."""
    if DIRECTORY is None:
        return train()
    path = os.path.join(DIRECTORY, name + ".pkl")
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    model = train()
    os.makedirs(DIRECTORY, exist_ok=True)
    with open(path + ".part", "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path + ".part", path)
    return model
