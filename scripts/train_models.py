"""Train and store every model dfl_sup_v10 (or, with --extra-features, dfl_sup_v11) uses, from the public DFL data only (no competition data).

Writes models/*.pkl (possession/model_store.py), which make_submission.py --model-dir models then
loads instead of training. The defaults mirror make_submission.py's for dfl_sup_v10.

    python scripts/train_models.py [--networks light_symsw] [--model-dir models] [--extra-features]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession import model_store  # noqa: E402
from possession.pipeline import PipelineConfig, dfl_models, sequence_models, turnover_model  # noqa: E402
from possession.proxy import PRIMARY_TAU  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--networks", default="light_symsw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--extra-features", action="store_true", help="the models of dfl_sup_v11")
    args = parser.parse_args()
    model_store.DIRECTORY = os.path.join(ROOT, args.model_dir)
    config = PipelineConfig(noise="n3", labels="ev", turnover_gamma=1.0, sequence=True, networks=args.networks,
                            extra_features=args.extra_features)
    for context in (False, True):
        dfl_models(PRIMARY_TAU, context, config.noise, config.labels, config.extra_metrica, config.extra_features)
        print("classifiers (context {}) ready".format(context), flush=True)
    turnover_model(config, PRIMARY_TAU)
    print("turnover detector ready", flush=True)
    sequence_models(config, PRIMARY_TAU)
    print("networks ready in {}".format(model_store.DIRECTORY), flush=True)
