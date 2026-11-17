"""Data in, submission out: features -> frame classifiers -> HMM -> CSV.

    bash scripts/fetch_dfl.sh                       # once; public DFL training data
    python scripts/build_features.py                # once; caches the competition feature table
    python scripts/make_submission.py               # writes evaluation/results/dfl_sup_v9.csv

Then validate and score it against the public window (a correctness check, not a selection tool):

    python evaluation/validate_submission.py evaluation/results/dfl_sup_v9.csv
    python evaluation/run_evaluation.py

The defaults are the configuration chosen on the DFL validation proxy (docs/MIGLIORAMENTI_SPELLF1.md,
sections 7-15): classifiers trained on the DFL open data's real labels, degraded with the n3 noise
(calibrated on label-free statistics of the match) and domain-randomised around it; dead ball from
the plain classifiers, the H/A split from classifiers with past/future context (sup+teamcx);
training labels in which short possessions backed by a DFL control event are kept (--labels ev,
possession/dfl_events.py); Viterbi decoding at evidence weight 0.25 with the H/A switching cost set
frame by frame by a turnover detector (--turnover-gamma 1.0, possession/turnover.py); the
probabilities are averaged with the mean of four temporal convolutional networks' (--sequence
--networks all4, possession/sequence.py); no touch snapping (sha256 aaa92d9c..., about 70 min). Other versions
(FIXED = --turnover-gamma 0 --evidence-weight 0.35 --no-sequence):

    python scripts/make_submission.py --name dfl_sup_v11 --networks light_symsw --extra-features   # submission
    python scripts/make_submission.py --name dfl_sup_v10 --networks light_symsw   # previous submission
    python scripts/make_submission.py --name dfl_sup_v8 --networks base   # previous default (f976dd76...)
    python scripts/make_submission.py --name dfl_sup_v6 --no-sequence     # previous default (963dbcf1...)

    python scripts/make_submission.py --name dfl_sup_v5 FIXED                # previous default (1bee7698...)
    python scripts/make_submission.py --name dfl_sup_v4 --labels tau FIXED   # (3650f97f...)
    python scripts/make_submission.py --name dfl_sup_v3 --source sup --labels tau FIXED   # (2ef78eed...)
    python scripts/make_submission.py --name dfl_sup_v1 --source sup --noise n2 --labels tau FIXED  # (01631bdb...)
    python scripts/make_submission.py --name dfl_sup_v2 --source sup --noise n2 --context --labels tau FIXED
    python scripts/make_submission.py --name hyb_v1 --source hybrid --labels tau FIXED   # dead ball from self-training
    python scripts/make_submission.py --name hmm_v7 --source self --evidence-weight 0.1 --turnover-gamma 0 --refine

v2 (context) wins on the proxy but is not used: on the competition match it loses dead balls far
beyond anything the proxy produces (docs/MIGLIORAMENTI_SPELLF1.md, section 6).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.features import build, cache_path  # noqa: E402
from possession.labeller import UNLABELLED  # noqa: E402
from possession.model import training_mask  # noqa: E402
from possession.labeller import pseudo_labels  # noqa: E402
from possession.pipeline import SOURCES, PipelineConfig, decode_labels, source_proba, turnover_scale  # noqa: E402
from possession.proxy import PRIMARY_TAU  # noqa: E402
from possession.split import SplitConfig  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")


def load_features(swap=False):
    path = cache_path(swap=swap)
    if os.path.isfile(path):
        return pd.read_parquet(path)
    features = build(swap=swap)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    features.to_parquet(path)
    return features


def add_extra_features(features):
    """`features` with the columns of possession/extra_features.py, computed from the same
    tracking and ball track (broadcast-filled) the feature table was built from. Cached next to it."""
    from possession.extra_features import extra_columns
    from possession.features import GAME, USE_BROADCAST, half_index, load_tracking, merged_ball
    path = cache_path().replace("_features.parquet", "_extra_features.parquet")
    if os.path.isfile(path):
        extra = pd.read_parquet(path)
    else:
        tracking = load_tracking("tactical", GAME)
        ball, _ = merged_ball(tracking, USE_BROADCAST)
        extra = extra_columns(dict(tracking, ball=ball), half_index(len(ball), GAME), features)
        extra.to_parquet(path)
    return pd.concat([features.reset_index(drop=True), extra], axis=1)


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def parse_split(text):
    if not text:
        return None
    fine, threshold, min_frames = text.split(",")
    return SplitConfig(fine_weight=float(fine), threshold=float(threshold), min_frames=int(min_frames))


def possession_count(labels):
    graded = labels[labels != "X"]
    alive = graded != "D"
    starts = alive & np.concatenate(([True], graded[1:] != graded[:-1]))
    return int(starts.sum())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="dfl_sup_v9")
    parser.add_argument("--evidence-weight", type=float, default=0.25,
                        help="scale of the frame evidence in the HMM (decode.py)")
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=False,
                        help="snap boundaries onto touches (helps the self-trained source only)")
    parser.add_argument("--source", choices=SOURCES, default="sup+teamcx",
                        help="where the frame probabilities come from (see possession/pipeline.py)")
    parser.add_argument("--method", choices=("viterbi", "posterior"), default="viterbi")
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=False,
                        help="past/future context features for the supervised sources (context.py)")
    parser.add_argument("--noise", choices=("n2", "n3"), default="n3",
                        help="DFL noise preset the supervised models train on (possession/proxy.py)")
    parser.add_argument("--labels", choices=("tau", "ev"), default="ev",
                        help="DFL training labels: duration rule or event-backed control (dfl_events.py)")
    parser.add_argument("--split", default=None, metavar="FINE,THRESHOLD,MIN_FRAMES",
                        help="re-open short turnovers merged by the HMM (possession/split.py), e.g. 1.0,0.6,12")
    parser.add_argument("--turnover-gamma", type=float, default=1.0,
                        help="> 0: a turnover detector sets the H/A switching cost (possession/turnover.py)")
    parser.add_argument("--extra-metrica", action=argparse.BooleanOptionalAction, default=False,
                        help="add Metrica's public sample matches to training (possession/metrica_sample.py)")
    parser.add_argument("--extra-features", action=argparse.BooleanOptionalAction, default=False,
                        help="append the four extra feature families (possession/extra_features.py, "
                             "docs/MIGLIORAMENTI_SPELLF1.md section 34); retrains every model")
    parser.add_argument("--sequence", action=argparse.BooleanOptionalAction, default=True,
                        help="sup+teamcx: average in the temporal network (possession/sequence.py)")
    parser.add_argument("--networks", choices=("base", "all4", "light_symsw", "all4_symsw"), default="all4",
                        help="with --sequence: one network (base), the mean of four (all4), or the _symsw sets: "
                             "symmetry copies and spell-balanced loss (sequence.ENSEMBLES)")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=False,
                        help="with --sequence: each network also reads the match with the teams swapped "
                             "and averages the two answers (possession/augment.py)")
    parser.add_argument("--model-dir", default=None,
                        help="load the trained models from this folder, or train and save them there "
                             "(possession/model_store.py; DFL-trained, no competition data)")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for training (default 1: bit-for-bit reproducible). More is "
                             "faster but not bit-for-bit -- for inspection runs, never the submission")
    parser.add_argument("--tau", type=float, default=PRIMARY_TAU,
                        help="DFL label definition for the supervised sources (dfl.controlled_labels)")
    args = parser.parse_args()

    start = time.time()
    config = PipelineConfig(use_refine=args.refine, method=args.method,
                            evidence_weight=args.evidence_weight, context=args.context,
                            noise=args.noise, split=parse_split(args.split), labels=args.labels,
                            turnover_gamma=args.turnover_gamma, extra_metrica=args.extra_metrica,
                            sequence=args.sequence, networks=args.networks,
                            extra_features=args.extra_features)
    if args.extra_features and (args.tta or args.extra_metrica):
        parser.error("--extra-features does not combine with --tta or --extra-metrica")
    if args.threads:
        from possession import sequence, supervised
        supervised.THREADS = sequence.THREADS = args.threads
        print("training on {} threads: not bit-for-bit reproducible".format(args.threads))
    if args.model_dir:
        from possession import model_store
        model_store.DIRECTORY = os.path.join(ROOT, args.model_dir)
    features = load_features()
    if config.extra_features:
        features = add_extra_features(features)
        print("features: + the extra columns of section 34 ({} columns in all)".format(features.shape[1]))
    labels = pseudo_labels(features, config.kickoff_frames)
    mask = training_mask(features, labels)
    print("pseudo-labels: {} frames, {} usable for self-training (H {} / A {} / D {})".format(
        int((labels != UNLABELLED).sum()), int(mask.sum()),
        *[int((labels[mask] == k).sum()) for k in range(3)]))
    print("source {}{} (noise {}, labels {}), decoder {} (evidence weight {}, turnover gamma {}), "
          "snapping {}, split {}".format(
              args.source, " + context" if config.context else "", config.noise, config.labels, config.method,
              config.evidence_weight, config.turnover_gamma, "on" if config.use_refine else "off",
              args.split or "off"))
    if config.extra_metrica:
        print("training data: 7 DFL matches + Metrica's 3 sample matches")
    if config.sequence:
        print("probabilities: mean of sup+teamcx and the temporal network ({})".format(config.networks))
    swapped = load_features(swap=True) if args.tta else None
    if args.tta:
        print("networks: averaged with their reading of the match with the teams swapped")
    proba, model_proba = source_proba(features, args.source, config, args.tau, swapped)
    # Kept next to the features (gitignored) for diagnostics; derived from tracking, never published.
    np.save(os.path.join(os.path.dirname(cache_path()), args.name + "_proba.npy"), proba)
    submission = decode_labels(proba, model_proba, features, config, turnover_scale(features, config, args.tau))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, args.name + ".csv")
    write_submission(path, submission)
    graded = submission[submission != "X"]
    print("{} graded frames: H {} / A {} / D {}; {} possessions".format(
        len(graded), *[int((graded == s).sum()) for s in "HAD"], possession_count(submission)))
    print("wrote {} in {:.0f}s".format(os.path.relpath(path, ROOT), time.time() - start))
