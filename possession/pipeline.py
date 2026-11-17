"""The possession pipeline as functions, so the competition match and the DFL proxy run the same code.

    features -> pseudo-labels -> classifiers -> rule blend -> HMM -> touch snapping -> labels

`scripts/make_submission.py` calls this on the competition match; `scripts/proxy_eval.py` calls it on
degraded DFL matches, where every frame has a label and the result can be scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from possession.decode import EVIDENCE_WEIGHT, decode
from possession.labeller import KICKOFF_FRAMES, UNLABELLED, pseudo_labels, whistle_stops
from possession.model import PUBLIC_WINDOW, predict_proba, train
from possession.refine import drop_fragments, refine, touch_events, undo_foul_regains
from possession.split import SplitConfig, insert_turnovers

RULE_WEIGHT = 0.5      # share of a ruled frame's probability that comes from the rule
STATES = np.array(["H", "A", "D"])


@dataclass
class PipelineConfig:
    evidence_weight: float = EVIDENCE_WEIGHT
    rule_weight: float = RULE_WEIGHT
    use_refine: bool = True
    method: str = "viterbi"           # decode.decode: "viterbi" or "posterior"
    context: bool = False             # supervised sources: add past/future context (context.py)
    noise: str = "n2"                 # supervised sources: DFL noise preset (proxy.NOISE_PRESETS)
    split: SplitConfig | None = None  # re-open short turnovers merged by the HMM (split.py)
    labels: str = "tau"               # DFL training labels: "tau" (duration rule) or "ev" (events)
    turnover_gamma: float = 0.0       # > 0: H <-> A switching cost set by the turnover detector
    extra_metrica: bool = False       # add Metrica's sample matches to training (metrica_sample.py)
    extra_features: bool = False      # append the columns of extra_features.py (section 34)
    sequence: bool = False            # sup+teamcx: average in the temporal network (sequence.py)
    networks: str = "base"            # with sequence: which set of networks (sequence.ENSEMBLES)
    kickoff_frames: dict = field(default_factory=lambda: dict(KICKOFF_FRAMES))
    exclude_window: tuple | None = PUBLIC_WINDOW


def self_trained_proba(features, config: PipelineConfig):
    """(proba, model_proba, rule_labels): classifiers trained on the match's own pseudo-labels.

    `proba` blends the rules back in (they are high-precision evidence of their own);
    `model_proba` is the classifiers alone, which is what boundary snapping reads.
    """
    labels = pseudo_labels(features, config.kickoff_frames)
    models = train(features, labels, config.exclude_window)
    model_proba = predict_proba(models, features)
    return blend_rules(model_proba, labels, config.rule_weight), model_proba, labels


def blend_rules(model_proba, rule_labels, rule_weight):
    proba = model_proba.copy()
    ruled = rule_labels != UNLABELLED
    proba[ruled] = (1 - rule_weight) * proba[ruled] + rule_weight * np.eye(3)[rule_labels[ruled]]
    return proba


def decode_labels(proba, model_proba, features, config: PipelineConfig, switch_scale=None):
    """Labels per frame: H / A / D in the graded halves, X elsewhere.

    switch_scale: optional per-frame multiplier of the H <-> A switch probability (turnover_scale).
    """
    half = features["half"].values.astype(int)
    # class_weight="balanced" trains against a flat prior, so that is what the posteriors divide by.
    prior = np.full(3, 1.0 / 3)
    labels = decode(proba, half, prior, evidence_weight=config.evidence_weight, method=config.method,
                    switch_scale=switch_scale)
    if config.split is not None:
        labels = insert_turnovers(labels, proba, half, prior, config.split)
    if config.use_refine:
        # Edges are placed on the model's own probabilities: the rule labels are deliberately
        # blunt about where a possession starts (a carry fills the whole gap between controls).
        labels = refine(labels, model_proba, half, touch_events(features))
        labels = drop_fragments(labels, proba, half)
        labels = undo_foul_regains(labels, whistle_stops(features), half)
    return force_pre_kickoff_dead(labels, half, config.kickoff_frames)


def force_pre_kickoff_dead(labels, half, kickoff_frames):
    """Graded frames before a published kickoff are dead ball (FAQ.md: 110,808-113,741 are all D)."""
    out = labels.copy()
    index = np.arange(len(out))
    for number, kickoff in kickoff_frames.items():
        out[(half == number) & (index < kickoff)] = "D"
    return out


def run(features, config: PipelineConfig | None = None):
    """(labels, proba) for one match, self-trained probabilities (hmm_v7)."""
    config = config or PipelineConfig()
    proba, model_proba, _ = self_trained_proba(features, config)
    return decode_labels(proba, model_proba, features, config), proba


# -- probabilities from the DFL open data -----------------------------------------------------

SOURCES = ("self", "sup", "sup+rules", "mix", "hybrid", "sup+teamcx")


def hybrid_proba(alive_dead_proba, team_proba):
    """Dead-ball probability from one model, the H-vs-A split of live play from another.

    P(D) = P_a(D);  P(H) = (1 - P_a(D)) * P_t(H) / (P_t(H) + P_t(A));  P(A) likewise.

    Why: on the competition match the DFL-trained classifiers name the team well but miss dead
    balls (public-window dead-ball recall 0.66, while every 5-minute window of the DFL proxy has
    >= 0.80 -- the proxy cannot represent how the ATD loses the ball at stoppages), whereas the
    self-trained classifiers, built on the ATD's own signals and the Smart Tagging, get dead balls
    right (0.98) and the team worse. scripts/proxy_window_noise.py documents the diagnosis.
    """
    live = team_proba[:, 0] + team_proba[:, 1]
    share_h = np.where(live > 0, team_proba[:, 0] / np.maximum(live, 1e-12), 0.5)
    dead = alive_dead_proba[:, 2]
    return np.stack([(1 - dead) * share_h, (1 - dead) * (1 - share_h), dead], axis=1)


def dfl_labels(match_id: str, raw, tau: float, labels: str = "tau"):
    """DFL training labels: the duration rule, or the event-backed rule (dfl_events.event_labels)."""
    from possession import dfl, dfl_events
    if labels == "tau":
        return dfl.controlled_labels(raw, tau)
    if labels == "ev":
        return dfl_events.event_labels(raw, dfl_events.control_frames(dfl_events.events(match_id)), tau)
    raise ValueError("unknown DFL labels: {}".format(labels))


def model_name(base: str, extra_metrica: bool, extra_features: bool) -> str:
    """Model-store name: `base` + _m0/_m1, + _xf with the extra features (so the committed v10
    models keep their names)."""
    return "{}_m{}{}".format(base, int(extra_metrica), "_xf" if extra_features else "")


def check_extras(extra_metrica: bool, extra_features: bool):
    if extra_metrica and extra_features:
        raise ValueError("extra_features needs the tracking of every training match; "
                         "Metrica's sample matches are not wired for it")


def dfl_models(tau: float, context: bool = False, noise: str = "n2", labels: str = "tau",
               extra_metrica: bool = False, extra_features: bool = False):
    """The classifiers of _dfl_training_matches, through possession.model_store."""
    from possession import model_store, supervised
    check_extras(extra_metrica, extra_features)
    name = model_name("sup_tau{}_cx{}_{}_{}".format(tau, int(context), noise, labels), extra_metrica, extra_features)
    return model_store.get(name, lambda: supervised.train(
        _dfl_training_matches(tau, context, noise, labels, extra_metrica, extra_features)))


def _dfl_training_matches(tau: float, context: bool = False, noise: str = "n2", labels: str = "tau",
                          extra_metrica: bool = False, extra_features: bool = False):
    """Training matches of the supervised classifiers: all seven DFL matches, each under every
    TRAINING_NOISE setting (domain randomisation). Needs scripts/fetch_dfl.sh. Deterministic: fixed noise seeds,
    fixed classifier seed, no early stopping, one thread. `context` adds the past/future columns;
    `extra_metrica` adds Metrica's sample matches (scripts/fetch_metrica_sample.sh)."""
    from possession import dfl, supervised
    from possession.context import add_context
    from possession.proxy import training_copies
    matches = []
    for match_id in dfl.MATCH_IDS:
        matches += [(add_context(features) if context else features, dfl_labels(match_id, raw, tau, labels))
                    for features, raw in training_copies(match_id, noise, extra_features)]
    if extra_metrica:
        from possession import metrica_sample
        inputs = {}                      # each match repeats: build its context columns once
        for features, labels_ in metrica_sample.training_matches():
            if id(features) not in inputs:
                inputs[id(features)] = add_context(features) if context else features
            matches.append((inputs[id(features)], labels_))
    return matches


def turnover_model(config: PipelineConfig, tau: float):
    """The turnover detector (possession/turnover.py) trained on all seven DFL matches, through
    possession.model_store."""
    from possession import dfl, model_store, turnover
    from possession.proxy import training_copies

    def rows():
        out = [turnover.training_rows(f, dfl_labels(match_id, raw, tau, config.labels))
               for match_id in dfl.MATCH_IDS
               for f, raw in training_copies(match_id, config.noise, config.extra_features)]
        if config.extra_metrica:
            from possession import metrica_sample
            out += [turnover.training_rows(f, labels) for f, labels in metrica_sample.training_matches()]
        return out

    check_extras(config.extra_metrica, config.extra_features)
    name = model_name("turnover_tau{}_{}_{}".format(tau, config.noise, config.labels),
                      config.extra_metrica, config.extra_features)
    return model_store.get(name, lambda: turnover.train(rows()))


def turnover_scale(features, config: PipelineConfig, tau: float):
    """Per-frame H <-> A switch multiplier from the turnover detector (turnover_model), or None
    when config.turnover_gamma is 0."""
    if config.turnover_gamma == 0:
        return None
    from possession import turnover
    from possession.context import add_context
    detector = turnover.predict(turnover_model(config, tau), add_context(features))
    return turnover.switch_scale(detector, config.turnover_gamma)


def sequence_models(config: PipelineConfig, tau: float):
    """The temporal networks of config.networks (possession/sequence.py), trained on the seven DFL
    matches under every training noise (plus their symmetry copies for sequence.WITH_SYMMETRY, plus
    Metrica's sample matches with extra_metrica), through possession.model_store."""
    from possession import dfl, model_store, sequence
    from possession.proxy import training_copies
    built = []

    def matches():
        if built:
            return built[0]
        out = [(f, dfl_labels(match_id, raw, tau, config.labels))
               for match_id in dfl.MATCH_IDS
               for f, raw in training_copies(match_id, config.noise, config.extra_features)]
        if config.networks in sequence.WITH_SYMMETRY:
            from possession.augment import swap_labels
            from possession.proxy import symmetry_copies
            for match_id in dfl.MATCH_IDS:
                for f, raw, swapped_teams in symmetry_copies(match_id, config.noise, config.extra_features):
                    labels = dfl_labels(match_id, raw, tau, config.labels)
                    out.append((f, swap_labels(labels) if swapped_teams else labels))
        if config.extra_metrica:
            from possession import metrica_sample
            out += metrica_sample.training_matches()
        built.append(out)
        return out

    check_extras(config.extra_metrica, config.extra_features)
    models = []
    for index, net_config in enumerate(sequence.ENSEMBLES[config.networks]):
        name = model_name("net_{}_{}_tau{}_{}_{}".format(config.networks, index, tau, config.noise, config.labels),
                          config.extra_metrica, config.extra_features)
        models.append(model_store.get(name, lambda c=net_config: sequence.train(matches(), c)))
    return models


def sequence_proba(features, config: PipelineConfig, tau: float, swapped=None):
    """P(H, A, D) per frame: the mean of the networks of sequence_models. `swapped`: features of the
    same match with the teams swapped (possession/augment.py); each network then averages its two
    readings."""
    from possession import sequence
    probas = [sequence.predict_proba_tta(model, features, swapped) for model in sequence_models(config, tau)]
    return probas[0] if len(probas) == 1 else np.mean(probas, axis=0)


def source_proba(features, source: str, config: PipelineConfig, tau: float, swapped=None):
    """(proba, model_proba) from one of SOURCES -- see scripts/proxy_experiments.py:

    self       classifiers self-trained on the match's pseudo-labels, rules blended in (hmm_v7)
    sup        classifiers trained on DFL real labels
    sup+teamcx dead ball from sup, the H/A split from sup with past/future context
    sup+rules  sup with the match's own rule labels blended in
    mix        mean of self and sup

    swapped    sup+teamcx with config.sequence only: features of the match with the teams swapped,
               for the networks' swapped-team averaging (sequence_proba)
    """
    from possession import supervised
    if source not in SOURCES:
        raise ValueError("unknown source {}; expected one of {}".format(source, SOURCES))
    if source == "self":
        proba, model_proba, _ = self_trained_proba(features, config)
        return proba, model_proba
    from possession.context import add_context
    if source == "sup+teamcx":
        plain = supervised.predict_proba(dfl_models(tau, False, config.noise, config.labels, config.extra_metrica,
                                         config.extra_features),
                                         features)
        team = supervised.predict_proba(dfl_models(tau, True, config.noise, config.labels, config.extra_metrica,
                                         config.extra_features),
                                        add_context(features))
        mixed = hybrid_proba(plain, team)
        if config.sequence:
            mixed = (mixed + sequence_proba(features, config, tau, swapped)) / 2
        return mixed, mixed
    inputs = add_context(features) if config.context else features
    sup = supervised.predict_proba(dfl_models(tau, config.context, config.noise, config.labels,
                                              config.extra_metrica, config.extra_features), inputs)
    if source == "sup":
        return sup, sup
    if source == "sup+rules":
        rules = pseudo_labels(features, config.kickoff_frames)
        return blend_rules(sup, rules, config.rule_weight), sup
    proba, model_proba, _ = self_trained_proba(features, config)
    if source == "hybrid":
        return hybrid_proba(proba, sup), hybrid_proba(model_proba, sup)
    return (proba + sup) / 2, (model_proba + sup) / 2


# -- the official baseline, on arrays ---------------------------------------------------------

def nearest_team_codes(tracking):
    """Per frame: 0 / 1 for the team of the player nearest the ball, -1 without a ball.

    The same rule as starter/baseline.py, computed on a tracking dict instead of the raw file.
    """
    distance = np.linalg.norm(tracking["pos"] - tracking["ball"][:, None, :], axis=2)
    safe = np.where(np.isnan(distance), np.inf, distance)
    nearest = np.argmin(safe, axis=1)
    found = np.isfinite(safe[np.arange(len(safe)), nearest])
    return np.where(found, tracking["team"][nearest], -1).astype(np.int8)


def baseline_labels(tracking, window):
    """starter/baseline.py's centred majority vote of the nearest team, D where nobody has the ball."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "starter"))
    from baseline import smooth
    return smooth(nearest_team_codes(tracking), window)
