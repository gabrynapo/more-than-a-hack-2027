"""Turn per-frame probabilities into possessions: a three-state HMM decoded with Viterbi.

Frame-by-frame argmax flickers, and every flicker is a false possession that spellF1 punishes. An
HMM makes each switch pay a fixed price, so a run of frames has to argue for a change together.

Two quantities set that trade-off, and both come from football rather than from tuning on the
public window:

    expected durations   a possession or a stoppage lasts ~15 s, so a state is left with
                         probability 1 / (15 * 25) per frame
    EVIDENCE_WEIGHT      consecutive frames are not independent observations -- the tracking
                         barely changes in 40 ms -- so log-evidence is scaled down by roughly the
                         number of frames that carry one independent look (~10, i.e. 0.4 s)
"""

import numpy as np

# Params
FPS = 25
MEAN_DURATION_S = {"H": 15.0, "A": 15.0, "D": 15.0}
# Where a live possession goes when it ends: mostly to the other side, sometimes to a stoppage.
TO_OTHER_SIDE, TO_DEAD = 0.75, 0.25
EVIDENCE_WEIGHT = 0.1
EPS = 1e-6
STATES = np.array(["H", "A", "D"])


def transition_matrix():
    leave = {state: 1.0 / (MEAN_DURATION_S[state] * FPS) for state in STATES}
    matrix = np.array([
        [1 - leave["H"], leave["H"] * TO_OTHER_SIDE, leave["H"] * TO_DEAD],
        [leave["A"] * TO_OTHER_SIDE, 1 - leave["A"], leave["A"] * TO_DEAD],
        [leave["D"] * 0.5, leave["D"] * 0.5, 1 - leave["D"]],
    ])
    return np.log(matrix)


MAX_SWITCH = 0.5        # a frame-dependent switch probability is capped here


def scaled_transitions(scale):
    """(n, 3, 3) log transition matrices whose H <-> A entries are multiplied by `scale` per frame.

    scale[t] applies to the step from frame t - 1 to frame t (possession/turnover.py). Leaving for a
    stoppage keeps its fixed probability; staying takes what is left.
    """
    base = np.exp(transition_matrix())
    matrices = np.repeat(base[None], len(scale), axis=0)
    for here, other in ((0, 1), (1, 0)):
        switch = np.minimum(base[here, other] * scale, MAX_SWITCH)
        matrices[:, here, other] = switch
        matrices[:, here, here] = 1 - switch - base[here, 2]
    return np.log(matrices)


def viterbi(proba, prior, evidence_weight=EVIDENCE_WEIGHT, switch_scale=None):
    """Most likely state sequence (indices into STATES) for one contiguous block of frames.

    Emissions are the classifier's posteriors divided by its training prior -- the scaled
    likelihood trick from hybrid HMMs -- raised to `evidence_weight`. `switch_scale` (per frame,
    optional) makes the H <-> A switching cost depend on the frame; None is the fixed-cost HMM.
    """
    log_emission = evidence_weight * (np.log(proba + EPS) - np.log(prior))
    fixed = transition_matrix()
    varying = None if switch_scale is None else scaled_transitions(np.asarray(switch_scale, dtype=float))
    n = len(proba)
    back = np.zeros((n, 3), dtype=np.int8)
    score = np.log(np.full(3, 1.0 / 3)) + log_emission[0]
    for t in range(1, n):
        log_transition = fixed if varying is None else varying[t]
        candidates = score[:, None] + log_transition          # from x to
        back[t] = np.argmax(candidates, axis=0)
        score = candidates[back[t], np.arange(3)] + log_emission[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def posterior(proba, prior, evidence_weight=EVIDENCE_WEIGHT):
    """(n, 3) state posteriors for one block of frames: forward-backward on the same model.

    Viterbi picks the single most likely sequence; when a short possession is likely but has many
    possible start and end points, each one of them is less likely than "no possession at all", and
    Viterbi drops it. The per-frame posterior sums over all of them.
    """
    from scipy.special import logsumexp
    log_emission = evidence_weight * (np.log(proba + EPS) - np.log(prior))
    log_transition = transition_matrix()
    n = len(proba)
    forward = np.empty((n, 3))
    backward = np.zeros((n, 3))
    forward[0] = np.log(1.0 / 3) + log_emission[0]
    for t in range(1, n):
        forward[t] = logsumexp(forward[t - 1][:, None] + log_transition, axis=0) + log_emission[t]
    for t in range(n - 2, -1, -1):
        backward[t] = logsumexp(log_transition + (log_emission[t + 1] + backward[t + 1])[None, :], axis=1)
    joint = forward + backward
    return np.exp(joint - logsumexp(joint, axis=1, keepdims=True))


def decode(proba, half, prior, evidence_weight=EVIDENCE_WEIGHT, method="viterbi", switch_scale=None):
    """Labels for every frame: H / A / D inside the graded halves, X outside.

    method "viterbi" (most likely sequence) or "posterior" (most likely state per frame).
    switch_scale: optional per-frame multiplier of the H <-> A switch probability (Viterbi only).
    """
    if switch_scale is not None and method != "viterbi":
        raise ValueError("a frame-dependent switching cost is implemented for Viterbi only")
    labels = np.full(len(proba), "X", dtype="<U1")
    for number in (1, 2):
        index = np.where(half == number)[0]
        if len(index) == 0:
            continue
        if method == "viterbi":
            scale = None if switch_scale is None else np.asarray(switch_scale)[index]
            labels[index] = STATES[viterbi(proba[index], prior, evidence_weight, scale)]
        elif method == "posterior":
            labels[index] = STATES[np.argmax(posterior(proba[index], prior, evidence_weight), axis=1)]
        else:
            raise ValueError("unknown decode method: {}".format(method))
    return labels
