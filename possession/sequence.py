"""A temporal convolutional network over the frame features: P(H), P(A), P(D) from the sequence.

The gradient-boosting classifiers see one frame at a time, plus a few hand-made rolling means and
past/future windows (features.py, context.py). Whether a one-second H between two A spells is a
real possession or a rebound depends on what happens over several seconds around it, in ways those
hand-made windows only approximate. A temporal convolutional network learns its own windows: a stack
of dilated, non-causal convolutions (as in MS-TCN, Farha & Gall, CVPR 2019) whose receptive field
grows with every layer.

    input     the features the classifiers use (no Smart Tagging) plus the ball-known flag,
              averaged over steps of STEP frames (0.2 s), standardised, missing values -> 0
    network   1x1 projection, LAYERS residual blocks with dilations 1, 2, 4, ... (receptive
              field ~ +/-25 s at the defaults), 1x1 output to three logits
    training  random crops of CHUNK steps from the halves of the training matches, class-balanced
              cross-entropy, Adam, a fixed number of iterations (no early stopping, nothing tuned
              on held-out data)
    output    per-frame probabilities (each step's softmax repeated over its frames), decoded by the
              same HMM as the other sources

Determinism. Every random draw comes from seeded generators, deterministic algorithms are enforced,
and the thread count is fixed (THREADS, 1 for the submission). The generic CPU code paths are
selected before torch loads (ATEN_CPU_CAPABILITY, MKL_CBWR), so the arithmetic does not depend on
which vector instructions the machine has.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from possession.model import ball_known, feature_columns  # noqa: E402

STEP = 5                    # frames per network step (0.2 s)
THREADS = 1                 # None: leave torch's default (measurement only, not bit-for-bit)
CODES = {"H": 0, "A": 1, "D": 2}


@dataclass(frozen=True)
class SequenceConfig:
    channels: int = 64
    layers: int = 7             # dilations 1..64: receptive field 1 + 2 * 127 steps (~ +/-25 s)
    kernel: int = 3
    dropout: float = 0.2
    chunk: int = 1024           # training crop, in steps (~205 s)
    batch: int = 16
    iterations: int = 1500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    stages: int = 1             # > 1: refinement stages read the previous stage's softmax (MS-TCN)
    refine_channels: int = 32
    refine_layers: int = 10     # dilations 1..512 in the refinement stages
    smooth_weight: float = 0.0  # MS-TCN's truncated MSE on log-probabilities of adjacent steps
    short_weight: float = 1.0   # > 1: steps of short possessions weigh up to this much more (spell_weights)
    short_ref_s: float = 5.0    # possessions at least this long keep weight 1


# Named sets of networks whose probabilities are averaged (docs/MIGLIORAMENTI_SPELLF1.md, section 15):
# "base" is v8's single network, "all4" adds another seed, a bigger network and a multi-stage one (v9).
# The "_symsw" sets are the same networks with the recipe of section 21: trained on the symmetry
# copies as well (possession/proxy.py: symmetry_copies), for twice the iterations, with the
# spell-balanced loss.
ENSEMBLES = {
    "base": (SequenceConfig(),),
    "all4": (SequenceConfig(),
             SequenceConfig(seed=1),
             SequenceConfig(channels=96, iterations=3000),
             SequenceConfig(stages=3, smooth_weight=0.15, iterations=3000)),
}


def with_recipe(config: SequenceConfig) -> SequenceConfig:
    return replace(config, iterations=2 * config.iterations, short_weight=4.0, short_ref_s=5.0)


ENSEMBLES["light_symsw"] = tuple(with_recipe(c) for c in ENSEMBLES["all4"][:2])
ENSEMBLES["all4_symsw"] = tuple(with_recipe(c) for c in ENSEMBLES["all4"])
WITH_SYMMETRY = ("light_symsw", "all4_symsw")      # sets trained on the symmetry copies too


def input_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in feature_columns(features, "seen") if not c.startswith("st_")]


def frame_inputs(features: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """(N, F + 1) float32: the feature columns and the ball-known flag."""
    values = features[columns].values.astype(np.float32)
    return np.concatenate([values, ball_known(features).astype(np.float32)[:, None]], axis=1)


def pool(values: np.ndarray, step: int = STEP) -> np.ndarray:
    """(ceil(n / step), F): NaN-aware mean over consecutive blocks of `step` rows."""
    n = len(values)
    padded = np.full((-(-n // step) * step, values.shape[1]), np.nan, dtype=np.float32)
    padded[:n] = values
    blocks = padded.reshape(-1, step, values.shape[1])
    count = np.sum(~np.isnan(blocks), axis=1)
    total = np.nansum(blocks, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def step_labels(labels: np.ndarray, step: int = STEP) -> np.ndarray:
    """(ceil(n / step),) int64 codes of each block's centre frame (or its last frame), -1 for X."""
    n = len(labels)
    centre = np.minimum(np.arange(0, n, step) + step // 2, n - 1)
    out = np.full(len(centre), -1, dtype=np.int64)
    for name, code in CODES.items():
        out[labels[centre] == name] = code
    return out


def spell_weights(labels: np.ndarray, max_weight: float, ref_s: float, fps: int = 25) -> np.ndarray:
    """(n,) per-frame loss weights that make short possessions count more.

    spellF1 gives a two-second possession the same say as a forty-second one, while a frame-wise
    loss gives it a twentieth. A frame of an H or A run lasting d seconds weighs ref_s / d, clipped
    to [1, max_weight]; dead-ball and ungraded frames weigh 1.
    """
    labels = np.asarray(labels)
    out = np.ones(len(labels), dtype=np.float32)
    if len(labels) == 0:
        return out
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(labels)]))
    for a, b in zip(starts, ends):
        if labels[a] in ("H", "A"):
            out[a:b] = np.clip(ref_s * fps / (b - a), 1.0, max_weight)
    return out


def step_labels_and_weights(labels: np.ndarray, config: "SequenceConfig", step: int = STEP) -> np.ndarray:
    """(ceil(n / step), 2) float32: each block's label code (step_labels) and its loss weight."""
    centre = np.minimum(np.arange(0, len(labels), step) + step // 2, len(labels) - 1)
    weights = spell_weights(labels, config.short_weight, config.short_ref_s)[centre]
    return np.stack([step_labels(labels, step).astype(np.float32), weights], axis=1)


def halves(features: pd.DataFrame) -> list[np.ndarray]:
    half = features["half"].values
    return [np.where(half == number)[0] for number in (1, 2) if (half == number).any()]


class Residual(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mix(torch.relu(self.conv(x))))


class TemporalNet(nn.Module):
    def __init__(self, n_inputs: int, config: SequenceConfig, outputs: int = 3):
        super().__init__()
        self.inp = nn.Conv1d(n_inputs, config.channels, 1)
        self.blocks = nn.ModuleList([Residual(config.channels, 2 ** k, config.kernel, config.dropout)
                                     for k in range(config.layers)])
        self.out = nn.Conv1d(config.channels, outputs, 1)

    def forward(self, x):                        # x: (batch, inputs, steps)
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.out(h)


class MultiStage(nn.Module):
    """MS-TCN: a first stage on the features, then stages that refine the previous softmax."""

    def __init__(self, n_inputs: int, config: SequenceConfig):
        super().__init__()
        refine = replace(config, channels=config.refine_channels, layers=config.refine_layers)
        self.first = TemporalNet(n_inputs, config)
        self.refiners = nn.ModuleList([TemporalNet(3, refine) for _ in range(config.stages - 1)])

    def stages(self, x):
        outputs = [self.first(x)]
        for refiner in self.refiners:
            outputs.append(refiner(torch.softmax(outputs[-1], dim=1)))
        return outputs

    def forward(self, x):
        return self.stages(x)[-1]


def smoothing_loss(logits, targets, clamp: float = 16.0):
    """MS-TCN's truncated MSE between log-probabilities of adjacent valid steps."""
    logp = torch.log_softmax(logits, dim=1)
    diff = (logp[:, :, 1:] - logp[:, :, :-1].detach()) ** 2
    valid = ((targets[:, 1:] >= 0) & (targets[:, :-1] >= 0)).unsqueeze(1).float()
    return (torch.clamp(diff, max=clamp) * valid).sum() / (valid.sum() * 3).clamp(min=1)


@dataclass
class SequenceModel:
    net: nn.Module
    columns: list[str]
    mean: np.ndarray
    std: np.ndarray


def _standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x - mean) / std
    return np.nan_to_num(np.clip(z, -5, 5), nan=0.0).astype(np.float32)


class _threads:
    """Fix torch's thread count inside the block (THREADS), restore it after."""

    def __enter__(self):
        self.before = torch.get_num_threads()
        if THREADS is not None:
            torch.set_num_threads(THREADS)

    def __exit__(self, *exc):
        torch.set_num_threads(self.before)


def _prepared(matches, target_of):
    """(columns, standardised [(steps x inputs, step targets)] per half, mean, std)."""
    columns = input_columns(matches[0][0])
    sequences = []
    for features, labels in matches:
        inputs = frame_inputs(features, columns)
        for index in halves(features):
            sequences.append((pool(inputs[index]), target_of(np.asarray(labels)[index])))
    stacked = np.concatenate([x for x, _ in sequences])
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    mean, std = np.nan_to_num(mean), np.where(np.nan_to_num(std) > 1e-6, np.nan_to_num(std), 1.0)
    return columns, [(_standardise(x, mean, std), y) for x, y in sequences], mean, std


def _crops(rng, sequences, lengths, config: SequenceConfig):
    """One training batch: (inputs, targets) tensors of `batch` random crops of `chunk` steps."""
    xs, ys = [], []
    for k in rng.choice(len(sequences), size=config.batch, p=lengths / lengths.sum()):
        x, y = sequences[k]
        start = int(rng.integers(0, max(len(y) - config.chunk, 0) + 1))
        cx, cy = x[start:start + config.chunk], y[start:start + config.chunk]
        if len(cy) < config.chunk:                     # a half shorter than a crop
            cx = np.pad(cx, ((0, config.chunk - len(cy)), (0, 0)))
            cy = np.pad(cy, ((0, config.chunk - len(cy)),) + ((0, 0),) * (cy.ndim - 1), constant_values=-1)
        xs.append(cx.T)
        ys.append(cy)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def _fit(net, sequences, config: SequenceConfig, rng, loss_of):
    """Adam for `iterations` batches; loss_of(net, inputs, targets) -> scalar loss."""
    lengths = np.array([len(y) for _, y in sequences], dtype=np.float64)
    optimiser = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    net.train()
    for _ in range(config.iterations):
        inputs, target = _crops(rng, sequences, lengths, config)
        loss = loss_of(net, inputs, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    net.eval()
    return net


def train(matches, config: SequenceConfig = SequenceConfig()) -> SequenceModel:
    """Network trained on [(features, H/A/D/X labels)] of several matches."""
    weighted = config.short_weight > 1
    target_of = (lambda labels: step_labels_and_weights(labels, config)) if weighted else step_labels
    columns, sequences, mean, std = _prepared(matches, target_of)
    targets = np.concatenate([y[:, 0] if weighted else y for _, y in sequences])
    counts = np.array([(targets == k).sum() for k in range(3)], dtype=np.float64)
    weights = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    plain_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

    def weighted_loss(logits, target):
        """Class-balanced cross-entropy with a weight per step; target is (batch, steps, 2)."""
        codes, valid = target[..., 0].long(), target[..., 0] >= 0
        losses = nn.functional.cross_entropy(logits, codes.clamp(min=0), reduction="none")
        step_weight = (weights[codes.clamp(min=0)] * target[..., 1])[valid]
        return (losses[valid] * step_weight).sum() / step_weight.sum()

    loss_fn = weighted_loss if weighted else plain_loss

    def loss_of(net, inputs, target):
        if config.stages == 1:
            return loss_fn(net(inputs), target)
        loss = 0
        for logits in net.stages(inputs):
            codes = target[..., 0].long() if weighted else target
            loss = loss + loss_fn(logits, target) + config.smooth_weight * smoothing_loss(logits, codes)
        return loss

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    n_inputs = sequences[0][0].shape[1]
    with _threads():
        net = TemporalNet(n_inputs, config) if config.stages == 1 else MultiStage(n_inputs, config)
        _fit(net, sequences, config, rng, loss_of)
    return SequenceModel(net=net, columns=columns, mean=mean, std=std)


def boundary_targets(labels: np.ndarray, step: int = STEP) -> np.ndarray:
    """(ceil(n / step),) per step: 1 near an H <-> A change (turnover.turnover_targets), 0, -1 on X."""
    from possession.turnover import turnover_targets
    near = turnover_targets(labels)
    centre = np.minimum(np.arange(0, len(labels), step) + step // 2, len(labels) - 1)
    return near[centre].astype(np.int64)


def train_boundary(matches, config: SequenceConfig = SequenceConfig()) -> SequenceModel:
    """A network with one output, "an H <-> A change is near" (ASRF's boundary branch, Ishikawa et
    al. 2021), trained like train() with class-balanced binary cross-entropy, so that its odds mean
    what the turnover detector's do."""
    columns, sequences, mean, std = _prepared(matches, boundary_targets)
    targets = np.concatenate([y for _, y in sequences])
    positive, negative = float((targets == 1).sum()), float((targets == 0).sum())
    pos_weight = torch.tensor(negative / max(positive, 1.0), dtype=torch.float32)

    def loss_of(net, inputs, target):
        logits = net(inputs)[:, 0]
        valid = target >= 0
        weight = torch.where(target == 1, pos_weight, torch.ones_like(pos_weight))[valid]
        loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid].float(), reduction="none")
        return (loss * weight).sum() / weight.sum()

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    with _threads():
        net = _fit(TemporalNet(sequences[0][0].shape[1], config, outputs=1), sequences, config, rng, loss_of)
    return SequenceModel(net=net, columns=columns, mean=mean, std=std)


def predict_boundary(model: SequenceModel, features: pd.DataFrame) -> np.ndarray:
    """(N,) P(an H <-> A change is near) per frame, reweighted to a balanced prior; 0.5 outside the halves."""
    inputs = frame_inputs(features, model.columns)
    out = np.full(len(features), 0.5, dtype=np.float64)
    with _threads(), torch.no_grad():
        for index in halves(features):
            x = _standardise(pool(inputs[index]), model.mean, model.std)
            steps = torch.sigmoid(model.net(torch.from_numpy(x.T[None]))[0, 0]).numpy().astype(np.float64)
            out[index] = np.repeat(steps, STEP)[:len(index)]
    return out


def predict_proba(model: SequenceModel, features: pd.DataFrame) -> np.ndarray:
    """(N, 3) H, A, D per frame; 1/3 each outside the halves."""
    inputs = frame_inputs(features, model.columns)
    out = np.full((len(features), 3), 1.0 / 3, dtype=np.float64)
    with _threads(), torch.no_grad():
        for index in halves(features):
            x = _standardise(pool(inputs[index]), model.mean, model.std)
            logits = model.net(torch.from_numpy(x.T[None]))[0].T
            steps = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            out[index] = np.repeat(steps, STEP, axis=0)[:len(index)]
    return out


def predict_proba_tta(model: SequenceModel, features: pd.DataFrame, swapped: pd.DataFrame | None) -> np.ndarray:
    """predict_proba, averaged with the network's reading of the same match with the teams swapped
    (`swapped`: its features, see possession.augment) when one is given."""
    proba = predict_proba(model, features)
    if swapped is None:
        return proba
    from possession.augment import swap_proba
    return (proba + swap_proba(predict_proba(model, swapped))) / 2


def ensemble_proba(matches, features: pd.DataFrame, configs=ENSEMBLES["base"],
                   swapped: pd.DataFrame | None = None) -> np.ndarray:
    """(N, 3) mean of the probabilities of one network per config, all trained on `matches`."""
    probas = [predict_proba_tta(train(matches, config), features, swapped) for config in configs]
    return probas[0] if len(probas) == 1 else np.mean(probas, axis=0)
