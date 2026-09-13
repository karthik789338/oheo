from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase2b_operation_channel_model import (
    OperationChannelModel,
    jensen_shannon_divergence,
)


PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase2b_ocm_smoke_test"
)


# ------------------------------------------------------------
# Frozen smoke-test settings
# ------------------------------------------------------------

SMOKE_SEED = 11

PRIMARY_NOISE_FRACTION = 0.25

OBSERVATION_DIM = 12
LATENT_STATE_COUNT = 8

BATCH_SIZE = 128

SMOKE_MAX_EPOCHS = 40
SMOKE_PATIENCE = 8
SMOKE_MIN_DELTA = 1e-5

LEARNING_RATE = 1e-3

LAMBDA_TRANS = 0.10
LAMBDA_ROLL = 0.50
LAMBDA_DET = 0.01

INITIAL_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.25

KMEANS_SAMPLE_COUNT = 20000
KMEANS_ITERATIONS = 25

EXPECTED_TRAIN_SEQUENCE_COUNT = 1376
EXPECTED_VAL_SEQUENCE_COUNT = 144

EXPECTED_TRAIN_TRAJECTORY_COUNT = (
    EXPECTED_TRAIN_SEQUENCE_COUNT * 8
)

EXPECTED_VAL_TRAJECTORY_COUNT = (
    EXPECTED_VAL_SEQUENCE_COUNT * 8
)


@dataclass
class TrajectoryRecord:
    trajectory_index: int
    sequence_index: int
    point_start: int
    point_count: int


class ContinuousTrajectoryDataset(Dataset):
    """
    Leakage-safe trajectory dataset.

    This class receives only:
        continuous observations,
        point ranges,
        primitive operation IDs.

    It has no access to symbolic states or structural labels.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        records: list[TrajectoryRecord],
        operations_by_sequence: dict[
            int,
            tuple[int, ...],
        ],
    ) -> None:
        self.vectors = vectors
        self.records = records
        self.operations_by_sequence = (
            operations_by_sequence
        )

    def __len__(self) -> int:
        return len(
            self.records
        )

    def __getitem__(
        self,
        index: int,
    ):
        record = self.records[
            index
        ]

        start = record.point_start
        end = (
            start
            + record.point_count
        )

        observation_path = (
            self.vectors[
                start:end
            ]
        )

        operation_ids = (
            self.operations_by_sequence[
                record.sequence_index
            ]
        )

        if (
            len(operation_ids)
            != record.point_count - 1
        ):
            raise AssertionError(
                "Operation count does not match "
                "trajectory point count."
            )

        return {
            "observations":
                torch.from_numpy(
                    observation_path
                ),

            "operations":
                torch.tensor(
                    operation_ids,
                    dtype=torch.long,
                ),
        }


def collate_trajectories(
    examples,
):
    batch_size = len(
        examples
    )

    maximum_points = max(
        example[
            "observations"
        ].shape[0]
        for example in examples
    )

    observation_dim = (
        examples[0][
            "observations"
        ].shape[1]
    )

    maximum_steps = (
        maximum_points - 1
    )

    observations = torch.zeros(
        batch_size,
        maximum_points,
        observation_dim,
        dtype=torch.float32,
    )

    point_mask = torch.zeros(
        batch_size,
        maximum_points,
        dtype=torch.bool,
    )

    operation_ids = torch.zeros(
        batch_size,
        maximum_steps,
        dtype=torch.long,
    )

    step_mask = torch.zeros(
        batch_size,
        maximum_steps,
        dtype=torch.bool,
    )

    lengths = torch.zeros(
        batch_size,
        dtype=torch.long,
    )

    for index, example in enumerate(
        examples
    ):
        path = example[
            "observations"
        ].float()

        operations = example[
            "operations"
        ]

        point_count = (
            path.shape[0]
        )

        step_count = (
            operations.shape[0]
        )

        observations[
            index,
            :point_count,
        ] = path

        point_mask[
            index,
            :point_count,
        ] = True

        if step_count > 0:
            operation_ids[
                index,
                :step_count,
            ] = operations

            step_mask[
                index,
                :step_count,
            ] = True

        lengths[
            index
        ] = point_count

    return {
        "observations":
            observations,

        "point_mask":
            point_mask,

        "operation_ids":
            operation_ids,

        "step_mask":
            step_mask,

        "lengths":
            lengths,
    }


def set_random_seed(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

        torch.backends.cudnn.deterministic = True


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def load_operation_vocabulary():
    return load_json(
        PHASE2A_DIR
        / "operation_vocabulary.json"
    )


def load_manifest(
    operation_vocabulary,
):
    """
    Load only the leakage-safe Phase 2A manifest.

    Test rows are counted but never returned for training/evaluation.
    """

    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    operations_by_sequence = {}
    split_by_sequence = {}

    for row in rows:
        split = row[
            "split"
        ]

        split_counts[
            split
        ] += 1

        # Test data is deliberately not prepared for use.
        if split == "test":
            continue

        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        operation_names = json.loads(
            row[
                "operation_sequence"
            ]
        )

        operation_ids = tuple(
            operation_vocabulary[
                name
            ]
            for name in operation_names
        )

        operations_by_sequence[
            sequence_index
        ] = operation_ids

        split_by_sequence[
            sequence_index
        ] = split

    if split_counts != {
        "train": 1376,
        "val": 144,
        "test": 144,
    }:
        raise AssertionError(
            "Frozen Phase 2A OOD split changed: "
            f"{split_counts}"
        )

    return (
        operations_by_sequence,
        split_by_sequence,
        split_counts,
    )


def load_visible_trajectory_records(
    split_by_sequence,
):
    """
    Use the Phase 2A trajectory index that has already removed
    symbolic initial/final state labels.
    """

    rows = load_csv(
        PHASE2A_DIR
        / "visible_trajectory_index.csv"
    )

    train_records = []
    val_records = []

    for row in rows:
        sequence_index = int(
            row[
                "sequence_index"
            ]
        )

        if (
            sequence_index
            not in split_by_sequence
        ):
            continue

        record = TrajectoryRecord(
            trajectory_index=int(
                row[
                    "trajectory_index"
                ]
            ),

            sequence_index=sequence_index,

            point_start=int(
                row[
                    "point_start"
                ]
            ),

            point_count=int(
                row[
                    "point_count"
                ]
            ),
        )

        split = split_by_sequence[
            sequence_index
        ]

        if split == "train":
            train_records.append(
                record
            )

        elif split == "val":
            val_records.append(
                record
            )

        else:
            raise AssertionError(
                "Unexpected split reached "
                "trajectory loader."
            )

    if (
        len(train_records)
        != EXPECTED_TRAIN_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Unexpected train trajectory count: "
            f"{len(train_records)}"
        )

    if (
        len(val_records)
        != EXPECTED_VAL_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Unexpected validation trajectory count: "
            f"{len(val_records)}"
        )

    return (
        train_records,
        val_records,
    )


def load_continuous_vectors():
    """
    Load only the continuous observations and noise metadata.

    Symbolic-state arrays stored in the NPZ are intentionally never
    accessed.
    """

    path = (
        PHASE1C_DIR
        / "continuous_observations.npz"
    )

    with np.load(
        path
    ) as data:

        noise_fractions = np.asarray(
            data[
                "noise_fractions"
            ],
            dtype=np.float64,
        )

        matching = np.where(
            np.isclose(
                noise_fractions,
                PRIMARY_NOISE_FRACTION,
            )
        )[0]

        if len(matching) != 1:
            raise AssertionError(
                "Could not uniquely locate "
                "the frozen 0.25 noise condition."
            )

        noise_index = int(
            matching[0]
        )

        vectors = np.array(
            data[
                "vectors"
            ][
                noise_index
            ],
            dtype=np.float32,
            copy=True,
        )

    if (
        vectors.ndim != 2
        or vectors.shape[1]
        != OBSERVATION_DIM
    ):
        raise AssertionError(
            "Continuous observation array "
            "has the wrong shape."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise AssertionError(
            "Continuous observation array "
            "contains NaN or Inf."
        )

    return (
        vectors,
        noise_index,
    )


def build_training_point_mask(
    vector_count: int,
    train_records,
):
    mask = np.zeros(
        vector_count,
        dtype=np.bool_,
    )

    for record in train_records:
        start = (
            record.point_start
        )

        end = (
            start
            + record.point_count
        )

        mask[
            start:end
        ] = True

    return mask


def kmeans_training_centers(
    vectors: np.ndarray,
    train_records,
    seed: int,
):
    """
    Small NumPy k-means initializer.

    It uses only training observations and no symbolic labels.
    """

    rng = np.random.default_rng(
        seed
    )

    training_mask = (
        build_training_point_mask(
            vector_count=len(
                vectors
            ),
            train_records=train_records,
        )
    )

    training_vectors = vectors[
        training_mask
    ]

    sample_count = min(
        KMEANS_SAMPLE_COUNT,
        len(training_vectors),
    )

    sample_indices = rng.choice(
        len(training_vectors),
        size=sample_count,
        replace=False,
    )

    sample = training_vectors[
        sample_indices
    ].astype(
        np.float64
    )

    # k-means++ initialization.
    centers = [
        sample[
            rng.integers(
                0,
                sample_count,
            )
        ]
    ]

    while (
        len(centers)
        < LATENT_STATE_COUNT
    ):
        center_array = np.asarray(
            centers
        )

        differences = (
            sample[
                :,
                None,
                :
            ]
            - center_array[
                None,
                :,
                :
            ]
        )

        squared_distances = (
            differences
            * differences
        ).sum(
            axis=2
        )

        nearest_squared = np.min(
            squared_distances,
            axis=1,
        )

        total = float(
            nearest_squared.sum()
        )

        if total <= 0.0:
            next_index = int(
                rng.integers(
                    0,
                    sample_count,
                )
            )

        else:
            probabilities = (
                nearest_squared
                / total
            )

            next_index = int(
                rng.choice(
                    sample_count,
                    p=probabilities,
                )
            )

        centers.append(
            sample[
                next_index
            ]
        )

    centers = np.asarray(
        centers,
        dtype=np.float64,
    )

    for _ in range(
        KMEANS_ITERATIONS
    ):
        differences = (
            sample[
                :,
                None,
                :
            ]
            - centers[
                None,
                :,
                :
            ]
        )

        squared_distances = (
            differences
            * differences
        ).sum(
            axis=2
        )

        assignments = np.argmin(
            squared_distances,
            axis=1,
        )

        updated = (
            centers.copy()
        )

        for cluster in range(
            LATENT_STATE_COUNT
        ):
            members = sample[
                assignments
                == cluster
            ]

            if len(members) > 0:
                updated[
                    cluster
                ] = members.mean(
                    axis=0
                )

        shift = np.linalg.norm(
            updated
            - centers
        )

        centers = updated

        if shift < 1e-7:
            break

    return torch.tensor(
        centers,
        dtype=torch.float32,
    )


def make_loader(
    vectors,
    records,
    operations_by_sequence,
    shuffle,
):
    dataset = (
        ContinuousTrajectoryDataset(
            vectors=vectors,
            records=records,
            operations_by_sequence=(
                operations_by_sequence
            ),
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        SMOKE_SEED
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=(
            collate_trajectories
        ),
        generator=generator,
    )


def masked_mean(
    values,
    mask,
):
    mask = mask.to(
        values.dtype
    )

    denominator = (
        mask.sum()
        .clamp_min(
            1.0
        )
    )

    return (
        values
        * mask
    ).sum() / denominator


def compute_losses(
    model,
    batch,
    temperature,
    device,
):
    observations = batch[
        "observations"
    ].to(
        device
    )

    point_mask = batch[
        "point_mask"
    ].to(
        device
    )

    operation_ids = batch[
        "operation_ids"
    ].to(
        device
    )

    step_mask = batch[
        "step_mask"
    ].to(
        device
    )

    lengths = batch[
        "lengths"
    ].to(
        device
    )

    latent = model.encode(
        observations,
        temperature=temperature,
    )

    reconstructed = (
        model.decode(
            latent
        )
    )

    reconstruction_error = (
        (
            reconstructed
            - observations
        )
        .pow(2)
        .mean(
            dim=-1
        )
    )

    loss_rec = masked_mean(
        reconstruction_error,
        point_mask,
    )

    current_latent = (
        latent[
            :,
            :-1,
            :
        ]
    )

    next_latent = (
        latent[
            :,
            1:,
            :
        ]
    )

    predicted_latent = (
        model.apply_operation(
            latent_distribution=(
                current_latent
            ),
            operation_ids=(
                operation_ids
            ),
            temperature=temperature,
        )
    )

    predicted_next_observation = (
        model.decode(
            predicted_latent
        )
    )

    next_observations = (
        observations[
            :,
            1:,
            :
        ]
    )

    one_step_error = (
        (
            predicted_next_observation
            - next_observations
        )
        .pow(2)
        .mean(
            dim=-1
        )
    )

    loss_step = masked_mean(
        one_step_error,
        step_mask,
    )

    transition_js = (
        jensen_shannon_divergence(
            predicted_latent,
            next_latent,
        )
    )

    loss_trans = masked_mean(
        transition_js,
        step_mask,
    )

    initial_latent = (
        latent[
            :,
            0,
            :
        ]
    )

    final_latent_prediction = (
        model.rollout(
            initial_distribution=(
                initial_latent
            ),
            operation_ids=(
                operation_ids
            ),
            step_mask=step_mask,
            temperature=temperature,
        )
    )

    final_prediction = (
        model.decode(
            final_latent_prediction
        )
    )

    batch_indices = torch.arange(
        observations.shape[0],
        device=device,
    )

    final_indices = (
        lengths - 1
    )

    final_target = observations[
        batch_indices,
        final_indices,
        :
    ]

    loss_roll = (
        (
            final_prediction
            - final_target
        )
        .pow(2)
        .mean()
    )

    loss_det = (
        model.operation_row_entropy(
            temperature=temperature
        )
    )

    total = (
        loss_rec
        + loss_step
        + LAMBDA_TRANS
        * loss_trans
        + LAMBDA_ROLL
        * loss_roll
        + LAMBDA_DET
        * loss_det
    )

    return {
        "total":
            total,

        "reconstruction":
            loss_rec,

        "one_step":
            loss_step,

        "transition":
            loss_trans,

        "rollout":
            loss_roll,

        "determinism":
            loss_det,
    }


@torch.no_grad()
def evaluate_rollout_mse(
    model,
    loader,
    temperature,
    device,
):
    model.eval()

    squared_error_sum = 0.0
    value_count = 0

    for batch in loader:
        observations = batch[
            "observations"
        ].to(
            device
        )

        operation_ids = batch[
            "operation_ids"
        ].to(
            device
        )

        step_mask = batch[
            "step_mask"
        ].to(
            device
        )

        lengths = batch[
            "lengths"
        ].to(
            device
        )

        initial_latent = (
            model.encode(
                observations[
                    :,
                    0,
                    :
                ],
                temperature=temperature,
            )
        )

        final_latent = (
            model.rollout(
                initial_distribution=(
                    initial_latent
                ),
                operation_ids=(
                    operation_ids
                ),
                step_mask=(
                    step_mask
                ),
                temperature=temperature,
            )
        )

        prediction = (
            model.decode(
                final_latent
            )
        )

        batch_indices = torch.arange(
            observations.shape[0],
            device=device,
        )

        final_target = observations[
            batch_indices,
            lengths - 1,
            :
        ]

        squared_error_sum += float(
            (
                prediction
                - final_target
            )
            .pow(2)
            .sum()
            .item()
        )

        value_count += int(
            final_target.numel()
        )

    return (
        squared_error_sum
        / value_count
    )


@torch.no_grad()
def evaluate_persistence_mse(
    loader,
    device,
):
    squared_error_sum = 0.0
    value_count = 0

    for batch in loader:
        observations = batch[
            "observations"
        ].to(
            device
        )

        lengths = batch[
            "lengths"
        ].to(
            device
        )

        initial = observations[
            :,
            0,
            :
        ]

        batch_indices = torch.arange(
            observations.shape[0],
            device=device,
        )

        final_target = observations[
            batch_indices,
            lengths - 1,
            :
        ]

        squared_error_sum += float(
            (
                initial
                - final_target
            )
            .pow(2)
            .sum()
            .item()
        )

        value_count += int(
            final_target.numel()
        )

    return (
        squared_error_sum
        / value_count
    )


def temperature_for_epoch(
    epoch: int,
):
    if SMOKE_MAX_EPOCHS <= 1:
        return FINAL_TEMPERATURE

    fraction = (
        (epoch - 1)
        / (
            SMOKE_MAX_EPOCHS - 1
        )
    )

    return (
        INITIAL_TEMPERATURE
        + fraction
        * (
            FINAL_TEMPERATURE
            - INITIAL_TEMPERATURE
        )
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    temperature,
    device,
):
    model.train()

    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "one_step": 0.0,
        "transition": 0.0,
        "rollout": 0.0,
        "determinism": 0.0,
    }

    batch_count = 0

    for batch in loader:
        optimizer.zero_grad(
            set_to_none=True
        )

        losses = compute_losses(
            model=model,
            batch=batch,
            temperature=temperature,
            device=device,
        )

        if not torch.isfinite(
            losses[
                "total"
            ]
        ):
            raise FloatingPointError(
                "Training loss became non-finite."
            )

        losses[
            "total"
        ].backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        for name in totals:
            totals[
                name
            ] += float(
                losses[
                    name
                ].detach().item()
            )

        batch_count += 1

    return {
        name:
            value
            / batch_count

        for name, value
        in totals.items()
    }


def write_training_history(
    rows,
):
    path = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def maximum_channel_row_sum_error(
    model,
    temperature,
):
    with torch.no_grad():
        channels = (
            model.operation_channels(
                temperature=temperature
            )
        )

        row_sums = channels.sum(
            dim=-1
        )

        error = (
            row_sums
            - 1.0
        ).abs().max()

    return float(
        error.item()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(
        SMOKE_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    (
        operations_by_sequence,
        split_by_sequence,
        split_counts,
    ) = load_manifest(
        operation_vocabulary
    )

    (
        train_records,
        val_records,
    ) = load_visible_trajectory_records(
        split_by_sequence
    )

    (
        vectors,
        noise_index,
    ) = load_continuous_vectors()

    codebook_centers = (
        kmeans_training_centers(
            vectors=vectors,
            train_records=train_records,
            seed=SMOKE_SEED,
        )
    )

    train_loader = make_loader(
        vectors=vectors,
        records=train_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=True,
    )

    val_loader = make_loader(
        vectors=vectors,
        records=val_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=False,
    )

    model = OperationChannelModel(
        observation_dim=(
            OBSERVATION_DIM
        ),
        latent_state_count=(
            LATENT_STATE_COUNT
        ),
        operation_count=len(
            operation_vocabulary
        ),
    ).to(
        device
    )

    model.initialize_codebook(
        codebook_centers
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    persistence_val_mse = (
        evaluate_persistence_mse(
            val_loader,
            device,
        )
    )

    initial_val_rollout_mse = (
        evaluate_rollout_mse(
            model=model,
            loader=val_loader,
            temperature=(
                INITIAL_TEMPERATURE
            ),
            device=device,
        )
    )

    best_val_mse = float(
        "inf"
    )

    best_epoch = None
    best_temperature = None

    epochs_without_improvement = 0

    history_rows = []

    checkpoint_path = (
        OUTPUT_DIR
        / "best_model.pt"
    )

    print(
        "Phase 2B OCM smoke test started."
    )

    print(
        json.dumps(
            {
                "device":
                    str(device),

                "noise_fraction":
                    PRIMARY_NOISE_FRACTION,

                "noise_index":
                    noise_index,

                "train_sequences":
                    split_counts[
                        "train"
                    ],

                "validation_sequences":
                    split_counts[
                        "val"
                    ],

                "test_sequences_not_used":
                    split_counts[
                        "test"
                    ],

                "train_trajectories":
                    len(
                        train_records
                    ),

                "validation_trajectories":
                    len(
                        val_records
                    ),

                "parameter_count":
                    parameter_count,

                "persistence_validation_mse":
                    persistence_val_mse,

                "initial_validation_rollout_mse":
                    initial_val_rollout_mse,
            },
            indent=2,
        )
    )

    for epoch in range(
        1,
        SMOKE_MAX_EPOCHS + 1,
    ):
        temperature = (
            temperature_for_epoch(
                epoch
            )
        )

        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                temperature=temperature,
                device=device,
            )
        )

        val_rollout_mse = (
            evaluate_rollout_mse(
                model=model,
                loader=val_loader,
                temperature=temperature,
                device=device,
            )
        )

        row = {
            "epoch":
                epoch,

            "temperature":
                temperature,

            "train_total_loss":
                train_metrics[
                    "total"
                ],

            "train_reconstruction_loss":
                train_metrics[
                    "reconstruction"
                ],

            "train_one_step_loss":
                train_metrics[
                    "one_step"
                ],

            "train_transition_loss":
                train_metrics[
                    "transition"
                ],

            "train_rollout_loss":
                train_metrics[
                    "rollout"
                ],

            "train_determinism_loss":
                train_metrics[
                    "determinism"
                ],

            "validation_rollout_mse":
                val_rollout_mse,
        }

        history_rows.append(
            row
        )

        print(
            "Epoch "
            f"{epoch:03d} | "
            f"T={temperature:.4f} | "
            f"train={train_metrics['total']:.6f} | "
            f"val_rollout={val_rollout_mse:.6f}"
        )

        improvement = (
            best_val_mse
            - val_rollout_mse
        )

        if (
            improvement
            > SMOKE_MIN_DELTA
        ):
            best_val_mse = (
                val_rollout_mse
            )

            best_epoch = epoch
            best_temperature = (
                temperature
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        best_epoch,

                    "temperature":
                        best_temperature,

                    "validation_rollout_mse":
                        best_val_mse,

                    "smoke_seed":
                        SMOKE_SEED,
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= SMOKE_PATIENCE
        ):
            print(
                "Smoke-test early stopping triggered."
            )
            break

    write_training_history(
        history_rows
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No validation checkpoint was saved."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    best_temperature = float(
        checkpoint[
            "temperature"
        ]
    )

    final_best_val_mse = (
        evaluate_rollout_mse(
            model=model,
            loader=val_loader,
            temperature=(
                best_temperature
            ),
            device=device,
        )
    )

    row_sum_error = (
        maximum_channel_row_sum_error(
            model=model,
            temperature=(
                best_temperature
            ),
        )
    )

    parameters_finite = all(
        torch.isfinite(
            parameter
        ).all().item()
        for parameter
        in model.parameters()
    )

    implementation_checks = (
        parameters_finite
        and row_sum_error < 1e-6
        and np.isfinite(
            final_best_val_mse
        )
    )

    improves_over_initial = (
        final_best_val_mse
        < initial_val_rollout_mse
    )

    beats_persistence = (
        final_best_val_mse
        < persistence_val_mse
    )

    smoke_passed = (
        implementation_checks
        and improves_over_initial
        and beats_persistence
    )

    summary = {
        "phase":
            "2B OCM implementation smoke test",

        "seed":
            SMOKE_SEED,

        "device":
            str(device),

        "noise_fraction":
            PRIMARY_NOISE_FRACTION,

        "noise_index":
            noise_index,

        "train_sequence_count":
            split_counts[
                "train"
            ],

        "validation_sequence_count":
            split_counts[
                "val"
            ],

        "test_sequence_count_not_used":
            split_counts[
                "test"
            ],

        "train_trajectory_count":
            len(
                train_records
            ),

        "validation_trajectory_count":
            len(
                val_records
            ),

        "parameter_count":
            parameter_count,

        "fixed_smoke_hyperparameters": {
            "learning_rate":
                LEARNING_RATE,

            "lambda_transition":
                LAMBDA_TRANS,

            "lambda_rollout":
                LAMBDA_ROLL,

            "lambda_determinism":
                LAMBDA_DET,

            "batch_size":
                BATCH_SIZE,

            "maximum_epochs":
                SMOKE_MAX_EPOCHS,

            "patience":
                SMOKE_PATIENCE,

            "initial_temperature":
                INITIAL_TEMPERATURE,

            "final_temperature":
                FINAL_TEMPERATURE,
        },

        "persistence_validation_mse":
            persistence_val_mse,

        "initial_validation_rollout_mse":
            initial_val_rollout_mse,

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "best_temperature":
            best_temperature,

        "best_validation_rollout_mse":
            final_best_val_mse,

        "improves_over_initial_model":
            improves_over_initial,

        "beats_persistence_validation":
            beats_persistence,

        "maximum_channel_row_sum_error":
            row_sum_error,

        "all_parameters_finite":
            parameters_finite,

        "implementation_checks_passed":
            implementation_checks,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "test_split_evaluated":
            False,

        "checkpoint_metric":
            "validation rollout MSE only",

        "smoke_test_status":
            (
                "passed"
                if smoke_passed
                else "failed"
            ),

        "interpretation": (
            "This is an implementation smoke test only. "
            "It is not a final model result and it does not "
            "evaluate structural recovery or the OOD test set."
        ),
    }

    with (
        OUTPUT_DIR
        / "phase2b_smoke_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(
        "Phase 2B OCM smoke test completed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
