from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase2b_run_ocm_smoke_test import (
    ContinuousTrajectoryDataset,
    TrajectoryRecord,
    collate_trajectories,
    load_operation_vocabulary,
    masked_mean,
    set_random_seed,
)

from phase2h_baseline_models import (
    AffineOperationModel,
    OperationConditionedMLP,
    GRUOperationSequenceModel,
    TransformerOperationSequenceModel,
    ContinuousLatentOperatorModel,
)


PHASE1C_DIR = Path(
    "outputs/phase1c_continuous_observations"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE2G_DIR = Path(
    "outputs/phase2g_baseline_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase2h_baseline_tuning"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "checkpoints"
)

RESULT_DIR = (
    OUTPUT_DIR / "configuration_results"
)

HISTORY_DIR = (
    OUTPUT_DIR / "training_histories"
)


TUNING_SEED = 11
DEVELOPMENT_NOISE = 0.25

OBSERVATION_DIM = 12

BATCH_SIZE = 128
MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5

EXPECTED_CONFIGURATION_COUNT = 19

EXPECTED_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}

EXPECTED_SEQUENCE_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

EXPECTED_TRAJECTORY_COUNTS = {
    "train": 11008,
    "val": 1152,
}

MAXIMUM_SEQUENCE_LENGTH = 25


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
):
    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device(
            "cuda"
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


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


def write_json(
    path: Path,
    value,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
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
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def validate_phase2g():
    summary = load_json(
        PHASE2G_DIR
        / "phase2g_summary.json"
    )

    if (
        summary[
            "status"
        ]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 2G protocol is not frozen."
        )

    if (
        summary[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2G sanity checks did not pass."
        )

    if (
        summary[
            "tuning_configuration_count"
        ]
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise AssertionError(
            "Phase 2G configuration count changed."
        )

    if (
        summary[
            "development_seed"
        ]
        != TUNING_SEED
    ):
        raise AssertionError(
            "Development seed changed."
        )

    if not np.isclose(
        summary[
            "development_noise_fraction"
        ],
        DEVELOPMENT_NOISE,
    ):
        raise AssertionError(
            "Development noise changed."
        )

    if (
        summary[
            "test_split_used_during_tuning"
        ]
        is not False
    ):
        raise AssertionError(
            "Phase 2G indicates test leakage."
        )

    if (
        summary[
            "clean_targets_used_during_training"
        ]
        is not False
    ):
        raise AssertionError(
            "Phase 2G allows clean training targets."
        )


def load_configuration_registry():
    registry = load_json(
        PHASE2G_DIR
        / "baseline_search_registry.json"
    )

    configurations = registry[
        "configurations"
    ]

    if (
        len(configurations)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise AssertionError(
            "Expected exactly 19 configurations."
        )

    counts = defaultdict(
        int
    )

    for configuration in configurations:
        counts[
            configuration[
                "baseline_id"
            ]
        ] += 1

    if dict(
        counts
    ) != EXPECTED_CONFIGURATION_COUNTS:
        raise AssertionError(
            "Baseline configuration counts changed: "
            f"{dict(counts)}"
        )

    return configurations


def load_training_manifest(
    operation_vocabulary,
):
    """
    Load operation sequences only for train and validation rows.

    Test rows are counted but their operation sequences are not parsed
    or prepared for evaluation.
    """

    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    counts = {
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

        counts[
            split
        ] += 1

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

        operations_by_sequence[
            sequence_index
        ] = tuple(
            operation_vocabulary[
                operation_name
            ]
            for operation_name
            in operation_names
        )

        split_by_sequence[
            sequence_index
        ] = split

    if counts != EXPECTED_SEQUENCE_COUNTS:
        raise AssertionError(
            "Frozen sequence counts changed: "
            f"{counts}"
        )

    return (
        operations_by_sequence,
        split_by_sequence,
        counts,
    )


def load_visible_records(
    split_by_sequence,
):
    rows = load_csv(
        PHASE2A_DIR
        / "visible_trajectory_index.csv"
    )

    records = {
        "train": [],
        "val": [],
    }

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

        split = split_by_sequence[
            sequence_index
        ]

        records[
            split
        ].append(
            TrajectoryRecord(
                trajectory_index=int(
                    row[
                        "trajectory_index"
                    ]
                ),

                sequence_index=(
                    sequence_index
                ),

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
        )

    counts = {
        split:
            len(
                split_records
            )
        for split, split_records
        in records.items()
    }

    if counts != EXPECTED_TRAJECTORY_COUNTS:
        raise AssertionError(
            "Frozen trajectory counts changed: "
            f"{counts}"
        )

    return records


def load_development_vectors():
    """
    Load only the frozen 0.25-noise observation array.

    The clean observation array is deliberately not accessed.
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
                DEVELOPMENT_NOISE,
            )
        )[0]

        if len(
            matching
        ) != 1:
            raise AssertionError(
                "Could not uniquely locate "
                "the 0.25 noise condition."
            )

        noise_index = int(
            matching[
                0
            ]
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
            "Development observation shape changed."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise FloatingPointError(
            "Development observations contain "
            "NaN or Inf."
        )

    return (
        vectors,
        noise_index,
    )


def make_loader(
    vectors,
    records,
    operations_by_sequence,
    shuffle,
):
    dataset = ContinuousTrajectoryDataset(
        vectors=vectors,
        records=records,
        operations_by_sequence=(
            operations_by_sequence
        ),
    )

    generator = torch.Generator()

    generator.manual_seed(
        TUNING_SEED
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


def final_targets(
    observations,
    lengths,
):
    batch_indices = torch.arange(
        observations.shape[0],
        device=observations.device,
    )

    return observations[
        batch_indices,
        lengths - 1,
        :
    ]


@torch.no_grad()
def evaluate_rollout_mse(
    model,
    loader,
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

        prediction = model.rollout(
            initial_observation=(
                observations[
                    :,
                    0,
                    :
                ]
            ),
            operation_ids=operation_ids,
            step_mask=step_mask,
        )

        target = final_targets(
            observations,
            lengths,
        )

        squared_error_sum += float(
            (
                prediction
                - target
            )
            .pow(2)
            .sum()
            .item()
        )

        value_count += int(
            target.numel()
        )

    return float(
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

        target = final_targets(
            observations,
            lengths,
        )

        squared_error_sum += float(
            (
                initial
                - target
            )
            .pow(2)
            .sum()
            .item()
        )

        value_count += int(
            target.numel()
        )

    return float(
        squared_error_sum
        / value_count
    )


def masked_observation_mse(
    prediction,
    target,
    mask,
):
    per_step = (
        prediction
        - target
    ).pow(
        2
    ).mean(
        dim=-1
    )

    return masked_mean(
        per_step,
        mask,
    )


def compute_neural_losses(
    baseline_id,
    model,
    batch,
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

    final_target = final_targets(
        observations,
        lengths,
    )

    initial_observation = observations[
        :,
        0,
        :
    ]

    if baseline_id == "B2":
        one_step_prediction = model.step(
            observations[
                :,
                :-1,
                :
            ],
            operation_ids,
        )

        primary_loss = (
            masked_observation_mse(
                one_step_prediction,
                observations[
                    :,
                    1:,
                    :
                ],
                step_mask,
            )
        )

        auxiliary_loss = torch.zeros(
            (),
            device=device,
        )

        final_prediction = model.rollout(
            initial_observation,
            operation_ids,
            step_mask,
        )

    elif baseline_id in {
        "B3",
        "B4",
    }:
        prefix_prediction = (
            model.predict_prefix(
                initial_observation,
                operation_ids,
                step_mask,
            )
        )

        primary_loss = (
            masked_observation_mse(
                prefix_prediction,
                observations[
                    :,
                    1:,
                    :
                ],
                step_mask,
            )
        )

        auxiliary_loss = torch.zeros(
            (),
            device=device,
        )

        final_prediction = model.rollout(
            initial_observation,
            operation_ids,
            step_mask,
        )

    elif baseline_id == "B5":
        latent = model.encode(
            observations
        )

        reconstruction = model.decode(
            latent
        )

        reconstruction_error = (
            reconstruction
            - observations
        ).pow(
            2
        ).mean(
            dim=-1
        )

        primary_loss = masked_mean(
            reconstruction_error,
            point_mask,
        )

        predicted_next_latent = (
            model.apply_operation(
                latent[
                    :,
                    :-1,
                    :
                ],
                operation_ids,
            )
        )

        latent_error = (
            predicted_next_latent
            - latent[
                :,
                1:,
                :
            ]
        ).pow(
            2
        ).mean(
            dim=-1
        )

        auxiliary_loss = masked_mean(
            latent_error,
            step_mask,
        )

        final_prediction = model.rollout(
            initial_observation,
            operation_ids,
            step_mask,
        )

    else:
        raise ValueError(
            f"Unsupported neural baseline: "
            f"{baseline_id}"
        )

    rollout_loss = F.mse_loss(
        final_prediction,
        final_target,
    )

    total_loss = (
        primary_loss
        + auxiliary_loss
        + rollout_loss
    )

    return {
        "total":
            total_loss,

        "primary":
            primary_loss,

        "auxiliary":
            auxiliary_loss,

        "rollout":
            rollout_loss,
    }


def train_one_epoch(
    baseline_id,
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    totals = {
        "total": 0.0,
        "primary": 0.0,
        "auxiliary": 0.0,
        "rollout": 0.0,
    }

    batch_count = 0

    for batch in loader:
        optimizer.zero_grad(
            set_to_none=True
        )

        losses = compute_neural_losses(
            baseline_id=baseline_id,
            model=model,
            batch=batch,
            device=device,
        )

        if not torch.isfinite(
            losses[
                "total"
            ]
        ):
            raise FloatingPointError(
                f"{baseline_id} produced "
                "a non-finite loss."
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
            value / batch_count
        for name, value
        in totals.items()
    }


def build_neural_model(
    configuration,
    operation_count,
):
    baseline_id = configuration[
        "baseline_id"
    ]

    if baseline_id == "B2":
        return OperationConditionedMLP(
            observation_dim=(
                OBSERVATION_DIM
            ),
            operation_count=(
                operation_count
            ),
            hidden_width=int(
                configuration[
                    "hidden_width"
                ]
            ),
            operation_embedding_dim=int(
                configuration[
                    "operation_embedding_dim"
                ]
            ),
        )

    if baseline_id == "B3":
        return GRUOperationSequenceModel(
            observation_dim=(
                OBSERVATION_DIM
            ),
            operation_count=(
                operation_count
            ),
            hidden_size=int(
                configuration[
                    "hidden_size"
                ]
            ),
            operation_embedding_dim=int(
                configuration[
                    "operation_embedding_dim"
                ]
            ),
        )

    if baseline_id == "B4":
        return TransformerOperationSequenceModel(
            observation_dim=(
                OBSERVATION_DIM
            ),
            operation_count=(
                operation_count
            ),
            model_dimension=int(
                configuration[
                    "model_dimension"
                ]
            ),
            attention_heads=int(
                configuration[
                    "attention_heads"
                ]
            ),
            layer_count=int(
                configuration[
                    "layer_count"
                ]
            ),
            feedforward_dimension=int(
                configuration[
                    "feedforward_dimension"
                ]
            ),
            maximum_sequence_length=(
                MAXIMUM_SEQUENCE_LENGTH
            ),
            dropout=float(
                configuration[
                    "dropout"
                ]
            ),
        )

    if baseline_id == "B5":
        return ContinuousLatentOperatorModel(
            observation_dim=(
                OBSERVATION_DIM
            ),
            operation_count=(
                operation_count
            ),
            latent_dimension=int(
                configuration[
                    "latent_dimension"
                ]
            ),
            encoder_width=int(
                configuration[
                    "encoder_width"
                ]
            ),
            operator_hidden_width=int(
                configuration[
                    "operator_hidden_width"
                ]
            ),
        )

    raise ValueError(
        f"Unsupported neural baseline: "
        f"{baseline_id}"
    )


def collect_affine_training_pairs(
    vectors,
    records,
    operations_by_sequence,
    operation_count,
):
    designs = [
        []
        for _ in range(
            operation_count
        )
    ]

    targets = [
        []
        for _ in range(
            operation_count
        )
    ]

    for record in records:
        operations = (
            operations_by_sequence[
                record.sequence_index
            ]
        )

        start = record.point_start

        for step, operation_id in enumerate(
            operations
        ):
            current = vectors[
                start + step
            ]

            next_observation = vectors[
                start + step + 1
            ]

            designs[
                operation_id
            ].append(
                current
            )

            targets[
                operation_id
            ].append(
                next_observation
            )

    output = []

    for operation_id in range(
        operation_count
    ):
        if not designs[
            operation_id
        ]:
            raise AssertionError(
                f"Operation {operation_id} "
                "has no training pairs."
            )

        output.append(
            (
                np.asarray(
                    designs[
                        operation_id
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    targets[
                        operation_id
                    ],
                    dtype=np.float64,
                ),
            )
        )

    return output


def fit_affine_model(
    training_pairs,
    ridge,
):
    matrices = []
    biases = []

    for inputs, targets in (
        training_pairs
    ):
        ones = np.ones(
            (
                len(inputs),
                1,
            ),
            dtype=np.float64,
        )

        design = np.concatenate(
            [
                inputs,
                ones,
            ],
            axis=1,
        )

        regularizer = np.eye(
            OBSERVATION_DIM + 1,
            dtype=np.float64,
        )

        # Do not regularize the affine bias.
        regularizer[
            -1,
            -1,
        ] = 0.0

        gram = (
            design.T
            @ design
            + ridge
            * regularizer
        )

        right_hand_side = (
            design.T
            @ targets
        )

        weights = np.linalg.solve(
            gram,
            right_hand_side,
        )

        matrices.append(
            weights[
                :-1,
                :
            ]
        )

        biases.append(
            weights[
                -1,
                :
            ]
        )

    return AffineOperationModel(
        matrices=torch.tensor(
            np.asarray(
                matrices
            ),
            dtype=torch.float32,
        ),
        biases=torch.tensor(
            np.asarray(
                biases
            ),
            dtype=torch.float32,
        ),
    )


def state_is_finite(
    model,
):
    return bool(
        all(
            torch.isfinite(
                value
            ).all().item()
            for value in (
                model.state_dict().values()
            )
        )
    )


def write_history(
    configuration_id,
    rows,
):
    write_csv(
        HISTORY_DIR
        / f"{configuration_id}.csv",
        rows,
    )


def run_affine_configuration(
    configuration,
    training_pairs,
    validation_loader,
    device,
    persistence_validation_mse,
):
    configuration_id = (
        configuration[
            "configuration_id"
        ]
    )

    result_path = (
        RESULT_DIR
        / f"{configuration_id}.json"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{configuration_id}.pt"
    )

    if (
        result_path.exists()
        and checkpoint_path.exists()
    ):
        existing = load_json(
            result_path
        )

        if (
            existing.get(
                "status"
            )
            == "completed"
        ):
            print(
                f"{configuration_id}: "
                "already completed, skipping."
            )

            return existing

    model = fit_affine_model(
        training_pairs=(
            training_pairs
        ),
        ridge=float(
            configuration[
                "ridge"
            ]
        ),
    ).to(
        device
    )

    validation_mse = (
        evaluate_rollout_mse(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    finite = state_is_finite(
        model
    )

    numerically_valid = bool(
        finite
        and np.isfinite(
            validation_mse
        )
    )

    torch.save(
        {
            "configuration":
                configuration,

            "model_state_dict":
                model.state_dict(),

            "validation_rollout_mse":
                validation_mse,
        },
        checkpoint_path,
    )

    parameter_count = sum(
        value.numel()
        for value in (
            model.state_dict().values()
        )
    )

    result = {
        **configuration,

        "seed":
            "deterministic",

        "noise_fraction":
            DEVELOPMENT_NOISE,

        "parameter_count":
            int(
                parameter_count
            ),

        "best_validation_rollout_mse":
            float(
                validation_mse
            ),

        "best_epoch":
            None,

        "epochs_executed":
            0,

        "persistence_validation_mse":
            float(
                persistence_validation_mse
            ),

        "beat_persistence":
            bool(
                validation_mse
                < persistence_validation_mse
            ),

        "all_state_values_finite":
            finite,

        "numerically_valid":
            numerically_valid,

        "selection_metric":
            "validation rollout MSE only",

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "clean_targets_read":
            False,

        "test_split_evaluated":
            False,

        "status":
            "completed",
    }

    write_json(
        result_path,
        result,
    )

    print(
        f"{configuration_id} completed | "
        f"validation MSE={validation_mse:.8f}"
    )

    return result


def run_neural_configuration(
    configuration,
    operation_count,
    vectors,
    train_records,
    validation_records,
    operations_by_sequence,
    device,
    persistence_validation_mse,
):
    baseline_id = configuration[
        "baseline_id"
    ]

    configuration_id = configuration[
        "configuration_id"
    ]

    result_path = (
        RESULT_DIR
        / f"{configuration_id}.json"
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{configuration_id}.pt"
    )

    if (
        result_path.exists()
        and checkpoint_path.exists()
    ):
        existing = load_json(
            result_path
        )

        if (
            existing.get(
                "status"
            )
            == "completed"
        ):
            print(
                f"{configuration_id}: "
                "already completed, skipping."
            )

            return existing

    set_random_seed(
        TUNING_SEED
    )

    train_loader = make_loader(
        vectors=vectors,
        records=train_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=True,
    )

    validation_loader = make_loader(
        vectors=vectors,
        records=validation_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=False,
    )

    model = build_neural_model(
        configuration=configuration,
        operation_count=operation_count,
    ).to(
        device
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    initial_validation_mse = (
        evaluate_rollout_mse(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            configuration[
                "learning_rate"
            ]
        ),
    )

    best_validation_mse = float(
        "inf"
    )

    best_epoch = None
    epochs_without_improvement = 0

    history = []

    print()
    print(
        "=" * 72
    )

    print(
        f"{configuration_id} | "
        f"baseline={baseline_id} | "
        f"parameters={parameter_count}"
    )

    print(
        "=" * 72
    )

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
    ):
        train_metrics = train_one_epoch(
            baseline_id=baseline_id,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_mse = (
            evaluate_rollout_mse(
                model=model,
                loader=validation_loader,
                device=device,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_total_loss":
                    train_metrics[
                        "total"
                    ],

                "train_primary_loss":
                    train_metrics[
                        "primary"
                    ],

                "train_auxiliary_loss":
                    train_metrics[
                        "auxiliary"
                    ],

                "train_rollout_loss":
                    train_metrics[
                        "rollout"
                    ],

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"{configuration_id} | "
            f"epoch={epoch:03d} | "
            f"train={train_metrics['total']:.6f} | "
            f"val={validation_mse:.6f}"
        )

        improvement = (
            best_validation_mse
            - validation_mse
        )

        if (
            improvement
            > EARLY_STOPPING_MIN_DELTA
        ):
            best_validation_mse = (
                validation_mse
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "configuration":
                        configuration,

                    "seed":
                        TUNING_SEED,

                    "noise_fraction":
                        DEVELOPMENT_NOISE,

                    "epoch":
                        best_epoch,

                    "validation_rollout_mse":
                        best_validation_mse,

                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                f"{configuration_id}: "
                "early stopping."
            )
            break

    write_history(
        configuration_id,
        history,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"{configuration_id} did not "
            "produce a checkpoint."
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

    final_validation_mse = (
        evaluate_rollout_mse(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    finite = state_is_finite(
        model
    )

    numerically_valid = bool(
        finite
        and np.isfinite(
            final_validation_mse
        )
    )

    result = {
        **configuration,

        "seed":
            TUNING_SEED,

        "noise_fraction":
            DEVELOPMENT_NOISE,

        "parameter_count":
            int(
                parameter_count
            ),

        "initial_validation_rollout_mse":
            float(
                initial_validation_mse
            ),

        "best_validation_rollout_mse":
            float(
                final_validation_mse
            ),

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "epochs_executed":
            len(
                history
            ),

        "persistence_validation_mse":
            float(
                persistence_validation_mse
            ),

        "beat_persistence":
            bool(
                final_validation_mse
                < persistence_validation_mse
            ),

        "all_state_values_finite":
            finite,

        "numerically_valid":
            numerically_valid,

        "selection_metric":
            "validation rollout MSE only",

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "clean_targets_read":
            False,

        "test_split_evaluated":
            False,

        "status":
            "completed",
    }

    write_json(
        result_path,
        result,
    )

    print(
        f"{configuration_id} completed | "
        f"best validation MSE="
        f"{final_validation_mse:.8f} | "
        f"epoch={result['best_epoch']}"
    )

    return result


def choose_selected_configurations(
    results,
):
    selected = {}

    for baseline_id in (
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        baseline_results = [
            row
            for row in results
            if (
                row[
                    "baseline_id"
                ]
                == baseline_id
                and row[
                    "numerically_valid"
                ]
                is True
            )
        ]

        if not baseline_results:
            raise RuntimeError(
                f"No valid configurations "
                f"for {baseline_id}."
            )

        winner = min(
            baseline_results,
            key=lambda row: (
                row[
                    "best_validation_rollout_mse"
                ],
                row[
                    "configuration_id"
                ],
            ),
        )

        selected[
            baseline_id
        ] = winner

    return selected


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_phase2g()

    configurations = (
        load_configuration_registry()
    )

    device = resolve_device(
        arguments.device
    )

    environment = {
        "python":
            platform.python_version(),

        "torch":
            torch.__version__,

        "torch_cuda_build":
            torch.version.cuda,

        "device":
            str(
                device
            ),
    }

    if device.type == "cuda":
        environment[
            "gpu_name"
        ] = torch.cuda.get_device_name(
            device
        )

    write_json(
        OUTPUT_DIR
        / "environment.json",
        environment,
    )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    (
        operations_by_sequence,
        split_by_sequence,
        sequence_counts,
    ) = load_training_manifest(
        operation_vocabulary
    )

    records = load_visible_records(
        split_by_sequence
    )

    (
        vectors,
        noise_index,
    ) = load_development_vectors()

    validation_loader = make_loader(
        vectors=vectors,
        records=records[
            "val"
        ],
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=False,
    )

    persistence_validation_mse = (
        evaluate_persistence_mse(
            validation_loader,
            device,
        )
    )

    affine_training_pairs = (
        collect_affine_training_pairs(
            vectors=vectors,
            records=records[
                "train"
            ],
            operations_by_sequence=(
                operations_by_sequence
            ),
            operation_count=len(
                operation_vocabulary
            ),
        )
    )

    print(
        "Phase 2H baseline tuning started."
    )

    print(
        json.dumps(
            {
                "device":
                    str(
                        device
                    ),

                "configuration_count":
                    len(
                        configurations
                    ),

                "development_seed":
                    TUNING_SEED,

                "development_noise":
                    DEVELOPMENT_NOISE,

                "noise_index":
                    noise_index,

                "train_sequences":
                    sequence_counts[
                        "train"
                    ],

                "validation_sequences":
                    sequence_counts[
                        "val"
                    ],

                "test_sequences_not_evaluated":
                    sequence_counts[
                        "test"
                    ],

                "train_trajectories":
                    len(
                        records[
                            "train"
                        ]
                    ),

                "validation_trajectories":
                    len(
                        records[
                            "val"
                        ]
                    ),

                "persistence_validation_mse":
                    persistence_validation_mse,

                "selection_metric":
                    "validation rollout MSE only",
            },
            indent=2,
        )
    )

    results = []

    for configuration in configurations:
        baseline_id = configuration[
            "baseline_id"
        ]

        if baseline_id == "B1":
            result = run_affine_configuration(
                configuration=(
                    configuration
                ),
                training_pairs=(
                    affine_training_pairs
                ),
                validation_loader=(
                    validation_loader
                ),
                device=device,
                persistence_validation_mse=(
                    persistence_validation_mse
                ),
            )

        else:
            result = run_neural_configuration(
                configuration=(
                    configuration
                ),
                operation_count=len(
                    operation_vocabulary
                ),
                vectors=vectors,
                train_records=records[
                    "train"
                ],
                validation_records=records[
                    "val"
                ],
                operations_by_sequence=(
                    operations_by_sequence
                ),
                device=device,
                persistence_validation_mse=(
                    persistence_validation_mse
                ),
            )

        results.append(
            result
        )

    if (
        len(results)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise AssertionError(
            "Not all 19 configurations completed."
        )

    selected = (
        choose_selected_configurations(
            results
        )
    )

    selected_output = {}

    for baseline_id, winner in (
        selected.items()
    ):
        selected_output[
            baseline_id
        ] = {
            key:
                value
            for key, value
            in winner.items()
            if key not in {
                "status",
                "all_state_values_finite",
                "numerically_valid",
                "structural_metrics_computed",
                "structural_labels_read",
                "blackwell_labels_read",
                "clean_targets_read",
                "test_split_evaluated",
            }
        }

        source_checkpoint = (
            CHECKPOINT_DIR
            / (
                winner[
                    "configuration_id"
                ]
                + ".pt"
            )
        )

        selected_checkpoint = (
            OUTPUT_DIR
            / (
                "selected_"
                + baseline_id
                + "_tuning_checkpoint.pt"
            )
        )

        shutil.copy2(
            source_checkpoint,
            selected_checkpoint,
        )

    write_json(
        OUTPUT_DIR
        / "selected_baseline_configurations.json",
        selected_output,
    )

    result_rows = sorted(
        results,
        key=lambda row:
            row[
                "configuration_id"
            ],
    )

    write_csv(
        OUTPUT_DIR
        / "tuning_results.csv",
        result_rows,
    )

    every_configuration_valid = bool(
        all(
            row[
                "numerically_valid"
            ]
            for row in results
        )
    )

    no_test_evaluation = bool(
        all(
            row[
                "test_split_evaluated"
            ]
            is False
            for row in results
        )
    )

    no_clean_targets = bool(
        all(
            row[
                "clean_targets_read"
            ]
            is False
            for row in results
        )
    )

    no_structural_metrics = bool(
        all(
            row[
                "structural_metrics_computed"
            ]
            is False
            for row in results
        )
    )

    phase_passed = bool(
        every_configuration_valid
        and no_test_evaluation
        and no_clean_targets
        and no_structural_metrics
        and len(
            selected_output
        ) == 5
    )

    summary = {
        "phase":
            "2H leakage-safe predictive baseline tuning",

        "configuration_count":
            EXPECTED_CONFIGURATION_COUNT,

        "completed_configuration_count":
            len(
                results
            ),

        "configuration_counts":
            EXPECTED_CONFIGURATION_COUNTS,

        "development_seed":
            TUNING_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "train_sequence_count":
            sequence_counts[
                "train"
            ],

        "validation_sequence_count":
            sequence_counts[
                "val"
            ],

        "test_sequence_count_not_evaluated":
            sequence_counts[
                "test"
            ],

        "train_trajectory_count":
            len(
                records[
                    "train"
                ]
            ),

        "validation_trajectory_count":
            len(
                records[
                    "val"
                ]
            ),

        "persistence_validation_mse":
            persistence_validation_mse,

        "selection_metric":
            "validation rollout MSE only",

        "selected_configurations":
            selected_output,

        "all_configurations_numerically_valid":
            every_configuration_valid,

        "test_split_evaluated":
            False,

        "clean_targets_read":
            False,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "phase2h_status":
            (
                "passed"
                if phase_passed
                else "failed"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase2h_summary.json",
        summary,
    )

    print()
    print(
        "Phase 2H baseline tuning completed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
