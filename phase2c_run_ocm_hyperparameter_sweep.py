from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
    jensen_shannon_divergence,
)

from phase2b_run_ocm_smoke_test import (
    OBSERVATION_DIM,
    LATENT_STATE_COUNT,
    PRIMARY_NOISE_FRACTION,
    set_random_seed,
    load_operation_vocabulary,
    load_manifest,
    load_visible_trajectory_records,
    load_continuous_vectors,
    kmeans_training_centers,
    make_loader,
    evaluate_persistence_mse,
    evaluate_rollout_mse,
    masked_mean,
    maximum_channel_row_sum_error,
)


PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase2c_hyperparameter_sweep"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR / "checkpoints"
)

HISTORY_DIR = (
    OUTPUT_DIR / "training_histories"
)

RESULT_DIR = (
    OUTPUT_DIR / "configuration_results"
)


# ------------------------------------------------------------
# Frozen Phase 2C protocol
# ------------------------------------------------------------

TUNING_SEED = 11

LEARNING_RATES = (
    0.001,
    0.0003,
)

TRANSITION_WEIGHTS = (
    0.10,
    0.25,
)

ROLLOUT_WEIGHTS = (
    0.25,
    0.50,
    1.00,
)

DETERMINISM_WEIGHTS = (
    0.001,
    0.01,
)

BATCH_SIZE = 128

MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-5

INITIAL_TEMPERATURE = 1.0
FINAL_TEMPERATURE = 0.25

# This preserves the annealing schedule used by the
# successful 40-epoch Phase 2B smoke test.
TEMPERATURE_ANNEAL_EPOCHS = 40

EXPECTED_CONFIGURATION_COUNT = 24

EXPECTED_TRAIN_SEQUENCE_COUNT = 1376
EXPECTED_VALIDATION_SEQUENCE_COUNT = 144
EXPECTED_UNUSED_TEST_SEQUENCE_COUNT = 144

EXPECTED_TRAIN_TRAJECTORY_COUNT = 11008
EXPECTED_VALIDATION_TRAJECTORY_COUNT = 1152


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Phase 2C OCM "
            "hyperparameter sweep."
        )
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help=(
            "Use cuda to fail immediately if "
            "CUDA is unavailable."
        ),
    )

    return parser.parse_args()


def resolve_device(
    requested: str,
) -> torch.device:

    if requested == "cpu":
        return torch.device(
            "cpu"
        )

    if requested == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested "
                "but torch.cuda.is_available() is False."
            )

        return torch.device(
            "cuda"
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def environment_description(
    device: torch.device,
):
    description = {
        "python":
            platform.python_version(),

        "torch":
            torch.__version__,

        "torch_cuda_build":
            torch.version.cuda,

        "device":
            str(device),
    }

    if device.type == "cuda":
        description[
            "gpu_name"
        ] = torch.cuda.get_device_name(
            device
        )

        description[
            "gpu_count"
        ] = torch.cuda.device_count()

    return description


def verify_phase2a_protocol() -> None:
    """
    Make sure the search space still matches the protocol
    frozen before Phase 2B.
    """

    summary = load_json(
        PHASE2A_DIR
        / "phase2a_summary.json"
    )

    protocol = load_json(
        PHASE2A_DIR
        / "model_protocol.json"
    )

    if (
        summary[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2A is not marked as passed."
        )

    if (
        summary[
            "status"
        ]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 2A protocol is not frozen."
        )

    if (
        summary[
            "primary_noise_fraction"
        ]
        != PRIMARY_NOISE_FRACTION
    ):
        raise AssertionError(
            "Primary noise condition changed."
        )

    selection = protocol[
        "hyperparameter_selection"
    ]

    if tuple(
        selection[
            "learning_rates"
        ]
    ) != LEARNING_RATES:
        raise AssertionError(
            "Learning-rate search space changed."
        )

    losses = protocol[
        "losses"
    ]

    if tuple(
        losses[
            "latent_transition_consistency"
        ][
            "candidate_weights"
        ]
    ) != TRANSITION_WEIGHTS:
        raise AssertionError(
            "Transition-loss search space changed."
        )

    if tuple(
        losses[
            "full_sequence_rollout"
        ][
            "candidate_weights"
        ]
    ) != ROLLOUT_WEIGHTS:
        raise AssertionError(
            "Rollout-loss search space changed."
        )

    if tuple(
        losses[
            "operation_row_entropy"
        ][
            "candidate_weights"
        ]
    ) != DETERMINISM_WEIGHTS:
        raise AssertionError(
            "Determinism-loss search space changed."
        )

    if (
        selection[
            "maximum_epochs"
        ]
        != MAXIMUM_EPOCHS
    ):
        raise AssertionError(
            "Maximum epoch count changed."
        )

    if (
        selection[
            "early_stopping_patience"
        ]
        != EARLY_STOPPING_PATIENCE
    ):
        raise AssertionError(
            "Early-stopping patience changed."
        )

    if not np.isclose(
        selection[
            "early_stopping_min_delta"
        ],
        EARLY_STOPPING_MIN_DELTA,
    ):
        raise AssertionError(
            "Early-stopping minimum delta changed."
        )

    if (
        selection[
            "selection_metric"
        ]
        != (
            "compositional validation "
            "final-observation MSE"
        )
    ):
        raise AssertionError(
            "Checkpoint-selection metric changed."
        )

    if (
        selection[
            "structural_metrics_used_for_selection"
        ]
        is not False
    ):
        raise AssertionError(
            "Structural metrics must not be used "
            "for hyperparameter selection."
        )

    if (
        selection[
            "test_metrics_used_for_selection"
        ]
        is not False
    ):
        raise AssertionError(
            "Test metrics must not be used "
            "for hyperparameter selection."
        )


def build_configuration_registry():
    rows = []

    combinations = product(
        LEARNING_RATES,
        TRANSITION_WEIGHTS,
        ROLLOUT_WEIGHTS,
        DETERMINISM_WEIGHTS,
    )

    for index, values in enumerate(
        combinations,
        start=1,
    ):
        (
            learning_rate,
            transition_weight,
            rollout_weight,
            determinism_weight,
        ) = values

        rows.append(
            {
                "configuration_id":
                    f"c{index:02d}",

                "learning_rate":
                    float(
                        learning_rate
                    ),

                "lambda_transition":
                    float(
                        transition_weight
                    ),

                "lambda_rollout":
                    float(
                        rollout_weight
                    ),

                "lambda_determinism":
                    float(
                        determinism_weight
                    ),
            }
        )

    if (
        len(rows)
        != EXPECTED_CONFIGURATION_COUNT
    ):
        raise AssertionError(
            "Hyperparameter grid does not contain "
            "exactly 24 configurations."
        )

    return rows


def temperature_for_epoch(
    epoch: int,
) -> float:
    """
    Linear 1.0 -> 0.25 annealing over epochs 1 through 40,
    followed by a fixed temperature of 0.25.
    """

    if epoch >= TEMPERATURE_ANNEAL_EPOCHS:
        return FINAL_TEMPERATURE

    fraction = (
        (epoch - 1)
        / (
            TEMPERATURE_ANNEAL_EPOCHS
            - 1
        )
    )

    return float(
        INITIAL_TEMPERATURE
        + fraction
        * (
            FINAL_TEMPERATURE
            - INITIAL_TEMPERATURE
        )
    )


def compute_losses(
    model,
    batch,
    temperature,
    device,
    lambda_transition,
    lambda_rollout,
    lambda_determinism,
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

    reconstructed = model.decode(
        latent
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

    loss_reconstruction = masked_mean(
        reconstruction_error,
        point_mask,
    )

    current_latent = latent[
        :,
        :-1,
        :
    ]

    next_latent = latent[
        :,
        1:,
        :
    ]

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

    next_observation = observations[
        :,
        1:,
        :
    ]

    one_step_error = (
        (
            predicted_next_observation
            - next_observation
        )
        .pow(2)
        .mean(
            dim=-1
        )
    )

    loss_one_step = masked_mean(
        one_step_error,
        step_mask,
    )

    transition_js = (
        jensen_shannon_divergence(
            predicted_latent,
            next_latent,
        )
    )

    loss_transition = masked_mean(
        transition_js,
        step_mask,
    )

    initial_latent = latent[
        :,
        0,
        :
    ]

    final_latent = model.rollout(
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

    final_prediction = model.decode(
        final_latent
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

    loss_rollout = (
        (
            final_prediction
            - final_target
        )
        .pow(2)
        .mean()
    )

    loss_determinism = (
        model.operation_row_entropy(
            temperature=temperature
        )
    )

    total = (
        loss_reconstruction
        + loss_one_step
        + lambda_transition
        * loss_transition
        + lambda_rollout
        * loss_rollout
        + lambda_determinism
        * loss_determinism
    )

    return {
        "total":
            total,

        "reconstruction":
            loss_reconstruction,

        "one_step":
            loss_one_step,

        "transition":
            loss_transition,

        "rollout":
            loss_rollout,

        "determinism":
            loss_determinism,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    temperature,
    device,
    configuration,
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
            lambda_transition=(
                configuration[
                    "lambda_transition"
                ]
            ),
            lambda_rollout=(
                configuration[
                    "lambda_rollout"
                ]
            ),
            lambda_determinism=(
                configuration[
                    "lambda_determinism"
                ]
            ),
        )

        if not torch.isfinite(
            losses[
                "total"
            ]
        ):
            raise FloatingPointError(
                "Training produced a non-finite loss."
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
            totals[name] += float(
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


def write_training_history(
    configuration_id,
    rows,
):
    path = (
        HISTORY_DIR
        / f"{configuration_id}.csv"
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


def run_configuration(
    configuration,
    device,
    vectors,
    train_records,
    val_records,
    operations_by_sequence,
    codebook_centers,
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

    # A completed configuration can be resumed safely.
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

    # Rebuild loaders for every configuration so every run receives
    # the same deterministic shuffle sequence.
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
            load_operation_vocabulary()
        ),
    ).to(
        device
    )

    model.initialize_codebook(
        codebook_centers
    )

    initial_validation_mse = float(
        evaluate_rollout_mse(
            model=model,
            loader=val_loader,
            temperature=(
                INITIAL_TEMPERATURE
            ),
            device=device,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration[
            "learning_rate"
        ],
    )

    best_validation_mse = float(
        "inf"
    )

    best_epoch = None
    best_temperature = None

    epochs_without_improvement = 0

    history = []

    print()
    print(
        "=" * 72
    )

    print(
        f"{configuration_id} | "
        f"lr={configuration['learning_rate']} | "
        f"trans={configuration['lambda_transition']} | "
        f"roll={configuration['lambda_rollout']} | "
        f"det={configuration['lambda_determinism']}"
    )

    print(
        "=" * 72
    )

    for epoch in range(
        1,
        MAXIMUM_EPOCHS + 1,
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
                configuration=configuration,
            )
        )

        validation_mse = float(
            evaluate_rollout_mse(
                model=model,
                loader=val_loader,
                temperature=temperature,
                device=device,
            )
        )

        history.append(
            {
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
                    validation_mse,
            }
        )

        print(
            f"{configuration_id} | "
            f"epoch={epoch:03d} | "
            f"T={temperature:.4f} | "
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

            best_temperature = (
                temperature
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "configuration":
                        configuration,

                    "model_state_dict":
                        model.state_dict(),

                    "seed":
                        TUNING_SEED,

                    "epoch":
                        best_epoch,

                    "temperature":
                        best_temperature,

                    "validation_rollout_mse":
                        best_validation_mse,
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

    write_training_history(
        configuration_id,
        history,
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"{configuration_id} did not save a checkpoint."
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

    final_best_validation_mse = float(
        evaluate_rollout_mse(
            model=model,
            loader=val_loader,
            temperature=float(
                checkpoint[
                    "temperature"
                ]
            ),
            device=device,
        )
    )

    row_sum_error = float(
        maximum_channel_row_sum_error(
            model=model,
            temperature=float(
                checkpoint[
                    "temperature"
                ]
            ),
        )
    )

    parameters_finite = bool(
        all(
            torch.isfinite(
                parameter
            ).all().item()

            for parameter
            in model.parameters()
        )
    )

    implementation_valid = bool(
        parameters_finite
        and row_sum_error < 1e-6
        and bool(
            np.isfinite(
                final_best_validation_mse
            )
        )
    )

    result = {
        **configuration,

        "seed":
            TUNING_SEED,

        "device":
            str(device),

        "initial_validation_rollout_mse":
            initial_validation_mse,

        "best_validation_rollout_mse":
            final_best_validation_mse,

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "best_temperature":
            float(
                checkpoint[
                    "temperature"
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

        "improved_over_initial":
            bool(
                final_best_validation_mse
                < initial_validation_mse
            ),

        "beat_persistence":
            bool(
                final_best_validation_mse
                < persistence_validation_mse
            ),

        "all_parameters_finite":
            parameters_finite,

        "maximum_channel_row_sum_error":
            row_sum_error,

        "implementation_valid":
            implementation_valid,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
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
        f"best val={final_best_validation_mse:.8f} | "
        f"epoch={result['best_epoch']}"
    )

    return result


def write_configuration_registry(
    configurations,
):
    write_json(
        OUTPUT_DIR
        / "configuration_registry.json",
        {
            "configuration_count":
                len(
                    configurations
                ),

            "seed":
                TUNING_SEED,

            "noise_fraction":
                PRIMARY_NOISE_FRACTION,

            "batch_size":
                BATCH_SIZE,

            "maximum_epochs":
                MAXIMUM_EPOCHS,

            "patience":
                EARLY_STOPPING_PATIENCE,

            "minimum_delta":
                EARLY_STOPPING_MIN_DELTA,

            "temperature_schedule": {
                "initial":
                    INITIAL_TEMPERATURE,

                "final":
                    FINAL_TEMPERATURE,

                "anneal_epochs":
                    TEMPERATURE_ANNEAL_EPOCHS,

                "after_annealing":
                    "hold at final temperature",
            },

            "selection_metric":
                "validation rollout MSE only",

            "configurations":
                configurations,
        },
    )


def write_sweep_results(
    results,
):
    rows = sorted(
        results,
        key=lambda row:
            row[
                "configuration_id"
            ],
    )

    fieldnames = [
        "configuration_id",
        "learning_rate",
        "lambda_transition",
        "lambda_rollout",
        "lambda_determinism",
        "best_validation_rollout_mse",
        "initial_validation_rollout_mse",
        "persistence_validation_mse",
        "best_epoch",
        "best_temperature",
        "epochs_executed",
        "improved_over_initial",
        "beat_persistence",
        "implementation_valid",
    ]

    path = (
        OUTPUT_DIR
        / "sweep_results.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def choose_winner(
    results,
):
    valid_results = [
        row
        for row in results
        if (
            row[
                "implementation_valid"
            ]
            is True
        )
    ]

    if not valid_results:
        raise RuntimeError(
            "No valid configurations completed."
        )

    return min(
        valid_results,
        key=lambda row: (
            row[
                "best_validation_rollout_mse"
            ],
            row[
                "configuration_id"
            ],
        ),
    )


def main() -> None:
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    verify_phase2a_protocol()

    device = resolve_device(
        arguments.device
    )

    environment = (
        environment_description(
            device
        )
    )

    write_json(
        OUTPUT_DIR
        / "environment.json",
        environment,
    )

    print(
        "Phase 2C environment:"
    )

    print(
        json.dumps(
            environment,
            indent=2,
        )
    )

    configurations = (
        build_configuration_registry()
    )

    write_configuration_registry(
        configurations
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

    if split_counts != {
        "train":
            EXPECTED_TRAIN_SEQUENCE_COUNT,

        "val":
            EXPECTED_VALIDATION_SEQUENCE_COUNT,

        "test":
            EXPECTED_UNUSED_TEST_SEQUENCE_COUNT,
    }:
        raise AssertionError(
            "Frozen OOD split changed."
        )

    (
        train_records,
        val_records,
    ) = load_visible_trajectory_records(
        split_by_sequence
    )

    if (
        len(train_records)
        != EXPECTED_TRAIN_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Training trajectory count changed."
        )

    if (
        len(val_records)
        != EXPECTED_VALIDATION_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Validation trajectory count changed."
        )

    (
        vectors,
        noise_index,
    ) = load_continuous_vectors()

    codebook_centers = (
        kmeans_training_centers(
            vectors=vectors,
            train_records=train_records,
            seed=TUNING_SEED,
        )
    )

    # Persistence is calculated once because it does not depend
    # on OCM hyperparameters.
    validation_loader = make_loader(
        vectors=vectors,
        records=val_records,
        operations_by_sequence=(
            operations_by_sequence
        ),
        shuffle=False,
    )

    persistence_validation_mse = float(
        evaluate_persistence_mse(
            validation_loader,
            device,
        )
    )

    print()
    print(
        "Phase 2C OCM hyperparameter sweep"
    )

    print(
        json.dumps(
            {
                "configuration_count":
                    len(
                        configurations
                    ),

                "seed":
                    TUNING_SEED,

                "noise_fraction":
                    PRIMARY_NOISE_FRACTION,

                "noise_index":
                    int(
                        noise_index
                    ),

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

        result = run_configuration(
            configuration=configuration,
            device=device,
            vectors=vectors,
            train_records=train_records,
            val_records=val_records,
            operations_by_sequence=(
                operations_by_sequence
            ),
            codebook_centers=(
                codebook_centers
            ),
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
            "Not all 24 configurations completed."
        )

    write_sweep_results(
        results
    )

    winner = choose_winner(
        results
    )

    selected_checkpoint = (
        CHECKPOINT_DIR
        / (
            winner[
                "configuration_id"
            ]
            + ".pt"
        )
    )

    selected_checkpoint_copy = (
        OUTPUT_DIR
        / "selected_tuning_checkpoint.pt"
    )

    shutil.copy2(
        selected_checkpoint,
        selected_checkpoint_copy,
    )

    selected_configuration = {
        "configuration_id":
            winner[
                "configuration_id"
            ],

        "learning_rate":
            winner[
                "learning_rate"
            ],

        "lambda_transition":
            winner[
                "lambda_transition"
            ],

        "lambda_rollout":
            winner[
                "lambda_rollout"
            ],

        "lambda_determinism":
            winner[
                "lambda_determinism"
            ],

        "tuning_seed":
            TUNING_SEED,

        "development_noise_fraction":
            PRIMARY_NOISE_FRACTION,

        "selection_metric":
            "validation rollout MSE only",

        "best_validation_rollout_mse":
            winner[
                "best_validation_rollout_mse"
            ],

        "best_epoch":
            winner[
                "best_epoch"
            ],

        "best_temperature":
            winner[
                "best_temperature"
            ],
    }

    write_json(
        OUTPUT_DIR
        / "selected_configuration.json",
        selected_configuration,
    )

    every_configuration_valid = bool(
        all(
            row[
                "implementation_valid"
            ]
            for row in results
        )
    )

    winner_beats_persistence = bool(
        winner[
            "best_validation_rollout_mse"
        ]
        < persistence_validation_mse
    )

    phase_passed = bool(
        every_configuration_valid
        and winner_beats_persistence
    )

    summary = {
        "phase":
            "2C blind OCM hyperparameter selection",

        "configuration_count":
            EXPECTED_CONFIGURATION_COUNT,

        "completed_configuration_count":
            len(
                results
            ),

        "tuning_seed":
            TUNING_SEED,

        "noise_fraction":
            PRIMARY_NOISE_FRACTION,

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

        "selection_metric":
            "validation rollout MSE only",

        "selected_configuration":
            selected_configuration,

        "persistence_validation_mse":
            persistence_validation_mse,

        "all_configurations_numerically_valid":
            every_configuration_valid,

        "selected_configuration_beats_persistence":
            winner_beats_persistence,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "blackwell_labels_read":
            False,

        "information_classes_read":
            False,

        "symbolic_states_read":
            False,

        "test_split_evaluated":
            False,

        "iid_test_evaluated":
            False,

        "phase2c_status":
            (
                "passed"
                if phase_passed
                else "failed"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase2c_summary.json",
        summary,
    )

    print()
    print(
        "Phase 2C hyperparameter sweep completed."
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
