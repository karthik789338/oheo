from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    set_random_seed,
)

from phase2h_baseline_models import (
    OperationConditionedMLP,
    GRUOperationSequenceModel,
    TransformerOperationSequenceModel,
    ContinuousLatentOperatorModel,
)

from phase2h_tune_predictive_baselines import (
    train_one_epoch,
)

from phase3c_run_tier_b_smoke_tests import (
    OBSERVATION_DIMENSION,
    LATENT_STATE_COUNT,
    DEVELOPMENT_SEED,
    DEVELOPMENT_NOISE,
    load_operation_vocabulary,
    load_cell,
    make_loader,
    evaluate_persistence,
    evaluate_affine_or_baseline,
    evaluate_ocm,
    fit_affine_model,
    try_constructor,
    state_is_finite,
    parameter_count,
    channel_row_sum_error,
    ocm_forward_losses,
)


PHASE3D_DIR = Path(
    "outputs/phase3d_tier_b_tuning"
)

CONFIGURATION_DIR = (
    PHASE3D_DIR / "configurations"
)

SELECTED_DIR = (
    PHASE3D_DIR / "selected"
)


EXPECTED_COUNTS = {
    "OCM": 24,
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}


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

    parser.add_argument(
        "--configuration-id",
        default="all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(path: Path, value):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def validate_registry():
    registry = load_json(
        PHASE3D_DIR
        / "tuning_registry.json"
    )

    if registry["status"] != "registry_frozen":
        raise AssertionError(
            "Phase 3D registry is not frozen."
        )

    if registry["configuration_count"] != 43:
        raise AssertionError(
            "Expected 43 tuning configurations."
        )

    counts = Counter(
        configuration["baseline_id"]
        for configuration
        in registry["configurations"]
    )

    if dict(counts) != EXPECTED_COUNTS:
        raise AssertionError(
            f"Configuration counts changed: {dict(counts)}"
        )

    if registry["training_cell"] != "train_joint":
        raise AssertionError(
            "Unexpected training cell."
        )

    if (
        registry["checkpoint_selection_cell"]
        != "val_joint"
    ):
        raise AssertionError(
            "Unexpected validation cell."
        )

    return registry


def build_model(
    configuration,
    operation_count,
    maximum_sequence_length,
):
    model_id = configuration[
        "baseline_id"
    ]

    if model_id == "B2":
        return try_constructor(
            OperationConditionedMLP,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        configuration[
                            "operation_embedding_dim"
                        ],

                    "hidden_width":
                        configuration[
                            "hidden_width"
                        ],
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        configuration[
                            "operation_embedding_dim"
                        ],

                    "hidden_width":
                        configuration[
                            "hidden_width"
                        ],
                },
            ],
        )

    if model_id == "B3":
        return try_constructor(
            GRUOperationSequenceModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        configuration[
                            "operation_embedding_dim"
                        ],

                    "hidden_size":
                        configuration[
                            "hidden_size"
                        ],
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "operation_embedding_dim":
                        configuration[
                            "operation_embedding_dim"
                        ],

                    "hidden_size":
                        configuration[
                            "hidden_size"
                        ],
                },
            ],
        )

    if model_id == "B4":
        common = {
            "operation_count":
                operation_count,

            "model_dimension":
                configuration[
                    "model_dimension"
                ],

            "attention_heads":
                configuration[
                    "attention_heads"
                ],

            "layer_count":
                configuration[
                    "layer_count"
                ],

            "feedforward_dimension":
                configuration[
                    "feedforward_dimension"
                ],

            "dropout":
                configuration[
                    "dropout"
                ],

            "maximum_sequence_length":
                maximum_sequence_length,
        }

        return try_constructor(
            TransformerOperationSequenceModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    **common,
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    **common,
                },
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    **{
                        key: value
                        for key, value
                        in common.items()
                        if key
                        != "maximum_sequence_length"
                    },
                },
            ],
        )

    if model_id == "B5":
        return try_constructor(
            ContinuousLatentOperatorModel,
            [
                {
                    "observation_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "latent_dimension":
                        configuration[
                            "latent_dimension"
                        ],

                    "encoder_width":
                        configuration[
                            "encoder_width"
                        ],

                    "operator_hidden_width":
                        configuration[
                            "operator_hidden_width"
                        ],
                },
                {
                    "obs_dim":
                        OBSERVATION_DIMENSION,

                    "operation_count":
                        operation_count,

                    "latent_dimension":
                        configuration[
                            "latent_dimension"
                        ],

                    "encoder_width":
                        configuration[
                            "encoder_width"
                        ],

                    "operator_hidden_width":
                        configuration[
                            "operator_hidden_width"
                        ],
                },
            ],
        )

    raise ValueError(
        f"Unsupported model: {model_id}"
    )


def train_baseline_configuration(
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    maximum_sequence_length,
    registry,
    device,
    configuration_dir,
):
    model_id = configuration[
        "baseline_id"
    ]

    set_random_seed(
        DEVELOPMENT_SEED
    )

    train_loader = make_loader(
        vectors=train_cell["vectors"],
        records=train_cell["records"],
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=True,
    )

    validation_loader = make_loader(
        vectors=validation_cell["vectors"],
        records=validation_cell["records"],
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=False,
    )

    model = build_model(
        configuration=configuration,
        operation_count=operation_count,
        maximum_sequence_length=(
            maximum_sequence_length
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            configuration[
                "learning_rate"
            ]
        ),
    )

    initial_validation_mse = (
        evaluate_affine_or_baseline(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    best_validation_mse = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    checkpoint_path = (
        configuration_dir
        / "best_model.pt"
    )

    history = []

    for epoch in range(
        1,
        int(
            registry[
                "maximum_epochs"
            ]
        )
        + 1,
    ):
        train_metrics = train_one_epoch(
            baseline_id=model_id,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        validation_mse = (
            evaluate_affine_or_baseline(
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
            f"{configuration['configuration_id']} | "
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
            > float(
                registry[
                    "early_stopping_min_delta"
                ]
            )
        ):
            best_validation_mse = float(
                validation_mse
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "phase":
                        "3D tuning only",

                    "eligible_for_final_use":
                        False,

                    "configuration":
                        configuration,

                    "epoch":
                        epoch,

                    "validation_rollout_mse":
                        validation_mse,

                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= int(
                registry[
                    "early_stopping_patience"
                ]
            )
        ):
            break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No validation-selected checkpoint saved."
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

    write_csv(
        configuration_dir
        / "training_history.csv",
        history,
    )

    finite = state_is_finite(model)

    return {
        "baseline_id":
            model_id,

        "configuration_id":
            configuration[
                "configuration_id"
            ],

        **configuration,

        "seed":
            DEVELOPMENT_SEED,

        "noise_fraction":
            DEVELOPMENT_NOISE,

        "parameter_count":
            parameter_count(model),

        "initial_validation_rollout_mse":
            float(
                initial_validation_mse
            ),

        "best_validation_rollout_mse":
            float(
                checkpoint[
                    "validation_rollout_mse"
                ]
            ),

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "epochs_executed":
            len(history),

        "all_state_values_finite":
            finite,

        "numerically_valid":
            bool(
                finite
                and np.isfinite(
                    checkpoint[
                        "validation_rollout_mse"
                    ]
                )
            ),
    }


def train_ocm_configuration(
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    registry,
    device,
    configuration_dir,
):
    set_random_seed(
        DEVELOPMENT_SEED
    )

    train_loader = make_loader(
        vectors=train_cell["vectors"],
        records=train_cell["records"],
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=True,
    )

    validation_loader = make_loader(
        vectors=validation_cell["vectors"],
        records=validation_cell["records"],
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=False,
    )

    model = OperationChannelModel(
        observation_dim=(
            OBSERVATION_DIMENSION
        ),
        latent_state_count=(
            LATENT_STATE_COUNT
        ),
        operation_count=(
            operation_count
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(
            configuration[
                "learning_rate"
            ]
        ),
    )

    temperature = float(
        configuration[
            "temperature"
        ]
    )

    initial_validation_mse = evaluate_ocm(
        model=model,
        loader=validation_loader,
        device=device,
        temperature=temperature,
    )

    best_validation_mse = float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    checkpoint_path = (
        configuration_dir
        / "best_model.pt"
    )

    history = []

    for epoch in range(
        1,
        int(
            registry[
                "maximum_epochs"
            ]
        )
        + 1,
    ):
        model.train()

        metric_sums = {
            "total": 0.0,
            "reconstruction": 0.0,
            "step": 0.0,
            "rollout": 0.0,
            "transition": 0.0,
            "determinism": 0.0,
        }

        batch_count = 0

        for batch in train_loader:
            batch = {
                key:
                    (
                        value.to(device)
                        if torch.is_tensor(value)
                        else value
                    )
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            losses = ocm_forward_losses(
                model=model,
                batch=batch,
                operation_count=operation_count,
                temperature=temperature,
            )

            total_loss = (
                losses[
                    "reconstruction_loss"
                ]
                + losses[
                    "step_loss"
                ]
                + float(
                    configuration[
                        "lambda_rollout"
                    ]
                )
                * losses[
                    "final_rollout_loss"
                ]
                + float(
                    configuration[
                        "lambda_transition"
                    ]
                )
                * losses[
                    "transition_loss"
                ]
                + float(
                    configuration[
                        "lambda_determinism"
                    ]
                )
                * losses[
                    "determinism_loss"
                ]
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            metric_sums["total"] += float(
                total_loss.detach().item()
            )

            metric_sums[
                "reconstruction"
            ] += float(
                losses[
                    "reconstruction_loss"
                ].detach().item()
            )

            metric_sums["step"] += float(
                losses[
                    "step_loss"
                ].detach().item()
            )

            metric_sums["rollout"] += float(
                losses[
                    "final_rollout_loss"
                ].detach().item()
            )

            metric_sums[
                "transition"
            ] += float(
                losses[
                    "transition_loss"
                ].detach().item()
            )

            metric_sums[
                "determinism"
            ] += float(
                losses[
                    "determinism_loss"
                ].detach().item()
            )

            batch_count += 1

        validation_mse = evaluate_ocm(
            model=model,
            loader=validation_loader,
            device=device,
            temperature=temperature,
        )

        row = {
            "epoch":
                epoch,

            "train_total_loss":
                metric_sums["total"]
                / batch_count,

            "train_reconstruction_loss":
                metric_sums[
                    "reconstruction"
                ]
                / batch_count,

            "train_step_loss":
                metric_sums["step"]
                / batch_count,

            "train_rollout_loss":
                metric_sums["rollout"]
                / batch_count,

            "train_transition_loss":
                metric_sums[
                    "transition"
                ]
                / batch_count,

            "train_determinism_loss":
                metric_sums[
                    "determinism"
                ]
                / batch_count,

            "validation_rollout_mse":
                validation_mse,
        }

        history.append(row)

        print(
            f"{configuration['configuration_id']} | "
            f"epoch={epoch:03d} | "
            f"train={row['train_total_loss']:.6f} | "
            f"val={validation_mse:.6f}"
        )

        improvement = (
            best_validation_mse
            - validation_mse
        )

        if (
            improvement
            > float(
                registry[
                    "early_stopping_min_delta"
                ]
            )
        ):
            best_validation_mse = float(
                validation_mse
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "phase":
                        "3D tuning only",

                    "eligible_for_final_use":
                        False,

                    "configuration":
                        configuration,

                    "epoch":
                        epoch,

                    "temperature":
                        temperature,

                    "validation_rollout_mse":
                        validation_mse,

                    "model_state_dict":
                        model.state_dict(),
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= int(
                registry[
                    "early_stopping_patience"
                ]
            )
        ):
            break

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No OCM checkpoint saved."
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

    write_csv(
        configuration_dir
        / "training_history.csv",
        history,
    )

    finite = state_is_finite(model)

    row_sum_error = channel_row_sum_error(
        model=model,
        operation_count=operation_count,
        temperature=temperature,
    )

    return {
        "baseline_id":
            "OCM",

        "configuration_id":
            configuration[
                "configuration_id"
            ],

        **configuration,

        "seed":
            DEVELOPMENT_SEED,

        "noise_fraction":
            DEVELOPMENT_NOISE,

        "parameter_count":
            parameter_count(model),

        "initial_validation_rollout_mse":
            float(
                initial_validation_mse
            ),

        "best_validation_rollout_mse":
            float(
                checkpoint[
                    "validation_rollout_mse"
                ]
            ),

        "best_epoch":
            int(
                checkpoint[
                    "epoch"
                ]
            ),

        "epochs_executed":
            len(history),

        "maximum_channel_row_sum_error":
            row_sum_error,

        "all_state_values_finite":
            finite,

        "numerically_valid":
            bool(
                finite
                and np.isfinite(
                    checkpoint[
                        "validation_rollout_mse"
                    ]
                )
                and (
                    row_sum_error is None
                    or row_sum_error < 1e-6
                )
            ),
    }


def run_affine_configuration(
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    device,
    configuration_dir,
):
    validation_loader = make_loader(
        vectors=validation_cell["vectors"],
        records=validation_cell["records"],
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=False,
    )

    model = fit_affine_model(
        vectors=train_cell["vectors"],
        records=train_cell["records"],
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        operation_count=operation_count,
        ridge=float(
            configuration["ridge"]
        ),
    ).to(device)

    validation_mse = evaluate_affine_or_baseline(
        model=model,
        loader=validation_loader,
        device=device,
    )

    torch.save(
        {
            "phase":
                "3D tuning only",

            "eligible_for_final_use":
                False,

            "configuration":
                configuration,

            "validation_rollout_mse":
                validation_mse,

            "model_state_dict":
                model.state_dict(),
        },
        configuration_dir
        / "best_model.pt",
    )

    finite = state_is_finite(model)

    return {
        "baseline_id":
            "B1",

        "configuration_id":
            configuration[
                "configuration_id"
            ],

        **configuration,

        "seed":
            "deterministic",

        "noise_fraction":
            DEVELOPMENT_NOISE,

        "parameter_count":
            int(
                model.matrices.numel()
                + model.biases.numel()
            ),

        "initial_validation_rollout_mse":
            None,

        "best_validation_rollout_mse":
            float(validation_mse),

        "best_epoch":
            None,

        "epochs_executed":
            0,

        "all_state_values_finite":
            finite,

        "numerically_valid":
            bool(
                finite
                and np.isfinite(
                    validation_mse
                )
            ),
    }


def finalize_results(
    registry,
    persistence_validation_mse,
):
    results = []

    for configuration in registry[
        "configurations"
    ]:
        result_path = (
            CONFIGURATION_DIR
            / configuration[
                "configuration_id"
            ]
            / "result.json"
        )

        if not result_path.exists():
            return None

        result = load_json(
            result_path
        )

        if result["status"] != "completed":
            return None

        if result["numerically_valid"] is not True:
            raise AssertionError(
                "Numerically invalid configuration: "
                + configuration[
                    "configuration_id"
                ]
            )

        results.append(result)

    selected = {}

    for model_id in (
        "OCM",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        candidates = [
            result
            for result in results
            if result[
                "baseline_id"
            ] == model_id
        ]

        winner = min(
            candidates,
            key=lambda result: (
                result[
                    "best_validation_rollout_mse"
                ],
                result[
                    "configuration_id"
                ],
            ),
        )

        selected[model_id] = winner

        selected_model_dir = (
            SELECTED_DIR / model_id
        )

        selected_model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_checkpoint = (
            CONFIGURATION_DIR
            / winner[
                "configuration_id"
            ]
            / "best_model.pt"
        )

        shutil.copy2(
            source_checkpoint,
            selected_model_dir
            / "tuning_checkpoint.pt",
        )

        write_json(
            selected_model_dir
            / "selected_result.json",
            winner,
        )

    write_csv(
        PHASE3D_DIR
        / "all_tuning_results.csv",
        results,
    )

    write_json(
        PHASE3D_DIR
        / "selected_configurations.json",
        selected,
    )

    summary = {
        "phase":
            "3D Tier B leakage-safe tuning",

        "configuration_count":
            len(results),

        "completed_configuration_count":
            len(results),

        "configuration_counts":
            EXPECTED_COUNTS,

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "train_sequence_count":
            1376,

        "validation_sequence_count":
            144,

        "train_trajectory_count":
            11008,

        "validation_trajectory_count":
            1152,

        "persistence_validation_mse":
            float(
                persistence_validation_mse
            ),

        "selection_metric":
            (
                "val_joint noisy-target "
                "rollout MSE only"
            ),

        "selected_configurations":
            selected,

        "all_configurations_numerically_valid":
            True,

        "test_cells_opened":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_metrics_computed":
            False,

        "structural_labels_read":
            False,

        "tuning_checkpoints_eligible_for_final_use":
            False,

        "phase3d_status":
            "passed",
    }

    write_json(
        PHASE3D_DIR
        / "phase3d_summary.json",
        summary,
    )

    return summary


def main():
    arguments = parse_arguments()

    CONFIGURATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SELECTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = validate_registry()

    device = resolve_device(
        arguments.device
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    operation_vocabulary = (
        load_operation_vocabulary()
    )

    operation_count = len(
        operation_vocabulary
    )

    train_cell = load_cell(
        cell_id="train_joint",
        operation_vocabulary=(
            operation_vocabulary
        ),
    )

    validation_cell = load_cell(
        cell_id="val_joint",
        operation_vocabulary=(
            operation_vocabulary
        ),
    )

    maximum_sequence_length = max(
        record.point_count - 1
        for record in (
            train_cell["records"]
            + validation_cell["records"]
        )
    )

    validation_loader = make_loader(
        vectors=validation_cell["vectors"],
        records=validation_cell["records"],
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=DEVELOPMENT_SEED,
        shuffle=False,
    )

    persistence_validation_mse = (
        evaluate_persistence(
            loader=validation_loader,
            device=device,
        )
    )

    requested = arguments.configuration_id

    configurations = [
        configuration
        for configuration
        in registry["configurations"]
        if (
            requested == "all"
            or configuration[
                "configuration_id"
            ] == requested
        )
    ]

    if not configurations:
        raise ValueError(
            "No matching configuration: "
            f"{requested}"
        )

    for index, configuration in enumerate(
        configurations,
        start=1,
    ):
        configuration_id = (
            configuration[
                "configuration_id"
            ]
        )

        configuration_dir = (
            CONFIGURATION_DIR
            / configuration_id
        )

        configuration_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_path = (
            configuration_dir
            / "result.json"
        )

        if (
            result_path.exists()
            and not arguments.force
        ):
            existing = load_json(
                result_path
            )

            if existing.get(
                "status"
            ) == "completed":
                print(
                    f"[{index}/{len(configurations)}] "
                    f"{configuration_id} already completed."
                )
                continue

        print()
        print(
            f"[{index}/{len(configurations)}] "
            f"Running {configuration_id}"
        )

        model_id = configuration[
            "baseline_id"
        ]

        if model_id == "B1":
            result = run_affine_configuration(
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                device=device,
                configuration_dir=(
                    configuration_dir
                ),
            )

        elif model_id == "OCM":
            result = train_ocm_configuration(
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                registry=registry,
                device=device,
                configuration_dir=(
                    configuration_dir
                ),
            )

        else:
            result = train_baseline_configuration(
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                maximum_sequence_length=(
                    maximum_sequence_length
                ),
                registry=registry,
                device=device,
                configuration_dir=(
                    configuration_dir
                ),
            )

        result.update(
            {
                "status":
                    "completed",

                "selection_metric":
                    (
                        "val_joint noisy-target "
                        "rollout MSE only"
                    ),

                "test_cells_opened":
                    False,

                "privileged_arrays_opened":
                    False,

                "clean_targets_read":
                    False,

                "structural_labels_read":
                    False,

                "eligible_for_final_use":
                    False,
            }
        )

        write_json(
            result_path,
            result,
        )

        print(
            json.dumps(
                result,
                indent=2,
            )
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = finalize_results(
        registry=registry,
        persistence_validation_mse=(
            persistence_validation_mse
        ),
    )

    if summary is None:
        completed_count = sum(
            (
                CONFIGURATION_DIR
                / configuration[
                    "configuration_id"
                ]
                / "result.json"
            ).exists()
            for configuration
            in registry[
                "configurations"
            ]
        )

        print()
        print(
            f"Phase 3D progress: "
            f"{completed_count}/43 configurations "
            f"have result files."
        )

    else:
        print()
        print(
            "Phase 3D tuning completed."
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print(
            f"Outputs written to: {PHASE3D_DIR}"
        )


if __name__ == "__main__":
    main()
