from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from phase2b_operation_channel_model import (
    OperationChannelModel,
)

from phase2b_run_ocm_smoke_test import (
    TrajectoryRecord,
    set_random_seed,
)

from phase2h_tune_predictive_baselines import (
    train_one_epoch,
)

from phase3c_run_tier_b_smoke_tests import (
    OBSERVATION_DIMENSION,
    LATENT_STATE_COUNT,
    BATCH_SIZE,
    load_operation_vocabulary,
    make_loader,
    evaluate_affine_or_baseline,
    evaluate_ocm,
    fit_affine_model,
    state_is_finite,
    parameter_count,
    channel_row_sum_error,
    ocm_forward_losses,
)

from phase3d_tune_tier_b_models import (
    build_model,
)


PHASE3B_DATA_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

VISIBLE_DIR = (
    PHASE3B_DATA_DIR / "visible"
)

PHASE3E_DIR = Path(
    "outputs/phase3e_tier_b_final_fits"
)

RUNS_DIR = PHASE3E_DIR / "runs"


EXPECTED_RUN_COUNTS = {
    "OCM": 25,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
}

ALLOWED_CELLS = {
    "train_joint",
    "val_joint",
}

EXPECTED_TRAJECTORY_COUNTS = {
    "train_joint": 11008,
    "val_joint": 1152,
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
        "--run-id",
        default="all",
    )

    parser.add_argument(
        "--model-id",
        choices=[
            "all",
            "OCM",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
        ],
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
                "CUDA was requested but is unavailable."
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


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


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


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(values.mean()),

        "std":
            (
                float(
                    values.std(ddof=1)
                )
                if len(values) > 1
                else 0.0
            ),

        "minimum":
            float(values.min()),

        "maximum":
            float(values.max()),
    }


def validate_registry():
    registry = load_json(
        PHASE3E_DIR
        / "final_fit_registry.json"
    )

    if registry["registry_status"] != "frozen":
        raise AssertionError(
            "Phase 3E registry is not frozen."
        )

    if registry["run_count"] != 130:
        raise AssertionError(
            "Expected 130 final-fit runs."
        )

    if int(registry["batch_size"]) != 128:
        raise AssertionError(
            "Frozen final-fit batch size changed."
        )

    if BATCH_SIZE != 128:
        raise AssertionError(
            "Imported loader batch size changed."
        )

    if BATCH_SIZE != int(
        registry["batch_size"]
    ):
        raise AssertionError(
            "Loader and registry batch sizes differ."
        )

    if registry["training_cell"] != "train_joint":
        raise AssertionError(
            "Unexpected training cell."
        )

    if (
        registry[
            "checkpoint_selection_cell"
        ]
        != "val_joint"
    ):
        raise AssertionError(
            "Unexpected validation cell."
        )

    counts = Counter(
        row["model_id"]
        for row in registry["runs"]
    )

    if dict(counts) != EXPECTED_RUN_COUNTS:
        raise AssertionError(
            f"Run counts changed: {dict(counts)}"
        )

    return registry


def parse_operations(value):
    parsed = json.loads(value)

    if not isinstance(parsed, list):
        raise TypeError(
            "Operation sequence must be a JSON list."
        )

    return tuple(
        str(item)
        for item in parsed
    )


def load_cell(
    cell_id,
    noise_fraction,
    operation_vocabulary,
):
    if cell_id not in ALLOWED_CELLS:
        raise ValueError(
            "Phase 3E may open only train_joint "
            "and val_joint."
        )

    observation_path = (
        VISIBLE_DIR
        / f"{cell_id}_observations.npz"
    )

    trajectory_path = (
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv"
    )

    sequence_path = (
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv"
    )

    with np.load(
        observation_path
    ) as data:
        if set(data.files) != {
            "noise_fractions",
            "vectors",
        }:
            raise AssertionError(
                f"{cell_id} contains unexpected arrays."
            )

        noise_levels = np.asarray(
            data["noise_fractions"],
            dtype=np.float64,
        )

        matches = np.where(
            np.isclose(
                noise_levels,
                noise_fraction,
            )
        )[0]

        if len(matches) != 1:
            raise AssertionError(
                f"Noise {noise_fraction} not found "
                f"for {cell_id}."
            )

        vectors = np.array(
            data["vectors"][
                int(matches[0])
            ],
            dtype=np.float32,
            copy=True,
        )

    if vectors.shape[1] != OBSERVATION_DIMENSION:
        raise AssertionError(
            f"{cell_id} observation dimension changed."
        )

    if not np.isfinite(vectors).all():
        raise FloatingPointError(
            f"{cell_id} contains non-finite values."
        )

    sequence_rows = load_csv(
        sequence_path
    )

    operations_by_sequence = {}

    for row in sequence_rows:
        sequence_index = int(
            row["sequence_index"]
        )

        operation_names = parse_operations(
            row["operation_sequence"]
        )

        unknown = [
            operation
            for operation in operation_names
            if operation not in operation_vocabulary
        ]

        if unknown:
            raise AssertionError(
                "Unknown operations: "
                + ", ".join(unknown)
            )

        operations_by_sequence[
            sequence_index
        ] = tuple(
            operation_vocabulary[
                operation
            ]
            for operation in operation_names
        )

    trajectory_rows = load_csv(
        trajectory_path
    )

    records = []

    for row in trajectory_rows:
        sequence_index = int(
            row["sequence_index"]
        )

        record = TrajectoryRecord(
            trajectory_index=int(
                row["trajectory_index"]
            ),

            sequence_index=sequence_index,

            point_start=int(
                row["point_start"]
            ),

            point_count=int(
                row["point_count"]
            ),
        )

        expected_points = (
            len(
                operations_by_sequence[
                    sequence_index
                ]
            )
            + 1
        )

        if record.point_count != expected_points:
            raise AssertionError(
                "Trajectory length and operation "
                "sequence disagree."
            )

        records.append(record)

    if (
        len(records)
        != EXPECTED_TRAJECTORY_COUNTS[
            cell_id
        ]
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    return {
        "vectors": vectors,
        "records": records,
        "operations_by_sequence":
            operations_by_sequence,
    }


def build_loaders(
    train_cell,
    validation_cell,
    seed,
):
    train_loader = make_loader(
        vectors=train_cell["vectors"],
        records=train_cell["records"],
        operations_by_sequence=(
            train_cell[
                "operations_by_sequence"
            ]
        ),
        seed=seed,
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
        seed=seed,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
    )


def train_baseline(
    run,
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    maximum_sequence_length,
    registry,
    device,
    run_dir,
):
    seed = int(run["seed"])

    set_random_seed(seed)

    (
        train_loader,
        validation_loader,
    ) = build_loaders(
        train_cell=train_cell,
        validation_cell=validation_cell,
        seed=seed,
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
        run_dir / "best_model.pt"
    )

    history = []

    for epoch in range(
        1,
        int(registry["maximum_epochs"]) + 1,
    ):
        train_metrics = train_one_epoch(
            baseline_id=run["model_id"],
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
                "epoch": epoch,

                "train_total_loss":
                    train_metrics["total"],

                "train_primary_loss":
                    train_metrics["primary"],

                "train_auxiliary_loss":
                    train_metrics["auxiliary"],

                "train_rollout_loss":
                    train_metrics["rollout"],

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"{run['run_id']} | "
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
                        "3E Tier B final fit",

                    "run_id":
                        run["run_id"],

                    "model_id":
                        run["model_id"],

                    "configuration":
                        configuration,

                    "seed":
                        seed,

                    "noise_fraction":
                        float(
                            run[
                                "noise_fraction"
                            ]
                        ),

                    "batch_size":
                        int(
                            registry[
                                "batch_size"
                            ]
                        ),

                    "epoch":
                        epoch,

                    "validation_rollout_mse":
                        validation_mse,

                    "eligible_for_final_evaluation":
                        True,

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

    selected_validation_mse = (
        evaluate_affine_or_baseline(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    write_csv(
        run_dir
        / "training_history.csv",
        history,
    )

    finite = state_is_finite(model)

    return {
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

        "reloaded_checkpoint_validation_mse":
            float(
                selected_validation_mse
            ),

        "best_epoch":
            int(
                checkpoint["epoch"]
            ),

        "epochs_executed":
            len(history),

        "all_state_values_finite":
            finite,

        "numerically_valid":
            bool(
                finite
                and np.isfinite(
                    selected_validation_mse
                )
            ),
    }


def train_ocm(
    run,
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    registry,
    device,
    run_dir,
):
    seed = int(run["seed"])

    set_random_seed(seed)

    (
        train_loader,
        validation_loader,
    ) = build_loaders(
        train_cell=train_cell,
        validation_cell=validation_cell,
        seed=seed,
    )

    model = OperationChannelModel(
        observation_dim=OBSERVATION_DIMENSION,
        latent_state_count=LATENT_STATE_COUNT,
        operation_count=operation_count,
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
        configuration["temperature"]
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
        run_dir / "best_model.pt"
    )

    history = []

    for epoch in range(
        1,
        int(registry["maximum_epochs"]) + 1,
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
                for key, value in batch.items()
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

        history.append(
            {
                "epoch":
                    epoch,

                "train_total_loss":
                    metric_sums["total"]
                    / batch_count,

                "train_reconstruction_loss":
                    metric_sums["reconstruction"]
                    / batch_count,

                "train_step_loss":
                    metric_sums["step"]
                    / batch_count,

                "train_rollout_loss":
                    metric_sums["rollout"]
                    / batch_count,

                "train_transition_loss":
                    metric_sums["transition"]
                    / batch_count,

                "train_determinism_loss":
                    metric_sums["determinism"]
                    / batch_count,

                "validation_rollout_mse":
                    validation_mse,
            }
        )

        print(
            f"{run['run_id']} | "
            f"epoch={epoch:03d} | "
            f"train="
            f"{history[-1]['train_total_loss']:.6f} | "
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
                        "3E Tier B final fit",

                    "run_id":
                        run["run_id"],

                    "model_id":
                        "OCM",

                    "configuration":
                        configuration,

                    "seed":
                        seed,

                    "noise_fraction":
                        float(
                            run[
                                "noise_fraction"
                            ]
                        ),

                    "batch_size":
                        int(
                            registry[
                                "batch_size"
                            ]
                        ),

                    "epoch":
                        epoch,

                    "temperature":
                        temperature,

                    "validation_rollout_mse":
                        validation_mse,

                    "eligible_for_final_evaluation":
                        True,

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
            "No OCM checkpoint was saved."
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

    selected_validation_mse = evaluate_ocm(
        model=model,
        loader=validation_loader,
        device=device,
        temperature=temperature,
    )

    write_csv(
        run_dir
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
        "parameter_count":
            parameter_count(model),

        "temperature":
            temperature,

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

        "reloaded_checkpoint_validation_mse":
            float(
                selected_validation_mse
            ),

        "best_epoch":
            int(
                checkpoint["epoch"]
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
                    selected_validation_mse
                )
                and (
                    row_sum_error is None
                    or row_sum_error < 1e-6
                )
            ),
    }


def fit_affine(
    run,
    configuration,
    train_cell,
    validation_cell,
    operation_count,
    registry,
    device,
    run_dir,
):
    validation_loader = make_loader(
        vectors=validation_cell["vectors"],
        records=validation_cell["records"],
        operations_by_sequence=(
            validation_cell[
                "operations_by_sequence"
            ]
        ),
        seed=11,
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

    validation_mse = (
        evaluate_affine_or_baseline(
            model=model,
            loader=validation_loader,
            device=device,
        )
    )

    checkpoint_path = (
        run_dir / "best_model.pt"
    )

    torch.save(
        {
            "phase":
                "3E Tier B final fit",

            "run_id":
                run["run_id"],

            "model_id":
                "B1",

            "configuration":
                configuration,

            "seed":
                "deterministic",

            "noise_fraction":
                float(
                    run[
                        "noise_fraction"
                    ]
                ),

            "batch_size":
                int(
                    registry[
                        "batch_size"
                    ]
                ),

            "validation_rollout_mse":
                validation_mse,

            "eligible_for_final_evaluation":
                True,

            "model_state_dict":
                model.state_dict(),
        },
        checkpoint_path,
    )

    finite = state_is_finite(model)

    return {
        "parameter_count":
            int(
                model.matrices.numel()
                + model.biases.numel()
            ),

        "initial_validation_rollout_mse":
            None,

        "best_validation_rollout_mse":
            float(validation_mse),

        "reloaded_checkpoint_validation_mse":
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


def finalize_results(registry):
    results = []

    for run in registry["runs"]:
        result_path = (
            RUNS_DIR
            / run["run_id"]
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
                "Numerically invalid final run: "
                + run["run_id"]
            )

        checkpoint_path = (
            RUNS_DIR
            / run["run_id"]
            / "best_model.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        results.append(result)

    counts = Counter(
        result["model_id"]
        for result in results
    )

    if dict(counts) != EXPECTED_RUN_COUNTS:
        raise AssertionError(
            f"Completed run counts changed: {dict(counts)}"
        )

    flat_rows = []

    for result in results:
        flat_rows.append(
            {
                "run_id":
                    result["run_id"],

                "model_id":
                    result["model_id"],

                "configuration_id":
                    result[
                        "configuration_id"
                    ],

                "seed":
                    result["seed"],

                "noise_fraction":
                    result[
                        "noise_fraction"
                    ],

                "parameter_count":
                    result[
                        "parameter_count"
                    ],

                "initial_validation_rollout_mse":
                    result.get(
                        "initial_validation_rollout_mse"
                    ),

                "best_validation_rollout_mse":
                    result[
                        "best_validation_rollout_mse"
                    ],

                "reloaded_checkpoint_validation_mse":
                    result[
                        "reloaded_checkpoint_validation_mse"
                    ],

                "best_epoch":
                    result.get(
                        "best_epoch"
                    ),

                "epochs_executed":
                    result[
                        "epochs_executed"
                    ],

                "batch_size":
                    result[
                        "batch_size"
                    ],

                "numerically_valid":
                    result[
                        "numerically_valid"
                    ],
            }
        )

    write_csv(
        PHASE3E_DIR
        / "all_final_fit_results.csv",
        flat_rows,
    )

    grouped = defaultdict(list)

    for result in results:
        grouped[
            (
                result["model_id"],
                float(
                    result[
                        "noise_fraction"
                    ]
                ),
            )
        ].append(
            float(
                result[
                    "best_validation_rollout_mse"
                ]
            )
        )

    validation_rows = []

    for (
        model_id,
        noise_fraction,
    ), values in sorted(
        grouped.items()
    ):
        statistics = summarize(values)

        validation_rows.append(
            {
                "model_id":
                    model_id,

                "noise_fraction":
                    noise_fraction,

                "fit_count":
                    len(values),

                "validation_mse_mean":
                    statistics["mean"],

                "validation_mse_std":
                    statistics["std"],

                "validation_mse_minimum":
                    statistics["minimum"],

                "validation_mse_maximum":
                    statistics["maximum"],
            }
        )

    write_csv(
        PHASE3E_DIR
        / "validation_summary.csv",
        validation_rows,
    )

    summary = {
        "phase":
            "3E frozen Tier B final fits",

        "registered_run_count":
            registry["run_count"],

        "completed_run_count":
            len(results),

        "completed_run_counts":
            dict(counts),

        "deterministic_fit_count":
            counts["B1"],

        "neural_fit_count":
            (
                len(results)
                - counts["B1"]
            ),

        "selected_configuration_ids": {
            model_id:
                registry[
                    "selected_configurations"
                ][model_id][
                    "configuration_id"
                ]
            for model_id in (
                "OCM",
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
            )
        },

        "seeds":
            registry["seeds"],

        "noise_levels":
            registry["noise_levels"],

        "batch_size":
            registry["batch_size"],

        "training_cell":
            registry["training_cell"],

        "checkpoint_selection_cell":
            registry[
                "checkpoint_selection_cell"
            ],

        "selection_metric":
            registry["selection_metric"],

        "fresh_initialization_used":
            True,

        "tuning_checkpoints_reused":
            False,

        "all_runs_numerically_valid":
            True,

        "final_checkpoints_eligible_for_evaluation":
            True,

        "test_cells_opened":
            False,

        "test_metrics_computed":
            False,

        "privileged_arrays_opened":
            False,

        "clean_targets_read":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_read":
            False,

        "phase2_outputs_modified":
            False,

        "phase3e_status":
            "passed",
    }

    write_json(
        PHASE3E_DIR
        / "phase3e_summary.json",
        summary,
    )

    return summary


def main():
    arguments = parse_arguments()

    RUNS_DIR.mkdir(
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

    selected_configurations = (
        registry[
            "selected_configurations"
        ]
    )

    requested_runs = [
        run
        for run in registry["runs"]
        if (
            (
                arguments.run_id == "all"
                or run["run_id"]
                == arguments.run_id
            )
            and (
                arguments.model_id == "all"
                or run["model_id"]
                == arguments.model_id
            )
        )
    ]

    if not requested_runs:
        raise ValueError(
            "No final-fit runs match the request."
        )

    current_noise = None
    train_cell = None
    validation_cell = None
    maximum_sequence_length = None

    for index, run in enumerate(
        requested_runs,
        start=1,
    ):
        run_id = run["run_id"]

        run_dir = (
            RUNS_DIR / run_id
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result_path = (
            run_dir / "result.json"
        )

        if (
            result_path.exists()
            and not arguments.force
        ):
            existing = load_json(
                result_path
            )

            if existing.get("status") == "completed":
                print(
                    f"[{index}/{len(requested_runs)}] "
                    f"{run_id} already completed."
                )
                continue

        noise_fraction = float(
            run["noise_fraction"]
        )

        if (
            current_noise is None
            or not np.isclose(
                current_noise,
                noise_fraction,
            )
        ):
            print()
            print(
                f"Loading visible train/validation "
                f"data for noise={noise_fraction}"
            )

            train_cell = load_cell(
                cell_id="train_joint",
                noise_fraction=noise_fraction,
                operation_vocabulary=(
                    operation_vocabulary
                ),
            )

            validation_cell = load_cell(
                cell_id="val_joint",
                noise_fraction=noise_fraction,
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

            current_noise = noise_fraction

        print()
        print(
            f"[{index}/{len(requested_runs)}] "
            f"Running {run_id}"
        )

        model_id = run["model_id"]

        configuration = (
            selected_configurations[
                model_id
            ]
        )

        if model_id == "B1":
            fit_result = fit_affine(
                run=run,
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                registry=registry,
                device=device,
                run_dir=run_dir,
            )

        elif model_id == "OCM":
            fit_result = train_ocm(
                run=run,
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                registry=registry,
                device=device,
                run_dir=run_dir,
            )

        else:
            fit_result = train_baseline(
                run=run,
                configuration=configuration,
                train_cell=train_cell,
                validation_cell=validation_cell,
                operation_count=operation_count,
                maximum_sequence_length=(
                    maximum_sequence_length
                ),
                registry=registry,
                device=device,
                run_dir=run_dir,
            )

        result = {
            "phase":
                "3E Tier B final fit",

            "run_id":
                run_id,

            "model_id":
                model_id,

            "configuration_id":
                configuration[
                    "configuration_id"
                ],

            "configuration":
                configuration,

            "seed":
                run["seed"],

            "noise_fraction":
                noise_fraction,

            "fit_type":
                run["fit_type"],

            "batch_size":
                int(
                    registry[
                        "batch_size"
                    ]
                ),

            **fit_result,

            "training_cell":
                "train_joint",

            "checkpoint_selection_cell":
                "val_joint",

            "selection_metric":
                registry[
                    "selection_metric"
                ],

            "fresh_initialization_used":
                True,

            "tuning_checkpoint_reused":
                False,

            "test_cells_opened":
                False,

            "test_metrics_computed":
                False,

            "privileged_arrays_opened":
                False,

            "clean_targets_read":
                False,

            "clean_targets_used_for_training":
                False,

            "clean_targets_used_for_selection":
                False,

            "structural_labels_read":
                False,

            "eligible_for_final_evaluation":
                True,

            "status":
                "completed",
        }

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
        registry
    )

    if summary is None:
        completed_count = sum(
            (
                RUNS_DIR
                / run["run_id"]
                / "result.json"
            ).exists()
            for run in registry["runs"]
        )

        print()
        print(
            "Phase 3E progress: "
            f"{completed_count}/130 result files present."
        )

    else:
        print()
        print(
            "Phase 3E final fits completed."
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        print(
            f"Outputs written to: {PHASE3E_DIR}"
        )


if __name__ == "__main__":
    main()
