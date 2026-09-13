from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4fr3b_run_tier_c_v4_tuning as tuning_runtime


PROTOCOL_VERSION = "tier_c_v4"

NOISE_FRACTIONS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_VAL_TRAJECTORY_COUNT = 1152
PHYSICAL_MICROBATCH_SIZE = 4

FINAL_RUN_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_final_run_registry.csv"
)

METRIC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_metric_registry.csv"
)

TEST_POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

FINAL_FIT_FREEZE_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "phase4gr3c_final_fit_freeze_summary.json"
)

COMPARATOR_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator/"
    "phase4gr3d1_primary_comparator_summary.json"
)

NORMALIZATION_BINDING_PATH = Path(
    "outputs/"
    "phase4gr3ar2_tier_c_v4_normalization_binding/"
    "phase4gr3ar2_normalization_binding_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3d2_tier_c_v4_b0_persistence"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value: Any,
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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(rows),
        f"No rows supplied for {path}.",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def close_float(
    left: float,
    right: float,
) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def validate_sources() -> list[dict[str, str]]:
    required_paths = (
        FINAL_RUN_REGISTRY_PATH,
        METRIC_REGISTRY_PATH,
        TEST_POLICY_PATH,
        FINAL_FIT_FREEZE_PATH,
        COMPARATOR_SUMMARY_PATH,
        NORMALIZATION_BINDING_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    freeze = load_json(
        FINAL_FIT_FREEZE_PATH
    )

    comparator = load_json(
        COMPARATOR_SUMMARY_PATH
    )

    test_policy = load_json(
        TEST_POLICY_PATH
    )

    normalization = load_json(
        NORMALIZATION_BINDING_PATH
    )

    require(
        freeze[
            "phase4gr3c_status"
        ] == "final_fit_artifacts_frozen",
        "Final-fit artifacts are not frozen.",
    )

    require(
        freeze[
            "audited_final_fit_count"
        ] == 130,
        "Expected 130 frozen trainable fits.",
    )

    require(
        freeze[
            "additional_final_fit_training_authorized"
        ] is False,
        "Additional final-fit training remains authorized.",
    )

    require(
        freeze[
            "test_data_read"
        ] is False,
        "Test data was already read.",
    )

    require(
        freeze[
            "test_open_count"
        ] == 0,
        "Test open count is not zero.",
    )

    require(
        comparator[
            "phase4gr3d1_status"
        ] == "primary_comparator_frozen",
        "Primary comparator is not frozen.",
    )

    require(
        comparator[
            "selected_comparator"
        ] == "B4",
        "Primary baseline comparator changed.",
    )

    require(
        comparator[
            "test_data_read"
        ] is False,
        "Comparator freeze read test data.",
    )

    require(
        comparator[
            "test_open_count"
        ] == 0,
        "Comparator freeze opened test data.",
    )

    require(
        test_policy[
            "test_generation_authorized_now"
        ] is False,
        "Test generation is already authorized.",
    )

    require(
        test_policy[
            "test_evaluation_authorized_now"
        ] is False,
        "Test evaluation is already authorized.",
    )

    require(
        test_policy[
            "test_open_count"
        ] == 0,
        "Test was already opened.",
    )

    require(
        normalization[
            "phase4gr3ar2_status"
        ] == "frozen_normalization_artifact_bound",
        "Normalization binding is not frozen.",
    )

    require(
        normalization[
            "visible_fields_already_normalized"
        ] is True,
        "Visible fields are not frozen as normalized.",
    )

    require(
        normalization[
            "normalization_applied_by_final_fit_runner"
        ] is False,
        "Double normalization occurred.",
    )

    metric_rows = load_csv(
        METRIC_REGISTRY_PATH
    )

    primary_metric_rows = [
        row
        for row in metric_rows
        if row[
            "metric_id"
        ] == "noisy_target_rollout_mse"
    ]

    require(
        len(primary_metric_rows) == 1,
        "Primary predictive metric is not unique.",
    )

    primary_metric = primary_metric_rows[0]

    require(
        primary_metric[
            "target"
        ] == "stored noisy observation",
        "B0 target changed.",
    )

    require(
        primary_metric[
            "aggregation"
        ]
        == (
            "Mean squared error over forecast points, "
            "channels, and pixels; then mean within sequence."
        ),
        "B0 aggregation rule changed.",
    )

    rows = load_csv(
        FINAL_RUN_REGISTRY_PATH
    )

    b0_rows = [
        row
        for row in rows
        if row[
            "model_id"
        ] == "B0"
    ]

    require(
        len(b0_rows) == 5,
        "Expected five B0 persistence rows.",
    )

    require(
        all(
            row[
                "run_type"
            ] == "evaluation_only"
            for row in b0_rows
        ),
        "B0 run type changed.",
    )

    require(
        all(
            row[
                "configuration_id"
            ] == "persistence"
            for row in b0_rows
        ),
        "B0 configuration changed.",
    )

    require(
        all(
            row[
                "seed"
            ] == "deterministic"
            for row in b0_rows
        ),
        "B0 seed policy changed.",
    )

    require(
        all(
            row[
                "training_allowed"
            ].strip().lower()
            == "false"
            for row in b0_rows
        ),
        "B0 unexpectedly permits training.",
    )

    observed_noises = sorted(
        float(
            row[
                "noise_fraction"
            ]
        )
        for row in b0_rows
    )

    require(
        observed_noises
        == list(NOISE_FRACTIONS),
        "B0 noise grid changed.",
    )

    return b0_rows


def compute_persistence_batch_mse(
    batch: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    observations = batch[
        "observations"
    ]

    step_mask = batch[
        "step_mask"
    ]

    require(
        observations.ndim == 5,
        (
            "Expected observations with shape "
            "[batch, points, channels, height, width]."
        ),
    )

    require(
        step_mask.ndim == 2,
        "Expected step_mask with shape [batch, steps].",
    )

    batch_size = observations.shape[0]
    point_count = observations.shape[1]

    require(
        point_count >= 2,
        (
            "Persistence validation received "
            "a zero-step trajectory."
        ),
    )

    forecast_step_count = (
        point_count - 1
    )

    require(
        step_mask.shape
        == (
            batch_size,
            forecast_step_count,
        ),
        "step_mask shape does not match observations.",
    )

    # B0 persistence semantics:
    # predict the trajectory's current/initial field
    # at every future rollout step.
    initial_field = observations[
        :,
        0:1,
        ...,
    ]

    targets = observations[
        :,
        1:,
        ...,
    ]

    predictions = initial_field.expand_as(
        targets
    )

    squared_error = (
        predictions
        - targets
    ).pow(2)

    # Mean over channels and pixels for each
    # forecast step.
    per_step_mse = squared_error.mean(
        dim=(2, 3, 4)
    )

    mask = step_mask.to(
        dtype=per_step_mse.dtype
    )

    valid_steps = mask.sum(
        dim=1
    )

    require(
        bool(
            torch.all(
                valid_steps > 0
            ).item()
        ),
        (
            "B0 validation encountered "
            "a zero-step trajectory."
        ),
    )

    # Mean over valid forecast steps within each
    # trajectory. This is the sequence-level
    # quantity that is subsequently averaged
    # across validation trajectories.
    per_trajectory_mse = (
        (
            per_step_mse
            * mask
        ).sum(
            dim=1
        )
        / valid_steps
    )

    require(
        bool(
            torch.isfinite(
                per_trajectory_mse
            ).all().item()
        ),
        "Non-finite B0 trajectory MSE.",
    )

    batch_mean = (
        per_trajectory_mse.mean()
    )

    require(
        bool(
            torch.isfinite(
                batch_mean
            ).item()
        ),
        "Non-finite B0 batch MSE.",
    )

    return (
        batch_mean,
        per_trajectory_mse,
    )


@torch.no_grad()
def evaluate_noise_condition(
    *,
    validation_cell: Any,
    noise_index: int,
    noise_fraction: float,
    device: torch.device,
) -> dict[str, Any]:
    tuning_runtime.NOISE_INDEX = (
        noise_index
    )

    tuning_runtime.NOISE_FRACTION = (
        noise_fraction
    )

    order = np.arange(
        len(
            validation_cell.rows
        ),
        dtype=np.int64,
    )

    (
        order,
        zero_batch_repairs,
    ) = (
        tuning_runtime.repair_zero_only_microbatches(
            order,
            validation_cell.lengths,
        )
    )

    require(
        zero_batch_repairs == 0,
        (
            "val_joint unexpectedly required "
            "zero-step microbatch repair."
        ),
    )

    weighted_mse_sum = 0.0
    trajectory_count = 0
    microbatch_count = 0

    per_trajectory_values: list[float] = []

    start_time = time.time()

    for start in range(
        0,
        len(order),
        PHYSICAL_MICROBATCH_SIZE,
    ):
        selected = order[
            start:
            start
            + PHYSICAL_MICROBATCH_SIZE
        ].tolist()

        batch = validation_cell.make_batch(
            selected,
            device,
        )

        (
            batch_mean,
            per_trajectory_mse,
        ) = compute_persistence_batch_mse(
            batch
        )

        batch_size = len(
            selected
        )

        # Match evaluate_validation():
        # scalar batch loss × batch size,
        # then divide by total trajectories.
        weighted_mse_sum += (
            float(
                batch_mean.detach().cpu()
            )
            * float(
                batch_size
            )
        )

        trajectory_count += (
            batch_size
        )

        microbatch_count += 1

        per_trajectory_values.extend(
            float(value)
            for value in (
                per_trajectory_mse
                .detach()
                .cpu()
                .tolist()
            )
        )

    require(
        trajectory_count
        == EXPECTED_VAL_TRAJECTORY_COUNT,
        (
            "Unexpected validation trajectory count: "
            f"{trajectory_count}"
        ),
    )

    require(
        len(
            per_trajectory_values
        )
        == EXPECTED_VAL_TRAJECTORY_COUNT,
        "Per-trajectory metric count mismatch.",
    )

    validation_mse = (
        weighted_mse_sum
        / float(
            trajectory_count
        )
    )

    direct_mean = (
        sum(
            per_trajectory_values
        )
        / len(
            per_trajectory_values
        )
    )

    # The two calculations use the same float32
    # per-trajectory values but different summation
    # order. A tiny numerical tolerance is allowed.
    require(
        math.isclose(
            validation_mse,
            direct_mean,
            rel_tol=0.0,
            abs_tol=1e-7,
        ),
        (
            "B0 aggregation audit failed: "
            f"{validation_mse} vs {direct_mean}"
        ),
    )

    require(
        math.isfinite(
            validation_mse
        ),
        "B0 validation metric is non-finite.",
    )

    values_array = np.asarray(
        per_trajectory_values,
        dtype=np.float64,
    )

    return {
        "model_id":
            "B0",

        "role":
            "persistence baseline",

        "configuration_id":
            "persistence",

        "run_type":
            "evaluation_only",

        "seed":
            "deterministic",

        "noise_fraction":
            noise_fraction,

        "noise_index":
            noise_index,

        "validation_cell":
            "val_joint",

        "validation_trajectory_count":
            trajectory_count,

        "validation_microbatch_count":
            microbatch_count,

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "zero_step_only_microbatch_repairs":
            zero_batch_repairs,

        "metric_id":
            "noisy_target_rollout_mse",

        "metric_direction":
            "minimize",

        "metric_aggregation":
            (
                "Mean squared error over valid "
                "forecast points, channels, and pixels "
                "within trajectory; then arithmetic "
                "mean across val_joint trajectories."
            ),

        "persistence_semantics":
            (
                "Predict the trajectory's initial "
                "stored noisy normalized field at "
                "every future rollout step."
            ),

        "validation_noisy_target_rollout_mse":
            validation_mse,

        "direct_trajectory_mean_audit":
            direct_mean,

        "minimum_trajectory_mse":
            float(
                values_array.min()
            ),

        "maximum_trajectory_mse":
            float(
                values_array.max()
            ),

        "mean_trajectory_mse":
            float(
                values_array.mean()
            ),

        "standard_deviation_trajectory_mse":
            float(
                values_array.std(
                    ddof=0
                )
            ),

        "elapsed_seconds":
            float(
                time.time()
                - start_time
            ),

        "model_initialized":
            False,

        "model_parameters_present":
            False,

        "optimizer_initialized":
            False,

        "optimizer_steps_performed":
            0,

        "training_batches_read":
            False,

        "validation_batches_read":
            True,

        "privileged_directory_read":
            False,

        "clean_targets_read":
            False,

        "privileged_state_ids_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "evaluation_status":
            "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/5] Validating frozen pre-test sources"
    )

    b0_registry_rows = (
        validate_sources()
    )

    device = (
        tuning_runtime.resolve_device(
            args.device
        )
    )

    if device.type == "cuda":
        print(
            "CUDA device:",
            torch.cuda.get_device_name(
                device
            ),
        )
    else:
        print(
            "Evaluation device:",
            device,
        )

    print(
        "[2/5] Loading visible val_joint trajectories"
    )

    validation_cell = (
        tuning_runtime.TrajectoryCell(
            cell_id="val_joint",
            expected_trajectory_count=(
                EXPECTED_VAL_TRAJECTORY_COUNT
            ),
        )
    )

    require(
        len(
            validation_cell.rows
        )
        == EXPECTED_VAL_TRAJECTORY_COUNT,
        "val_joint trajectory count changed.",
    )

    require(
        int(
            np.sum(
                validation_cell.lengths
                == 0
            )
        )
        == 0,
        (
            "val_joint unexpectedly contains "
            "zero-step trajectories."
        ),
    )

    print(
        "[3/5] Evaluating five B0 persistence conditions"
    )

    results = []

    for noise_index, noise_fraction in enumerate(
        NOISE_FRACTIONS
    ):
        print(
            f"B0 | noise={noise_fraction:g}"
        )

        result = evaluate_noise_condition(
            validation_cell=(
                validation_cell
            ),
            noise_index=noise_index,
            noise_fraction=(
                noise_fraction
            ),
            device=device,
        )

        results.append(
            result
        )

        result_path = (
            OUTPUT_DIR
            / (
                "B0_noise_"
                + (
                    f"{noise_fraction:g}"
                    .replace(
                        ".",
                        "p",
                    )
                )
                + "_validation_result.json"
            )
        )

        write_json(
            result_path,
            result,
        )

        print(
            "  val rollout MSE:",
            result[
                "validation_noisy_target_rollout_mse"
            ],
        )

    print(
        "[4/5] Freezing B0 validation registry"
    )

    csv_rows = [
        {
            "model_id":
                result[
                    "model_id"
                ],

            "noise_fraction":
                result[
                    "noise_fraction"
                ],

            "seed":
                result[
                    "seed"
                ],

            "validation_cell":
                result[
                    "validation_cell"
                ],

            "trajectory_count":
                result[
                    "validation_trajectory_count"
                ],

            "metric_id":
                result[
                    "metric_id"
                ],

            "validation_noisy_target_rollout_mse":
                result[
                    "validation_noisy_target_rollout_mse"
                ],

            "minimum_trajectory_mse":
                result[
                    "minimum_trajectory_mse"
                ],

            "maximum_trajectory_mse":
                result[
                    "maximum_trajectory_mse"
                ],

            "evaluation_status":
                result[
                    "evaluation_status"
                ],
        }
        for result in results
    ]

    b0_registry_path = (
        OUTPUT_DIR
        / "b0_persistence_validation_registry.csv"
    )

    write_csv(
        b0_registry_path,
        csv_rows,
    )

    print(
        "[5/5] Writing Phase 4G-R3D2 summary"
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in (
            FINAL_RUN_REGISTRY_PATH,
            METRIC_REGISTRY_PATH,
            TEST_POLICY_PATH,
            FINAL_FIT_FREEZE_PATH,
            COMPARATOR_SUMMARY_PATH,
            NORMALIZATION_BINDING_PATH,
        )
    }

    write_json(
        OUTPUT_DIR
        / "phase4gr3d2_source_hashes.json",
        source_hashes,
    )

    summary = {
        "phase":
            (
                "4G-R3D2 Tier C v4 B0 "
                "persistence validation evaluation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "b0_evaluation_count":
            len(
                results
            ),

        "expected_b0_evaluation_count":
            5,

        "noise_fractions":
            list(
                NOISE_FRACTIONS
            ),

        "seed_policy":
            "deterministic",

        "training_performed":
            False,

        "model_parameters_initialized":
            False,

        "optimizer_initialized":
            False,

        "optimizer_steps_performed":
            0,

        "training_batches_read":
            False,

        "validation_cell":
            "val_joint",

        "validation_trajectory_count_per_condition":
            EXPECTED_VAL_TRAJECTORY_COUNT,

        "metric_id":
            "noisy_target_rollout_mse",

        "metric_direction":
            "minimize",

        "metric_values":
            {
                str(
                    result[
                        "noise_fraction"
                    ]
                ):
                    result[
                        "validation_noisy_target_rollout_mse"
                    ]
                for result in results
            },

        "all_b0_evaluations_completed":
            True,

        "b0_persistence_registry_path":
            str(
                b0_registry_path
            ),

        "b0_persistence_registry_sha256":
            sha256_file(
                b0_registry_path
            ),

        "primary_comparator_frozen":
            True,

        "primary_comparator":
            "B4",

        "privileged_directory_read":
            False,

        "clean_targets_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "phase4gr3d2_status":
            "b0_persistence_evaluations_frozen",

        "next_phase":
            (
                "4G-R3E Tier C v4 "
                "pre-test authorization audit"
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4gr3d2_b0_persistence_summary.json"
    )

    write_json(
        summary_path,
        summary,
    )

    print(
        "Phase 4G-R3D2 B0 persistence "
        "evaluation complete."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
