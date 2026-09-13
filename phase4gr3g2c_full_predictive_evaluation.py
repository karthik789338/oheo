from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4fr3b_run_tier_c_v4_tuning as runtime
import phase4gr3g2b_primary_confirmatory_test as primary_eval
import phase4gr3d2_evaluate_b0_persistence as frozen_b0


PROTOCOL_VERSION = "tier_c_v4"

TEST_CELLS = [
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
]

NOISE_LEVELS = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
]

NOISE_TO_INDEX = {
    0.0: 0,
    0.1: 1,
    0.25: 2,
    0.5: 3,
    1.0: 4,
}

MODEL_IDS = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
]

EXPECTED_MODEL_COUNTS = {
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}

EXPECTED_SEEDS = [
    11,
    23,
    37,
    53,
    71,
]

EXPECTED_SEQUENCE_COUNT = 144
EXPECTED_TRAJECTORY_COUNT = 1152
EXPECTED_TRAJECTORIES_PER_SEQUENCE = 8

EVALUATION_BATCH_SIZE = 16

# Frozen 4G-R3D2 B0 validation results.
#
# These are used only to verify that the implementation
# below is exactly the already-frozen persistence baseline
# before B0 is applied to the sealed test.
FROZEN_B0_VAL_JOINT = {
    0.0:
        1.013115149560488,

    0.1:
        1.0248534069396555,

    0.25:
        1.0822331263269815,

    0.5:
        1.2713734747634993,

    1.0:
        1.8827527927027807,
}


G1_ROOT = Path(
    "outputs/"
    "phase4gr3g1_tier_c_v4_sealed_test"
)

G1_SUMMARY = (
    G1_ROOT
    / "phase4gr3g1_sealed_test_generation_summary.json"
)

G1_ARTIFACT_MANIFEST = (
    G1_ROOT
    / "sealed_test_artifact_manifest.csv"
)

G2A_REPORT = Path(
    "outputs/"
    "phase4gr3g2a_tier_c_v4_evaluation_interfaces/"
    "phase4gr3g2a_evaluation_interface_report.json"
)

G2B_RESULT = Path(
    "outputs/"
    "phase4gr3g2b_tier_c_v4_primary_confirmatory/"
    "primary_confirmatory_result.json"
)

FINAL_REGISTRY = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

B0_REGISTRY = Path(
    "outputs/"
    "phase4gr3d2_tier_c_v4_b0_persistence/"
    "b0_persistence_validation_registry.csv"
)

TEST_POLICY = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g2c_tier_c_v4_full_predictive"
)

RUN_DIR = (
    OUTPUT_DIR
    / "runs"
)

B0_RUN_DIR = (
    OUTPUT_DIR
    / "b0_runs"
)

SEQUENCE_RESULTS_PATH = (
    OUTPUT_DIR
    / "predictive_sequence_metrics.csv"
)

RUN_SUMMARY_PATH = (
    OUTPUT_DIR
    / "predictive_run_summary.csv"
)

CONDITION_SUMMARY_PATH = (
    OUTPUT_DIR
    / "predictive_model_condition_summary.csv"
)

B0_AUDIT_PATH = (
    OUTPUT_DIR
    / "b0_persistence_validation_audit.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase4gr3g2c_full_predictive_summary.json"
)

HASHES_PATH = (
    OUTPUT_DIR
    / "phase4gr3g2c_result_hashes.json"
)


METRIC_NAMES = [
    "noisy_target_rollout_mse",
    "clean_target_rollout_mse",
    "noisy_target_one_step_mse",
    "clean_target_one_step_mse",
]


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


def write_csv(
    path: Path,
    rows: list[
        dict[str, Any]
    ],
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
        writer.writerows(
            rows
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
            digest.update(
                chunk
            )

    return digest.hexdigest()


def noise_key(
    noise: float,
) -> str:
    if float(noise) == 0.0:
        return "0"

    return str(
        float(noise)
    ).replace(
        ".",
        "p",
    )


def safe_name(
    value: str,
) -> str:
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        value,
    )


def determine_training_maximum_length(
) -> int:
    train_cell = (
        runtime.TrajectoryCell(
            "train_joint",
            11008,
        )
    )

    # The frozen R3D2 evaluator mutates the shared tuning
    # runtime globals while cycling through noise conditions.
    # Preserve the incoming frozen development state and restore
    # it before G2C resolves model configurations.
    original_noise_index = getattr(
        runtime,
        "NOISE_INDEX",
        None,
    )

    original_noise_fraction = getattr(
        runtime,
        "NOISE_FRACTION",
        None,
    )

    validation_cell = (
        runtime.TrajectoryCell(
            "val_joint",
            1152,
        )
    )

    result = max(
        int(
            train_cell
            .maximum_sequence_length
        ),
        int(
            validation_cell
            .maximum_sequence_length
        ),
    )

    del train_cell
    del validation_cell

    return result


def resolve_configuration_map(
) -> dict[
    str,
    dict[str, Any],
]:
    records = (
        runtime.load_configurations()
    )

    result = {}

    for record in records:
        configuration_id = str(
            record[
                "configuration_id"
            ]
        )

        require(
            configuration_id
            not in result,
            (
                "Duplicate configuration ID: "
                f"{configuration_id}"
            ),
        )

        result[
            configuration_id
        ] = record[
            "configuration"
        ]

    return result


def verify_protocol_state() -> dict[str, str]:
    print(
        "[1/10] Verifying frozen experimental state"
    )

    required = [
        G1_SUMMARY,
        G1_ARTIFACT_MANIFEST,
        G2A_REPORT,
        G2B_RESULT,
        FINAL_REGISTRY,
        B0_REGISTRY,
        TEST_POLICY,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    g1 = load_json(
        G1_SUMMARY
    )

    require(
        g1[
            "phase4gr3g1_status"
        ] == "sealed_test_generation_frozen",
        "G1 is no longer frozen.",
    )

    require(
        g1[
            "test_open_count"
        ] == 1,
        "Test-open count changed.",
    )

    require(
        g1[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test-open count changed.",
    )

    require(
        g1[
            "single_opening_consumed"
        ] is True,
        "Single opening is not consumed.",
    )

    require(
        g1[
            "test_regeneration_authorized"
        ] is False,
        "Test regeneration is authorized.",
    )

    g2a = load_json(
        G2A_REPORT
    )

    require(
        g2a[
            "phase4gr3g2a_status"
        ] == "evaluation_interfaces_frozen",
        "G2A is not frozen.",
    )

    g2b = load_json(
        G2B_RESULT
    )

    require(
        g2b[
            "phase4gr3g2b_status"
        ] == "primary_confirmatory_result_frozen",
        "G2B primary result is not frozen.",
    )

    primary_analysis = (
        g2b[
            "primary_analysis"
        ]
    )

    require(
        primary_analysis[
            "primary_model"
        ] == "OCM",
        "Primary model changed.",
    )

    require(
        primary_analysis[
            "primary_comparator"
        ] == "B4",
        "Primary comparator changed.",
    )

    require(
        primary_analysis[
            "predictive_superiority_supported"
        ] is False,
        (
            "Stored G2B result no longer "
            "matches observed primary result."
        ),
    )

    policy = load_json(
        TEST_POLICY
    )

    secondary = (
        policy[
            "secondary_conditions"
        ]
    )

    require(
        secondary[
            "cells"
        ] == TEST_CELLS,
        "Frozen secondary cell set changed.",
    )

    require(
        [
            float(value)
            for value
            in secondary[
                "noise_fractions"
            ]
        ] == NOISE_LEVELS,
        "Frozen secondary noise grid changed.",
    )

    for metric in METRIC_NAMES:
        require(
            metric
            in secondary[
                "metrics"
            ],
            (
                "Predictive secondary metric "
                f"missing from policy: {metric}"
            ),
        )

    return {
        "g1_summary_sha256":
            sha256_file(
                G1_SUMMARY
            ),

        "g1_artifact_manifest_sha256":
            sha256_file(
                G1_ARTIFACT_MANIFEST
            ),

        "g2a_report_sha256":
            sha256_file(
                G2A_REPORT
            ),

        "g2b_result_sha256":
            sha256_file(
                G2B_RESULT
            ),

        "final_registry_sha256":
            sha256_file(
                FINAL_REGISTRY
            ),

        "b0_registry_sha256":
            sha256_file(
                B0_REGISTRY
            ),

        "test_policy_sha256":
            sha256_file(
                TEST_POLICY
            ),
    }


def verify_final_registry(
) -> list[dict[str, str]]:
    print(
        "[2/10] Re-verifying 130 frozen neural checkpoints"
    )

    rows = load_csv(
        FINAL_REGISTRY
    )

    require(
        len(rows) == 130,
        (
            "Expected 130 frozen final fits; "
            f"found {len(rows)}."
        ),
    )

    counts = Counter(
        row[
            "model_id"
        ]
        for row in rows
    )

    require(
        dict(counts)
        == EXPECTED_MODEL_COUNTS,
        (
            "Frozen model counts changed: "
            f"{dict(counts)}"
        ),
    )

    noise_counts = Counter(
        float(
            row[
                "noise_fraction"
            ]
        )
        for row in rows
    )

    require(
        noise_counts
        == Counter(
            {
                0.0: 26,
                0.1: 26,
                0.25: 26,
                0.5: 26,
                1.0: 26,
            }
        ),
        (
            "Frozen noise counts changed: "
            f"{dict(noise_counts)}"
        ),
    )

    for row in rows:
        require(
            int(
                row[
                    "test_open_count"
                ]
            ) == 0,
            (
                "A final-fit registry row "
                "claims test access."
            ),
        )

        checkpoint = Path(
            row[
                "best_checkpoint_path"
            ]
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        require(
            sha256_file(
                checkpoint
            )
            == row[
                "best_checkpoint_sha256"
            ],
            (
                "Frozen checkpoint changed: "
                f"{checkpoint}"
            ),
        )

    return rows


def load_test_cells(
) -> dict[
    str,
    primary_eval.SealedTestCell,
]:
    print(
        "[3/10] Loading four immutable sealed test cells"
    )

    result = {}

    for cell_id in TEST_CELLS:
        cell = (
            primary_eval.SealedTestCell(
                cell_id
            )
        )

        require(
            len(
                cell.rows
            )
            == EXPECTED_TRAJECTORY_COUNT,
            (
                f"{cell_id} trajectory "
                "count changed."
            ),
        )

        require(
            len(
                cell.sequence_lookup
            )
            == EXPECTED_SEQUENCE_COUNT,
            (
                f"{cell_id} sequence "
                "count changed."
            ),
        )

        result[
            cell_id
        ] = cell

        print(
            f"  {cell_id}: "
            f"{len(cell.sequence_lookup)} sequences, "
            f"{len(cell.rows)} trajectories, "
            f"max_len={cell.maximum_sequence_length}"
        )

    return result


def b0_metrics_from_batch(
    batch: dict[str, Any],
) -> dict[
    str,
    np.ndarray,
]:
    """
    Evaluate the deterministic B0 persistence baseline.

    The noisy-target rollout metric is delegated directly
    to the already-frozen Phase 4G-R3D2 implementation:

        initial noisy field -> every future rollout point

    followed by mean MSE over valid future steps within
    each trajectory.

    Secondary clean-target and one-step quantities are
    deterministic extensions of the same persistence rule.
    """

    noisy = batch[
        "observations"
    ]

    clean = batch[
        "clean_observations"
    ]

    step_mask = batch[
        "step_mask"
    ]

    require(
        noisy.shape == clean.shape,
        (
            "Noisy and clean B0 observations "
            "have different shapes."
        ),
    )

    require(
        step_mask.shape
        == (
            noisy.shape[0],
            noisy.shape[1] - 1,
        ),
        (
            "B0 step mask does not match "
            "trajectory observations."
        ),
    )

    # --------------------------------------------------------------
    # Frozen noisy-target rollout metric.
    #
    # IMPORTANT:
    # Do not reimplement this quantity. Call the exact function
    # that produced the frozen Phase 4G-R3D2 validation registry.
    # --------------------------------------------------------------

    (
        frozen_batch_mean,
        noisy_rollout,
    ) = (
        frozen_b0
        .compute_persistence_batch_mse(
            {
                "observations":
                    noisy,

                "step_mask":
                    step_mask,
            }
        )
    )

    require(
        noisy_rollout.shape
        == (
            noisy.shape[0],
        ),
        (
            "Frozen B0 rollout metric returned "
            "unexpected trajectory shape."
        ),
    )

    require(
        bool(
            torch.isfinite(
                noisy_rollout
            ).all().item()
        ),
        "Frozen B0 rollout metric is non-finite.",
    )

    # Independent batch-mean audit.
    require(
        abs(
            float(
                frozen_batch_mean
                .detach()
                .cpu()
            )
            - float(
                noisy_rollout
                .mean()
                .detach()
                .cpu()
            )
        )
        < 1e-6,
        (
            "Frozen B0 batch mean and direct "
            "trajectory mean disagree."
        ),
    )

    mask = step_mask.to(
        dtype=noisy.dtype
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
            "B0 secondary evaluation encountered "
            "a zero-step trajectory."
        ),
    )

    # --------------------------------------------------------------
    # Clean-target rollout counterpart.
    #
    # Predictor remains the observed INITIAL NOISY field.
    # Only the evaluation target changes to the corresponding
    # clean future fields.
    # --------------------------------------------------------------

    initial_noisy = noisy[
        :,
        0:1,
        ...,
    ]

    clean_future = clean[
        :,
        1:,
        ...,
    ]

    initial_predictions = (
        initial_noisy.expand_as(
            clean_future
        )
    )

    clean_rollout_per_step = (
        (
            initial_predictions
            - clean_future
        )
        .pow(2)
        .mean(
            dim=(2, 3, 4)
        )
    )

    clean_rollout = (
        (
            clean_rollout_per_step
            * mask
        ).sum(
            dim=1
        )
        / valid_steps
    )

    # --------------------------------------------------------------
    # One-step persistence counterpart.
    #
    # At each observed step, persistence predicts that the
    # immediately following field remains equal to the CURRENT
    # noisy field.
    # --------------------------------------------------------------

    current_noisy = noisy[
        :,
        :-1,
        ...,
    ]

    next_noisy = noisy[
        :,
        1:,
        ...,
    ]

    next_clean = clean[
        :,
        1:,
        ...,
    ]

    noisy_one_step_per_step = (
        (
            current_noisy
            - next_noisy
        )
        .pow(2)
        .mean(
            dim=(2, 3, 4)
        )
    )

    clean_one_step_per_step = (
        (
            current_noisy
            - next_clean
        )
        .pow(2)
        .mean(
            dim=(2, 3, 4)
        )
    )

    noisy_one_step = (
        (
            noisy_one_step_per_step
            * mask
        ).sum(
            dim=1
        )
        / valid_steps
    )

    clean_one_step = (
        (
            clean_one_step_per_step
            * mask
        ).sum(
            dim=1
        )
        / valid_steps
    )

    for name, values in (
        (
            "noisy_target_rollout_mse",
            noisy_rollout,
        ),
        (
            "clean_target_rollout_mse",
            clean_rollout,
        ),
        (
            "noisy_target_one_step_mse",
            noisy_one_step,
        ),
        (
            "clean_target_one_step_mse",
            clean_one_step,
        ),
    ):
        require(
            bool(
                torch.isfinite(
                    values
                ).all().item()
            ),
            (
                "Non-finite B0 metric: "
                f"{name}"
            ),
        )

    return {
        "noisy_target_rollout_mse":
            noisy_rollout
            .detach()
            .cpu()
            .numpy(),

        "clean_target_rollout_mse":
            clean_rollout
            .detach()
            .cpu()
            .numpy(),

        "noisy_target_one_step_mse":
            noisy_one_step
            .detach()
            .cpu()
            .numpy(),

        "clean_target_one_step_mse":
            clean_one_step
            .detach()
            .cpu()
            .numpy(),
    }

@torch.no_grad()
def audit_b0_against_frozen_validation(
    device: torch.device,
) -> dict[str, Any]:
    print(
        "[4/10] Auditing B0 persistence semantics "
        "against exact frozen R3D2 implementation"
    )

    # Preserve the frozen development runtime state because
    # the exact R3D2 B0 evaluator mutates these shared globals.
    original_noise_index = getattr(
        runtime,
        "NOISE_INDEX",
        None,
    )

    original_noise_fraction = getattr(
        runtime,
        "NOISE_FRACTION",
        None,
    )

    # Protect the already-frozen B0 registry itself.
    expected_registry_sha256 = (
        "0bafb4fb9fde9b5515db1391069e09dae2bd3cf"
        "3849944d0673808676d466497"
    )

    observed_registry_sha256 = (
        sha256_file(
            B0_REGISTRY
        )
    )

    require(
        observed_registry_sha256
        == expected_registry_sha256,
        (
            "Frozen B0 validation registry hash changed: "
            f"{observed_registry_sha256}"
        ),
    )

    validation_cell = (
        runtime.TrajectoryCell(
            "val_joint",
            1152,
        )
    )

    require(
        len(
            validation_cell.rows
        )
        == EXPECTED_TRAJECTORY_COUNT,
        (
            "Frozen val_joint trajectory "
            "count changed."
        ),
    )

    observed = {}

    for noise in NOISE_LEVELS:
        noise_index = (
            NOISE_TO_INDEX[
                noise
            ]
        )

        # Call the exact evaluator that produced
        # phase4gr3d2_tier_c_v4_b0_persistence.
        result = (
            frozen_b0
            .evaluate_noise_condition(
                validation_cell=(
                    validation_cell
                ),
                noise_index=(
                    noise_index
                ),
                noise_fraction=(
                    noise
                ),
                device=device,
            )
        )

        mean_value = float(
            result[
                "validation_noisy_target_rollout_mse"
            ]
        )

        expected = float(
            FROZEN_B0_VAL_JOINT[
                noise
            ]
        )

        absolute_error = abs(
            mean_value
            - expected
        )

        require(
            absolute_error < 1e-10,
            (
                "Exact frozen R3D2 B0 evaluator "
                "does not reproduce its frozen "
                f"registry at noise={noise}: "
                f"observed={mean_value}, "
                f"expected={expected}, "
                f"abs_error={absolute_error}"
            ),
        )

        direct_mean = float(
            result[
                "direct_trajectory_mean_audit"
            ]
        )

        require(
            abs(
                direct_mean
                - mean_value
            ) < 1e-7,
            (
                "Frozen B0 direct-trajectory "
                "aggregation audit failed."
            ),
        )

        observed[
            str(
                noise
            )
        ] = {
            "observed_validation_mse":
                mean_value,

            "frozen_validation_mse":
                expected,

            "absolute_error":
                absolute_error,

            "direct_trajectory_mean_audit":
                direct_mean,

            "minimum_trajectory_mse":
                float(
                    result[
                        "minimum_trajectory_mse"
                    ]
                ),

            "maximum_trajectory_mse":
                float(
                    result[
                        "maximum_trajectory_mse"
                    ]
                ),

            "passed":
                True,
        }

        print(
            f"  noise={noise}: "
            f"observed={mean_value:.12f} "
            f"frozen={expected:.12f} "
            "PASS"
        )

    # Restore the exact runtime state that existed before
    # the validation-only B0 audit. The B0 evaluator and the
    # tuning runner reference the same imported runtime module.
    if original_noise_index is not None:
        runtime.NOISE_INDEX = (
            original_noise_index
        )

    if original_noise_fraction is not None:
        runtime.NOISE_FRACTION = (
            original_noise_fraction
        )

    print(
        "  Restored runtime noise state:",
        "NOISE_INDEX=",
        getattr(
            runtime,
            "NOISE_INDEX",
            None,
        ),
        "NOISE_FRACTION=",
        getattr(
            runtime,
            "NOISE_FRACTION",
            None,
        ),
    )

    audit = {
        "baseline":
            "B0",

        "frozen_source_module":
            (
                "phase4gr3d2_evaluate_b0_persistence.py"
            ),

        "frozen_registry_path":
            str(
                B0_REGISTRY
            ),

        "frozen_registry_sha256":
            observed_registry_sha256,

        "persistence_semantics":
            (
                "Predict the trajectory's initial "
                "stored noisy normalized field at "
                "every valid future rollout step."
            ),

        "noisy_target_rollout_aggregation":
            (
                "Mean squared error over channels and "
                "pixels at every valid future forecast "
                "step; mean over valid forecast steps "
                "within each trajectory."
            ),

        "secondary_clean_rollout_extension":
            (
                "Use the same initial noisy persistence "
                "prediction but evaluate against clean "
                "future fields."
            ),

        "secondary_one_step_extension":
            (
                "At each valid source step, use the "
                "current noisy field as the persistence "
                "prediction for the immediately following "
                "noisy or clean field."
            ),

        "validation_cell":
            "val_joint",

        "frozen_validation_results_reproduced":
            True,

        "noise_results":
            observed,

        "test_data_read_during_audit":
            False,
    }

    write_json(
        B0_AUDIT_PATH,
        audit,
    )

    del validation_cell

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return audit

def checkpoint_run_paths(
    row: dict[str, str],
    cell_id: str,
) -> tuple[
    Path,
    Path,
]:
    run_name = safe_name(
        f"{row['run_id']}__{cell_id}"
    )

    return (
        RUN_DIR
        / f"{run_name}.csv",

        RUN_DIR
        / f"{run_name}.json",
    )


def b0_run_paths(
    noise: float,
    cell_id: str,
) -> tuple[
    Path,
    Path,
]:
    run_name = (
        f"B0_noise_{noise_key(noise)}"
        f"__{cell_id}"
    )

    return (
        B0_RUN_DIR
        / f"{run_name}.csv",

        B0_RUN_DIR
        / f"{run_name}.json",
    )


def aggregate_trajectory_rows_to_sequences(
    rows: list[
        dict[str, Any]
    ],
    base_record: dict[str, Any],
) -> list[dict[str, Any]]:
    groups = defaultdict(list)

    for row in rows:
        groups[
            (
                int(
                    row[
                        "cell_sequence_index"
                    ]
                ),
                str(
                    row[
                        "source_sequence_id"
                    ]
                ),
            )
        ].append(
            row
        )

    require(
        len(groups)
        == EXPECTED_SEQUENCE_COUNT,
        (
            "Expected 144 sequence groups; "
            f"found {len(groups)}."
        ),
    )

    result = []

    for (
        sequence_index,
        source_sequence_id,
    ), group_rows in sorted(
        groups.items()
    ):
        require(
            len(
                group_rows
            )
            == EXPECTED_TRAJECTORIES_PER_SEQUENCE,
            (
                "Expected eight trajectories "
                "per sequence."
            ),
        )

        record = dict(
            base_record
        )

        record.update(
            {
                "cell_sequence_index":
                    sequence_index,

                "source_sequence_id":
                    source_sequence_id,

                "trajectory_count":
                    len(
                        group_rows
                    ),
            }
        )

        for metric in METRIC_NAMES:
            record[
                metric
            ] = float(
                np.mean(
                    [
                        float(
                            row[
                                metric
                            ]
                        )
                        for row in group_rows
                    ]
                )
            )

        result.append(
            record
        )

    require(
        len(result)
        == EXPECTED_SEQUENCE_COUNT,
        "Sequence aggregation count changed.",
    )

    return result


@torch.no_grad()
def evaluate_neural_run(
    registry_row: dict[str, str],
    cell_id: str,
    cell: primary_eval.SealedTestCell,
    configuration_map: dict[
        str,
        dict[str, Any],
    ],
    maximum_sequence_length: int,
    device: torch.device,
    source_signature: dict[str, str],
) -> list[dict[str, Any]]:
    result_csv, marker_path = (
        checkpoint_run_paths(
            registry_row,
            cell_id,
        )
    )

    checkpoint_path = Path(
        registry_row[
            "best_checkpoint_path"
        ]
    )

    checkpoint_sha = (
        registry_row[
            "best_checkpoint_sha256"
        ]
    )

    if (
        marker_path.exists()
        and result_csv.exists()
    ):
        marker = load_json(
            marker_path
        )

        require(
            marker[
                "status"
            ] == "completed",
            (
                "Existing neural run marker "
                "is not complete."
            ),
        )

        require(
            marker[
                "checkpoint_sha256"
            ] == checkpoint_sha,
            (
                "Existing neural run belongs "
                "to a different checkpoint."
            ),
        )

        require(
            marker[
                "g1_artifact_manifest_sha256"
            ]
            == source_signature[
                "g1_artifact_manifest_sha256"
            ],
            (
                "Existing neural run belongs "
                "to different test artifacts."
            ),
        )

        require(
            marker[
                "result_csv_sha256"
            ] == sha256_file(
                result_csv
            ),
            (
                "Existing neural run CSV "
                "hash changed."
            ),
        )

        rows = load_csv(
            result_csv
        )

        require(
            len(rows)
            == EXPECTED_SEQUENCE_COUNT,
            (
                "Completed neural run does "
                "not contain 144 sequences."
            ),
        )

        print(
            f"  SKIP completed: "
            f"{registry_row['run_id']} "
            f"{cell_id}"
        )

        return rows

    model_id = (
        registry_row[
            "model_id"
        ]
    )

    configuration_id = (
        registry_row[
            "configuration_id"
        ]
    )

    require(
        configuration_id
        in configuration_map,
        (
            "Configuration missing: "
            f"{configuration_id}"
        ),
    )

    configuration = (
        configuration_map[
            configuration_id
        ]
    )

    noise = float(
        registry_row[
            "noise_fraction"
        ]
    )

    require(
        noise in NOISE_TO_INDEX,
        (
            "Unexpected final-fit noise: "
            f"{noise}"
        ),
    )

    noise_index = (
        NOISE_TO_INDEX[
            noise
        ]
    )

    seed = int(
        registry_row[
            "effective_seed"
        ]
    )

    require(
        checkpoint_path.exists(),
        f"Missing checkpoint: {checkpoint_path}",
    )

    require(
        sha256_file(
            checkpoint_path
        )
        == checkpoint_sha,
        (
            "Frozen checkpoint hash changed: "
            f"{checkpoint_path}"
        ),
    )

    if hasattr(
        runtime,
        "set_global_determinism",
    ):
        runtime.set_global_determinism(
            seed
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    codec, model, optimizer = (
        runtime.build_model_bundle(
            model_id=model_id,
            configuration=configuration,
            maximum_sequence_length=(
                maximum_sequence_length
            ),
            device=device,
        )
    )

    del optimizer

    codec.load_state_dict(
        checkpoint[
            "codec_state_dict"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    codec.eval()
    model.eval()

    trajectory_rows = []
    runtime_consistency_checked = False

    for start in range(
        0,
        EXPECTED_TRAJECTORY_COUNT,
        EVALUATION_BATCH_SIZE,
    ):
        indices = list(
            range(
                start,
                min(
                    start
                    + EVALUATION_BATCH_SIZE,
                    EXPECTED_TRAJECTORY_COUNT,
                ),
            )
        )

        batch = cell.make_batch(
            trajectory_indices=indices,
            noise_index=noise_index,
            device=device,
        )

        (
            predicted_steps,
            final_prediction,
        ) = (
            primary_eval.predict_fields(
                model_id=model_id,
                model=model,
                codec=codec,
                configuration=configuration,
                batch=batch,
            )
        )

        metrics = (
            primary_eval
            .per_trajectory_metrics(
                predicted_step_fields=(
                    predicted_steps
                ),
                final_prediction=(
                    final_prediction
                ),
                batch=batch,
            )
        )

        if not runtime_consistency_checked:
            accepted_batch = {
                "observations":
                    batch[
                        "observations"
                    ],

                "operation_ids":
                    batch[
                        "operation_ids"
                    ],

                "point_mask":
                    batch[
                        "point_mask"
                    ],

                "step_mask":
                    batch[
                        "step_mask"
                    ],

                "sequence_lengths":
                    batch[
                        "sequence_lengths"
                    ],
            }

            accepted = (
                runtime.compute_losses(
                    model_id=model_id,
                    model=model,
                    codec=codec,
                    batch=accepted_batch,
                    configuration=configuration,
                )
            )

            observed_rollout = float(
                np.mean(
                    metrics[
                        "noisy_target_rollout_mse"
                    ]
                )
            )

            observed_one_step = float(
                np.mean(
                    metrics[
                        "noisy_target_one_step_mse"
                    ]
                )
            )

            accepted_rollout = float(
                accepted[
                    "rollout_field_loss"
                ]
                .detach()
                .cpu()
            )

            accepted_one_step = float(
                accepted[
                    "one_step_field_loss"
                ]
                .detach()
                .cpu()
            )

            require(
                abs(
                    observed_rollout
                    - accepted_rollout
                ) < 5e-5,
                (
                    "Rollout metric does not "
                    "reproduce frozen runtime."
                ),
            )

            require(
                abs(
                    observed_one_step
                    - accepted_one_step
                ) < 5e-5,
                (
                    "One-step metric does not "
                    "reproduce frozen runtime."
                ),
            )

            runtime_consistency_checked = True

        for local_index in range(
            len(indices)
        ):
            trajectory_rows.append(
                {
                    "cell_sequence_index":
                        batch[
                            "cell_sequence_indices"
                        ][local_index],

                    "source_sequence_id":
                        batch[
                            "source_sequence_ids"
                        ][local_index],

                    **{
                        metric:
                            float(
                                metrics[
                                    metric
                                ][local_index]
                            )
                        for metric
                        in METRIC_NAMES
                    },
                }
            )

    require(
        runtime_consistency_checked,
        "Runtime consistency was not checked.",
    )

    require(
        len(
            trajectory_rows
        )
        == EXPECTED_TRAJECTORY_COUNT,
        (
            "Neural evaluation trajectory "
            "count changed."
        ),
    )

    base_record = {
        "model_id":
            model_id,

        "configuration_id":
            configuration_id,

        "effective_seed":
            seed,

        "training_noise_fraction":
            noise,

        "test_noise_fraction":
            noise,

        "cell_id":
            cell_id,

        "checkpoint_run_id":
            registry_row[
                "run_id"
            ],
    }

    sequence_rows = (
        aggregate_trajectory_rows_to_sequences(
            rows=trajectory_rows,
            base_record=base_record,
        )
    )

    write_csv(
        result_csv,
        sequence_rows,
    )

    marker = {
        "status":
            "completed",

        "model_id":
            model_id,

        "configuration_id":
            configuration_id,

        "effective_seed":
            seed,

        "noise_fraction":
            noise,

        "cell_id":
            cell_id,

        "sequence_count":
            len(
                sequence_rows
            ),

        "checkpoint_path":
            str(
                checkpoint_path
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "g1_artifact_manifest_sha256":
            source_signature[
                "g1_artifact_manifest_sha256"
            ],

        "runtime_consistency_check":
            True,

        "result_csv_path":
            str(
                result_csv
            ),

        "result_csv_sha256":
            sha256_file(
                result_csv
            ),

        "training_performed":
            False,

        "checkpoint_reselection_performed":
            False,
    }

    write_json(
        marker_path,
        marker,
    )

    print(
        f"  DONE {model_id} "
        f"seed={seed} "
        f"noise={noise} "
        f"{cell_id}"
    )

    del checkpoint
    del codec
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return sequence_rows


@torch.no_grad()
def evaluate_b0_run(
    noise: float,
    cell_id: str,
    cell: primary_eval.SealedTestCell,
    device: torch.device,
    source_signature: dict[str, str],
) -> list[dict[str, Any]]:
    result_csv, marker_path = (
        b0_run_paths(
            noise,
            cell_id,
        )
    )

    if (
        result_csv.exists()
        and marker_path.exists()
    ):
        marker = load_json(
            marker_path
        )

        require(
            marker[
                "status"
            ] == "completed",
            "Existing B0 run is incomplete.",
        )

        require(
            marker[
                "g1_artifact_manifest_sha256"
            ]
            == source_signature[
                "g1_artifact_manifest_sha256"
            ],
            (
                "Existing B0 run belongs "
                "to different test artifacts."
            ),
        )

        require(
            marker[
                "result_csv_sha256"
            ] == sha256_file(
                result_csv
            ),
            "Existing B0 run CSV changed.",
        )

        rows = load_csv(
            result_csv
        )

        require(
            len(rows)
            == EXPECTED_SEQUENCE_COUNT,
            (
                "Completed B0 run does not "
                "contain 144 sequences."
            ),
        )

        print(
            f"  SKIP completed: "
            f"B0 noise={noise} {cell_id}"
        )

        return rows

    noise_index = (
        NOISE_TO_INDEX[
            noise
        ]
    )

    trajectory_rows = []

    for start in range(
        0,
        EXPECTED_TRAJECTORY_COUNT,
        64,
    ):
        indices = list(
            range(
                start,
                min(
                    start + 64,
                    EXPECTED_TRAJECTORY_COUNT,
                ),
            )
        )

        batch = cell.make_batch(
            trajectory_indices=indices,
            noise_index=noise_index,
            device=device,
        )

        metrics = (
            b0_metrics_from_batch(
                batch
            )
        )

        for local_index in range(
            len(indices)
        ):
            trajectory_rows.append(
                {
                    "cell_sequence_index":
                        batch[
                            "cell_sequence_indices"
                        ][local_index],

                    "source_sequence_id":
                        batch[
                            "source_sequence_ids"
                        ][local_index],

                    **{
                        metric:
                            float(
                                metrics[
                                    metric
                                ][local_index]
                            )
                        for metric
                        in METRIC_NAMES
                    },
                }
            )

    require(
        len(
            trajectory_rows
        )
        == EXPECTED_TRAJECTORY_COUNT,
        (
            "B0 evaluation trajectory "
            "count changed."
        ),
    )

    base_record = {
        "model_id":
            "B0",

        "configuration_id":
            "persistence",

        "effective_seed":
            "",

        "training_noise_fraction":
            noise,

        "test_noise_fraction":
            noise,

        "cell_id":
            cell_id,

        "checkpoint_run_id":
            "B0_persistence",
    }

    sequence_rows = (
        aggregate_trajectory_rows_to_sequences(
            rows=trajectory_rows,
            base_record=base_record,
        )
    )

    write_csv(
        result_csv,
        sequence_rows,
    )

    marker = {
        "status":
            "completed",

        "model_id":
            "B0",

        "configuration_id":
            "persistence",

        "effective_seed":
            None,

        "noise_fraction":
            noise,

        "cell_id":
            cell_id,

        "sequence_count":
            len(
                sequence_rows
            ),

        "g1_artifact_manifest_sha256":
            source_signature[
                "g1_artifact_manifest_sha256"
            ],

        "b0_definition_verified_against_frozen_validation":
            True,

        "result_csv_path":
            str(
                result_csv
            ),

        "result_csv_sha256":
            sha256_file(
                result_csv
            ),

        "training_performed":
            False,
    }

    write_json(
        marker_path,
        marker,
    )

    print(
        f"  DONE B0 "
        f"noise={noise} "
        f"{cell_id}"
    )

    return sequence_rows


def summarize_runs(
    sequence_rows: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    groups = defaultdict(list)

    for row in sequence_rows:
        key = (
            row[
                "model_id"
            ],
            row[
                "configuration_id"
            ],
            str(
                row[
                    "effective_seed"
                ]
            ),
            float(
                row[
                    "test_noise_fraction"
                ]
            ),
            row[
                "cell_id"
            ],
            row[
                "checkpoint_run_id"
            ],
        )

        groups[
            key
        ].append(
            row
        )

    result = []

    for (
        model_id,
        configuration_id,
        seed,
        noise,
        cell_id,
        checkpoint_run_id,
    ), rows in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][3],
            item[0][4],
            item[0][2],
        ),
    ):
        require(
            len(rows)
            == EXPECTED_SEQUENCE_COUNT,
            (
                "Run summary expected "
                "144 sequence rows."
            ),
        )

        record = {
            "model_id":
                model_id,

            "configuration_id":
                configuration_id,

            "effective_seed":
                seed,

            "noise_fraction":
                noise,

            "cell_id":
                cell_id,

            "checkpoint_run_id":
                checkpoint_run_id,

            "sequence_count":
                len(
                    rows
                ),
        }

        for metric in METRIC_NAMES:
            values = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row
                    in rows
                ],
                dtype=np.float64,
            )

            record[
                metric
            ] = float(
                values.mean()
            )

            record[
                metric
                + "_sequence_sd"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        result.append(
            record
        )

    require(
        len(result) == 540,
        (
            "Expected 540 run-condition summaries "
            f"(520 neural + 20 B0); "
            f"found {len(result)}."
        ),
    )

    return result


def summarize_model_conditions(
    sequence_rows: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    # First average neural seeds within each
    # sequence, exactly preserving sequence as
    # the statistical unit.
    seed_groups = defaultdict(list)

    for row in sequence_rows:
        key = (
            row[
                "model_id"
            ],
            float(
                row[
                    "test_noise_fraction"
                ]
            ),
            row[
                "cell_id"
            ],
            int(
                row[
                    "cell_sequence_index"
                ]
            ),
            str(
                row[
                    "source_sequence_id"
                ]
            ),
        )

        seed_groups[
            key
        ].append(
            row
        )

    seed_averaged = []

    for (
        model_id,
        noise,
        cell_id,
        sequence_index,
        source_sequence_id,
    ), rows in seed_groups.items():
        expected_seed_count = (
            1
            if model_id
            in {
                "B0",
                "B1",
            }
            else 5
        )

        require(
            len(rows)
            == expected_seed_count,
            (
                f"{model_id} {cell_id} noise={noise} "
                f"sequence={sequence_index}: "
                f"expected {expected_seed_count} "
                f"seed rows, found {len(rows)}."
            ),
        )

        record = {
            "model_id":
                model_id,

            "noise_fraction":
                noise,

            "cell_id":
                cell_id,

            "cell_sequence_index":
                sequence_index,

            "source_sequence_id":
                source_sequence_id,

            "seed_count":
                expected_seed_count,
        }

        for metric in METRIC_NAMES:
            record[
                metric
            ] = float(
                np.mean(
                    [
                        float(
                            row[
                                metric
                            ]
                        )
                        for row
                        in rows
                    ]
                )
            )

        seed_averaged.append(
            record
        )

    condition_groups = defaultdict(list)

    for row in seed_averaged:
        condition_groups[
            (
                row[
                    "model_id"
                ],
                float(
                    row[
                        "noise_fraction"
                    ]
                ),
                row[
                    "cell_id"
                ],
            )
        ].append(
            row
        )

    result = []

    for (
        model_id,
        noise,
        cell_id,
    ), rows in sorted(
        condition_groups.items()
    ):
        require(
            len(rows)
            == EXPECTED_SEQUENCE_COUNT,
            (
                "Model-condition summary "
                "expected 144 sequences."
            ),
        )

        seed_count = (
            1
            if model_id
            in {
                "B0",
                "B1",
            }
            else 5
        )

        record = {
            "model_id":
                model_id,

            "noise_fraction":
                noise,

            "cell_id":
                cell_id,

            "seed_count":
                seed_count,

            "sequence_count":
                len(
                    rows
                ),
        }

        for metric in METRIC_NAMES:
            values = np.asarray(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row in rows
                ],
                dtype=np.float64,
            )

            record[
                metric
            ] = float(
                values.mean()
            )

            record[
                metric
                + "_sequence_sd"
            ] = float(
                values.std(
                    ddof=1
                )
            )

        result.append(
            record
        )

    require(
        len(result)
        == (
            7
            * 5
            * 4
        ),
        (
            "Expected 140 model-condition rows; "
            f"found {len(result)}."
        ),
    )

    return result


def verify_primary_reproduction(
    condition_summary: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    print(
        "[9/10] Reproducing frozen G2B primary means"
    )

    g2b = load_json(
        G2B_RESULT
    )

    primary = g2b[
        "primary_analysis"
    ]

    lookup = {
        (
            row[
                "model_id"
            ],
            row[
                "cell_id"
            ],
            float(
                row[
                    "noise_fraction"
                ]
            ),
        ):
            row
        for row
        in condition_summary
    }

    ocm = lookup[
        (
            "OCM",
            "test_joint",
            0.25,
        )
    ][
        "noisy_target_rollout_mse"
    ]

    b4 = lookup[
        (
            "B4",
            "test_joint",
            0.25,
        )
    ][
        "noisy_target_rollout_mse"
    ]

    require(
        abs(
            float(ocm)
            - float(
                primary[
                    "ocm_mean"
                ]
            )
        ) < 1e-8,
        (
            "G2C OCM primary mean does not "
            "reproduce frozen G2B within "
            "the frozen numerical audit tolerance."
        ),
    )

    require(
        abs(
            float(b4)
            - float(
                primary[
                    "b4_mean"
                ]
            )
        ) < 1e-8,
        (
            "G2C B4 primary mean does not "
            "reproduce frozen G2B within "
            "the frozen numerical audit tolerance."
        ),
    )

    record = {
        "g2b_primary_reproduced":
            True,

        "ocm_g2c":
            float(
                ocm
            ),

        "ocm_g2b":
            float(
                primary[
                    "ocm_mean"
                ]
            ),

        "b4_g2c":
            float(
                b4
            ),

        "b4_g2b":
            float(
                primary[
                    "b4_mean"
                ]
            ),
    }

    print(
        json.dumps(
            record,
            indent=2,
        )
    )

    return record


def main() -> None:
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    B0_RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_signature = (
        verify_protocol_state()
    )

    registry_rows = (
        verify_final_registry()
    )

    cells = (
        load_test_cells()
    )

    device = (
        runtime.resolve_device(
            "cuda"
        )
    )

    print(
        "Device:",
        device,
    )

    b0_audit = (
        audit_b0_against_frozen_validation(
            device=device,
        )
    )

    require(
        b0_audit[
            "frozen_validation_results_reproduced"
        ] is True,
        "B0 validation audit failed.",
    )

    print(
        "[5/10] Resolving frozen model configurations"
    )

    configuration_map = (
        resolve_configuration_map()
    )

    maximum_sequence_length = (
        determine_training_maximum_length()
    )

    require(
        all(
            cell.maximum_sequence_length
            <= maximum_sequence_length
            for cell in cells.values()
        ),
        (
            "A sealed test sequence exceeds "
            "the frozen architecture length."
        ),
    )

    print(
        "Frozen architecture maximum length:",
        maximum_sequence_length,
    )

    print(
        "[6/10] Evaluating 520 frozen neural "
        "checkpoint/cell conditions"
    )

    all_sequence_rows = []

    total_neural_evaluations = (
        len(
            registry_rows
        )
        * len(
            TEST_CELLS
        )
    )

    evaluation_index = 0

    registry_rows.sort(
        key=lambda row: (
            MODEL_IDS.index(
                row[
                    "model_id"
                ]
            ),
            float(
                row[
                    "noise_fraction"
                ]
            ),
            int(
                row[
                    "effective_seed"
                ]
            ),
        )
    )

    for registry_row in registry_rows:
        for cell_id in TEST_CELLS:
            evaluation_index += 1

            print()
            print(
                f"[neural "
                f"{evaluation_index}/"
                f"{total_neural_evaluations}] "
                f"{registry_row['model_id']} "
                f"seed={registry_row['effective_seed']} "
                f"noise={registry_row['noise_fraction']} "
                f"{cell_id}"
            )

            rows = evaluate_neural_run(
                registry_row=(
                    registry_row
                ),
                cell_id=cell_id,
                cell=cells[
                    cell_id
                ],
                configuration_map=(
                    configuration_map
                ),
                maximum_sequence_length=(
                    maximum_sequence_length
                ),
                device=device,
                source_signature=(
                    source_signature
                ),
            )

            all_sequence_rows.extend(
                rows
            )

    expected_neural_sequence_rows = (
        520
        * EXPECTED_SEQUENCE_COUNT
    )

    require(
        len(
            all_sequence_rows
        )
        == expected_neural_sequence_rows,
        (
            "Unexpected neural sequence "
            "result count."
        ),
    )

    print(
        "[7/10] Evaluating 20 frozen B0 "
        "persistence conditions"
    )

    for noise in NOISE_LEVELS:
        for cell_id in TEST_CELLS:
            rows = evaluate_b0_run(
                noise=noise,
                cell_id=cell_id,
                cell=cells[
                    cell_id
                ],
                device=device,
                source_signature=(
                    source_signature
                ),
            )

            all_sequence_rows.extend(
                rows
            )

    expected_total_sequence_rows = (
        (
            520
            + 20
        )
        * EXPECTED_SEQUENCE_COUNT
    )

    require(
        len(
            all_sequence_rows
        )
        == expected_total_sequence_rows,
        (
            "Unexpected total sequence result "
            f"count: {len(all_sequence_rows)}"
        ),
    )

    print(
        "[8/10] Freezing predictive result tables"
    )

    # Normalize types after possible CSV resume.
    normalized_sequence_rows = []

    for row in all_sequence_rows:
        normalized = dict(
            row
        )

        normalized[
            "training_noise_fraction"
        ] = float(
            row[
                "training_noise_fraction"
            ]
        )

        normalized[
            "test_noise_fraction"
        ] = float(
            row[
                "test_noise_fraction"
            ]
        )

        normalized[
            "cell_sequence_index"
        ] = int(
            row[
                "cell_sequence_index"
            ]
        )

        normalized[
            "trajectory_count"
        ] = int(
            row[
                "trajectory_count"
            ]
        )

        for metric in METRIC_NAMES:
            normalized[
                metric
            ] = float(
                row[
                    metric
                ]
            )

        normalized_sequence_rows.append(
            normalized
        )

    normalized_sequence_rows.sort(
        key=lambda row: (
            row[
                "model_id"
            ],
            float(
                row[
                    "test_noise_fraction"
                ]
            ),
            row[
                "cell_id"
            ],
            str(
                row[
                    "effective_seed"
                ]
            ),
            int(
                row[
                    "cell_sequence_index"
                ]
            ),
        )
    )

    write_csv(
        SEQUENCE_RESULTS_PATH,
        normalized_sequence_rows,
    )

    run_summary = summarize_runs(
        normalized_sequence_rows
    )

    write_csv(
        RUN_SUMMARY_PATH,
        run_summary,
    )

    condition_summary = (
        summarize_model_conditions(
            normalized_sequence_rows
        )
    )

    write_csv(
        CONDITION_SUMMARY_PATH,
        condition_summary,
    )

    primary_reproduction = (
        verify_primary_reproduction(
            condition_summary
        )
    )

    print(
        "[10/10] Freezing G2C summary and hashes"
    )

    summary = {
        "phase":
            (
                "4G-R3G2C Tier C v4 full "
                "secondary predictive evaluation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "test_open_count":
            1,

        "maximum_test_open_count":
            1,

        "test_regeneration_authorized":
            False,

        "additional_training_performed":
            False,

        "additional_tuning_performed":
            False,

        "checkpoint_reselection_performed":
            False,

        "comparator_reselection_performed":
            False,

        "sealed_test_cells":
            TEST_CELLS,

        "noise_levels":
            NOISE_LEVELS,

        "predictive_metrics":
            METRIC_NAMES,

        "neural_checkpoint_count":
            len(
                registry_rows
            ),

        "neural_checkpoint_cell_evaluation_count":
            520,

        "b0_condition_evaluation_count":
            20,

        "total_run_condition_count":
            540,

        "sequence_result_count":
            len(
                normalized_sequence_rows
            ),

        "run_summary_count":
            len(
                run_summary
            ),

        "model_condition_summary_count":
            len(
                condition_summary
            ),

        "b0_persistence_definition_verified":
            True,

        "b0_validation_audit_path":
            str(
                B0_AUDIT_PATH
            ),

        "primary_g2b_reproduction":
            primary_reproduction,

        "primary_predictive_superiority_supported":
            False,

        "sequence_results_path":
            str(
                SEQUENCE_RESULTS_PATH
            ),

        "run_summary_path":
            str(
                RUN_SUMMARY_PATH
            ),

        "condition_summary_path":
            str(
                CONDITION_SUMMARY_PATH
            ),

        "source_signature":
            source_signature,

        "elapsed_seconds":
            float(
                time.time()
                - start_time
            ),

        "phase4gr3g2c_status":
            "full_predictive_results_frozen",

        "next_phase":
            (
                "4G-R3G2D Tier C v4 "
                "structural/process-memory evaluation"
            ),
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    artifacts = [
        B0_AUDIT_PATH,
        SEQUENCE_RESULTS_PATH,
        RUN_SUMMARY_PATH,
        CONDITION_SUMMARY_PATH,
        SUMMARY_PATH,
    ]

    hashes = {
        str(path):
            sha256_file(
                path
            )
        for path in artifacts
    }

    write_json(
        HASHES_PATH,
        hashes,
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 4G-R3G2C COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        json.dumps(
            {
                "neural_checkpoint_cell_evaluations":
                    520,

                "b0_condition_evaluations":
                    20,

                "total_run_conditions":
                    540,

                "sequence_results":
                    len(
                        normalized_sequence_rows
                    ),

                "model_condition_rows":
                    len(
                        condition_summary
                    ),

                "b0_validation_reproduced":
                    True,

                "g2b_primary_reproduced":
                    True,

                "primary_predictive_superiority_supported":
                    False,

                "status":
                    "full_predictive_results_frozen",
            },
            indent=2,
        )
    )

    print()
    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
