from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

NOISE_FRACTIONS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_RUN_COUNT = 130

EXPECTED_MODEL_COUNTS = {
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}

EXPECTED_NOISE_COUNTS = {
    0.0: 26,
    0.1: 26,
    0.25: 26,
    0.5: 26,
    1.0: 26,
}

EXPECTED_SEED_COUNTS = {
    11: 30,
    23: 25,
    37: 25,
    53: 25,
    71: 25,
}

EXPECTED_NORMALIZATION_SHA256 = (
    "3bbfd2c72e774d32efead77287778004"
    "a498051028441838970c83baf406012c"
)


FINAL_FIT_ROOT = Path(
    "outputs/"
    "phase4gr3b_tier_c_v4_final_fits"
)

FINAL_FIT_STATUS_PATH = (
    FINAL_FIT_ROOT
    / "phase4gr3b_final_fit_status.json"
)

RESOLVED_REGISTRY_PATH = Path(
    "outputs/"
    "phase4gr3_tier_c_v4_final_fit_preflight/"
    "resolved_trainable_final_fit_registry.csv"
)

NORMALIZATION_BINDING_PATH = Path(
    "outputs/"
    "phase4gr3ar2_tier_c_v4_normalization_binding/"
    "phase4gr3ar2_normalization_binding_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze"
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

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        FINAL_FIT_STATUS_PATH,
        RESOLVED_REGISTRY_PATH,
        NORMALIZATION_BINDING_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        "[1/6] Validating Phase 4G-R3B completion"
    )

    status = load_json(
        FINAL_FIT_STATUS_PATH
    )

    require(
        status[
            "phase4gr3b_status"
        ] == "completed",
        "Phase 4G-R3B is not completed.",
    )

    require(
        status[
            "authorized_run_count"
        ] == EXPECTED_RUN_COUNT,
        "Authorized run count is not 130.",
    )

    require(
        status[
            "completed_run_count"
        ] == EXPECTED_RUN_COUNT,
        "Completed run count is not 130.",
    )

    require(
        status[
            "incomplete_run_count"
        ] == 0,
        "Incomplete final fits remain.",
    )

    require(
        status[
            "interrupted_resumable_run_count"
        ] == 0,
        "Interrupted final fits remain.",
    )

    require(
        status[
            "all_trainable_final_fits_completed"
        ] is True,
        "Not all final fits completed.",
    )

    require(
        status[
            "selected_tuning_checkpoint_weights_reused"
        ] is False,
        "Tuning checkpoint weights were reused.",
    )

    require(
        status[
            "privileged_directory_read"
        ] is False,
        "Privileged directory was read.",
    )

    require(
        status[
            "clean_targets_read"
        ] is False,
        "Clean targets were read.",
    )

    require(
        status[
            "test_data_generated"
        ] is False,
        "Test data was generated.",
    )

    require(
        status[
            "test_data_read"
        ] is False,
        "Test data was read.",
    )

    require(
        status[
            "test_metrics_computed"
        ] is False,
        "Test metrics were computed.",
    )

    require(
        status[
            "test_open_count"
        ] == 0,
        "Test open count is nonzero.",
    )

    print(
        "[2/6] Validating normalization binding"
    )

    normalization_binding = load_json(
        NORMALIZATION_BINDING_PATH
    )

    require(
        normalization_binding[
            "phase4gr3ar2_status"
        ] == "frozen_normalization_artifact_bound",
        "Normalization binding is not frozen.",
    )

    require(
        normalization_binding[
            "normalization_artifact_sha256"
        ] == EXPECTED_NORMALIZATION_SHA256,
        "Normalization SHA256 changed.",
    )

    require(
        normalization_binding[
            "visible_fields_already_normalized"
        ] is True,
        "Visible fields are not marked normalized.",
    )

    require(
        normalization_binding[
            "normalization_applied_by_final_fit_runner"
        ] is False,
        "Final-fit runner reapplied normalization.",
    )

    require(
        normalization_binding[
            "double_normalization_prohibited"
        ] is True,
        "Double-normalization prohibition missing.",
    )

    print(
        "[3/6] Loading resolved 130-run registry"
    )

    registry_rows = load_csv(
        RESOLVED_REGISTRY_PATH
    )

    require(
        len(registry_rows)
        == EXPECTED_RUN_COUNT,
        "Resolved registry does not contain 130 runs.",
    )

    registry_by_run = {
        row[
            "run_id"
        ]:
            row
        for row in registry_rows
    }

    require(
        len(registry_by_run)
        == EXPECTED_RUN_COUNT,
        "Resolved registry contains duplicate run IDs.",
    )

    print(
        "[4/6] Auditing all 130 completed fits"
    )

    frozen_rows: list[dict[str, Any]] = []

    model_counts = Counter()
    noise_counts = Counter()
    seed_counts = Counter()

    validation_values = defaultdict(
        list
    )

    for registry_row in registry_rows:
        run_id = registry_row[
            "run_id"
        ]

        model_id = registry_row[
            "model_id"
        ]

        noise_fraction = float(
            registry_row[
                "noise_fraction"
            ]
        )

        effective_seed = int(
            registry_row[
                "effective_seed"
            ]
        )

        run_directory = Path(
            registry_row[
                "output_directory"
            ]
        )

        result_path = (
            run_directory
            / "final_fit_result.json"
        )

        history_path = (
            run_directory
            / "epoch_history.json"
        )

        best_checkpoint_path = (
            run_directory
            / "best_checkpoint.pt"
        )

        last_checkpoint_path = (
            run_directory
            / "last_checkpoint.pt"
        )

        for path in (
            result_path,
            history_path,
            best_checkpoint_path,
            last_checkpoint_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        result = load_json(
            result_path
        )

        history = load_json(
            history_path
        )

        require(
            result[
                "final_fit_status"
            ] == "completed",
            f"{run_id} is not completed.",
        )

        require(
            result[
                "protocol_version"
            ] == PROTOCOL_VERSION,
            f"{run_id} protocol changed.",
        )

        require(
            result[
                "run_id"
            ] == run_id,
            f"{run_id} result ID mismatch.",
        )

        require(
            result[
                "model_id"
            ] == model_id,
            f"{run_id} model mismatch.",
        )

        require(
            close_float(
                result[
                    "noise_fraction"
                ],
                noise_fraction,
            ),
            f"{run_id} noise mismatch.",
        )

        require(
            int(
                result[
                    "effective_seed"
                ]
            ) == effective_seed,
            f"{run_id} seed mismatch.",
        )

        require(
            result[
                "configuration_id"
            ]
            == registry_row[
                "selected_configuration_id"
            ],
            f"{run_id} configuration changed.",
        )

        require(
            result[
                "fresh_seeded_initialization"
            ] is True,
            f"{run_id} was not freshly initialized.",
        )

        require(
            result[
                "initialize_from_tuning_checkpoint"
            ] is False,
            f"{run_id} initialized from tuning checkpoint.",
        )

        require(
            result[
                "selected_tuning_checkpoint_weights_loaded"
            ] is False,
            f"{run_id} loaded tuning weights.",
        )

        require(
            result[
                "selected_tuning_checkpoint_usage"
            ] == "provenance_only",
            f"{run_id} tuning checkpoint use changed.",
        )

        require(
            result[
                "selected_tuning_epoch_used_as_fixed_duration"
            ] is False,
            f"{run_id} reused tuning epoch duration.",
        )

        require(
            result[
                "independent_final_fit_early_stopping"
            ] is True,
            f"{run_id} did not use independent stopping.",
        )

        require(
            result[
                "training_cell"
            ] == "train_joint",
            f"{run_id} training cell changed.",
        )

        require(
            result[
                "validation_cell"
            ] == "val_joint",
            f"{run_id} validation cell changed.",
        )

        require(
            result[
                "checkpoint_selection_direction"
            ] == "minimize",
            f"{run_id} checkpoint direction changed.",
        )

        require(
            result[
                "normalization_artifact_sha256"
            ] == EXPECTED_NORMALIZATION_SHA256,
            f"{run_id} normalization hash changed.",
        )

        require(
            result[
                "privileged_directory_read"
            ] is False,
            f"{run_id} records privileged access.",
        )

        require(
            result[
                "clean_targets_read"
            ] is False,
            f"{run_id} records clean-target access.",
        )

        require(
            result[
                "privileged_state_ids_read"
            ] is False,
            f"{run_id} records state-ID access.",
        )

        require(
            result[
                "test_data_generated"
            ] is False,
            f"{run_id} generated test data.",
        )

        require(
            result[
                "test_data_read"
            ] is False,
            f"{run_id} read test data.",
        )

        require(
            result[
                "test_metrics_computed"
            ] is False,
            f"{run_id} computed test metrics.",
        )

        require(
            result[
                "test_open_count"
            ] == 0,
            f"{run_id} opened test data.",
        )

        require(
            result[
                "scientific_test_comparison_performed"
            ] is False,
            f"{run_id} performed a test comparison.",
        )

        epochs_completed = int(
            result[
                "epochs_completed"
            ]
        )

        require(
            len(history)
            == epochs_completed,
            f"{run_id} history length mismatch.",
        )

        best_epoch = int(
            result[
                "best_epoch"
            ]
        )

        require(
            1 <= best_epoch <= epochs_completed,
            f"{run_id} best epoch invalid.",
        )

        best_metric = float(
            result[
                "best_validation_rollout_mse"
            ]
        )

        require(
            math.isfinite(
                best_metric
            ),
            f"{run_id} best metric non-finite.",
        )

        history_metrics = [
            float(
                epoch_row[
                    "validation"
                ][
                    "selection_metric"
                ]
            )
            for epoch_row in history
        ]

        require(
            all(
                math.isfinite(value)
                for value
                in history_metrics
            ),
            f"{run_id} history has non-finite metric.",
        )

        history_minimum = min(
            history_metrics
        )

        require(
            close_float(
                history_minimum,
                best_metric,
            ),
            (
                f"{run_id} stored best metric "
                "is not history minimum."
            ),
        )

        best_epoch_rows = [
            epoch_row
            for epoch_row in history
            if int(
                epoch_row[
                    "epoch"
                ]
            ) == best_epoch
        ]

        require(
            len(best_epoch_rows) == 1,
            f"{run_id} best epoch missing.",
        )

        require(
            close_float(
                best_epoch_rows[0][
                    "validation"
                ][
                    "selection_metric"
                ],
                best_metric,
            ),
            f"{run_id} best epoch metric mismatch.",
        )

        observed_best_sha = sha256_file(
            best_checkpoint_path
        )

        observed_last_sha = sha256_file(
            last_checkpoint_path
        )

        observed_history_sha = sha256_file(
            history_path
        )

        observed_result_sha = sha256_file(
            result_path
        )

        require(
            observed_best_sha
            == result[
                "best_checkpoint_sha256"
            ],
            f"{run_id} best checkpoint hash changed.",
        )

        require(
            observed_last_sha
            == result[
                "last_checkpoint_sha256"
            ],
            f"{run_id} last checkpoint hash changed.",
        )

        require(
            observed_history_sha
            == result[
                "history_sha256"
            ],
            f"{run_id} history hash changed.",
        )

        model_counts[
            model_id
        ] += 1

        noise_counts[
            noise_fraction
        ] += 1

        seed_counts[
            effective_seed
        ] += 1

        validation_values[
            (
                model_id,
                noise_fraction,
            )
        ].append(
            best_metric
        )

        frozen_rows.append(
            {
                "execution_index":
                    int(
                        registry_row[
                            "execution_index"
                        ]
                    ),

                "run_id":
                    run_id,

                "model_id":
                    model_id,

                "configuration_id":
                    result[
                        "configuration_id"
                    ],

                "noise_fraction":
                    noise_fraction,

                "effective_seed":
                    effective_seed,

                "best_epoch":
                    best_epoch,

                "epochs_completed":
                    epochs_completed,

                "stop_reason":
                    result[
                        "stop_reason"
                    ],

                "best_validation_rollout_mse":
                    best_metric,

                "total_optimizer_steps":
                    int(
                        result[
                            "total_optimizer_steps"
                        ]
                    ),

                "best_checkpoint_path":
                    str(
                        best_checkpoint_path
                    ),

                "best_checkpoint_sha256":
                    observed_best_sha,

                "last_checkpoint_path":
                    str(
                        last_checkpoint_path
                    ),

                "last_checkpoint_sha256":
                    observed_last_sha,

                "history_path":
                    str(
                        history_path
                    ),

                "history_sha256":
                    observed_history_sha,

                "result_path":
                    str(
                        result_path
                    ),

                "result_sha256":
                    observed_result_sha,

                "source_signature_sha256":
                    result[
                        "source_signature_sha256"
                    ],

                "normalization_artifact_sha256":
                    result[
                        "normalization_artifact_sha256"
                    ],

                "test_open_count":
                    0,

                "artifact_audit_passed":
                    True,
            }
        )

    require(
        dict(model_counts)
        == EXPECTED_MODEL_COUNTS,
        (
            "Model counts changed: "
            f"{dict(model_counts)}"
        ),
    )

    require(
        dict(noise_counts)
        == EXPECTED_NOISE_COUNTS,
        (
            "Noise counts changed: "
            f"{dict(noise_counts)}"
        ),
    )

    require(
        dict(seed_counts)
        == EXPECTED_SEED_COUNTS,
        (
            "Seed counts changed: "
            f"{dict(seed_counts)}"
        ),
    )

    print(
        "[5/6] Writing frozen final-fit registry"
    )

    frozen_rows.sort(
        key=lambda row: row[
            "execution_index"
        ]
    )

    frozen_registry_path = (
        OUTPUT_DIR
        / "frozen_final_fit_registry.csv"
    )

    write_csv(
        frozen_registry_path,
        frozen_rows,
    )

    validation_summary_rows = []

    for model_id in MODEL_IDS:
        for noise_fraction in NOISE_FRACTIONS:
            values = validation_values[
                (
                    model_id,
                    noise_fraction,
                )
            ]

            if not values:
                continue

            validation_summary_rows.append(
                {
                    "model_id":
                        model_id,

                    "noise_fraction":
                        noise_fraction,

                    "run_count":
                        len(values),

                    "mean_best_validation_rollout_mse":
                        sum(values)
                        / len(values),

                    "minimum_best_validation_rollout_mse":
                        min(values),

                    "maximum_best_validation_rollout_mse":
                        max(values),
                }
            )

    validation_summary_path = (
        OUTPUT_DIR
        / "final_fit_validation_summary.csv"
    )

    write_csv(
        validation_summary_path,
        validation_summary_rows,
    )

    print(
        "[6/6] Freezing artifact audit summary"
    )

    summary = {
        "phase":
            (
                "4G-R3C Tier C v4 130-run "
                "final-fit artifact freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "audited_final_fit_count":
            len(
                frozen_rows
            ),

        "expected_final_fit_count":
            EXPECTED_RUN_COUNT,

        "all_130_final_fit_results_present":
            True,

        "all_130_best_checkpoints_present":
            True,

        "all_130_last_checkpoints_present":
            True,

        "all_130_histories_present":
            True,

        "all_result_hashes_verified":
            True,

        "all_best_checkpoint_hashes_verified":
            True,

        "all_last_checkpoint_hashes_verified":
            True,

        "all_history_hashes_verified":
            True,

        "all_best_metrics_match_history_minima":
            True,

        "model_counts":
            dict(
                sorted(
                    model_counts.items()
                )
            ),

        "noise_counts":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    noise_counts.items()
                )
            },

        "seed_counts":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    seed_counts.items()
                )
            },

        "normalization_artifact_sha256":
            EXPECTED_NORMALIZATION_SHA256,

        "selected_tuning_checkpoint_weights_reused":
            False,

        "fresh_seeded_initialization_verified":
            True,

        "independent_final_fit_early_stopping_verified":
            True,

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

        "b0_persistence_evaluations_completed":
            False,

        "primary_validation_baseline_selected":
            False,

        "frozen_final_fit_registry_path":
            str(
                frozen_registry_path
            ),

        "frozen_final_fit_registry_sha256":
            sha256_file(
                frozen_registry_path
            ),

        "validation_summary_path":
            str(
                validation_summary_path
            ),

        "validation_summary_sha256":
            sha256_file(
                validation_summary_path
            ),

        "additional_final_fit_training_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "final_fit_artifacts_frozen":
            True,

        "phase4gr3c_status":
            "final_fit_artifacts_frozen",

        "next_phase":
            (
                "4G-R3D Tier C v4 B0 persistence "
                "evaluation and primary comparator freeze"
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4gr3c_final_fit_freeze_summary.json"
    )

    write_json(
        summary_path,
        summary,
    )

    print(
        "Phase 4G-R3C final-fit artifact freeze passed."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
