from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
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

EXPECTED_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
    "OCM": 24,
}

EXPECTED_CONFIGURATION_COUNT = 43
EXPECTED_SELECTED_CONFIGURATION_COUNT = 6

SELECTION_METRIC = (
    "val_joint noisy-target rollout field MSE"
)

SELECTION_DIRECTION = "minimize"


TUNING_ROOT = Path(
    "outputs/"
    "phase4fr3b_tier_c_v4_development_tuning"
)

TUNING_STATUS_PATH = (
    TUNING_ROOT
    / "phase4fr3b_tuning_status.json"
)

TUNING_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_tuning_registry.csv"
)

IMPLEMENTATION_ACCEPTANCE_PATH = Path(
    "outputs/"
    "phase4er3d_tier_c_v4_implementation_acceptance/"
    "phase4er3d_implementation_acceptance_summary.json"
)

REPAIRED_PREFLIGHT_PATH = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes/"
    "phase4fr3ar1_repaired_tuning_preflight_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
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

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

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


def sha256_file(path: Path) -> str:
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


def load_registry() -> list[dict[str, Any]]:
    with TUNING_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    require(
        len(rows)
        == EXPECTED_CONFIGURATION_COUNT,
        (
            "Unexpected tuning-registry row count: "
            f"{len(rows)}"
        ),
    )

    parsed = []

    for registry_index, row in enumerate(
        rows
    ):
        model_id = row["model_id"]

        require(
            model_id in MODEL_IDS,
            (
                "Unexpected model ID in registry: "
                f"{model_id}"
            ),
        )

        configuration = json.loads(
            row[
                "source_configuration_json"
            ]
        )

        require(
            configuration[
                "configuration_id"
            ]
            == row[
                "configuration_id"
            ],
            "Configuration ID mismatch.",
        )

        require(
            row[
                "training_cell"
            ] == "train_joint",
            "Registry training cell changed.",
        )

        require(
            row[
                "checkpoint_selection_cell"
            ] == "val_joint",
            "Registry validation cell changed.",
        )

        require(
            float(
                row[
                    "tuning_noise_fraction"
                ]
            ) == 0.25,
            "Registry tuning noise changed.",
        )

        require(
            row[
                "test_access_allowed"
            ].strip().lower()
            in {
                "false",
                "0",
            },
            "Registry unexpectedly permits test access.",
        )

        parsed.append(
            {
                "registry_index":
                    registry_index,

                "model_id":
                    model_id,

                "configuration_id":
                    row[
                        "configuration_id"
                    ],

                "configuration":
                    configuration,

                "registry_row":
                    row,
            }
        )

    counts = Counter(
        row["model_id"]
        for row in parsed
    )

    require(
        dict(counts)
        == EXPECTED_CONFIGURATION_COUNTS,
        (
            "Registry configuration counts changed: "
            f"{dict(counts)}"
        ),
    )

    configuration_ids = [
        row["configuration_id"]
        for row in parsed
    ]

    require(
        len(configuration_ids)
        == len(set(configuration_ids)),
        "Configuration IDs are not unique.",
    )

    return parsed


def validate_protocol_sources() -> dict[str, Any]:
    required_paths = (
        TUNING_STATUS_PATH,
        TUNING_REGISTRY_PATH,
        IMPLEMENTATION_ACCEPTANCE_PATH,
        REPAIRED_PREFLIGHT_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    tuning_status = load_json(
        TUNING_STATUS_PATH
    )

    acceptance = load_json(
        IMPLEMENTATION_ACCEPTANCE_PATH
    )

    repaired_preflight = load_json(
        REPAIRED_PREFLIGHT_PATH
    )

    require(
        tuning_status[
            "phase4fr3b_status"
        ] == "completed",
        "Development tuning is not completed.",
    )

    require(
        tuning_status[
            "all_configurations_completed"
        ] is True,
        "Not every frozen configuration completed.",
    )

    require(
        tuning_status[
            "configuration_count"
        ] == EXPECTED_CONFIGURATION_COUNT,
        "Tuning status has the wrong configuration count.",
    )

    require(
        tuning_status[
            "completed_configuration_count"
        ] == EXPECTED_CONFIGURATION_COUNT,
        "Completed configuration count is not 43.",
    )

    require(
        tuning_status[
            "incomplete_configuration_count"
        ] == 0,
        "Some tuning configurations remain incomplete.",
    )

    require(
        tuning_status[
            "privileged_directory_read"
        ] is False,
        "Tuning status records privileged-directory access.",
    )

    require(
        tuning_status[
            "test_data_read"
        ] is False,
        "Tuning status records test access.",
    )

    require(
        tuning_status[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        tuning_status[
            "final_fit_performed"
        ] is False,
        "Final fitting was already performed.",
    )

    require(
        acceptance[
            "phase4er3d_status"
        ] == "implementation_accepted",
        "Implementation acceptance is not frozen.",
    )

    require(
        acceptance[
            "tuning_execution_authorized"
        ] is True,
        "Tuning was not authorized.",
    )

    require(
        acceptance[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting was already authorized.",
    )

    require(
        acceptance[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation was already authorized.",
    )

    require(
        acceptance[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation was already authorized.",
    )

    require(
        repaired_preflight[
            "phase4fr3ar1_status"
        ]
        == (
            "ready_for_development_tuning_"
            "with_derived_visible_structural_indexes"
        ),
        "Repaired development preflight is not accepted.",
    )

    return {
        "required_paths":
            required_paths,

        "tuning_status":
            tuning_status,

        "acceptance":
            acceptance,

        "repaired_preflight":
            repaired_preflight,
    }


def load_and_validate_result(
    registry_record: dict[str, Any],
) -> dict[str, Any]:
    model_id = registry_record[
        "model_id"
    ]

    configuration_id = registry_record[
        "configuration_id"
    ]

    configuration_dir = (
        TUNING_ROOT
        / model_id
        / configuration_id
    )

    result_path = (
        configuration_dir
        / "tuning_result.json"
    )

    best_checkpoint_path = (
        configuration_dir
        / "best_checkpoint.pt"
    )

    last_checkpoint_path = (
        configuration_dir
        / "last_checkpoint.pt"
    )

    history_path = (
        configuration_dir
        / "epoch_history.json"
    )

    for path in (
        result_path,
        best_checkpoint_path,
        last_checkpoint_path,
        history_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    result = load_json(
        result_path
    )

    require(
        result[
            "protocol_version"
        ] == PROTOCOL_VERSION,
        f"{configuration_id} protocol version changed.",
    )

    require(
        result[
            "model_id"
        ] == model_id,
        f"{configuration_id} model ID changed.",
    )

    require(
        result[
            "configuration_id"
        ] == configuration_id,
        f"{configuration_id} configuration ID changed.",
    )

    require(
        result[
            "configuration"
        ] == registry_record[
            "configuration"
        ],
        f"{configuration_id} configuration changed.",
    )

    require(
        result[
            "tuning_status"
        ] == "completed",
        f"{configuration_id} did not complete.",
    )

    require(
        result[
            "training_cell"
        ] == "train_joint",
        f"{configuration_id} training cell changed.",
    )

    require(
        result[
            "validation_cell"
        ] == "val_joint",
        f"{configuration_id} validation cell changed.",
    )

    require(
        float(
            result[
                "noise_fraction"
            ]
        ) == 0.25,
        f"{configuration_id} noise fraction changed.",
    )

    require(
        result[
            "checkpoint_selection_metric"
        ] == SELECTION_METRIC,
        (
            f"{configuration_id} selection metric "
            "changed."
        ),
    )

    require(
        result[
            "checkpoint_selection_direction"
        ] == SELECTION_DIRECTION,
        (
            f"{configuration_id} selection "
            "direction changed."
        ),
    )

    metric = float(
        result[
            "best_validation_rollout_mse"
        ]
    )

    require(
        math.isfinite(metric),
        (
            f"{configuration_id} has a "
            "non-finite selection metric."
        ),
    )

    best_epoch = int(
        result[
            "best_epoch"
        ]
    )

    epochs_completed = int(
        result[
            "epochs_completed"
        ]
    )

    require(
        1 <= best_epoch <= epochs_completed,
        (
            f"{configuration_id} has an "
            "invalid best epoch."
        ),
    )

    require(
        result[
            "privileged_directory_read"
        ] is False,
        (
            f"{configuration_id} records "
            "privileged-directory access."
        ),
    )

    require(
        result[
            "clean_targets_read"
        ] is False,
        (
            f"{configuration_id} records "
            "clean-target access."
        ),
    )

    require(
        result[
            "privileged_state_ids_read"
        ] is False,
        (
            f"{configuration_id} records "
            "privileged-state access."
        ),
    )

    require(
        result[
            "test_data_generated"
        ] is False,
        (
            f"{configuration_id} generated "
            "test data."
        ),
    )

    require(
        result[
            "test_data_read"
        ] is False,
        (
            f"{configuration_id} read "
            "test data."
        ),
    )

    require(
        result[
            "test_metrics_computed"
        ] is False,
        (
            f"{configuration_id} computed "
            "test metrics."
        ),
    )

    require(
        result[
            "test_open_count"
        ] == 0,
        (
            f"{configuration_id} opened "
            "test data."
        ),
    )

    require(
        result[
            "final_fit_performed"
        ] is False,
        (
            f"{configuration_id} performed "
            "a final fit."
        ),
    )

    require(
        sha256_file(
            best_checkpoint_path
        )
        == result[
            "best_checkpoint_sha256"
        ],
        (
            f"{configuration_id} best "
            "checkpoint hash changed."
        ),
    )

    history = load_json(
        history_path
    )

    require(
        len(history)
        == epochs_completed,
        (
            f"{configuration_id} history length "
            "does not match epochs completed."
        ),
    )

    best_history_rows = [
        row
        for row in history
        if int(
            row["epoch"]
        ) == best_epoch
    ]

    require(
        len(best_history_rows) == 1,
        (
            f"{configuration_id} best epoch "
            "is absent from history."
        ),
    )

    history_best_metric = float(
        best_history_rows[0][
            "validation"
        ][
            "selection_metric"
        ]
    )

    require(
        math.isclose(
            history_best_metric,
            metric,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        (
            f"{configuration_id} result metric "
            "does not match epoch history."
        ),
    )

    observed_history_minimum = min(
        float(
            row[
                "validation"
            ][
                "selection_metric"
            ]
        )
        for row in history
    )

    require(
        math.isclose(
            observed_history_minimum,
            metric,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        (
            f"{configuration_id} stored best metric "
            "is not the history minimum."
        ),
    )

    return {
        "registry_index":
            registry_record[
                "registry_index"
            ],

        "model_id":
            model_id,

        "configuration_id":
            configuration_id,

        "configuration":
            registry_record[
                "configuration"
            ],

        "tuning_seed":
            int(
                result[
                    "tuning_seed"
                ]
            ),

        "best_epoch":
            best_epoch,

        "best_validation_rollout_mse":
            metric,

        "epochs_completed":
            epochs_completed,

        "stop_reason":
            result[
                "stop_reason"
            ],

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
            sha256_file(
                best_checkpoint_path
            ),

        "last_checkpoint_path":
            str(
                last_checkpoint_path
            ),

        "last_checkpoint_sha256":
            sha256_file(
                last_checkpoint_path
            ),

        "history_path":
            str(
                history_path
            ),

        "history_sha256":
            sha256_file(
                history_path
            ),

        "result_path":
            str(
                result_path
            ),

        "result_sha256":
            sha256_file(
                result_path
            ),

        "source_signature_sha256":
            result[
                "source_signature_sha256"
            ],

        "test_open_count":
            0,

        "eligible_for_selection":
            True,
    }


def select_configuration(
    model_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    require(
        bool(model_rows),
        "Cannot select from an empty model family.",
    )

    # Primary rule: minimum frozen validation metric.
    #
    # Administrative tie rule:
    # use frozen tuning-registry order only if metrics
    # are numerically identical. Registry order existed
    # before tuning outcomes and is not outcome-derived.
    ranked = sorted(
        model_rows,
        key=lambda row: (
            row[
                "best_validation_rollout_mse"
            ],
            row[
                "registry_index"
            ],
        ),
    )

    selected = dict(
        ranked[0]
    )

    selected[
        "family_rank"
    ] = 1

    selected[
        "selected"
    ] = True

    selected[
        "selection_rule"
    ] = (
        "minimum val_joint noisy-target "
        "rollout field MSE"
    )

    selected[
        "tie_break_rule"
    ] = (
        "frozen tuning-registry order for "
        "exact numerical ties only"
    )

    selected[
        "selection_metric_direction"
    ] = "minimize"

    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4fr3c_tuning_selection_summary.json"
    )

    if summary_path.exists():
        existing = load_json(
            summary_path
        )

        if existing.get(
            "phase4fr3c_status"
        ) == "tuning_selection_frozen":
            print(
                "Phase 4F-R3C tuning selection "
                "is already frozen."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    print(
        "[1/6] Validating completed tuning phase"
    )

    protocol_sources = (
        validate_protocol_sources()
    )

    print(
        "[2/6] Loading frozen tuning registry"
    )

    registry_records = load_registry()

    print(
        "[3/6] Validating all 43 tuning results"
    )

    validated_results = [
        load_and_validate_result(
            registry_record
        )
        for registry_record
        in registry_records
    ]

    require(
        len(validated_results)
        == EXPECTED_CONFIGURATION_COUNT,
        "Did not validate all 43 results.",
    )

    print(
        "[4/6] Ranking configurations within each model family"
    )

    ranking_rows = []
    selected_rows = []

    for model_id in MODEL_IDS:
        family_rows = [
            row
            for row in validated_results
            if row[
                "model_id"
            ] == model_id
        ]

        require(
            len(family_rows)
            == EXPECTED_CONFIGURATION_COUNTS[
                model_id
            ],
            (
                f"{model_id} has the wrong number "
                "of validated results."
            ),
        )

        ranked_family = sorted(
            family_rows,
            key=lambda row: (
                row[
                    "best_validation_rollout_mse"
                ],
                row[
                    "registry_index"
                ],
            ),
        )

        for family_rank, row in enumerate(
            ranked_family,
            start=1,
        ):
            ranking_rows.append(
                {
                    "model_id":
                        model_id,

                    "family_rank":
                        family_rank,

                    "configuration_id":
                        row[
                            "configuration_id"
                        ],

                    "best_validation_rollout_mse":
                        row[
                            "best_validation_rollout_mse"
                        ],

                    "best_epoch":
                        row[
                            "best_epoch"
                        ],

                    "epochs_completed":
                        row[
                            "epochs_completed"
                        ],

                    "stop_reason":
                        row[
                            "stop_reason"
                        ],

                    "registry_index":
                        row[
                            "registry_index"
                        ],

                    "selected":
                        family_rank == 1,

                    "best_checkpoint_path":
                        row[
                            "best_checkpoint_path"
                        ],

                    "best_checkpoint_sha256":
                        row[
                            "best_checkpoint_sha256"
                        ],

                    "test_data_read":
                        False,
                }
            )

        selected_rows.append(
            select_configuration(
                family_rows
            )
        )

    require(
        len(selected_rows)
        == EXPECTED_SELECTED_CONFIGURATION_COUNT,
        "Expected exactly six selected configurations.",
    )

    require(
        {
            row["model_id"]
            for row in selected_rows
        } == set(MODEL_IDS),
        "Selected registry does not cover all model families.",
    )

    write_csv(
        OUTPUT_DIR
        / "all_tuning_results_ranked.csv",
        ranking_rows,
    )

    print(
        "[5/6] Freezing selected configuration registry"
    )

    selected_registry = {
        "phase":
            (
                "4F-R3C Tier C v4 selected "
                "development-tuning registry"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "selection_cell":
            "val_joint",

        "selection_noise_fraction":
            0.25,

        "selection_metric":
            SELECTION_METRIC,

        "selection_direction":
            SELECTION_DIRECTION,

        "selection_rule":
            (
                "Select the configuration with the "
                "minimum val_joint noisy-target "
                "rollout field MSE independently "
                "within each model family."
            ),

        "tie_break_rule":
            (
                "Use frozen tuning-registry order "
                "only for exact numerical ties."
            ),

        "selected_configuration_count":
            len(selected_rows),

        "selected_configurations":
            selected_rows,

        "validation_selection_completed":
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

        "final_fit_performed":
            False,
    }

    selected_registry_path = (
        OUTPUT_DIR
        / "selected_tuning_registry.json"
    )

    write_json(
        selected_registry_path,
        selected_registry,
    )

    selected_csv_rows = []

    for row in selected_rows:
        selected_csv_rows.append(
            {
                "model_id":
                    row[
                        "model_id"
                    ],

                "configuration_id":
                    row[
                        "configuration_id"
                    ],

                "registry_index":
                    row[
                        "registry_index"
                    ],

                "tuning_seed":
                    row[
                        "tuning_seed"
                    ],

                "best_epoch":
                    row[
                        "best_epoch"
                    ],

                "best_validation_rollout_mse":
                    row[
                        "best_validation_rollout_mse"
                    ],

                "epochs_completed":
                    row[
                        "epochs_completed"
                    ],

                "stop_reason":
                    row[
                        "stop_reason"
                    ],

                "best_checkpoint_path":
                    row[
                        "best_checkpoint_path"
                    ],

                "best_checkpoint_sha256":
                    row[
                        "best_checkpoint_sha256"
                    ],

                "configuration_json":
                    json.dumps(
                        row[
                            "configuration"
                        ],
                        sort_keys=True,
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "selected_tuning_registry.csv",
        selected_csv_rows,
    )

    print(
        "[6/6] Writing immutable selection summary"
    )

    source_paths = list(
        protocol_sources[
            "required_paths"
        ]
    )

    source_paths.extend(
        Path(
            row[
                "result_path"
            ]
        )
        for row in validated_results
    )

    source_paths.extend(
        Path(
            row[
                "best_checkpoint_path"
            ]
        )
        for row in validated_results
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in source_paths
    }

    write_json(
        OUTPUT_DIR
        / "tuning_selection_source_hashes.json",
        source_hashes,
    )

    summary = {
        "phase":
            (
                "4F-R3C Tier C v4 development-only "
                "tuning selection and registry freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_tuning_status":
            protocol_sources[
                "tuning_status"
            ][
                "phase4fr3b_status"
            ],

        "source_configuration_count":
            len(validated_results),

        "source_configuration_counts":
            EXPECTED_CONFIGURATION_COUNTS,

        "validated_configuration_count":
            len(validated_results),

        "selected_configuration_count":
            len(selected_rows),

        "selected_model_ids":
            [
                row[
                    "model_id"
                ]
                for row in selected_rows
            ],

        "selected_configuration_ids":
            {
                row[
                    "model_id"
                ]:
                    row[
                        "configuration_id"
                    ]
                for row in selected_rows
            },

        "selected_validation_metrics":
            {
                row[
                    "model_id"
                ]:
                    row[
                        "best_validation_rollout_mse"
                    ]
                for row in selected_rows
            },

        "selection_cell":
            "val_joint",

        "selection_noise_fraction":
            0.25,

        "selection_metric":
            SELECTION_METRIC,

        "selection_direction":
            SELECTION_DIRECTION,

        "selection_performed_independently_by_model_family":
            True,

        "tie_break_rule":
            (
                "frozen tuning-registry order "
                "for exact numerical ties only"
            ),

        "selected_registry_path":
            str(
                selected_registry_path
            ),

        "selected_registry_sha256":
            sha256_file(
                selected_registry_path
            ),

        "all_tuning_results_validated":
            True,

        "all_selected_checkpoints_exist":
            True,

        "all_selected_checkpoint_hashes_frozen":
            True,

        "validation_selection_performed":
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

        "scientific_test_comparison_performed":
            False,

        "additional_tuning_authorized":
            False,

        "selected_registry_frozen":
            True,

        "final_fit_execution_authorized":
            True,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "phase4fr3c_status":
            "tuning_selection_frozen",

        "next_phase":
            (
                "4G-R3 Tier C v4 final-fit "
                "execution preflight"
            ),
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        "Phase 4F-R3C tuning selection frozen."
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
