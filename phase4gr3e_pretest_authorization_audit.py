from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_TUNING_COUNT = 43
EXPECTED_SELECTED_COUNT = 6
EXPECTED_FINAL_FIT_COUNT = 130
EXPECTED_B0_COUNT = 5

PRIMARY_MODEL = "OCM"
PRIMARY_COMPARATOR = "B4"
PRIMARY_CELL = "test_joint"
PRIMARY_NOISE = 0.25
PRIMARY_METRIC = "noisy_target_rollout_mse"


TUNING_STATUS_PATH = Path(
    "outputs/"
    "phase4fr3b_tier_c_v4_development_tuning/"
    "phase4fr3b_tuning_status.json"
)

TUNING_SELECTION_PATH = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection/"
    "phase4fr3c_tuning_selection_summary.json"
)

SELECTED_REGISTRY_PATH = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection/"
    "selected_tuning_registry.json"
)

FINAL_FIT_STATUS_PATH = Path(
    "outputs/"
    "phase4gr3b_tier_c_v4_final_fits/"
    "phase4gr3b_final_fit_status.json"
)

FINAL_FIT_FREEZE_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "phase4gr3c_final_fit_freeze_summary.json"
)

FROZEN_FINAL_FIT_REGISTRY_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

COMPARATOR_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator/"
    "phase4gr3d1_primary_comparator_summary.json"
)

COMPARATOR_ARTIFACT_PATH = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator/"
    "primary_validation_comparator.json"
)

B0_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3d2_tier_c_v4_b0_persistence/"
    "phase4gr3d2_b0_persistence_summary.json"
)

B0_REGISTRY_PATH = Path(
    "outputs/"
    "phase4gr3d2_tier_c_v4_b0_persistence/"
    "b0_persistence_validation_registry.csv"
)

NORMALIZATION_BINDING_PATH = Path(
    "outputs/"
    "phase4gr3ar2_tier_c_v4_normalization_binding/"
    "phase4gr3ar2_normalization_binding_summary.json"
)

TEST_POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3e_tier_c_v4_pretest_authorization_audit"
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
    left: Any,
    right: Any,
) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def check_zero_test_access(
    *,
    source_name: str,
    record: dict[str, Any],
) -> None:
    if "test_data_generated" in record:
        require(
            record[
                "test_data_generated"
            ] is False,
            (
                f"{source_name} records "
                "test-data generation."
            ),
        )

    if "test_data_read" in record:
        require(
            record[
                "test_data_read"
            ] is False,
            (
                f"{source_name} records "
                "test-data access."
            ),
        )

    if "test_metrics_computed" in record:
        require(
            record[
                "test_metrics_computed"
            ] is False,
            (
                f"{source_name} records "
                "test metrics."
            ),
        )

    if "test_open_count" in record:
        require(
            int(
                record[
                    "test_open_count"
                ]
            ) == 0,
            (
                f"{source_name} has "
                "nonzero test_open_count."
            ),
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        TUNING_STATUS_PATH,
        TUNING_SELECTION_PATH,
        SELECTED_REGISTRY_PATH,
        FINAL_FIT_STATUS_PATH,
        FINAL_FIT_FREEZE_PATH,
        FROZEN_FINAL_FIT_REGISTRY_PATH,
        COMPARATOR_SUMMARY_PATH,
        COMPARATOR_ARTIFACT_PATH,
        B0_SUMMARY_PATH,
        B0_REGISTRY_PATH,
        NORMALIZATION_BINDING_PATH,
        TEST_POLICY_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    evidence = []

    print(
        "[1/7] Auditing development tuning"
    )

    tuning_status = load_json(
        TUNING_STATUS_PATH
    )

    require(
        tuning_status[
            "phase4fr3b_status"
        ] == "completed",
        "Tuning phase is not completed.",
    )

    require(
        tuning_status[
            "configuration_count"
        ] == EXPECTED_TUNING_COUNT,
        "Expected 43 tuning configurations.",
    )

    require(
        tuning_status[
            "completed_configuration_count"
        ] == EXPECTED_TUNING_COUNT,
        "Not all 43 tuning configurations completed.",
    )

    require(
        tuning_status[
            "incomplete_configuration_count"
        ] == 0,
        "Incomplete tuning configurations remain.",
    )

    require(
        tuning_status[
            "all_configurations_completed"
        ] is True,
        "Tuning completion flag is false.",
    )

    require(
        tuning_status[
            "privileged_directory_read"
        ] is False,
        "Tuning accessed privileged directory.",
    )

    require(
        tuning_status[
            "final_fit_performed"
        ] is False,
        "Tuning phase unexpectedly performed final fitting.",
    )

    check_zero_test_access(
        source_name="tuning_status",
        record=tuning_status,
    )

    evidence.append(
        {
            "requirement":
                "All 43 tuning runs completed.",

            "passed":
                True,

            "source":
                str(
                    TUNING_STATUS_PATH
                ),
        }
    )

    print(
        "[2/7] Auditing selected configuration freeze"
    )

    selection = load_json(
        TUNING_SELECTION_PATH
    )

    selected_registry = load_json(
        SELECTED_REGISTRY_PATH
    )

    require(
        selection[
            "phase4fr3c_status"
        ] == "tuning_selection_frozen",
        "Tuning selection is not frozen.",
    )

    require(
        selection[
            "selected_configuration_count"
        ] == EXPECTED_SELECTED_COUNT,
        "Expected six selected configurations.",
    )

    require(
        selection[
            "selected_registry_frozen"
        ] is True,
        "Selected registry is not frozen.",
    )

    require(
        selection[
            "additional_tuning_authorized"
        ] is False,
        "Additional tuning remains authorized.",
    )

    require(
        selected_registry[
            "selected_configuration_count"
        ] == EXPECTED_SELECTED_COUNT,
        "Selected registry does not contain six models.",
    )

    require(
        {
            row[
                "model_id"
            ]
            for row in selected_registry[
                "selected_configurations"
            ]
        }
        == {
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "OCM",
        },
        "Selected model family set changed.",
    )

    check_zero_test_access(
        source_name="tuning_selection",
        record=selection,
    )

    check_zero_test_access(
        source_name="selected_registry",
        record=selected_registry,
    )

    evidence.append(
        {
            "requirement":
                (
                    "One selected configuration per "
                    "trainable model is frozen."
                ),

            "passed":
                True,

            "source":
                str(
                    TUNING_SELECTION_PATH
                ),
        }
    )

    print(
        "[3/7] Auditing 130 final fits and checksums"
    )

    final_status = load_json(
        FINAL_FIT_STATUS_PATH
    )

    final_freeze = load_json(
        FINAL_FIT_FREEZE_PATH
    )

    frozen_final_rows = load_csv(
        FROZEN_FINAL_FIT_REGISTRY_PATH
    )

    require(
        final_status[
            "phase4gr3b_status"
        ] == "completed",
        "Final-fit execution is not completed.",
    )

    require(
        final_status[
            "completed_run_count"
        ] == EXPECTED_FINAL_FIT_COUNT,
        "Expected 130 completed final fits.",
    )

    require(
        final_status[
            "incomplete_run_count"
        ] == 0,
        "Incomplete final fits remain.",
    )

    require(
        final_status[
            "all_trainable_final_fits_completed"
        ] is True,
        "Final-fit completion flag is false.",
    )

    require(
        final_freeze[
            "phase4gr3c_status"
        ] == "final_fit_artifacts_frozen",
        "Final-fit artifacts are not frozen.",
    )

    require(
        final_freeze[
            "audited_final_fit_count"
        ] == EXPECTED_FINAL_FIT_COUNT,
        "Expected 130 audited final fits.",
    )

    require(
        final_freeze[
            "all_130_final_fit_results_present"
        ] is True,
        "Final-fit result files are incomplete.",
    )

    require(
        final_freeze[
            "all_130_best_checkpoints_present"
        ] is True,
        "Best checkpoints are incomplete.",
    )

    require(
        final_freeze[
            "all_result_hashes_verified"
        ] is True,
        "Result hashes were not all verified.",
    )

    require(
        final_freeze[
            "all_best_checkpoint_hashes_verified"
        ] is True,
        "Best-checkpoint hashes were not all verified.",
    )

    require(
        final_freeze[
            "all_last_checkpoint_hashes_verified"
        ] is True,
        "Last-checkpoint hashes were not all verified.",
    )

    require(
        final_freeze[
            "all_history_hashes_verified"
        ] is True,
        "History hashes were not all verified.",
    )

    require(
        final_freeze[
            "all_best_metrics_match_history_minima"
        ] is True,
        "Stored best metrics do not all match histories.",
    )

    require(
        len(
            frozen_final_rows
        ) == EXPECTED_FINAL_FIT_COUNT,
        "Frozen final-fit registry does not contain 130 rows.",
    )

    require(
        all(
            int(
                row[
                    "test_open_count"
                ]
            ) == 0
            for row in frozen_final_rows
        ),
        "A frozen final-fit row has nonzero test access.",
    )

    require(
        final_freeze[
            "additional_final_fit_training_authorized"
        ] is False,
        "Additional final-fit training remains authorized.",
    )

    check_zero_test_access(
        source_name="final_fit_status",
        record=final_status,
    )

    check_zero_test_access(
        source_name="final_fit_freeze",
        record=final_freeze,
    )

    evidence.append(
        {
            "requirement":
                (
                    "All 130 trainable final fits are "
                    "completed and checksummed."
                ),

            "passed":
                True,

            "source":
                str(
                    FINAL_FIT_FREEZE_PATH
                ),
        }
    )

    evidence.append(
        {
            "requirement":
                (
                    "No final checkpoint was selected "
                    "using a test artifact."
                ),

            "passed":
                True,

            "source":
                str(
                    FINAL_FIT_FREEZE_PATH
                ),
        }
    )

    print(
        "[4/7] Auditing B0 persistence evaluations"
    )

    b0_summary = load_json(
        B0_SUMMARY_PATH
    )

    b0_rows = load_csv(
        B0_REGISTRY_PATH
    )

    require(
        b0_summary[
            "phase4gr3d2_status"
        ] == "b0_persistence_evaluations_frozen",
        "B0 persistence evaluations are not frozen.",
    )

    require(
        b0_summary[
            "b0_evaluation_count"
        ] == EXPECTED_B0_COUNT,
        "Expected five B0 evaluations.",
    )

    require(
        b0_summary[
            "all_b0_evaluations_completed"
        ] is True,
        "Not all B0 evaluations completed.",
    )

    require(
        b0_summary[
            "training_performed"
        ] is False,
        "B0 unexpectedly performed training.",
    )

    require(
        b0_summary[
            "optimizer_steps_performed"
        ] == 0,
        "B0 performed optimizer steps.",
    )

    require(
        len(
            b0_rows
        ) == EXPECTED_B0_COUNT,
        "B0 registry does not contain five rows.",
    )

    require(
        all(
            row[
                "evaluation_status"
            ] == "completed"
            for row in b0_rows
        ),
        "A B0 evaluation is incomplete.",
    )

    require(
        sorted(
            float(
                row[
                    "noise_fraction"
                ]
            )
            for row in b0_rows
        )
        == [
            0.0,
            0.1,
            0.25,
            0.5,
            1.0,
        ],
        "B0 noise grid changed.",
    )

    check_zero_test_access(
        source_name="b0_summary",
        record=b0_summary,
    )

    evidence.append(
        {
            "requirement":
                (
                    "All five B0 persistence "
                    "evaluations are completed."
                ),

            "passed":
                True,

            "source":
                str(
                    B0_SUMMARY_PATH
                ),
        }
    )

    print(
        "[5/7] Auditing primary comparator freeze"
    )

    comparator_summary = load_json(
        COMPARATOR_SUMMARY_PATH
    )

    comparator = load_json(
        COMPARATOR_ARTIFACT_PATH
    )

    require(
        comparator_summary[
            "phase4gr3d1_status"
        ] == "primary_comparator_frozen",
        "Primary comparator is not frozen.",
    )

    require(
        comparator_summary[
            "selected_comparator"
        ] == PRIMARY_COMPARATOR,
        "Primary comparator is not B4.",
    )

    require(
        comparator[
            "primary_comparator_frozen"
        ] is True,
        "Comparator artifact is not frozen.",
    )

    require(
        comparator[
            "selected_comparator_model_id"
        ] == PRIMARY_COMPARATOR,
        "Comparator artifact does not select B4.",
    )

    require(
        comparator[
            "selection_cell"
        ] == "val_joint",
        "Comparator was not selected on val_joint.",
    )

    require(
        close_float(
            comparator[
                "selection_noise_fraction"
            ],
            PRIMARY_NOISE,
        ),
        "Comparator selection noise is not 0.25.",
    )

    require(
        comparator[
            "selection_metric"
        ] == PRIMARY_METRIC,
        "Comparator metric changed.",
    )

    require(
        comparator[
            "selection_direction"
        ] == "minimize",
        "Comparator selection direction changed.",
    )

    require(
        comparator[
            "primary_model"
        ] == PRIMARY_MODEL,
        "Primary model changed.",
    )

    require(
        comparator[
            "primary_model_used_for_comparator_selection"
        ] is False,
        (
            "Primary model improperly influenced "
            "baseline comparator selection."
        ),
    )

    require(
        comparator[
            "test_artifact_used_for_selection"
        ] is False,
        "Test artifact influenced comparator selection.",
    )

    require(
        comparator[
            "selection_frozen_before_test_generation"
        ] is True,
        "Comparator was not frozen before test generation.",
    )

    check_zero_test_access(
        source_name="comparator_summary",
        record=comparator_summary,
    )

    check_zero_test_access(
        source_name="comparator_artifact",
        record=comparator,
    )

    evidence.append(
        {
            "requirement":
                (
                    "Validation-selected best B1-B5 "
                    "baseline is frozen."
                ),

            "passed":
                True,

            "source":
                str(
                    COMPARATOR_ARTIFACT_PATH
                ),
        }
    )

    print(
        "[6/7] Auditing frozen test-opening policy"
    )

    test_policy = load_json(
        TEST_POLICY_PATH
    )

    require(
        test_policy[
            "test_generation_before_tuning"
        ] is False,
        "Test was permitted before tuning.",
    )

    require(
        test_policy[
            "test_generation_before_final_fits"
        ] is False,
        "Test was permitted before final fits.",
    )

    require(
        test_policy[
            "test_generation_authorized_now"
        ] is False,
        (
            "Existing frozen policy already says "
            "test generation is authorized."
        ),
    )

    require(
        test_policy[
            "test_evaluation_authorized_now"
        ] is False,
        (
            "Existing frozen policy already says "
            "test evaluation is authorized."
        ),
    )

    require(
        test_policy[
            "test_open_count"
        ] == 0,
        "Frozen test policy has nonzero open count.",
    )

    require(
        test_policy[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test opening count is not one.",
    )

    primary = test_policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary[
            "cell"
        ] == PRIMARY_CELL,
        "Primary test cell changed.",
    )

    require(
        close_float(
            primary[
                "noise_fraction"
            ],
            PRIMARY_NOISE,
        ),
        "Primary test noise changed.",
    )

    require(
        primary[
            "metric"
        ] == PRIMARY_METRIC,
        "Primary confirmatory metric changed.",
    )

    require(
        primary[
            "primary_model"
        ] == PRIMARY_MODEL,
        "Primary model changed.",
    )

    require(
        primary[
            "bootstrap_replicates"
        ] == 10000,
        "Bootstrap replicate count changed.",
    )

    require(
        primary[
            "bootstrap_seed"
        ] == 73011,
        "Bootstrap seed changed.",
    )

    require(
        primary[
            "statistical_unit"
        ] == "sequence",
        "Primary statistical unit changed.",
    )

    require(
        (
            "upper endpoint"
            in primary[
                "predictive_superiority_supported_if"
            ]
        ),
        "Primary superiority rule changed.",
    )

    print(
        "[7/7] Freezing pre-test authorization readiness"
    )

    normalization = load_json(
        NORMALIZATION_BINDING_PATH
    )

    require(
        normalization[
            "phase4gr3ar2_status"
        ] == "frozen_normalization_artifact_bound",
        "Normalization binding is not frozen.",
    )

    require(
        normalization[
            "double_normalization_prohibited"
        ] is True,
        "Double-normalization prohibition missing.",
    )

    check_zero_test_access(
        source_name="normalization_binding",
        record=normalization,
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in required_paths
    }

    source_hash_path = (
        OUTPUT_DIR
        / "pretest_source_hashes.json"
    )

    write_json(
        source_hash_path,
        source_hashes,
    )

    requirements_from_policy = (
        test_policy[
            "requirements_before_test_generation"
        ]
    )

    require(
        len(
            requirements_from_policy
        ) == 7,
        (
            "Expected seven frozen requirements "
            "before test generation."
        ),
    )

    readiness = {
        "phase":
            (
                "4G-R3E Tier C v4 "
                "pre-test authorization audit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "frozen_test_policy_requirements":
            requirements_from_policy,

        "requirement_evidence":
            evidence,

        "tuning_runs_completed":
            43,

        "selected_configuration_count":
            6,

        "trainable_final_fit_count":
            130,

        "b0_persistence_evaluation_count":
            5,

        "primary_model":
            PRIMARY_MODEL,

        "primary_comparator":
            PRIMARY_COMPARATOR,

        "primary_test_cell":
            PRIMARY_CELL,

        "primary_noise_fraction":
            PRIMARY_NOISE,

        "primary_metric":
            PRIMARY_METRIC,

        "primary_statistical_unit":
            "sequence",

        "bootstrap_replicates":
            10000,

        "bootstrap_seed":
            73011,

        "all_tuning_decisions_frozen":
            True,

        "all_model_configurations_frozen":
            True,

        "all_final_fit_artifacts_frozen":
            True,

        "all_final_fit_hashes_verified":
            True,

        "all_b0_evaluations_frozen":
            True,

        "primary_comparator_frozen":
            True,

        "normalization_binding_frozen":
            True,

        "additional_tuning_authorized":
            False,

        "additional_final_fit_training_authorized":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "maximum_test_open_count":
            1,

        "current_frozen_policy_test_generation_authorized":
            False,

        "current_frozen_policy_test_evaluation_authorized":
            False,

        "pretest_audit_passed":
            True,

        "ready_to_create_one_time_test_authorization":
            True,

        "one_time_test_authorization_created":
            False,

        "test_generation_authorized_by_this_artifact":
            False,

        "test_evaluation_authorized_by_this_artifact":
            False,

        "pretest_source_hashes_path":
            str(
                source_hash_path
            ),

        "pretest_source_hashes_sha256":
            sha256_file(
                source_hash_path
            ),

        "phase4gr3e_status":
            "ready_for_one_time_test_authorization",

        "next_phase":
            (
                "4G-R3F Tier C v4 one-time "
                "test-opening authorization freeze"
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "phase4gr3e_pretest_authorization_audit.json"
    )

    write_json(
        summary_path,
        readiness,
    )

    print(
        "Phase 4G-R3E pre-test authorization "
        "audit passed."
    )

    print(
        json.dumps(
            readiness,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
