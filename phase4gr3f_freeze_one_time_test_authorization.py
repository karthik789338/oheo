from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_TEST_CARRIER_COUNT = 32
EXPECTED_FINAL_FIT_COUNT = 130

EXPECTED_TEST_CELLS = [
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
]

EXPECTED_TEST_CARRIER_IDS = [
    5, 6, 8, 10, 11, 12, 14, 15,
    19, 31, 34, 44, 46, 55, 61, 62,
    78, 81, 84, 87, 95, 107, 110, 115,
    123, 125, 137, 139, 143, 145, 146, 154,
]

PHASE4A_ROOT = Path(
    "outputs/"
    "phase4ar3_tier_c_v4_protocol"
)

PHASE4A_EXPECTED_HASHES = {
    "acceptance_gates_v4.json":
        "abfa88d7be9866a4324a01a02fbf69775d4e97a0167c881e536e4a1084fa9a11",

    "carrier_parameter_schema_v4.csv":
        "e842d6bf243e04b28c39ca26cb921af27338f198ca959bb13245f0dcb20e9c87",

    "carrier_split_v4.csv":
        "cf86ba2356a5c7057acb027224bcbaf219e7f410661c27418fb54d683f19251b",

    "coarsening_protocol_v4.json":
        "02e434e58c0ff82730f9f68f7bbe6fa300db6b9538f5eb370d0b82146f5b406b",

    "hypothesis_registry_v4.json":
        "4508d42eb06983cbe00282d8ce036d458b75bbb1934a6eadd13414103f78e201",

    "input_hashes.json":
        "48233859c07a82d48fcc01d4a35a926112d45b518572a88ce79be9ed839fb3c8",

    "model_family_registry_v4.csv":
        "f4c4e817b85ab015c66a27e8a6deacb76387f9b036ea5fc739be009d2552e3f9",

    "phase4ar3_summary.json":
        "e24617a71d952a0383aa1141183b06127dcb347fc97d7160fc57d67c4c029228",

    "phase4_v4_plan.json":
        "a48eba095d1ba6edc58f283c4a6469d7004eae7b15b6dbe94ba30774a4116a5d",

    "state_morphology_regimes_v4.csv":
        "f9fcca6b68a03f937977f66d12402f60b16a02a719193b0f5ef94cdd0e180880",

    "terminal_revision_policy.json":
        "42e8e0d04ce8708b7dc19d78c0f17b1ec5ce4c684ff59c0e7faa6b585cb456e1",

    "test_sealing_policy.json":
        "f9db182e4c27d3e9506ecb2edd6b73ecb487560c324c427a58554be140f05389",

    "tier_c_v4_protocol.json":
        "b6173a9e0f5f012f2f085d1ec126b7092f09ece168d0a4b123b8ddf1011d53fc",

    "training_protocol_v4.json":
        "4f7819dccf192f611c0c9b820ed799a8d75b07b7ddb6768c534b6110c442aa13",
}

PRETEST_AUDIT_PATH = Path(
    "outputs/"
    "phase4gr3e_tier_c_v4_pretest_authorization_audit/"
    "phase4gr3e_pretest_authorization_audit.json"
)

FROZEN_FINAL_FIT_REGISTRY_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

FINAL_FIT_FREEZE_PATH = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "phase4gr3c_final_fit_freeze_summary.json"
)

COMPARATOR_PATH = Path(
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

TEST_OPENING_POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

METRIC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_metric_registry.csv"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3f_tier_c_v4_test_authorization"
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


def parse_false(value: str) -> bool:
    return (
        str(value)
        .strip()
        .lower()
        in {
            "false",
            "0",
            "no",
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/8] Verifying immutable Phase 4A-R3 protocol"
    )

    phase4a_hashes = {}

    for filename, expected_sha in (
        PHASE4A_EXPECTED_HASHES.items()
    ):
        path = (
            PHASE4A_ROOT
            / filename
        )

        if not path.exists():
            raise FileNotFoundError(path)

        observed_sha = sha256_file(
            path
        )

        require(
            observed_sha == expected_sha,
            (
                f"Phase 4A-R3 artifact changed: "
                f"{path}"
            ),
        )

        phase4a_hashes[
            str(path)
        ] = observed_sha

    phase4a_summary = load_json(
        PHASE4A_ROOT
        / "phase4ar3_summary.json"
    )

    tier_c_protocol = load_json(
        PHASE4A_ROOT
        / "tier_c_v4_protocol.json"
    )

    test_sealing = load_json(
        PHASE4A_ROOT
        / "test_sealing_policy.json"
    )

    training_protocol = load_json(
        PHASE4A_ROOT
        / "training_protocol_v4.json"
    )

    require(
        phase4a_summary[
            "protocol_version"
        ] == PROTOCOL_VERSION,
        "Phase 4A-R3 protocol version changed.",
    )

    require(
        phase4a_summary[
            "test_carrier_source"
        ] == "unopened Tier C v3 test IDs",
        "Frozen test carrier source changed.",
    )

    require(
        phase4a_summary[
            "test_carrier_parameters_generated"
        ] is False,
        (
            "Phase 4A-R3 says test carrier parameters "
            "were already generated."
        ),
    )

    require(
        phase4a_summary[
            "test_fields_generated"
        ] is False,
        (
            "Phase 4A-R3 says test fields "
            "were already generated."
        ),
    )

    require(
        phase4a_summary[
            "test_manifests_generated"
        ] is False,
        (
            "Phase 4A-R3 says test manifests "
            "were already generated."
        ),
    )

    require(
        phase4a_summary[
            "test_metrics_computed"
        ] is False,
        (
            "Phase 4A-R3 says test metrics "
            "were already computed."
        ),
    )

    require(
        phase4a_summary[
            "tier_c_v4_is_final_revision"
        ] is True,
        "Tier C v4 is not marked final.",
    )

    require(
        phase4a_summary[
            "tier_c_v5_authorized"
        ] is False,
        "Tier C v5 is unexpectedly authorized.",
    )

    require(
        int(
            phase4a_summary[
                "validation_test_overlap_count"
            ]
        ) == 0,
        "Validation/test carrier overlap is nonzero.",
    )

    print(
        "[2/8] Freezing unopened test carrier IDs"
    )

    carrier_rows = load_csv(
        PHASE4A_ROOT
        / "carrier_split_v4.csv"
    )

    test_carrier_rows = [
        row
        for row in carrier_rows
        if row[
            "split"
        ] == "test"
    ]

    test_carrier_rows.sort(
        key=lambda row: int(
            row[
                "split_position"
            ]
        )
    )

    require(
        len(
            test_carrier_rows
        )
        == EXPECTED_TEST_CARRIER_COUNT,
        "Expected 32 frozen test carriers.",
    )

    require(
        [
            int(
                row[
                    "split_position"
                ]
            )
            for row
            in test_carrier_rows
        ]
        == list(
            range(
                EXPECTED_TEST_CARRIER_COUNT
            )
        ),
        "Test split positions are not 0..31.",
    )

    test_carrier_ids = [
        int(
            row[
                "carrier_id"
            ]
        )
        for row
        in test_carrier_rows
    ]

    require(
        test_carrier_ids
        == EXPECTED_TEST_CARRIER_IDS,
        (
            "Frozen Tier C v4 test carrier IDs "
            "changed."
        ),
    )

    require(
        len(
            set(
                test_carrier_ids
            )
        )
        == EXPECTED_TEST_CARRIER_COUNT,
        "Duplicate test carrier IDs found.",
    )

    require(
        all(
            row[
                "test_source"
            ]
            == "unopened_tier_c_v3_test"
            for row
            in test_carrier_rows
        ),
        "Test carrier source labels changed.",
    )

    require(
        all(
            parse_false(
                row[
                    "fresh_v4_parameters_generated"
                ]
            )
            for row
            in test_carrier_rows
        ),
        (
            "A test carrier row says fresh "
            "parameters were already generated."
        ),
    )

    require(
        all(
            parse_false(
                row[
                    "fresh_v4_fields_generated"
                ]
            )
            for row
            in test_carrier_rows
        ),
        (
            "A test carrier row says fresh "
            "fields were already generated."
        ),
    )

    print(
        "[3/8] Verifying test sealing policy"
    )

    require(
        test_sealing[
            "test_carrier_ids_source"
        ]
        == "unopened Tier C v3 test split",
        "Test carrier ID source changed.",
    )

    require(
        test_sealing[
            "sealed_test_cells"
        ] == EXPECTED_TEST_CELLS,
        "Sealed test cells changed.",
    )

    require(
        test_sealing[
            "test_carrier_parameters_generated_before_checkpoint_freeze"
        ] is False,
        (
            "Test carrier parameters were generated "
            "before checkpoint freeze."
        ),
    )

    require(
        test_sealing[
            "test_clean_fields_generated_before_checkpoint_freeze"
        ] is False,
        (
            "Test clean fields were generated "
            "before checkpoint freeze."
        ),
    )

    require(
        test_sealing[
            "test_noisy_fields_generated_before_checkpoint_freeze"
        ] is False,
        (
            "Test noisy fields were generated "
            "before checkpoint freeze."
        ),
    )

    require(
        test_sealing[
            "test_trajectory_manifests_generated_before_checkpoint_freeze"
        ] is False,
        (
            "Test manifests were generated "
            "before checkpoint freeze."
        ),
    )

    require(
        test_sealing[
            "test_metrics_computed_before_checkpoint_freeze"
        ] is False,
        (
            "Test metrics were computed "
            "before checkpoint freeze."
        ),
    )

    require(
        test_sealing[
            "test_fields_used_for_dataset_acceptance"
        ] is False,
        "Test fields affected dataset acceptance.",
    )

    require(
        test_sealing[
            "test_fields_used_for_model_selection"
        ] is False,
        "Test fields affected model selection.",
    )

    require(
        test_sealing[
            "test_generation_authorization_condition"
        ]
        == (
            "All 130 final Tier C v4 model "
            "checkpoints must be complete and frozen."
        ),
        "Test-generation authorization condition changed.",
    )

    require(
        test_sealing[
            "test_generation_phase"
        ] == "4G-R3",
        "Frozen test-generation phase changed.",
    )

    require(
        training_protocol[
            "test_cells"
        ] == EXPECTED_TEST_CELLS,
        "Training protocol test cells changed.",
    )

    require(
        training_protocol[
            "test_cells_generated_during_tuning"
        ] is False,
        "Test cells were generated during tuning.",
    )

    require(
        training_protocol[
            "test_cells_generated_during_final_fitting"
        ] is False,
        "Test cells were generated during final fitting.",
    )

    require(
        training_protocol[
            "final_checkpoint_freeze_required_before_test_generation"
        ] is True,
        "Checkpoint freeze is no longer required.",
    )

    print(
        "[4/8] Verifying pre-test audit"
    )

    pretest = load_json(
        PRETEST_AUDIT_PATH
    )

    require(
        pretest[
            "phase4gr3e_status"
        ]
        == "ready_for_one_time_test_authorization",
        "Pre-test audit is not ready.",
    )

    require(
        pretest[
            "pretest_audit_passed"
        ] is True,
        "Pre-test audit did not pass.",
    )

    require(
        pretest[
            "ready_to_create_one_time_test_authorization"
        ] is True,
        "One-time authorization is not permitted.",
    )

    require(
        pretest[
            "one_time_test_authorization_created"
        ] is False,
        (
            "Pre-test state says an authorization "
            "already exists."
        ),
    )

    require(
        pretest[
            "test_open_count"
        ] == 0,
        "Test has already been opened.",
    )

    require(
        pretest[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test-open count changed.",
    )

    print(
        "[5/8] Re-verifying all 130 frozen checkpoints"
    )

    final_freeze = load_json(
        FINAL_FIT_FREEZE_PATH
    )

    require(
        final_freeze[
            "phase4gr3c_status"
        ] == "final_fit_artifacts_frozen",
        "Final-fit artifacts are not frozen.",
    )

    frozen_rows = load_csv(
        FROZEN_FINAL_FIT_REGISTRY_PATH
    )

    require(
        len(
            frozen_rows
        ) == EXPECTED_FINAL_FIT_COUNT,
        "Expected 130 frozen final fits.",
    )

    authorized_checkpoint_rows = []

    for row in frozen_rows:
        checkpoint_path = Path(
            row[
                "best_checkpoint_path"
            ]
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        observed_sha = sha256_file(
            checkpoint_path
        )

        require(
            observed_sha
            == row[
                "best_checkpoint_sha256"
            ],
            (
                "Frozen checkpoint hash changed: "
                f"{row['run_id']}"
            ),
        )

        require(
            int(
                row[
                    "test_open_count"
                ]
            ) == 0,
            (
                "A frozen final-fit row records "
                "test access."
            ),
        )

        authorized_checkpoint_rows.append(
            {
                "run_id":
                    row[
                        "run_id"
                    ],

                "model_id":
                    row[
                        "model_id"
                    ],

                "noise_fraction":
                    row[
                        "noise_fraction"
                    ],

                "effective_seed":
                    row[
                        "effective_seed"
                    ],

                "configuration_id":
                    row[
                        "configuration_id"
                    ],

                "best_checkpoint_path":
                    str(
                        checkpoint_path
                    ),

                "best_checkpoint_sha256":
                    observed_sha,
            }
        )

    checkpoint_manifest_path = (
        OUTPUT_DIR
        / "authorized_final_checkpoint_manifest.csv"
    )

    write_csv(
        checkpoint_manifest_path,
        authorized_checkpoint_rows,
    )

    print(
        "[6/8] Binding comparator, B0, normalization, and metrics"
    )

    comparator = load_json(
        COMPARATOR_PATH
    )

    require(
        comparator[
            "primary_comparator_frozen"
        ] is True,
        "Primary comparator is not frozen.",
    )

    require(
        comparator[
            "selected_comparator_model_id"
        ] == "B4",
        "Primary comparator is not B4.",
    )

    require(
        comparator[
            "test_artifact_used_for_selection"
        ] is False,
        "Test artifact influenced comparator selection.",
    )

    b0_summary = load_json(
        B0_SUMMARY_PATH
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
        ] == 5,
        "Expected five B0 evaluations.",
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

    metric_rows = load_csv(
        METRIC_REGISTRY_PATH
    )

    require(
        any(
            row[
                "metric_id"
            ]
            == "noisy_target_rollout_mse"
            for row in metric_rows
        ),
        "Primary predictive metric missing.",
    )

    print(
        "[7/8] Binding frozen test-opening policy"
    )

    test_policy = load_json(
        TEST_OPENING_POLICY_PATH
    )

    require(
        test_policy[
            "sealed_test_cells"
        ] == EXPECTED_TEST_CELLS,
        "Predictive test cells changed.",
    )

    require(
        test_policy[
            "test_carrier_count"
        ] == EXPECTED_TEST_CARRIER_COUNT,
        "Predictive test carrier count changed.",
    )

    require(
        test_policy[
            "test_open_count"
        ] == 0,
        "Test has already been opened.",
    )

    require(
        test_policy[
            "maximum_test_open_count"
        ] == 1,
        "Maximum test-open count changed.",
    )

    require(
        test_policy[
            "test_generation_authorized_now"
        ] is False,
        (
            "Original frozen policy was modified "
            "to authorize test generation."
        ),
    )

    require(
        test_policy[
            "test_evaluation_authorized_now"
        ] is False,
        (
            "Original frozen policy was modified "
            "to authorize test evaluation."
        ),
    )

    primary = test_policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary[
            "cell"
        ] == "test_joint",
        "Primary test cell changed.",
    )

    require(
        float(
            primary[
                "noise_fraction"
            ]
        ) == 0.25,
        "Primary test noise changed.",
    )

    require(
        primary[
            "metric"
        ] == "noisy_target_rollout_mse",
        "Primary metric changed.",
    )

    require(
        primary[
            "primary_model"
        ] == "OCM",
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
        "Statistical unit changed.",
    )

    print(
        "[8/8] Creating one-time authorization artifact"
    )

    frozen_generation_seeds = {
        "sealed_test_manifest_seed":
            phase4a_summary[
                "sealed_test_manifest_seed"
            ],

        "fresh_carrier_parameter_seed":
            phase4a_summary[
                "fresh_carrier_parameter_seed"
            ],

        "fresh_field_initialization_seed":
            phase4a_summary[
                "fresh_field_initialization_seed"
            ],

        "fresh_noise_seed":
            phase4a_summary[
                "fresh_noise_seed"
            ],

        "fresh_solver_seed":
            phase4a_summary[
                "fresh_solver_seed"
            ],
    }

    require(
        all(
            value is not None
            for value
            in frozen_generation_seeds.values()
        ),
        "One or more frozen generation seeds are missing.",
    )

    source_hashes = {
        str(
            PRETEST_AUDIT_PATH
        ):
            sha256_file(
                PRETEST_AUDIT_PATH
            ),

        str(
            FINAL_FIT_FREEZE_PATH
        ):
            sha256_file(
                FINAL_FIT_FREEZE_PATH
            ),

        str(
            FROZEN_FINAL_FIT_REGISTRY_PATH
        ):
            sha256_file(
                FROZEN_FINAL_FIT_REGISTRY_PATH
            ),

        str(
            COMPARATOR_PATH
        ):
            sha256_file(
                COMPARATOR_PATH
            ),

        str(
            B0_SUMMARY_PATH
        ):
            sha256_file(
                B0_SUMMARY_PATH
            ),

        str(
            B0_REGISTRY_PATH
        ):
            sha256_file(
                B0_REGISTRY_PATH
            ),

        str(
            NORMALIZATION_BINDING_PATH
        ):
            sha256_file(
                NORMALIZATION_BINDING_PATH
            ),

        str(
            TEST_OPENING_POLICY_PATH
        ):
            sha256_file(
                TEST_OPENING_POLICY_PATH
            ),

        str(
            METRIC_REGISTRY_PATH
        ):
            sha256_file(
                METRIC_REGISTRY_PATH
            ),

        str(
            checkpoint_manifest_path
        ):
            sha256_file(
                checkpoint_manifest_path
            ),
    }

    source_hashes.update(
        phase4a_hashes
    )

    source_hashes_path = (
        OUTPUT_DIR
        / "one_time_authorization_source_hashes.json"
    )

    write_json(
        source_hashes_path,
        source_hashes,
    )

    authorization = {
        "phase":
            (
                "4G-R3F Tier C v4 one-time "
                "test-opening authorization freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "authorization_type":
            "single_sealed_test_opening",

        "authorization_created":
            True,

        "authorization_scope_frozen":
            True,

        "test_open_count_before_authorized_opening":
            0,

        "maximum_test_open_count":
            1,

        "authorized_open_transition":
            "0 -> 1 exactly once",

        "second_test_opening_authorized":
            False,

        "test_generation_authorized":
            True,

        "test_evaluation_authorized":
            True,

        "test_evaluation_authorized_only_under_same_single_opening":
            True,

        "test_artifacts_generated_by_this_phase":
            False,

        "test_data_read_by_this_phase":
            False,

        "test_metrics_computed_by_this_phase":
            False,

        "test_open_count_after_this_phase":
            0,

        "sealed_test_cells":
            EXPECTED_TEST_CELLS,

        "test_carrier_count":
            EXPECTED_TEST_CARRIER_COUNT,

        "test_carrier_ids":
            test_carrier_ids,

        "test_carrier_ids_source":
            "unopened Tier C v3 test split",

        "frozen_generation_seeds":
            frozen_generation_seeds,

        "phase4ar3_fresh_seeds":
            tier_c_protocol[
                "fresh_seeds"
            ],

        "test_seed_source_policy":
            test_policy[
                "test_seed_source"
            ],

        "primary_confirmatory_condition":
            primary,

        "secondary_conditions":
            test_policy[
                "secondary_conditions"
            ],

        "primary_model":
            "OCM",

        "primary_comparator":
            "B4",

        "primary_comparator_selected_before_test_generation":
            True,

        "authorized_trainable_checkpoint_count":
            len(
                authorized_checkpoint_rows
            ),

        "authorized_checkpoint_manifest_path":
            str(
                checkpoint_manifest_path
            ),

        "authorized_checkpoint_manifest_sha256":
            sha256_file(
                checkpoint_manifest_path
            ),

        "b0_persistence_evaluations_frozen":
            True,

        "normalization_binding_frozen":
            True,

        "metric_registry_frozen":
            True,

        "additional_tuning_authorized":
            False,

        "additional_final_fit_training_authorized":
            False,

        "checkpoint_reselection_after_test_opening_authorized":
            False,

        "comparator_reselection_after_test_opening_authorized":
            False,

        "hyperparameter_changes_after_test_opening_authorized":
            False,

        "new_model_family_after_test_opening_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "one_time_test_authorization_frozen":
            True,

        "authorization_source_hashes_path":
            str(
                source_hashes_path
            ),

        "authorization_source_hashes_sha256":
            sha256_file(
                source_hashes_path
            ),

        "phase4gr3f_status":
            "one_time_test_opening_authorized",

        "next_phase":
            (
                "4G-R3G Tier C v4 one-time "
                "sealed-test generation and evaluation"
            ),
    }

    authorization_path = (
        OUTPUT_DIR
        / "one_time_test_opening_authorization.json"
    )

    write_json(
        authorization_path,
        authorization,
    )

    summary = {
        "phase":
            (
                "4G-R3F Tier C v4 one-time "
                "test authorization freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "test_carrier_count":
            EXPECTED_TEST_CARRIER_COUNT,

        "test_carrier_ids":
            test_carrier_ids,

        "sealed_test_cells":
            EXPECTED_TEST_CELLS,

        "authorized_trainable_checkpoint_count":
            len(
                authorized_checkpoint_rows
            ),

        "primary_model":
            "OCM",

        "primary_comparator":
            "B4",

        "primary_cell":
            "test_joint",

        "primary_noise_fraction":
            0.25,

        "primary_metric":
            "noisy_target_rollout_mse",

        "bootstrap_replicates":
            10000,

        "bootstrap_seed":
            73011,

        "test_open_count":
            0,

        "maximum_test_open_count":
            1,

        "test_generation_authorized":
            True,

        "test_evaluation_authorized":
            True,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "one_time_test_authorization_created":
            True,

        "one_time_test_authorization_frozen":
            True,

        "authorization_path":
            str(
                authorization_path
            ),

        "authorization_sha256":
            sha256_file(
                authorization_path
            ),

        "phase4gr3f_status":
            "one_time_test_opening_authorized",
    }

    write_json(
        OUTPUT_DIR
        / "phase4gr3f_test_authorization_summary.json",
        summary,
    )

    print(
        "Phase 4G-R3F one-time test-opening "
        "authorization frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
