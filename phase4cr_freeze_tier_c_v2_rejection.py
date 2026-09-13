from __future__ import annotations

import csv
import json
from pathlib import Path


PHASE4AR_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

PHASE4BR_DIR = Path(
    "outputs/phase4br_tier_c_v2_development_data"
)

PHASE4CR_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

PHASE4C_V1_DIR = Path(
    "outputs/phase4c_tier_c_audit"
)


PROTOCOL_VERSION = "tier_c_v2"


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
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


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def main():
    phase4ar = load_json(
        PHASE4AR_DIR
        / "phase4ar_summary.json"
    )

    phase4br_fields = load_json(
        PHASE4BR_DIR
        / "phase4br_field_bank_summary.json"
    )

    phase4br_data = load_json(
        PHASE4BR_DIR
        / "phase4br_summary.json"
    )

    phase4cr = load_json(
        PHASE4CR_DIR
        / "phase4cr_summary.json"
    )

    physics = load_json(
        PHASE4CR_DIR
        / "physics_audit.json"
    )

    manifold = load_json(
        PHASE4CR_DIR
        / "manifold_basic_audit.json"
    )

    decoder = load_json(
        PHASE4CR_DIR
        / "state_decoder_summary.json"
    )

    global_statistics = load_json(
        PHASE4CR_DIR
        / "global_statistics_classifier_metrics.json"
    )

    affine = load_json(
        PHASE4CR_DIR
        / "affine_shortcut_audit.json"
    )

    integrity = load_json(
        PHASE4CR_DIR
        / "data_integrity_audit.json"
    )

    gate_rows = load_csv(
        PHASE4CR_DIR
        / "acceptance_gate_results.csv"
    )

    v1_rejection = load_json(
        PHASE4C_V1_DIR
        / "phase4c_rejection_summary.json"
    )

    if (
        phase4ar["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R is not frozen."
        )

    if (
        phase4ar["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v2 protocol version changed."
        )

    if (
        phase4br_fields[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v2 development field bank "
            "is incomplete."
        )

    if (
        phase4br_data["phase4br_status"]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v2 development dataset "
            "is incomplete."
        )

    if (
        phase4cr["phase4cr_status"]
        != "failed_acceptance_gates"
    ):
        raise AssertionError(
            "Phase 4C-R did not record an "
            "acceptance failure."
        )

    if (
        phase4cr[
            "tier_c_v2_development_data_accepted"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v2 was unexpectedly accepted."
        )

    if (
        phase4cr["test_fields_generated"]
        is not False
        or phase4cr["test_manifests_generated"]
        is not False
        or phase4cr["test_fields_read"]
        is not False
        or phase4cr["test_metrics_computed"]
        is not False
    ):
        raise AssertionError(
            "Tier C v2 sealed-test policy "
            "was violated."
        )

    if (
        v1_rejection[
            "phase4c_rejection_freeze_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Tier C v1 rejection is not frozen."
        )

    failed_gates = [
        row["gate"]
        for row in gate_rows
        if not parse_bool(
            row["passed"]
        )
    ]

    expected_failed_gates = {
        "three_seed_state_separability",
    }

    if set(failed_gates) != expected_failed_gates:
        raise AssertionError(
            "Unexpected Tier C v2 failed gates: "
            f"{failed_gates}"
        )

    v1_validation_accuracy = float(
        v1_rejection[
            "state_decoder_validation_accuracy"
        ]
    )

    v2_validation_accuracy = float(
        decoder[
            "mean_validation_accuracy"
        ]
    )

    validation_improvement = (
        v2_validation_accuracy
        - v1_validation_accuracy
    )

    record = {
        "phase":
            "4C-R Tier C v2 rejection freeze",

        "protocol_version":
            PROTOCOL_VERSION,

        "tier":
            "Tier C v2 physical-field bridge",

        "source_status": {
            "phase4ar":
                phase4ar["phase4ar_status"],

            "phase4br_field_bank":
                phase4br_fields[
                    "field_bank_generation_status"
                ],

            "phase4br_dataset":
                phase4br_data["phase4br_status"],

            "phase4cr":
                phase4cr["phase4cr_status"],
        },

        "tier_c_v2_accepted":
            False,

        "tier_c_v2_rejected":
            True,

        "rejection_is_final":
            True,

        "failed_acceptance_gates":
            failed_gates,

        "state_separability_failure": {
            "diagnostic_model":
                decoder["diagnostic_model"],

            "diagnostic_seeds":
                decoder["diagnostic_seeds"],

            "mean_training_accuracy":
                decoder[
                    "mean_training_accuracy"
                ],

            "mean_validation_accuracy":
                decoder[
                    "mean_validation_accuracy"
                ],

            "validation_accuracy_standard_deviation":
                decoder[
                    "validation_accuracy_standard_deviation"
                ],

            "worst_seed_validation_accuracy":
                decoder[
                    "worst_seed_validation_accuracy"
                ],

            "best_seed_validation_accuracy":
                decoder[
                    "best_seed_validation_accuracy"
                ],

            "required_mean_validation_accuracy":
                decoder[
                    "minimum_required_mean_validation_accuracy"
                ],

            "required_worst_seed_validation_accuracy":
                decoder[
                    "minimum_required_worst_seed_validation_accuracy"
                ],

            "gate_passed":
                decoder[
                    "state_separability_gate_passed"
                ],

            "interpretation":
                (
                    "The revised spatial morphology contains "
                    "substantial state information, but the frozen "
                    "three-seed convolutional diagnostic does not "
                    "recover the eight states at the required "
                    "validation accuracy."
                ),
        },

        "comparison_with_tier_c_v1": {
            "v1_validation_state_accuracy":
                v1_validation_accuracy,

            "v2_mean_validation_state_accuracy":
                v2_validation_accuracy,

            "absolute_validation_accuracy_improvement":
                validation_improvement,

            "percentage_point_improvement":
                100.0
                * validation_improvement,

            "interpretation":
                (
                    "Tier C v2 improves state separability "
                    "substantially over Tier C v1 but still does "
                    "not satisfy the frozen acceptance threshold."
                ),
        },

        "passed_controls": {
            "numerical_physics":
                physics[
                    "physics_gate_passed"
                ],

            "energy_nonincrease_fraction":
                physics[
                    "energy_nonincrease_fraction"
                ],

            "phase_range":
                physics[
                    "phase_range_passed"
                ],

            "defect_range":
                physics[
                    "defect_range_passed"
                ],

            "basic_manifold_quality":
                manifold[
                    "basic_manifold_gate_passed"
                ],

            "exact_duplicate_field_count":
                manifold[
                    "exact_duplicate_field_count"
                ],

            "global_statistics_shortcut_absent":
                global_statistics[
                    "global_statistics_shortcut_absent"
                ],

            "global_statistics_validation_accuracy":
                global_statistics[
                    "validation_accuracy"
                ],

            "affine_shortcut_removed":
                affine[
                    "affine_shortcut_removed"
                ],

            "affine_validation_continuous_mse":
                affine[
                    "validation_primitive_continuous_mse"
                ],

            "affine_validation_exact_primitive_operation_rate":
                affine[
                    "validation_exact_primitive_operation_rate"
                ],

            "data_integrity":
                integrity[
                    "data_integrity_gate_passed"
                ],

            "sealed_test_files_absent":
                integrity[
                    "sealed_test_files_absent"
                ],
        },

        "negative_result_summary":
            (
                "Tier C v2 resolves the Tier C v1 numerical-range "
                "failure, preserves sealed test data, removes the "
                "frozen global-statistics and PCA-affine shortcuts, "
                "and improves validation state decoding. However, "
                "mean and worst-seed validation state accuracy remain "
                "below the frozen identifiability thresholds."
            ),

        "phase4dr_predictive_training_authorized":
            False,

        "allowed_next_action": {
            "phase_id":
                "4C-RD",

            "name":
                "Tier C v2 development-only failure diagnosis",

            "allowed_operations": [
                (
                    "Evaluate the already-fitted diagnostic "
                    "checkpoints on training and validation fields."
                ),
                (
                    "Compute state confusion matrices and "
                    "factor-wise accuracies."
                ),
                (
                    "Measure which coarsening, anisotropy, or "
                    "defect-recovery factor is insufficiently "
                    "separable."
                ),
                (
                    "Use only existing training and validation "
                    "development fields."
                ),
            ],

            "new_model_fitting_allowed":
                False,

            "test_generation_allowed":
                False,

            "test_field_access_allowed":
                False,
        },

        "prohibited_actions": [
            (
                "Do not lower the frozen state-separability "
                "thresholds."
            ),
            (
                "Do not replace or tune the diagnostic architecture "
                "and reinterpret Tier C v2 as accepted."
            ),
            (
                "Do not begin predictive model training."
            ),
            (
                "Do not generate test-carrier parameters, fields, "
                "or manifests."
            ),
            (
                "Do not overwrite Tier C v1 or Tier C v2 outputs."
            ),
        ],

        "sealed_test_status": {
            "test_carrier_parameters_generated":
                False,

            "test_fields_generated":
                False,

            "test_manifests_generated":
                False,

            "test_fields_read":
                False,

            "test_metrics_computed":
                False,

            "test_set_remains_eligible_for_future_use":
                True,
        },

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,

        "rejection_freeze_status":
            "passed",
    }

    write_json(
        PHASE4CR_DIR
        / "phase4cr_rejection_record.json",
        record,
    )

    summary = {
        "phase":
            "4C-R Tier C v2 rejection freeze",

        "protocol_version":
            PROTOCOL_VERSION,

        "failed_gate_count":
            len(failed_gates),

        "failed_acceptance_gates":
            failed_gates,

        "mean_validation_state_accuracy":
            decoder[
                "mean_validation_accuracy"
            ],

        "worst_seed_validation_state_accuracy":
            decoder[
                "worst_seed_validation_accuracy"
            ],

        "required_mean_validation_state_accuracy":
            decoder[
                "minimum_required_mean_validation_accuracy"
            ],

        "required_worst_seed_validation_state_accuracy":
            decoder[
                "minimum_required_worst_seed_validation_accuracy"
            ],

        "validation_accuracy_improvement_over_v1":
            validation_improvement,

        "tier_c_v2_rejected":
            True,

        "phase4dr_predictive_training_authorized":
            False,

        "sealed_test_set_remains_unopened":
            True,

        "next_phase":
            (
                "4C-RD Tier C v2 development-only "
                "failure diagnosis"
            ),

        "prior_outputs_modified":
            False,

        "phase4cr_rejection_freeze_status":
            "passed",
    }

    write_json(
        PHASE4CR_DIR
        / "phase4cr_rejection_summary.json",
        summary,
    )

    print(
        "Phase 4C-R Tier C v2 rejection frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
