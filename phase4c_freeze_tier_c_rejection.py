from __future__ import annotations

import csv
import json
from pathlib import Path


PHASE4A_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

PHASE4B_DIR = Path(
    "outputs/phase4b_tier_c_data"
)

PHASE4C_DIR = Path(
    "outputs/phase4c_tier_c_audit"
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
    phase4a = load_json(
        PHASE4A_DIR
        / "phase4a_summary.json"
    )

    phase4b = load_json(
        PHASE4B_DIR
        / "phase4b_summary.json"
    )

    phase4c = load_json(
        PHASE4C_DIR
        / "phase4c_summary.json"
    )

    physics = load_json(
        PHASE4C_DIR
        / "physics_audit.json"
    )

    manifold = load_json(
        PHASE4C_DIR
        / "manifold_basic_audit.json"
    )

    state_decoder = load_json(
        PHASE4C_DIR
        / "privileged_state_decoder_metrics.json"
    )

    global_statistics = load_json(
        PHASE4C_DIR
        / "global_statistics_classifier_metrics.json"
    )

    affine = load_json(
        PHASE4C_DIR
        / "affine_shortcut_audit.json"
    )

    integrity = load_json(
        PHASE4C_DIR
        / "data_integrity_audit.json"
    )

    gate_rows = load_csv(
        PHASE4C_DIR
        / "acceptance_gate_results.csv"
    )

    if (
        phase4a["phase4a_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A is not frozen."
        )

    if (
        phase4b["phase4b_status"]
        != "completed"
    ):
        raise AssertionError(
            "Phase 4B is incomplete."
        )

    if (
        phase4c["phase4c_status"]
        != "failed_acceptance_gates"
    ):
        raise AssertionError(
            "Phase 4C did not record an acceptance failure."
        )

    if phase4c["tier_c_data_accepted"] is not False:
        raise AssertionError(
            "Tier C v1 was unexpectedly accepted."
        )

    failed_gates = [
        row["gate"]
        for row in gate_rows
        if not parse_bool(row["passed"])
    ]

    physics_checks = {
        "all_values_finite":
            physics["all_values_finite"],

        "energy_nonincrease_fraction_passed":
            (
                physics[
                    "energy_nonincrease_fraction"
                ]
                >= physics[
                    "minimum_required_energy_nonincrease_fraction"
                ]
            ),

        "phase_mean_drift_passed":
            (
                physics[
                    "maximum_phase_mean_drift"
                ]
                <= physics[
                    "maximum_allowed_phase_mean_drift"
                ]
            ),

        "phase_range_passed":
            physics["phase_range_passed"],

        "defect_range_passed":
            physics["defect_range_passed"],
    }

    failed_physics_checks = [
        name
        for name, passed
        in physics_checks.items()
        if passed is not True
    ]

    state_accuracy = {
        "train":
            state_decoder["train_accuracy"],

        "validation":
            state_decoder["validation_accuracy"],

        "test":
            state_decoder["test_accuracy"],

        "required_validation":
            state_decoder[
                "minimum_required_validation_accuracy"
            ],

        "required_test":
            state_decoder[
                "minimum_required_test_accuracy"
            ],

        "chance":
            1.0 / phase4c["state_count"],
    }

    rejection_record = {
        "phase":
            "4C Tier C v1 rejection freeze",

        "tier":
            "Tier C v1 physical-field bridge",

        "source_status": {
            "phase4a":
                phase4a["phase4a_status"],

            "phase4b":
                phase4b["phase4b_status"],

            "phase4c":
                phase4c["phase4c_status"],
        },

        "tier_c_v1_accepted":
            False,

        "tier_c_v1_rejected":
            True,

        "rejection_is_final":
            True,

        "failed_acceptance_gates":
            failed_gates,

        "physics_failure": {
            "gate_passed":
                physics["physics_gate_passed"],

            "checks":
                physics_checks,

            "failed_checks":
                failed_physics_checks,

            "energy_nonincrease_fraction":
                physics[
                    "energy_nonincrease_fraction"
                ],

            "maximum_phase_mean_drift":
                physics[
                    "maximum_phase_mean_drift"
                ],

            "preclip_phase_minimum":
                physics[
                    "preclip_phase_minimum"
                ],

            "preclip_phase_maximum":
                physics[
                    "preclip_phase_maximum"
                ],

            "preclip_defect_minimum":
                physics[
                    "preclip_defect_minimum"
                ],

            "preclip_defect_maximum":
                physics[
                    "preclip_defect_maximum"
                ],
        },

        "state_separability_failure": {
            "gate_passed":
                state_decoder[
                    "state_separability_gate_passed"
                ],

            "accuracy":
                state_accuracy,

            "interpretation":
                (
                    "The frozen diagnostic does not recover "
                    "the eight state regimes reliably across "
                    "carriers. Training accuracy is also below "
                    "the acceptance threshold, so this is not "
                    "only a held-out-carrier failure."
                ),
        },

        "passed_controls": {
            "all_training_channels_nonconstant":
                manifold[
                    "all_training_channels_nonconstant"
                ],

            "every_state_has_carrier_variation":
                manifold[
                    "every_state_has_carrier_variation"
                ],

            "no_exact_duplicate_fields":
                manifold[
                    "no_exact_duplicate_fields"
                ],

            "global_statistics_shortcut_absent":
                global_statistics[
                    "global_statistics_shortcut_absent"
                ],

            "affine_shortcut_removed":
                affine[
                    "affine_shortcut_removed"
                ],

            "data_integrity_gate_passed":
                integrity[
                    "data_integrity_gate_passed"
                ],
        },

        "negative_result_summary":
            (
                "The first physics-informed spatial bridge "
                "successfully removes simple global-statistics "
                "and low-rank affine shortcuts, but its eight "
                "history states are insufficiently identifiable "
                "under the frozen convolutional diagnostic and "
                "one or more numerical field-range requirements "
                "are violated."
            ),

        "phase4d_training_authorized":
            False,

        "prohibited_actions": [
            (
                "Do not lower the frozen acceptance thresholds."
            ),
            (
                "Do not tune or replace the diagnostic decoder "
                "and reinterpret Tier C v1 as accepted."
            ),
            (
                "Do not alter the Tier C v1 simulator and reuse "
                "the opened test-carrier audit as an independent test."
            ),
            (
                "Do not overwrite Phase 4A, 4B, or 4C outputs."
            ),
        ],

        "next_protocol": {
            "phase_id":
                "4A-R",

            "name":
                "Tier C v2 revision protocol freeze",

            "requirements": [
                (
                    "Use fresh carrier, initialization, solver, "
                    "and noise seeds."
                ),
                (
                    "Use train and validation carriers only for "
                    "dataset acceptance and simulator development."
                ),
                (
                    "Keep all new test fields sealed until final "
                    "model checkpoints are frozen."
                ),
                (
                    "Strengthen spatial state signatures through "
                    "physically interpretable morphology, anisotropy, "
                    "defect, and relaxation controls."
                ),
                (
                    "Retain the global-statistics shortcut ceiling "
                    "and low-rank affine shortcut audit."
                ),
            ],
        },

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,

        "rejection_freeze_status":
            "passed",
    }

    output_path = (
        PHASE4C_DIR
        / "phase4c_rejection_record.json"
    )

    write_json(
        output_path,
        rejection_record,
    )

    summary = {
        "phase":
            "4C Tier C v1 rejection freeze",

        "failed_gate_count":
            len(failed_gates),

        "failed_acceptance_gates":
            failed_gates,

        "failed_physics_checks":
            failed_physics_checks,

        "state_decoder_validation_accuracy":
            state_accuracy["validation"],

        "state_decoder_test_accuracy":
            state_accuracy["test"],

        "tier_c_v1_rejected":
            True,

        "phase4d_training_authorized":
            False,

        "next_phase":
            "4A-R Tier C v2 revision protocol freeze",

        "prior_outputs_modified":
            False,

        "phase4c_rejection_freeze_status":
            "passed",
    }

    write_json(
        PHASE4C_DIR
        / "phase4c_rejection_summary.json",
        summary,
    )

    print(
        "Phase 4C Tier C v1 rejection frozen."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
