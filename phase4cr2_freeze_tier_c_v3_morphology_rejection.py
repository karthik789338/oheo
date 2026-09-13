from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PHASE4C_V1_DIR = Path(
    "outputs/phase4c_tier_c_audit"
)

PHASE4CR_V2_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

PHASE4CRD_V2_DIR = Path(
    "outputs/phase4crd_tier_c_v2_failure_diagnosis"
)

PHASE4AR2_DIR = Path(
    "outputs/phase4ar2_tier_c_v3_protocol"
)

PHASE4BR2_DIR = Path(
    "outputs/phase4br2_tier_c_v3_development_data"
)

OUTPUT_DIR = Path(
    "outputs/phase4cr2_tier_c_v3_predecoder_audit"
)


PROTOCOL_VERSION = "tier_c_v3"

MINIMUM_ENERGY_NONINCREASE_FRACTION = 0.99

MAXIMUM_PHASE_MEAN_DRIFT = 1e-5
MAXIMUM_DEFECT_MEAN_DRIFT = 1e-5

PHASE_MINIMUM = -1.000001
PHASE_MAXIMUM = 1.000001

DEFECT_MINIMUM = -0.000001
DEFECT_MAXIMUM = 1.000001

MINIMUM_MEAN_LENGTH_RATIO = 2.50
MINIMUM_WORST_LENGTH_RATIO = 1.80
MAXIMUM_MEAN_INTERFACE_RATIO = 0.70

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_DIFFERENCE = 1e-7


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value,
):
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
    rows,
):
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
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path):
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


def validate_prior_results():
    v1_rejection = load_json(
        PHASE4C_V1_DIR
        / "phase4c_rejection_summary.json"
    )

    v2_rejection = load_json(
        PHASE4CR_V2_DIR
        / "phase4cr_rejection_summary.json"
    )

    v2_rejection_record = load_json(
        PHASE4CR_V2_DIR
        / "phase4cr_rejection_record.json"
    )

    v2_diagnosis = load_json(
        PHASE4CRD_V2_DIR
        / "phase4crd_summary.json"
    )

    v3_protocol = load_json(
        PHASE4AR2_DIR
        / "phase4ar2_summary.json"
    )

    field_summary = load_json(
        PHASE4BR2_DIR
        / "phase4br2_field_bank_summary.json"
    )

    morphology = load_json(
        PHASE4BR2_DIR
        / "coarsening_morphology_summary.json"
    )

    dataset_summary = load_json(
        PHASE4BR2_DIR
        / "phase4br2_summary.json"
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

    v2_summary_status = v2_rejection.get(
        "phase4cr_rejection_freeze_status"
    )

    v2_record_status = v2_rejection_record.get(
        "rejection_freeze_status"
    )

    if (
        v2_summary_status != "passed"
        and v2_record_status != "passed"
    ):
        raise AssertionError(
            "Tier C v2 rejection is not frozen."
        )

    if (
        v2_rejection["tier_c_v2_rejected"]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection changed."
        )

    if (
        v2_diagnosis["phase4crd_status"]
        != "passed"
    ):
        raise AssertionError(
            "Tier C v2 diagnosis has not passed."
        )

    if (
        v2_diagnosis[
            "weakest_validation_factor"
        ]
        != "coarsening_level"
    ):
        raise AssertionError(
            "Tier C v2 diagnosis no longer identifies "
            "coarsening as the weak factor."
        )

    if (
        v3_protocol["phase4ar2_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v3 protocol is not frozen."
        )

    if (
        v3_protocol["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v3 protocol version changed."
        )

    if (
        v3_protocol[
            "targeted_revision_factor"
        ]
        != "coarsening_level"
    ):
        raise AssertionError(
            "Tier C v3 is not a coarsening-only revision."
        )

    if (
        field_summary[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v3 development field generation "
            "is incomplete."
        )

    if (
        dataset_summary["phase4br2_status"]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v3 development dataset "
            "materialization is incomplete."
        )

    return {
        "v1_rejection":
            v1_rejection,

        "v2_rejection":
            v2_rejection,

        "v2_rejection_record":
            v2_rejection_record,

        "v2_diagnosis":
            v2_diagnosis,

        "v3_protocol":
            v3_protocol,

        "field_summary":
            field_summary,

        "morphology":
            morphology,

        "dataset_summary":
            dataset_summary,
    }


def audit_physics(field_summary):
    checks = {
        "all_values_finite":
            field_summary[
                "all_values_finite"
            ] is True,

        "energy_nonincrease_fraction":
            (
                float(
                    field_summary[
                        "energy_nonincrease_fraction"
                    ]
                )
                >= MINIMUM_ENERGY_NONINCREASE_FRACTION
            ),

        "phase_mean_drift":
            (
                float(
                    field_summary[
                        "maximum_phase_mean_drift"
                    ]
                )
                <= MAXIMUM_PHASE_MEAN_DRIFT
            ),

        "defect_mean_drift":
            (
                float(
                    field_summary[
                        "maximum_defect_mean_drift"
                    ]
                )
                <= MAXIMUM_DEFECT_MEAN_DRIFT
            ),

        "phase_minimum":
            (
                float(
                    field_summary["phase_minimum"]
                )
                >= PHASE_MINIMUM
            ),

        "phase_maximum":
            (
                float(
                    field_summary["phase_maximum"]
                )
                <= PHASE_MAXIMUM
            ),

        "defect_minimum":
            (
                float(
                    field_summary["defect_minimum"]
                )
                >= DEFECT_MINIMUM
            ),

        "defect_maximum":
            (
                float(
                    field_summary["defect_maximum"]
                )
                <= DEFECT_MAXIMUM
            ),
    }

    return {
        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "passed":
            bool(
                all(checks.values())
            ),
    }


def audit_morphology(morphology):
    checks = {
        "mean_coarse_to_fine_characteristic_length_ratio":
            (
                float(
                    morphology[
                        "mean_coarse_to_fine_characteristic_length_ratio"
                    ]
                )
                >= MINIMUM_MEAN_LENGTH_RATIO
            ),

        "minimum_coarse_to_fine_characteristic_length_ratio":
            (
                float(
                    morphology[
                        "minimum_coarse_to_fine_characteristic_length_ratio"
                    ]
                )
                >= MINIMUM_WORST_LENGTH_RATIO
            ),

        "mean_coarse_to_fine_interface_density_ratio":
            (
                float(
                    morphology[
                        "mean_coarse_to_fine_interface_density_ratio"
                    ]
                )
                <= MAXIMUM_MEAN_INTERFACE_RATIO
            ),

        "sorted_phase_multiset_preserved":
            (
                float(
                    morphology[
                        "maximum_sorted_phase_difference"
                    ]
                )
                <= MAXIMUM_SORTED_PHASE_DIFFERENCE
            ),

        "sorted_defect_multiset_preserved":
            (
                float(
                    morphology[
                        "maximum_sorted_defect_difference"
                    ]
                )
                <= MAXIMUM_SORTED_DEFECT_DIFFERENCE
            ),

        "sorted_orientation_multiset_preserved":
            (
                float(
                    morphology[
                        "maximum_sorted_orientation_angle_difference"
                    ]
                )
                <= MAXIMUM_SORTED_ORIENTATION_DIFFERENCE
            ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    expected_failure = [
        "mean_coarse_to_fine_interface_density_ratio"
    ]

    if failed_checks != expected_failure:
        raise AssertionError(
            "Unexpected Tier C v3 morphology failures: "
            f"{failed_checks}"
        )

    return {
        "checks":
            checks,

        "failed_checks":
            failed_checks,

        "passed":
            False,
    }


def audit_sealing_and_training(
    field_summary,
    dataset_summary,
):
    checks = {
        "test_carrier_count_generated":
            (
                int(
                    field_summary[
                        "test_carrier_count_generated"
                    ]
                )
                == 0
            ),

        "test_carrier_parameters_generated":
            (
                field_summary[
                    "test_carrier_parameters_generated"
                ]
                is False
                and dataset_summary[
                    "test_carrier_parameters_generated"
                ]
                is False
            ),

        "test_fields_generated":
            (
                field_summary[
                    "test_fields_generated"
                ]
                is False
                and dataset_summary[
                    "test_fields_generated"
                ]
                is False
            ),

        "test_manifests_generated":
            (
                field_summary[
                    "test_manifests_generated"
                ]
                is False
                and dataset_summary[
                    "test_manifests_generated"
                ]
                is False
            ),

        "sealed_test_files_absent":
            (
                dataset_summary[
                    "sealed_test_files_absent"
                ]
                is True
            ),

        "test_metrics_computed":
            (
                field_summary[
                    "test_metrics_computed"
                ]
                is False
                and dataset_summary[
                    "test_metrics_computed"
                ]
                is False
            ),

        "diagnostic_decoder_not_fitted":
            (
                field_summary[
                    "diagnostic_decoder_fitted"
                ]
                is False
                and dataset_summary[
                    "diagnostic_decoder_fitted"
                ]
                is False
            ),

        "predictive_training_not_performed":
            (
                field_summary[
                    "predictive_training_performed"
                ]
                is False
                and dataset_summary[
                    "predictive_training_performed"
                ]
                is False
            ),

        "checkpoint_selection_not_performed":
            (
                field_summary[
                    "checkpoint_selection_performed"
                ]
                is False
                and dataset_summary[
                    "checkpoint_selection_performed"
                ]
                is False
            ),
    }

    return {
        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "passed":
            bool(
                all(checks.values())
            ),
    }


def write_input_hashes():
    input_paths = {
        "phase4ar2_summary":
            PHASE4AR2_DIR
            / "phase4ar2_summary.json",

        "phase4br2_field_bank_summary":
            PHASE4BR2_DIR
            / "phase4br2_field_bank_summary.json",

        "coarsening_morphology_summary":
            PHASE4BR2_DIR
            / "coarsening_morphology_summary.json",

        "phase4br2_dataset_summary":
            PHASE4BR2_DIR
            / "phase4br2_summary.json",

        "tier_c_v2_diagnosis":
            PHASE4CRD_V2_DIR
            / "phase4crd_summary.json",
    }

    hashes = {}

    for name, path in input_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_prior_results()

    field_summary = sources[
        "field_summary"
    ]

    morphology = sources[
        "morphology"
    ]

    dataset_summary = sources[
        "dataset_summary"
    ]

    physics_audit = audit_physics(
        field_summary
    )

    morphology_audit = audit_morphology(
        morphology
    )

    sealing_audit = (
        audit_sealing_and_training(
            field_summary=field_summary,
            dataset_summary=dataset_summary,
        )
    )

    if physics_audit["passed"] is not True:
        raise AssertionError(
            "Tier C v3 has an unexpected physics failure: "
            f"{physics_audit['failed_checks']}"
        )

    if sealing_audit["passed"] is not True:
        raise AssertionError(
            "Tier C v3 sealing or training-state audit failed: "
            f"{sealing_audit['failed_checks']}"
        )

    if (
        morphology[
            "preliminary_coarsening_morphology_gate_passed"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v3 preliminary morphology gate "
            "was expected to fail."
        )

    gate_rows = [
        {
            "gate":
                "numerical_physics",

            "passed":
                physics_audit["passed"],
        },
        {
            "gate":
                "coarsening_morphology",

            "passed":
                morphology_audit["passed"],
        },
        {
            "gate":
                "development_data_and_test_sealing",

            "passed":
                sealing_audit["passed"],
        },
    ]

    write_csv(
        OUTPUT_DIR
        / "predecoder_gate_results.csv",
        gate_rows,
    )

    write_json(
        OUTPUT_DIR
        / "physics_predecoder_audit.json",
        physics_audit,
    )

    write_json(
        OUTPUT_DIR
        / "morphology_predecoder_audit.json",
        morphology_audit,
    )

    write_json(
        OUTPUT_DIR
        / "sealing_predecoder_audit.json",
        sealing_audit,
    )

    write_input_hashes()

    rejection_record = {
        "phase":
            (
                "4C-R2 Tier C v3 pre-decoder "
                "acceptance audit and rejection freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "tier_c_v1_rejection_frozen":
            True,

        "tier_c_v2_rejection_frozen":
            True,

        "tier_c_v3_accepted":
            False,

        "tier_c_v3_rejected":
            True,

        "rejection_is_final":
            True,

        "failed_acceptance_gates": [
            "coarsening_morphology"
        ],

        "failed_morphology_checks":
            morphology_audit[
                "failed_checks"
            ],

        "passed_results": {
            "numerical_physics":
                physics_audit["passed"],

            "mean_characteristic_length_ratio":
                (
                    morphology_audit[
                        "checks"
                    ][
                        "mean_coarse_to_fine_characteristic_length_ratio"
                    ]
                ),

            "worst_characteristic_length_ratio":
                (
                    morphology_audit[
                        "checks"
                    ][
                        "minimum_coarse_to_fine_characteristic_length_ratio"
                    ]
                ),

            "phase_multiset_preserved":
                (
                    morphology_audit[
                        "checks"
                    ][
                        "sorted_phase_multiset_preserved"
                    ]
                ),

            "defect_multiset_preserved":
                (
                    morphology_audit[
                        "checks"
                    ][
                        "sorted_defect_multiset_preserved"
                    ]
                ),

            "orientation_multiset_preserved":
                (
                    morphology_audit[
                        "checks"
                    ][
                        "sorted_orientation_multiset_preserved"
                    ]
                ),

            "test_sealing":
                sealing_audit["passed"],
        },

        "observed_morphology": {
            "mean_coarse_to_fine_characteristic_length_ratio":
                morphology[
                    "mean_coarse_to_fine_characteristic_length_ratio"
                ],

            "minimum_coarse_to_fine_characteristic_length_ratio":
                morphology[
                    "minimum_coarse_to_fine_characteristic_length_ratio"
                ],

            "maximum_coarse_to_fine_characteristic_length_ratio":
                morphology[
                    "maximum_coarse_to_fine_characteristic_length_ratio"
                ],

            "mean_coarse_to_fine_interface_density_ratio":
                morphology[
                    "mean_coarse_to_fine_interface_density_ratio"
                ],

            "minimum_coarse_to_fine_interface_density_ratio":
                morphology[
                    "minimum_coarse_to_fine_interface_density_ratio"
                ],

            "maximum_coarse_to_fine_interface_density_ratio":
                morphology[
                    "maximum_coarse_to_fine_interface_density_ratio"
                ],
        },

        "frozen_requirements": {
            "minimum_mean_characteristic_length_ratio":
                MINIMUM_MEAN_LENGTH_RATIO,

            "minimum_worst_characteristic_length_ratio":
                MINIMUM_WORST_LENGTH_RATIO,

            "maximum_mean_interface_density_ratio":
                MAXIMUM_MEAN_INTERFACE_RATIO,
        },

        "scientific_interpretation":
            (
                "The targeted spectral-band construction creates "
                "the required separation in characteristic length "
                "while preserving carrier-specific field-value "
                "distributions. However, it does not reduce periodic "
                "phase-interface density to the frozen degree. Some "
                "nominally coarse fields retain as much or slightly "
                "more interface than their paired fine fields."
            ),

        "state_decoder_fitting_authorized":
            False,

        "predictive_training_authorized":
            False,

        "test_generation_authorized":
            False,

        "allowed_next_phase": {
            "phase_id":
                "4C-R2D",

            "name":
                (
                    "Tier C v3 development-only "
                    "interface-density failure diagnosis"
                ),

            "new_model_fitting_allowed":
                False,

            "test_access_allowed":
                False,

            "allowed_inputs": [
                (
                    "Existing development morphology "
                    "diagnostics."
                ),
                (
                    "Existing development raw carrier-state "
                    "fields."
                ),
                (
                    "Frozen Tier C v3 state and carrier "
                    "registries."
                ),
            ],
        },

        "prohibited_actions": [
            (
                "Do not fit the three state-separability "
                "diagnostic decoders."
            ),
            (
                "Do not begin predictive spatial-model "
                "training."
            ),
            (
                "Do not lower or reinterpret the frozen "
                "interface-density threshold."
            ),
            (
                "Do not overwrite Tier C v3 fields or "
                "development cells."
            ),
            (
                "Do not generate or access Tier C v3 "
                "test artifacts."
            ),
        ],

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "diagnostic_decoder_fitted":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,

        "rejection_freeze_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr2_rejection_record.json",
        rejection_record,
    )

    summary = {
        "phase":
            (
                "4C-R2 Tier C v3 pre-decoder "
                "rejection freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "physics_gate_passed":
            physics_audit["passed"],

        "coarsening_morphology_gate_passed":
            False,

        "development_data_and_test_sealing_passed":
            sealing_audit["passed"],

        "failed_gate_count":
            1,

        "failed_acceptance_gates": [
            "coarsening_morphology"
        ],

        "failed_morphology_checks":
            morphology_audit[
                "failed_checks"
            ],

        "mean_coarse_to_fine_characteristic_length_ratio":
            morphology[
                "mean_coarse_to_fine_characteristic_length_ratio"
            ],

        "minimum_coarse_to_fine_characteristic_length_ratio":
            morphology[
                "minimum_coarse_to_fine_characteristic_length_ratio"
            ],

        "mean_coarse_to_fine_interface_density_ratio":
            morphology[
                "mean_coarse_to_fine_interface_density_ratio"
            ],

        "required_maximum_mean_interface_density_ratio":
            MAXIMUM_MEAN_INTERFACE_RATIO,

        "tier_c_v3_rejected":
            True,

        "state_decoder_fitting_authorized":
            False,

        "predictive_training_authorized":
            False,

        "sealed_test_set_remains_unopened":
            True,

        "next_phase":
            (
                "4C-R2D Tier C v3 development-only "
                "interface-density failure diagnosis"
            ),

        "prior_outputs_modified":
            False,

        "phase4cr2_status":
            "rejection_frozen",
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr2_summary.json",
        summary,
    )

    print(
        "Phase 4C-R2 Tier C v3 morphology "
        "rejection frozen."
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
