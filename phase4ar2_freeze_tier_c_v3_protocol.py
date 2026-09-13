from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE3H_DIR = Path(
    "outputs/phase3h_tier_b_synthesis"
)

PHASE4A_V1_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

PHASE4C_V1_DIR = Path(
    "outputs/phase4c_tier_c_audit"
)

PHASE4AR_V2_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

PHASE4CR_V2_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

PHASE4CRD_V2_DIR = Path(
    "outputs/phase4crd_tier_c_v2_failure_diagnosis"
)

OUTPUT_DIR = Path(
    "outputs/phase4ar2_tier_c_v3_protocol"
)


PROTOCOL_VERSION = "tier_c_v3"

STATE_COUNT = 8
PRIMITIVE_OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104
INFORMATION_CLASS_COUNT = 9

CARRIER_COUNT = 160

CARRIER_SPLIT_COUNTS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

TRANSFORMATION_SPLIT_COUNTS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32
PHYSICAL_CARRIER_DIMENSION = 8

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

FROZEN_FINAL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE_FRACTION = 0.25

CARRIER_SPLIT_SEED = 61026
CARRIER_PARAMETER_SEED = 61027
FIELD_INITIALIZATION_SEED = 61028
SOLVER_SEED = 61029
NOISE_SEED = 61030
DEVELOPMENT_MANIFEST_SEED = 61032
SEALED_TEST_MANIFEST_SEED = 61033

DIAGNOSTIC_SEEDS = (
    62011,
    62023,
    62037,
)

INITIAL_TIME_STEP = 0.005
MINIMUM_TIME_STEP = (
    INITIAL_TIME_STEP / 32.0
)
MAXIMUM_STEP_HALVINGS = 5

FINE_COARSENING_LENGTH_RANGE = (
    1.75,
    2.50,
)

COARSE_COARSENING_LENGTH_RANGE = (
    6.50,
    8.00,
)

COARSENING_BANDWIDTH_FRACTION = 0.25

MINIMUM_MEAN_VALIDATION_STATE_ACCURACY = 0.80
MINIMUM_WORST_VALIDATION_STATE_ACCURACY = 0.75

MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY = 0.90
MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY = 0.85

MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY = 0.95

MINIMUM_MEAN_COARSE_TO_FINE_LENGTH_RATIO = 2.50
MINIMUM_WORST_CARRIER_LENGTH_RATIO = 1.80
MAXIMUM_MEAN_COARSE_TO_FINE_INTERFACE_RATIO = 0.70

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_ANGLE_DIFFERENCE = 1e-7

OCM_TUNING_CONFIGURATION_COUNT = 24

BASELINE_TUNING_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}

FINAL_FIT_COUNTS = {
    "OCM": 25,
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
}


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


def validate_prior_phases():
    phase3h = load_json(
        PHASE3H_DIR
        / "phase3h_summary.json"
    )

    v1_rejection = load_json(
        PHASE4C_V1_DIR
        / "phase4c_rejection_summary.json"
    )

    v2_protocol = load_json(
        PHASE4AR_V2_DIR
        / "phase4ar_summary.json"
    )

    v2_rejection = load_json(
        PHASE4CR_V2_DIR
        / "phase4cr_rejection_summary.json"
    )

    v2_rejection_record = load_json(
        PHASE4CR_V2_DIR
        / "phase4cr_rejection_record.json"
    )

    diagnosis = load_json(
        PHASE4CRD_V2_DIR
        / "phase4crd_summary.json"
    )

    diagnosis_findings = load_json(
        PHASE4CRD_V2_DIR
        / "diagnosis_findings.json"
    )

    if phase3h["phase3h_status"] != "passed":
        raise AssertionError(
            "Phase 3H has not passed."
        )

    if (
        phase3h["phase3_results_frozen"]
        is not True
    ):
        raise AssertionError(
            "Phase 3 is not frozen."
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

    if (
        v2_protocol["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v2 protocol is not frozen."
        )

    if (
        v2_rejection[
            "tier_c_v2_rejected"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection is not recorded."
        )

    if (
        v2_rejection_record[
            "rejection_is_final"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection is not final."
        )

    if (
        v2_rejection[
            "phase4dr_predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v2 predictive training was "
            "incorrectly authorized."
        )

    if (
        v2_rejection[
            "sealed_test_set_remains_unopened"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 test set is not sealed."
        )

    if (
        diagnosis["phase4crd_status"]
        != "passed"
    ):
        raise AssertionError(
            "Tier C v2 diagnosis has not passed."
        )

    if (
        diagnosis[
            "validation_accuracy_reproduction_error"
        ]
        != 0.0
    ):
        raise AssertionError(
            "Tier C v2 diagnostic accuracy "
            "was not reproduced exactly."
        )

    if (
        diagnosis[
            "weakest_validation_factor"
        ]
        != "coarsening_level"
    ):
        raise AssertionError(
            "Tier C v2 weakest factor changed."
        )

    if (
        diagnosis[
            "strongest_validation_factor"
        ]
        != "anisotropy_level"
    ):
        raise AssertionError(
            "Unexpected strongest Tier C v2 factor."
        )

    dominant = diagnosis[
        "dominant_validation_confusion"
    ]

    if (
        dominant["coarsening_differs"] != 1
        or dominant["anisotropy_differs"] != 0
        or dominant["defect_recovery_differs"] != 0
    ):
        raise AssertionError(
            "Dominant confusion is not a pure "
            "coarsening error."
        )

    if (
        diagnosis[
            "sealed_test_files_absent"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 sealed test files exist."
        )

    if (
        diagnosis["test_fields_read"]
        is not False
        or diagnosis["test_metrics_computed"]
        is not False
    ):
        raise AssertionError(
            "Tier C v2 test-sealing policy "
            "was violated."
        )

    return {
        "phase3h": phase3h,
        "v1_rejection": v1_rejection,
        "v2_protocol": v2_protocol,
        "v2_rejection": v2_rejection,
        "v2_rejection_record": v2_rejection_record,
        "diagnosis": diagnosis,
        "diagnosis_findings": diagnosis_findings,
    }


def load_split_lookup(path: Path):
    rows = load_csv(path)

    lookup = {
        int(row["carrier_id"]):
            row["split"]
        for row in rows
    }

    if set(lookup) != set(
        range(CARRIER_COUNT)
    ):
        raise AssertionError(
            f"Carrier IDs changed in {path}."
        )

    counts = Counter(
        lookup.values()
    )

    if dict(counts) != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            f"Carrier counts changed in {path}: "
            f"{dict(counts)}"
        )

    return lookup


def create_fresh_v3_carrier_split():
    v1_lookup = load_split_lookup(
        PHASE4A_V1_DIR
        / "carrier_reuse_manifest.csv"
    )

    v2_lookup = load_split_lookup(
        PHASE4AR_V2_DIR
        / "carrier_split_v2.csv"
    )

    all_ids = set(range(CARRIER_COUNT))

    v1_test_ids = {
        carrier_id
        for carrier_id, split
        in v1_lookup.items()
        if split == "test"
    }

    v2_validation_ids = {
        carrier_id
        for carrier_id, split
        in v2_lookup.items()
        if split == "val"
    }

    v2_test_ids = {
        carrier_id
        for carrier_id, split
        in v2_lookup.items()
        if split == "test"
    }

    generator = np.random.default_rng(
        CARRIER_SPLIT_SEED
    )

    v3_test_candidates = sorted(
        all_ids
        - v1_test_ids
        - v2_validation_ids
        - v2_test_ids
    )

    if len(v3_test_candidates) < 32:
        raise AssertionError(
            "Insufficient clean candidates for "
            "the Tier C v3 test split."
        )

    v3_test_ids = set(
        int(value)
        for value in generator.choice(
            np.asarray(
                v3_test_candidates,
                dtype=np.int64,
            ),
            size=32,
            replace=False,
        )
    )

    v3_validation_candidates = sorted(
        all_ids
        - v1_test_ids
        - v2_validation_ids
        - v2_test_ids
        - v3_test_ids
    )

    if len(v3_validation_candidates) < 32:
        raise AssertionError(
            "Insufficient clean candidates for "
            "the Tier C v3 validation split."
        )

    v3_validation_ids = set(
        int(value)
        for value in generator.choice(
            np.asarray(
                v3_validation_candidates,
                dtype=np.int64,
            ),
            size=32,
            replace=False,
        )
    )

    v3_train_ids = (
        all_ids
        - v3_test_ids
        - v3_validation_ids
    )

    if len(v3_train_ids) != 96:
        raise AssertionError(
            "Tier C v3 training count changed."
        )

    split_members = {
        "train": v3_train_ids,
        "val": v3_validation_ids,
        "test": v3_test_ids,
    }

    rows = []

    for split in (
        "train",
        "val",
        "test",
    ):
        for split_position, carrier_id in enumerate(
            sorted(split_members[split])
        ):
            rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "split":
                        split,

                    "split_position":
                        split_position,

                    "v1_split":
                        v1_lookup[
                            carrier_id
                        ],

                    "v2_split":
                        v2_lookup[
                            carrier_id
                        ],

                    "fresh_v3_parameters_generated":
                        False,

                    "fresh_v3_fields_generated":
                        False,
                }
            )

    rows.sort(
        key=lambda row:
            row["carrier_id"]
    )

    counts = dict(
        Counter(
            row["split"]
            for row in rows
        )
    )

    if counts != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            "Tier C v3 split counts changed."
        )

    overlap_summary = {
        "v3_test_vs_v1_test":
            len(
                v3_test_ids
                & v1_test_ids
            ),

        "v3_test_vs_v2_test":
            len(
                v3_test_ids
                & v2_test_ids
            ),

        "v3_test_vs_v2_validation":
            len(
                v3_test_ids
                & v2_validation_ids
            ),

        "v3_validation_vs_v1_test":
            len(
                v3_validation_ids
                & v1_test_ids
            ),

        "v3_validation_vs_v2_test":
            len(
                v3_validation_ids
                & v2_test_ids
            ),

        "v3_validation_vs_v2_validation":
            len(
                v3_validation_ids
                & v2_validation_ids
            ),
    }

    if any(
        value != 0
        for value in overlap_summary.values()
    ):
        raise AssertionError(
            "Tier C v3 validation or test split "
            "overlaps a prohibited prior split."
        )

    return rows, overlap_summary


def validate_transformation_split():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    counts = dict(
        Counter(
            row["split"]
            for row in rows
        )
    )

    if counts != TRANSFORMATION_SPLIT_COUNTS:
        raise AssertionError(
            "Transformation split changed."
        )

    if len(rows) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return rows


def create_carrier_parameter_schema():
    source_rows = load_csv(
        PHASE4AR_V2_DIR
        / "carrier_parameter_schema_v2.csv"
    )

    source_rows.sort(
        key=lambda row:
            int(row["parameter_index"])
    )

    output = []

    for row in source_rows:
        value = dict(row)

        value["protocol_version"] = (
            PROTOCOL_VERSION
        )

        value[
            "changed_from_v2"
        ] = False

        value[
            "fresh_parameter_seed"
        ] = CARRIER_PARAMETER_SEED

        output.append(value)

    if len(output) != PHYSICAL_CARRIER_DIMENSION:
        raise AssertionError(
            "Carrier parameter dimension changed."
        )

    return output


def create_targeted_state_regimes():
    source_rows = load_csv(
        PHASE4AR_V2_DIR
        / "state_morphology_regimes_v2.csv"
    )

    source_rows.sort(
        key=lambda row:
            int(row["state_id"])
    )

    rows = []

    for source in source_rows:
        state_id = int(
            source["state_id"]
        )

        coarsening_level = int(
            source["coarsening_level"]
        )

        if coarsening_level == 0:
            length_minimum = (
                FINE_COARSENING_LENGTH_RANGE[0]
            )

            length_maximum = (
                FINE_COARSENING_LENGTH_RANGE[1]
            )

            band_label = "fine"

        else:
            length_minimum = (
                COARSE_COARSENING_LENGTH_RANGE[0]
            )

            length_maximum = (
                COARSE_COARSENING_LENGTH_RANGE[1]
            )

            band_label = "coarse"

        row = dict(source)

        row["protocol_version"] = (
            PROTOCOL_VERSION
        )

        row[
            "coarsening_target_family"
        ] = (
            "state_controlled_spectral_band_"
            "rank_transport"
        )

        row[
            "coarsening_band_label"
        ] = band_label

        row[
            "target_characteristic_length_minimum_pixels"
        ] = length_minimum

        row[
            "target_characteristic_length_maximum_pixels"
        ] = length_maximum

        row[
            "coarsening_bandwidth_fraction"
        ] = COARSENING_BANDWIDTH_FRACTION

        row[
            "carrier_length_scale_controls_final_coarsening"
        ] = False

        row[
            "carrier_latent_controls_within_band_position"
        ] = True

        row[
            "phase_value_multiset_preserved"
        ] = True

        row[
            "defect_value_multiset_preserved"
        ] = True

        row[
            "orientation_pair_multiset_preserved"
        ] = True

        row[
            "anisotropy_parameters_changed_from_v2"
        ] = False

        row[
            "defect_recovery_parameters_changed_from_v2"
        ] = False

        row[
            "coarsening_parameters_changed_from_v2"
        ] = True

        rows.append(row)

    if len(rows) != STATE_COUNT:
        raise AssertionError(
            "Tier C v3 state count changed."
        )

    fine_rows = [
        row
        for row in rows
        if int(
            row["coarsening_level"]
        ) == 0
    ]

    coarse_rows = [
        row
        for row in rows
        if int(
            row["coarsening_level"]
        ) == 1
    ]

    if len(fine_rows) != 4:
        raise AssertionError(
            "Expected four fine states."
        )

    if len(coarse_rows) != 4:
        raise AssertionError(
            "Expected four coarse states."
        )

    fine_maximum = max(
        float(
            row[
                "target_characteristic_length_maximum_pixels"
            ]
        )
        for row in fine_rows
    )

    coarse_minimum = min(
        float(
            row[
                "target_characteristic_length_minimum_pixels"
            ]
        )
        for row in coarse_rows
    )

    if fine_maximum >= coarse_minimum:
        raise AssertionError(
            "Fine and coarse spatial-scale bands overlap."
        )

    return rows


def create_revision_decision(
    prior_results,
):
    diagnosis = prior_results[
        "diagnosis"
    ]

    findings = prior_results[
        "diagnosis_findings"
    ]

    return {
        "phase":
            "Tier C v3 targeted revision decision",

        "protocol_version":
            PROTOCOL_VERSION,

        "source_protocol":
            "tier_c_v2",

        "source_diagnosis_phase":
            diagnosis["phase"],

        "source_diagnosis_status":
            diagnosis[
                "phase4crd_status"
            ],

        "v2_mean_validation_state_accuracy":
            diagnosis[
                "frozen_mean_validation_state_accuracy"
            ],

        "v2_weakest_factor":
            diagnosis[
                "weakest_validation_factor"
            ],

        "v2_weakest_factor_accuracy":
            diagnosis[
                "weakest_validation_factor_accuracy_mean"
            ],

        "v2_strongest_factor":
            diagnosis[
                "strongest_validation_factor"
            ],

        "v2_strongest_factor_accuracy":
            diagnosis[
                "strongest_validation_factor_accuracy_mean"
            ],

        "v2_validation_ensemble_accuracy":
            diagnosis[
                "validation_probability_ensemble_accuracy"
            ],

        "v2_validation_any_seed_correct_fraction":
            diagnosis[
                "validation_any_seed_correct_fraction"
            ],

        "v2_validation_all_seeds_correct_fraction":
            diagnosis[
                "validation_all_seeds_correct_fraction"
            ],

        "dominant_confusion":
            diagnosis[
                "dominant_validation_confusion"
            ],

        "targeted_revision":
            "coarsening morphology only",

        "coarsening_controls_changed":
            True,

        "anisotropy_controls_changed":
            False,

        "defect_recovery_controls_changed":
            False,

        "carrier_parameter_ranges_changed":
            False,

        "numerical_stabilization_changed":
            False,

        "global_marginal_preservation_changed":
            False,

        "reason":
            (
                "Coarsening is the only morphology factor "
                "substantially below the acceptance threshold. "
                "Anisotropy and defect recovery are already "
                "decoded near perfectly, and the dominant state "
                "confusions are pure coarsening flips."
            ),

        "v2_diagnostic_findings_hash":
            sha256_file(
                PHASE4CRD_V2_DIR
                / "diagnosis_findings.json"
            ),

        "v3_design_scope_locked":
            True,
    }


def create_coarsening_protocol():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "revision_scope":
            "coarsening factor only",

        "target_family":
            (
                "State-controlled spectral-band "
                "rank transport."
            ),

        "fine_characteristic_length_range_pixels":
            list(
                FINE_COARSENING_LENGTH_RANGE
            ),

        "coarse_characteristic_length_range_pixels":
            list(
                COARSE_COARSENING_LENGTH_RANGE
            ),

        "bands_overlap":
            False,

        "spectral_bandwidth_fraction":
            COARSENING_BANDWIDTH_FRACTION,

        "carrier_variation_rule":
            (
                "A fresh carrier latent coordinate selects "
                "a continuous location inside the frozen "
                "state-specific length-scale band."
            ),

        "carrier_base_length_scale_role":
            (
                "Controls initialization only and does not "
                "determine the final coarsening class."
            ),

        "phase_target_rule":
            (
                "Generate a band-limited spatial score at the "
                "state-specific characteristic length, then "
                "assign the carrier's unchanged phase values by "
                "stable rank transport."
            ),

        "defect_target_rule":
            (
                "Preserve the carrier's defect-value multiset "
                "while spatially arranging values according to "
                "the unchanged v2 defect-recovery rule on the "
                "revised phase morphology."
            ),

        "orientation_target_rule":
            (
                "Preserve the carrier's orientation-vector "
                "multiset while retaining the unchanged v2 "
                "anisotropy construction."
            ),

        "global_value_distribution_preservation": {
            "phase_sorted_values_preserved":
                True,

            "defect_sorted_values_preserved":
                True,

            "orientation_pair_multiset_preserved":
                True,

            "state_dependent_phase_mean":
                False,

            "state_dependent_phase_variance":
                False,

            "state_dependent_defect_mean":
                False,

            "state_dependent_defect_variance":
                False,
        },

        "frozen_morphology_acceptance_metrics": {
            "minimum_mean_coarse_to_fine_characteristic_length_ratio":
                MINIMUM_MEAN_COARSE_TO_FINE_LENGTH_RATIO,

            "minimum_worst_carrier_characteristic_length_ratio":
                MINIMUM_WORST_CARRIER_LENGTH_RATIO,

            "maximum_mean_coarse_to_fine_interface_density_ratio":
                MAXIMUM_MEAN_COARSE_TO_FINE_INTERFACE_RATIO,

            "maximum_sorted_phase_difference":
                MAXIMUM_SORTED_PHASE_DIFFERENCE,

            "maximum_sorted_defect_difference":
                MAXIMUM_SORTED_DEFECT_DIFFERENCE,

            "maximum_sorted_orientation_angle_difference":
                MAXIMUM_SORTED_ORIENTATION_ANGLE_DIFFERENCE,
        },

        "test_fields_used_for_morphology_acceptance":
            False,
    }


def create_numerical_protocol():
    source = load_json(
        PHASE4AR_V2_DIR
        / "numerical_stabilization_protocol.json"
    )

    output = dict(source)

    output["protocol_version"] = (
        PROTOCOL_VERSION
    )

    output[
        "changed_from_tier_c_v2"
    ] = False

    output[
        "initial_time_step"
    ] = INITIAL_TIME_STEP

    output[
        "minimum_time_step"
    ] = MINIMUM_TIME_STEP

    output[
        "maximum_step_halvings"
    ] = MAXIMUM_STEP_HALVINGS

    return output


def create_acceptance_gates():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "acceptance_scope":
            (
                "Fresh Tier C v3 training and validation "
                "carriers only. Test parameters, fields, "
                "manifests, and metrics must not exist."
            ),

        "numerical_physics": {
            "required":
                True,

            "all_values_finite":
                True,

            "minimum_energy_nonincrease_fraction":
                0.99,

            "maximum_phase_mean_drift":
                1e-5,

            "maximum_defect_mean_drift":
                1e-5,

            "phase_minimum":
                -1.000001,

            "phase_maximum":
                1.000001,

            "defect_minimum":
                -0.000001,

            "defect_maximum":
                1.000001,

            "maximum_unresolved_simulation_failures":
                0,
        },

        "basic_manifold_quality": {
            "required":
                True,

            "all_training_channels_nonconstant":
                True,

            "every_state_has_positive_carrier_variation":
                True,

            "maximum_exact_duplicate_count":
                0,
        },

        "coarsening_morphology": {
            "required":
                True,

            "characteristic_length_estimator":
                (
                    "Radially averaged phase structure "
                    "factor with zero-frequency component "
                    "excluded."
                ),

            "interface_density_estimator":
                (
                    "Mean periodic phase-gradient magnitude."
                ),

            "paired_comparison_rule":
                (
                    "Compare states differing only in "
                    "coarsening level for every development "
                    "carrier and each anisotropy/defect pair."
                ),

            "minimum_mean_coarse_to_fine_characteristic_length_ratio":
                MINIMUM_MEAN_COARSE_TO_FINE_LENGTH_RATIO,

            "minimum_worst_carrier_characteristic_length_ratio":
                MINIMUM_WORST_CARRIER_LENGTH_RATIO,

            "maximum_mean_coarse_to_fine_interface_density_ratio":
                MAXIMUM_MEAN_COARSE_TO_FINE_INTERFACE_RATIO,

            "maximum_sorted_phase_difference":
                MAXIMUM_SORTED_PHASE_DIFFERENCE,

            "maximum_sorted_defect_difference":
                MAXIMUM_SORTED_DEFECT_DIFFERENCE,

            "maximum_sorted_orientation_angle_difference":
                MAXIMUM_SORTED_ORIENTATION_ANGLE_DIFFERENCE,

            "test_fields_used":
                False,
        },

        "state_separability": {
            "required":
                True,

            "diagnostic_model":
                (
                    "Frozen residual convolutional "
                    "state decoder unchanged from v2."
                ),

            "diagnostic_architecture_changed_from_v2":
                False,

            "diagnostic_seeds":
                list(
                    DIAGNOSTIC_SEEDS
                ),

            "minimum_mean_validation_state_accuracy":
                MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

            "minimum_worst_seed_validation_state_accuracy":
                MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

            "minimum_mean_validation_coarsening_accuracy":
                MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

            "minimum_worst_seed_validation_coarsening_accuracy":
                MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,

            "training_data":
                (
                    "Clean Tier C v3 training-carrier "
                    "fields only."
                ),

            "selection_data":
                (
                    "Clean Tier C v3 validation-carrier "
                    "fields only."
                ),

            "test_fields_used":
                False,

            "diagnostic_checkpoints_eligible_for_predictive_use":
                False,
        },

        "global_shortcut_control": {
            "required":
                True,

            "diagnostic_features": [
                "per-channel mean",
                "per-channel variance",
                "per-channel skewness",
                "per-channel excess kurtosis",
                "per-channel 10th percentile",
                "per-channel median",
                "per-channel 90th percentile",
            ],

            "maximum_validation_accuracy":
                MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY,

            "test_fields_used":
                False,
        },

        "affine_shortcut_control": {
            "required":
                True,

            "diagnostic":
                (
                    "Training-only PCA-128 plus ridge "
                    "affine primitive operators."
                ),

            "minimum_validation_continuous_mse":
                1e-8,

            "validation_exact_primitive_rate_must_be_below":
                1.0,

            "fraction_validation_carriers_all_104_exact_must_be_below":
                1.0,

            "test_fields_used":
                False,
        },

        "data_integrity": {
            "required":
                True,

            "carrier_splits_disjoint":
                True,

            "transformation_split_unchanged":
                True,

            "v3_validation_excludes_v2_validation":
                True,

            "v3_validation_excludes_v1_and_v2_test":
                True,

            "v3_test_excludes_v1_and_v2_test":
                True,

            "v3_test_excludes_v2_validation":
                True,

            "test_parameter_files_absent":
                True,

            "test_field_files_absent":
                True,

            "test_manifest_files_absent":
                True,

            "validation_statistics_used_for_normalization":
                False,

            "test_statistics_used_for_normalization":
                False,
        },

        "decision_rule": {
            "all_required_gates_must_pass":
                True,

            "thresholds_may_be_lowered_after_execution":
                False,

            "diagnostic_architecture_may_be_changed_after_execution":
                False,

            "failed_v3_must_be_frozen_as_a_new_negative_result":
                True,
        },
    }


def create_test_sealing_policy():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "test_carrier_split_frozen":
            True,

        "test_carrier_parameters_generated_before_checkpoint_freeze":
            False,

        "test_clean_fields_generated_before_checkpoint_freeze":
            False,

        "test_noisy_fields_generated_before_checkpoint_freeze":
            False,

        "test_trajectory_manifests_generated_before_checkpoint_freeze":
            False,

        "test_metrics_computed_before_checkpoint_freeze":
            False,

        "development_cells_available_before_checkpoint_freeze": [
            "train_joint",
            "val_composition",
            "val_carrier",
            "val_joint",
        ],

        "sealed_cells": [
            "test_iid_pairing",
            "test_composition",
            "test_carrier",
            "test_joint",
        ],

        "development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "sealed_test_manifest_seed":
            SEALED_TEST_MANIFEST_SEED,

        "test_generation_authorization_condition":
            (
                "All 130 Tier C v3 final model checkpoints "
                "must be complete, valid, and frozen."
            ),

        "test_generation_phase":
            "4G-R2",

        "test_fields_used_for_dataset_acceptance":
            False,

        "test_fields_used_for_model_selection":
            False,
    }


def create_training_protocol():
    baseline_count = sum(
        BASELINE_TUNING_CONFIGURATION_COUNTS.values()
    )

    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE_FRACTION,

        "training_cell":
            "train_joint",

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            (
                "val_joint noisy-target spatial "
                "rollout MSE only"
            ),

        "development_cells": [
            "train_joint",
            "val_composition",
            "val_carrier",
            "val_joint",
        ],

        "test_cells": [
            "test_iid_pairing",
            "test_composition",
            "test_carrier",
            "test_joint",
        ],

        "test_cells_generated_during_tuning":
            False,

        "test_cells_generated_during_final_fitting":
            False,

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_used_for_training":
            False,

        "physics_diagnostics_used_for_model_selection":
            False,

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_counts":
            BASELINE_TUNING_CONFIGURATION_COUNTS,

        "total_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT
            + baseline_count,

        "final_fit_counts":
            FINAL_FIT_COUNTS,

        "total_final_fit_count":
            sum(
                FINAL_FIT_COUNTS.values()
            ),

        "final_seeds":
            list(
                FROZEN_FINAL_SEEDS
            ),

        "noise_levels":
            list(
                FROZEN_NOISE_LEVELS
            ),

        "fresh_initialization_required":
            True,

        "tuning_checkpoints_eligible_for_final_use":
            False,

        "final_checkpoint_freeze_required_before_test_generation":
            True,
    }


def create_model_registry():
    source_rows = load_csv(
        PHASE4AR_V2_DIR
        / "model_family_registry_v2.csv"
    )

    if len(source_rows) != 7:
        raise AssertionError(
            "Model-family count changed."
        )

    output = []

    for row in source_rows:
        value = dict(row)

        value["protocol_version"] = (
            PROTOCOL_VERSION
        )

        value[
            "architecture_changed_from_v2"
        ] = False

        value[
            "test_fields_available_during_development"
        ] = False

        output.append(value)

    return output


def create_hypothesis_registry():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "registry_status":
            "frozen",

        "hypotheses": [
            {
                "hypothesis_id":
                    "H4R2-1",

                "statement":
                    (
                        "Nonoverlapping state-controlled "
                        "coarsening-scale bands will raise "
                        "validation coarsening-factor decoding "
                        "above 90 percent without changing the "
                        "carrier-specific global value "
                        "distributions."
                    ),

                "primary_metric":
                    (
                        "Mean validation coarsening-factor "
                        "accuracy across three fresh diagnostic "
                        "seeds."
                    ),

                "acceptance_threshold":
                    (
                        MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY
                    ),
            },
            {
                "hypothesis_id":
                    "H4R2-2",

                "statement":
                    (
                        "Strengthening only coarsening will raise "
                        "eight-state validation accuracy above the "
                        "frozen Tier C threshold while retaining "
                        "near-perfect anisotropy and defect-recovery "
                        "decoding."
                    ),

                "primary_metric":
                    (
                        "Mean and worst-seed validation "
                        "eight-state accuracy."
                    ),
            },
            {
                "hypothesis_id":
                    "H4R2-3",

                "statement":
                    (
                        "Exact marginal-value preservation will "
                        "prevent the richer global-statistics "
                        "classifier from identifying Tier C v3 "
                        "states."
                    ),

                "primary_metric":
                    (
                        "Validation accuracy of the frozen "
                        "28-feature global-statistics classifier."
                    ),
            },
            {
                "hypothesis_id":
                    "H4R2-4",

                "statement":
                    (
                        "The stronger coarsening morphology will "
                        "remain nontrivial for the frozen PCA-128 "
                        "affine diagnostic."
                    ),

                "primary_metric":
                    (
                        "Validation affine continuous MSE and "
                        "exact transformation recovery."
                    ),
            },
            {
                "hypothesis_id":
                    "H4R2-5",

                "statement":
                    (
                        "After final checkpoints are frozen, "
                        "unseen carrier morphology will remain a "
                        "larger predictive challenge than unseen "
                        "operation composition."
                    ),

                "evaluated_before_checkpoint_freeze":
                    False,
            },
        ],
    }


def create_phase_plan():
    return {
        "4A-R2":
            (
                "Freeze the targeted Tier C v3 coarsening "
                "revision, fresh split, fresh seeds, acceptance "
                "gates, and test-sealing policy."
            ),

        "4B-R2":
            (
                "Generate only Tier C v3 training and validation "
                "carrier-state fields and four development cells."
            ),

        "4C-R2":
            (
                "Audit numerical physics, explicit coarsening "
                "morphology, state and factor separability, "
                "global shortcuts, affine shortcuts, and test "
                "sealing."
            ),

        "4D-R2":
            (
                "Implement or reuse the frozen spatial model "
                "families and perform smoke tests using only "
                "train_joint and val_joint."
            ),

        "4E-R2":
            (
                "Execute the 43 frozen tuning configurations."
            ),

        "4F-R2":
            (
                "Run and freeze all 130 final model fits without "
                "generating test artifacts."
            ),

        "4G-R2":
            (
                "Generate sealed Tier C v3 test artifacts from "
                "the frozen protocol and perform final predictive "
                "and structural evaluation."
            ),

        "4H-R2":
            (
                "Freeze Tier C v3 claims and synthesize Tier A, "
                "Tier B, Tier C v1, Tier C v2, and Tier C v3."
            ),
    }


def write_input_hashes():
    paths = {
        "phase3h_summary":
            PHASE3H_DIR
            / "phase3h_summary.json",

        "tier_c_v1_rejection":
            PHASE4C_V1_DIR
            / "phase4c_rejection_summary.json",

        "tier_c_v2_protocol":
            PHASE4AR_V2_DIR
            / "phase4ar_summary.json",

        "tier_c_v2_rejection":
            PHASE4CR_V2_DIR
            / "phase4cr_rejection_summary.json",

        "tier_c_v2_rejection_record":
            PHASE4CR_V2_DIR
            / "phase4cr_rejection_record.json",

        "tier_c_v2_diagnosis":
            PHASE4CRD_V2_DIR
            / "phase4crd_summary.json",

        "tier_c_v2_diagnosis_findings":
            PHASE4CRD_V2_DIR
            / "diagnosis_findings.json",

        "tier_c_v2_factor_accuracy":
            PHASE4CRD_V2_DIR
            / "factor_accuracy_summary.csv",

        "tier_c_v2_top_confusions":
            PHASE4CRD_V2_DIR
            / "validation_top_state_confusions.csv",

        "transformation_split":
            PHASE1D_DIR
            / "transformation_split.csv",
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
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

    prior_results = (
        validate_prior_phases()
    )

    (
        carrier_split_rows,
        overlap_summary,
    ) = create_fresh_v3_carrier_split()

    transformation_rows = (
        validate_transformation_split()
    )

    carrier_parameter_schema = (
        create_carrier_parameter_schema()
    )

    state_regimes = (
        create_targeted_state_regimes()
    )

    revision_decision = (
        create_revision_decision(
            prior_results
        )
    )

    coarsening_protocol = (
        create_coarsening_protocol()
    )

    numerical_protocol = (
        create_numerical_protocol()
    )

    acceptance_gates = (
        create_acceptance_gates()
    )

    test_sealing_policy = (
        create_test_sealing_policy()
    )

    training_protocol = (
        create_training_protocol()
    )

    model_registry = (
        create_model_registry()
    )

    hypothesis_registry = (
        create_hypothesis_registry()
    )

    phase_plan = create_phase_plan()

    write_input_hashes()

    write_csv(
        OUTPUT_DIR
        / "carrier_split_v3.csv",
        carrier_split_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "carrier_parameter_schema_v3.csv",
        carrier_parameter_schema,
    )

    write_csv(
        OUTPUT_DIR
        / "state_morphology_regimes_v3.csv",
        state_regimes,
    )

    write_csv(
        OUTPUT_DIR
        / "model_family_registry_v3.csv",
        model_registry,
    )

    write_json(
        OUTPUT_DIR
        / "revision_decision.json",
        revision_decision,
    )

    write_json(
        OUTPUT_DIR
        / "coarsening_protocol.json",
        coarsening_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "numerical_stabilization_protocol.json",
        numerical_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "acceptance_gates_v3.json",
        acceptance_gates,
    )

    write_json(
        OUTPUT_DIR
        / "test_sealing_policy.json",
        test_sealing_policy,
    )

    write_json(
        OUTPUT_DIR
        / "training_protocol_v3.json",
        training_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "hypothesis_registry_v3.json",
        hypothesis_registry,
    )

    write_json(
        OUTPUT_DIR
        / "phase4_v3_plan.json",
        phase_plan,
    )

    carrier_counts = dict(
        Counter(
            row["split"]
            for row in carrier_split_rows
        )
    )

    transformation_counts = dict(
        Counter(
            row["split"]
            for row in transformation_rows
        )
    )

    protocol = {
        "phase":
            (
                "4A-R2 Tier C v3 targeted "
                "coarsening protocol freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "scientific_scope":
            (
                "Fresh physics-informed synthetic spatial "
                "bridge with a targeted coarsening revision. "
                "It is not real microscopy data and is not "
                "a calibrated alloy model."
            ),

        "revision_decision":
            revision_decision,

        "carrier_split": {
            "carrier_count":
                CARRIER_COUNT,

            "split_counts":
                carrier_counts,

            "split_seed":
                CARRIER_SPLIT_SEED,

            "prohibited_overlap_counts":
                overlap_summary,
        },

        "transformation_split_counts":
            transformation_counts,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "carrier_parameter_dimension":
            PHYSICAL_CARRIER_DIMENSION,

        "coarsening_protocol":
            coarsening_protocol,

        "state_regimes":
            state_regimes,

        "numerical_protocol":
            numerical_protocol,

        "acceptance_gates":
            acceptance_gates,

        "test_sealing_policy":
            test_sealing_policy,

        "training_protocol":
            training_protocol,

        "hypothesis_registry":
            hypothesis_registry,

        "model_families": {
            row["model_id"]: row
            for row in model_registry
        },

        "fresh_seeds": {
            "carrier_split":
                CARRIER_SPLIT_SEED,

            "carrier_parameters":
                CARRIER_PARAMETER_SEED,

            "field_initialization":
                FIELD_INITIALIZATION_SEED,

            "solver":
                SOLVER_SEED,

            "measurement_noise":
                NOISE_SEED,

            "development_manifests":
                DEVELOPMENT_MANIFEST_SEED,

            "sealed_test_manifests":
                SEALED_TEST_MANIFEST_SEED,

            "diagnostic_seeds":
                list(DIAGNOSTIC_SEEDS),
        },

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "tier_c_v3_protocol.json",
        protocol,
    )

    summary = {
        "phase":
            (
                "4A-R2 Tier C v3 targeted "
                "coarsening protocol freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "tier_c_v1_rejection_frozen":
            True,

        "tier_c_v2_rejection_frozen":
            True,

        "tier_c_v2_diagnosis_passed":
            True,

        "targeted_revision_factor":
            "coarsening_level",

        "anisotropy_controls_changed":
            False,

        "defect_recovery_controls_changed":
            False,

        "state_count":
            STATE_COUNT,

        "primitive_operation_count":
            PRIMITIVE_OPERATION_COUNT,

        "semigroup_size":
            SEMIGROUP_SIZE,

        "information_class_count":
            INFORMATION_CLASS_COUNT,

        "carrier_count":
            CARRIER_COUNT,

        "carrier_split_counts":
            carrier_counts,

        "prohibited_split_overlap_counts":
            overlap_summary,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "fine_coarsening_length_range_pixels":
            list(
                FINE_COARSENING_LENGTH_RANGE
            ),

        "coarse_coarsening_length_range_pixels":
            list(
                COARSE_COARSENING_LENGTH_RANGE
            ),

        "coarsening_length_bands_overlap":
            False,

        "phase_value_multiset_preserved":
            True,

        "defect_value_multiset_preserved":
            True,

        "orientation_pair_multiset_preserved":
            True,

        "minimum_mean_validation_state_accuracy":
            MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

        "minimum_worst_validation_state_accuracy":
            MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

        "minimum_mean_validation_coarsening_accuracy":
            MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

        "minimum_worst_validation_coarsening_accuracy":
            MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,

        "maximum_global_shortcut_validation_accuracy":
            MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY,

        "fresh_carrier_split_seed":
            CARRIER_SPLIT_SEED,

        "fresh_carrier_parameter_seed":
            CARRIER_PARAMETER_SEED,

        "fresh_field_initialization_seed":
            FIELD_INITIALIZATION_SEED,

        "fresh_solver_seed":
            SOLVER_SEED,

        "fresh_noise_seed":
            NOISE_SEED,

        "fresh_development_manifest_seed":
            DEVELOPMENT_MANIFEST_SEED,

        "sealed_test_manifest_seed":
            SEALED_TEST_MANIFEST_SEED,

        "diagnostic_seeds":
            list(DIAGNOSTIC_SEEDS),

        "tuning_configuration_count":
            training_protocol[
                "total_tuning_configuration_count"
            ],

        "final_fit_count":
            training_protocol[
                "total_final_fit_count"
            ],

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,

        "phase4ar2_status":
            "protocol_frozen",
    }

    write_json(
        OUTPUT_DIR
        / "phase4ar2_summary.json",
        summary,
    )

    print(
        "Phase 4A-R2 Tier C v3 targeted "
        "coarsening protocol frozen successfully."
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
