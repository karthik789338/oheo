from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE3H_DIR = Path(
    "outputs/phase3h_tier_b_synthesis"
)

PHASE4AR_V2_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

PHASE4CR_V2_DIR = Path(
    "outputs/phase4cr_tier_c_v2_audit"
)

PHASE4AR2_V3_DIR = Path(
    "outputs/phase4ar2_tier_c_v3_protocol"
)

PHASE4BR2_V3_DIR = Path(
    "outputs/phase4br2_tier_c_v3_development_data"
)

PHASE4CR2_V3_DIR = Path(
    "outputs/phase4cr2_tier_c_v3_predecoder_audit"
)

PHASE4CR2D_V3_DIR = Path(
    "outputs/phase4cr2d_tier_c_v3_interface_diagnosis"
)

OUTPUT_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)


PROTOCOL_VERSION = "tier_c_v4"

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

CARRIER_SPLIT_SEED = 71026
CARRIER_PARAMETER_SEED = 71027
FIELD_INITIALIZATION_SEED = 71028
SOLVER_SEED = 71029
NOISE_SEED = 71030
DEVELOPMENT_MANIFEST_SEED = 71032
SEALED_TEST_MANIFEST_SEED = 71033

DIAGNOSTIC_SEEDS = (
    72011,
    72023,
    72037,
)

FINAL_MODEL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE_FRACTION = 0.25

FINE_DIFFUSION_SIGMA = 0.75
COARSE_DIFFUSION_SIGMA = 1.25
ANISOTROPIC_PARALLEL_MULTIPLIER = 1.50
ANISOTROPIC_PERPENDICULAR_MULTIPLIER = 0.80

MINIMUM_COARSE_ITERATIONS = 2
MAXIMUM_COARSE_ITERATIONS = 32

INTERNAL_TARGET_INTERFACE_RATIO = 0.62
INTERNAL_TARGET_LENGTH_RATIO = 2.60

ACCEPTANCE_MAXIMUM_MEAN_INTERFACE_RATIO = 0.70
ACCEPTANCE_MAXIMUM_PAIR_INTERFACE_RATIO = 0.85
ACCEPTANCE_MINIMUM_PAIR_PASS_FRACTION = 0.95

ACCEPTANCE_MINIMUM_MEAN_LENGTH_RATIO = 2.50
ACCEPTANCE_MINIMUM_WORST_LENGTH_RATIO = 1.80

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_DIFFERENCE = 1e-7

MINIMUM_MEAN_VALIDATION_STATE_ACCURACY = 0.80
MINIMUM_WORST_VALIDATION_STATE_ACCURACY = 0.75

MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY = 0.90
MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY = 0.85

MAXIMUM_GLOBAL_SHORTCUT_VALIDATION_ACCURACY = 0.95

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

    counts = dict(
        Counter(
            lookup.values()
        )
    )

    if counts != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            f"Carrier split counts changed in {path}: "
            f"{counts}"
        )

    return lookup


def validate_prior_results():
    phase3h = load_json(
        PHASE3H_DIR
        / "phase3h_summary.json"
    )

    v2_protocol = load_json(
        PHASE4AR_V2_DIR
        / "phase4ar_summary.json"
    )

    v2_rejection = load_json(
        PHASE4CR_V2_DIR
        / "phase4cr_rejection_summary.json"
    )

    v3_protocol = load_json(
        PHASE4AR2_V3_DIR
        / "phase4ar2_summary.json"
    )

    v3_dataset = load_json(
        PHASE4BR2_V3_DIR
        / "phase4br2_summary.json"
    )

    v3_rejection = load_json(
        PHASE4CR2_V3_DIR
        / "phase4cr2_summary.json"
    )

    v3_rejection_record = load_json(
        PHASE4CR2_V3_DIR
        / "phase4cr2_rejection_record.json"
    )

    v3_diagnosis = load_json(
        PHASE4CR2D_V3_DIR
        / "phase4cr2d_summary.json"
    )

    failure_pattern = load_json(
        PHASE4CR2D_V3_DIR
        / "failure_pattern_diagnosis.json"
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
            "Phase 3 results are not frozen."
        )

    if (
        v2_protocol["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v2 protocol is not frozen."
        )

    if (
        v2_rejection["tier_c_v2_rejected"]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 rejection changed."
        )

    if (
        v2_rejection[
            "sealed_test_set_remains_unopened"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v2 test set was opened."
        )

    if (
        v3_protocol["phase4ar2_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v3 protocol is not frozen."
        )

    if (
        v3_dataset[
            "sealed_test_files_absent"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 sealed test files exist."
        )

    if (
        v3_rejection["phase4cr2_status"]
        != "rejection_frozen"
    ):
        raise AssertionError(
            "Tier C v3 rejection is not frozen."
        )

    if (
        v3_rejection["tier_c_v3_rejected"]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 rejection changed."
        )

    if (
        v3_rejection_record[
            "rejection_is_final"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 rejection is not final."
        )

    if (
        v3_rejection[
            "state_decoder_fitting_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "A Tier C v3 state decoder was authorized."
        )

    if (
        v3_diagnosis["phase4cr2d_status"]
        != "passed"
    ):
        raise AssertionError(
            "Tier C v3 diagnosis has not passed."
        )

    if (
        v3_diagnosis["failure_scope"]
        != (
            "systematic_across_anisotropy_and_"
            "defect_recovery_combinations"
        )
    ):
        raise AssertionError(
            "Tier C v3 failure is no longer systematic."
        )

    if (
        v3_diagnosis[
            "all_factor_combination_means_exceed_threshold"
        ]
        is not True
    ):
        raise AssertionError(
            "Not all Tier C v3 factor combinations fail."
        )

    if (
        failure_pattern[
            "strongest_continuous_association"
        ][
            "variable"
        ]
        != "base_length_scale_pixels"
    ):
        raise AssertionError(
            "Unexpected strongest Tier C v3 association."
        )

    if (
        v3_diagnosis[
            "test_fields_read"
        ]
        is not False
        or v3_diagnosis[
            "test_metrics_computed"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v3 test-sealing policy was violated."
        )

    return {
        "phase3h":
            phase3h,

        "v2_protocol":
            v2_protocol,

        "v2_rejection":
            v2_rejection,

        "v3_protocol":
            v3_protocol,

        "v3_dataset":
            v3_dataset,

        "v3_rejection":
            v3_rejection,

        "v3_rejection_record":
            v3_rejection_record,

        "v3_diagnosis":
            v3_diagnosis,

        "failure_pattern":
            failure_pattern,
    }


def create_v4_carrier_split():
    v2_lookup = load_split_lookup(
        PHASE4AR_V2_DIR
        / "carrier_split_v2.csv"
    )

    v3_lookup = load_split_lookup(
        PHASE4AR2_V3_DIR
        / "carrier_split_v3.csv"
    )

    validation_ids = {
        carrier_id
        for carrier_id, split
        in v2_lookup.items()
        if split == "test"
    }

    test_ids = {
        carrier_id
        for carrier_id, split
        in v3_lookup.items()
        if split == "test"
    }

    if len(validation_ids) != 32:
        raise AssertionError(
            "Expected 32 sealed Tier C v2 test IDs."
        )

    if len(test_ids) != 32:
        raise AssertionError(
            "Expected 32 sealed Tier C v3 test IDs."
        )

    if validation_ids & test_ids:
        raise AssertionError(
            "Tier C v4 validation and test IDs overlap."
        )

    all_ids = set(
        range(CARRIER_COUNT)
    )

    training_ids = (
        all_ids
        - validation_ids
        - test_ids
    )

    if len(training_ids) != 96:
        raise AssertionError(
            "Tier C v4 training count changed."
        )

    split_members = {
        "train":
            training_ids,

        "val":
            validation_ids,

        "test":
            test_ids,
    }

    rows = []

    for split in (
        "train",
        "val",
        "test",
    ):
        for split_position, carrier_id in enumerate(
            sorted(
                split_members[split]
            )
        ):
            rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "split":
                        split,

                    "split_position":
                        split_position,

                    "v2_split":
                        v2_lookup[
                            carrier_id
                        ],

                    "v3_split":
                        v3_lookup[
                            carrier_id
                        ],

                    "validation_source":
                        (
                            "unopened_tier_c_v2_test"
                            if split == "val"
                            else ""
                        ),

                    "test_source":
                        (
                            "unopened_tier_c_v3_test"
                            if split == "test"
                            else ""
                        ),

                    "fresh_v4_parameters_generated":
                        False,

                    "fresh_v4_fields_generated":
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
            "Tier C v4 carrier counts changed."
        )

    overlap_summary = {
        "v4_validation_vs_v2_test":
            len(
                validation_ids
                & {
                    carrier_id
                    for carrier_id, split
                    in v2_lookup.items()
                    if split == "test"
                }
            ),

        "v4_validation_vs_v3_validation":
            len(
                validation_ids
                & {
                    carrier_id
                    for carrier_id, split
                    in v3_lookup.items()
                    if split == "val"
                }
            ),

        "v4_validation_vs_v3_test":
            len(
                validation_ids
                & test_ids
            ),

        "v4_test_vs_v3_test":
            len(
                test_ids
                & {
                    carrier_id
                    for carrier_id, split
                    in v3_lookup.items()
                    if split == "test"
                }
            ),

        "v4_test_vs_v2_test":
            len(
                test_ids
                & validation_ids
            ),
    }

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
        PHASE4AR2_V3_DIR
        / "carrier_parameter_schema_v3.csv"
    )

    source_rows.sort(
        key=lambda row:
            int(
                row["parameter_index"]
            )
    )

    output = []

    for row in source_rows:
        value = dict(row)

        value["protocol_version"] = (
            PROTOCOL_VERSION
        )

        value[
            "fresh_parameter_seed"
        ] = CARRIER_PARAMETER_SEED

        value[
            "changed_from_v3"
        ] = False

        value[
            "controls_final_coarsening_scale"
        ] = False

        output.append(value)

    if len(output) != PHYSICAL_CARRIER_DIMENSION:
        raise AssertionError(
            "Carrier parameter dimension changed."
        )

    return output


def create_state_regimes():
    source_rows = load_csv(
        PHASE4AR2_V3_DIR
        / "state_morphology_regimes_v3.csv"
    )

    source_rows.sort(
        key=lambda row:
            int(row["state_id"])
    )

    output = []

    for source in source_rows:
        row = dict(source)

        row["protocol_version"] = (
            PROTOCOL_VERSION
        )

        row[
            "coarsening_target_family"
        ] = (
            "volume_preserving_diffusion_threshold"
        )

        row[
            "carrier_length_scale_controls_final_coarsening"
        ] = False

        row[
            "shared_fine_morphology_per_carrier_and_anisotropy"
        ] = True

        row[
            "minimum_coarse_iterations"
        ] = MINIMUM_COARSE_ITERATIONS

        row[
            "maximum_coarse_iterations"
        ] = MAXIMUM_COARSE_ITERATIONS

        row[
            "internal_target_interface_ratio"
        ] = INTERNAL_TARGET_INTERFACE_RATIO

        row[
            "internal_target_length_ratio"
        ] = INTERNAL_TARGET_LENGTH_RATIO

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
            "anisotropy_parameters_changed_from_v3"
        ] = False

        row[
            "defect_recovery_parameters_changed_from_v3"
        ] = False

        row[
            "coarsening_construction_changed_from_v3"
        ] = True

        output.append(row)

    if len(output) != STATE_COUNT:
        raise AssertionError(
            "Tier C v4 state count changed."
        )

    return output


def create_coarsening_protocol():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "revision_scope":
            "coarsening construction only",

        "construction_name":
            (
                "Volume-preserving diffusion-threshold "
                "coarsening with exact rank transport."
            ),

        "motivation_from_v3": {
            "overall_interface_gate_failure_fraction":
                0.9296875,

            "failure_scope":
                (
                    "systematic across anisotropy and "
                    "defect-recovery combinations"
                ),

            "fraction_carriers_all_four_pairs_failing":
                0.8671875,

            "spearman_length_ratio_vs_interface_ratio":
                -0.18274633783474475,

            "strongest_continuous_association":
                "base_length_scale_pixels",

            "strongest_association_spearman":
                -0.7231826090841283,
        },

        "fine_construction": {
            "shared_seed_field":
                True,

            "diffusion_sigma":
                FINE_DIFFUSION_SIGMA,

            "rank_transport_preserves_phase_values":
                True,
        },

        "coarse_construction": {
            "initial_score":
                (
                    "The same fine score used by the "
                    "paired fine state."
                ),

            "isotropic_diffusion_sigma":
                COARSE_DIFFUSION_SIGMA,

            "anisotropic_parallel_multiplier":
                ANISOTROPIC_PARALLEL_MULTIPLIER,

            "anisotropic_perpendicular_multiplier":
                ANISOTROPIC_PERPENDICULAR_MULTIPLIER,

            "minimum_iterations":
                MINIMUM_COARSE_ITERATIONS,

            "maximum_iterations":
                MAXIMUM_COARSE_ITERATIONS,

            "rank_transport_after_every_iteration":
                True,

            "internal_interface_ratio_target":
                INTERNAL_TARGET_INTERFACE_RATIO,

            "internal_characteristic_length_ratio_target":
                INTERNAL_TARGET_LENGTH_RATIO,

            "selection_rule":
                (
                    "Choose the earliest deterministic diffusion "
                    "iteration satisfying both internal targets."
                ),

            "failure_rule":
                (
                    "Reject the complete dataset generation if "
                    "any development pair does not satisfy both "
                    "internal targets within the frozen iteration "
                    "limit."
                ),
        },

        "independent_verification": {
            "metrics_recomputed_from_saved_raw_fields":
                True,

            "constructor_intermediate_metrics_not_used_as_final_audit":
                True,

            "verification_code_separate_from_construction_code":
                True,
        },

        "marginal_preservation": {
            "phase_value_multiset_preserved":
                True,

            "defect_value_multiset_preserved":
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

        "carrier_scale_decoupling": {
            "base_length_scale_used_for_initialization":
                True,

            "base_length_scale_controls_final_coarse_iteration":
                False,

            "coarse_stopping_rule_uses_pairwise_morphology":
                True,
        },
    }


def create_acceptance_gates():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "acceptance_scope":
            (
                "Fresh Tier C v4 training and validation "
                "carrier instances only."
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

        "coarsening_construction": {
            "required":
                True,

            "all_pairs_find_valid_iteration":
                True,

            "maximum_selected_iteration":
                MAXIMUM_COARSE_ITERATIONS,

            "minimum_selected_iteration":
                MINIMUM_COARSE_ITERATIONS,

            "internal_target_interface_ratio":
                INTERNAL_TARGET_INTERFACE_RATIO,

            "internal_target_length_ratio":
                INTERNAL_TARGET_LENGTH_RATIO,
        },

        "independent_coarsening_morphology": {
            "required":
                True,

            "minimum_mean_characteristic_length_ratio":
                ACCEPTANCE_MINIMUM_MEAN_LENGTH_RATIO,

            "minimum_worst_pair_characteristic_length_ratio":
                ACCEPTANCE_MINIMUM_WORST_LENGTH_RATIO,

            "maximum_mean_interface_density_ratio":
                ACCEPTANCE_MAXIMUM_MEAN_INTERFACE_RATIO,

            "maximum_pair_interface_density_ratio":
                ACCEPTANCE_MAXIMUM_PAIR_INTERFACE_RATIO,

            "minimum_fraction_pairs_below_interface_threshold":
                ACCEPTANCE_MINIMUM_PAIR_PASS_FRACTION,

            "maximum_sorted_phase_difference":
                MAXIMUM_SORTED_PHASE_DIFFERENCE,

            "maximum_sorted_defect_difference":
                MAXIMUM_SORTED_DEFECT_DIFFERENCE,

            "maximum_sorted_orientation_difference":
                MAXIMUM_SORTED_ORIENTATION_DIFFERENCE,

            "test_fields_used":
                False,
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

        "state_separability": {
            "required":
                True,

            "diagnostic_model":
                (
                    "Frozen residual convolutional state "
                    "decoder unchanged from Tier C v2."
                ),

            "diagnostic_architecture_changed":
                False,

            "diagnostic_seeds":
                list(DIAGNOSTIC_SEEDS),

            "minimum_mean_validation_state_accuracy":
                MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

            "minimum_worst_seed_validation_state_accuracy":
                MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

            "minimum_mean_validation_coarsening_accuracy":
                MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

            "minimum_worst_seed_validation_coarsening_accuracy":
                MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,

            "test_fields_used":
                False,

            "diagnostic_checkpoints_eligible_for_predictive_use":
                False,
        },

        "global_shortcut_control": {
            "required":
                True,

            "feature_count":
                28,

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

            "v4_validation_source":
                "unopened Tier C v2 test IDs",

            "v4_test_source":
                "unopened Tier C v3 test IDs",

            "v4_validation_and_test_disjoint":
                True,

            "fresh_carrier_parameter_seed":
                True,

            "fresh_field_initialization_seed":
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

            "thresholds_may_be_changed_after_execution":
                False,

            "diagnostic_architecture_may_be_changed_after_execution":
                False,

            "failed_v4_must_be_frozen":
                True,

            "tier_c_v5_allowed":
                False,
        },
    }


def create_test_sealing_policy():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "test_carrier_ids_source":
            "unopened Tier C v3 test split",

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

        "development_cells": [
            "train_joint",
            "val_composition",
            "val_carrier",
            "val_joint",
        ],

        "sealed_test_cells": [
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
                "All 130 final Tier C v4 model checkpoints "
                "must be complete and frozen."
            ),

        "test_generation_phase":
            "4G-R3",

        "test_fields_used_for_dataset_acceptance":
            False,

        "test_fields_used_for_model_selection":
            False,
    }


def create_terminal_revision_policy():
    return {
        "protocol_version":
            PROTOCOL_VERSION,

        "tier_c_v4_is_final_revision":
            True,

        "tier_c_v5_authorized":
            False,

        "reason":
            (
                "Tier C v1, v2, and v3 already supplied three "
                "independent development-stage negative results. "
                "A final revision using fresh unopened validation "
                "and test carrier pools is allowed, but additional "
                "adaptive revisions would create excessive "
                "development-set engineering risk."
            ),

        "termination_conditions": [
            (
                "Any required pre-decoder morphology or physics "
                "gate fails."
            ),
            (
                "The frozen state-separability diagnostic fails."
            ),
            (
                "The global-statistics shortcut gate fails."
            ),
            (
                "The affine-shortcut gate fails."
            ),
            (
                "The development-data integrity gate fails."
            ),
        ],

        "action_after_terminal_failure":
            (
                "Freeze the failure, do not train predictive "
                "models, and report Tier C as an unresolved "
                "negative bridge result."
            ),

        "threshold_lowering_after_failure_allowed":
            False,

        "new_diagnostic_architecture_after_failure_allowed":
            False,

        "additional_simulator_revision_after_failure_allowed":
            False,
    }


def create_training_protocol():
    baseline_tuning_count = sum(
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
                "val_joint noisy-target spatial rollout "
                "MSE only"
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

        "physics_metrics_used_for_model_selection":
            False,

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_counts":
            BASELINE_TUNING_CONFIGURATION_COUNTS,

        "total_tuning_configuration_count":
            (
                OCM_TUNING_CONFIGURATION_COUNT
                + baseline_tuning_count
            ),

        "final_fit_counts":
            FINAL_FIT_COUNTS,

        "total_final_fit_count":
            sum(
                FINAL_FIT_COUNTS.values()
            ),

        "final_model_seeds":
            list(FINAL_MODEL_SEEDS),

        "noise_levels":
            list(NOISE_LEVELS),

        "fresh_initialization_required":
            True,

        "tuning_checkpoints_eligible_for_final_use":
            False,

        "final_checkpoint_freeze_required_before_test_generation":
            True,
    }


def create_model_registry():
    source_rows = load_csv(
        PHASE4AR2_V3_DIR
        / "model_family_registry_v3.csv"
    )

    if len(source_rows) != 7:
        raise AssertionError(
            "Tier C model-family count changed."
        )

    output = []

    for source in source_rows:
        row = dict(source)

        row["protocol_version"] = (
            PROTOCOL_VERSION
        )

        row[
            "architecture_changed_from_v3"
        ] = False

        row[
            "test_fields_available_during_development"
        ] = False

        output.append(row)

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
                    "H4R3-1",

                "statement":
                    (
                        "Volume-preserving diffusion-threshold "
                        "coarsening will satisfy both characteristic-"
                        "length and interface-density requirements "
                        "across development carriers."
                    ),

                "primary_metrics": [
                    (
                        "Independent mean coarse-to-fine "
                        "interface-density ratio."
                    ),
                    (
                        "Independent mean and worst-pair "
                        "characteristic-length ratios."
                    ),
                ],
            },
            {
                "hypothesis_id":
                    "H4R3-2",

                "statement":
                    (
                        "Decoupling the final coarsening stopping "
                        "rule from carrier base length will eliminate "
                        "the systematic carrier-scale dependence "
                        "observed in Tier C v3."
                    ),

                "primary_metric":
                    (
                        "Spearman association between carrier "
                        "base length and interface-density ratio."
                    ),
            },
            {
                "hypothesis_id":
                    "H4R3-3",

                "statement":
                    (
                        "The final Tier C v4 morphology will satisfy "
                        "the frozen eight-state and coarsening-factor "
                        "validation-accuracy thresholds."
                    ),

                "primary_metrics": [
                    (
                        "Three-seed mean and worst-seed "
                        "eight-state validation accuracy."
                    ),
                    (
                        "Three-seed mean and worst-seed "
                        "coarsening-factor validation accuracy."
                    ),
                ],
            },
            {
                "hypothesis_id":
                    "H4R3-4",

                "statement":
                    (
                        "Exact marginal-value preservation will keep "
                        "global-statistics state decoding below the "
                        "frozen shortcut threshold."
                    ),

                "primary_metric":
                    (
                        "Validation accuracy of the frozen "
                        "28-feature global-statistics diagnostic."
                    ),
            },
            {
                "hypothesis_id":
                    "H4R3-5",

                "statement":
                    (
                        "The final spatial fields will remain "
                        "nontrivial under the frozen PCA-128 affine "
                        "shortcut diagnostic."
                    ),

                "primary_metric":
                    (
                        "Validation affine continuous MSE and "
                        "exact structural recovery."
                    ),
            },
        ],
    }


def create_phase_plan():
    return {
        "4A-R3":
            (
                "Freeze final Tier C v4 construction, carrier "
                "split, acceptance gates, and terminal rule."
            ),

        "4B-R3":
            (
                "Generate fresh training and validation fields, "
                "independently verify morphology, and materialize "
                "four development cells."
            ),

        "4C-R3":
            (
                "Run pre-decoder gates; only after they pass, fit "
                "the three fresh frozen diagnostic decoders and "
                "run shortcut and integrity audits."
            ),

        "4D-R3":
            (
                "Run spatial-model implementation smoke tests "
                "using development data only."
            ),

        "4E-R3":
            (
                "Execute the 43 frozen tuning configurations."
            ),

        "4F-R3":
            (
                "Execute and freeze all 130 final model fits "
                "without generating test artifacts."
            ),

        "4G-R3":
            (
                "Generate the sealed test fields and manifests, "
                "then perform final evaluation exactly once."
            ),

        "4H-R3":
            (
                "Freeze the final Tier C claims and synthesize "
                "all positive and negative Tier AC results."
            ),
    }


def write_input_hashes():
    paths = {
        "phase3h_summary":
            PHASE3H_DIR
            / "phase3h_summary.json",

        "tier_c_v2_protocol":
            PHASE4AR_V2_DIR
            / "phase4ar_summary.json",

        "tier_c_v2_rejection":
            PHASE4CR_V2_DIR
            / "phase4cr_rejection_summary.json",

        "tier_c_v3_protocol":
            PHASE4AR2_V3_DIR
            / "phase4ar2_summary.json",

        "tier_c_v3_dataset":
            PHASE4BR2_V3_DIR
            / "phase4br2_summary.json",

        "tier_c_v3_rejection":
            PHASE4CR2_V3_DIR
            / "phase4cr2_summary.json",

        "tier_c_v3_rejection_record":
            PHASE4CR2_V3_DIR
            / "phase4cr2_rejection_record.json",

        "tier_c_v3_diagnosis":
            PHASE4CR2D_V3_DIR
            / "phase4cr2d_summary.json",

        "tier_c_v3_failure_pattern":
            PHASE4CR2D_V3_DIR
            / "failure_pattern_diagnosis.json",

        "transformation_split":
            PHASE1D_DIR
            / "transformation_split.csv",
    }

    hashes = {}

    for name, path in paths.items():
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

    prior_results = validate_prior_results()

    (
        carrier_split_rows,
        split_overlap_summary,
    ) = create_v4_carrier_split()

    transformation_rows = (
        validate_transformation_split()
    )

    carrier_parameter_schema = (
        create_carrier_parameter_schema()
    )

    state_regimes = (
        create_state_regimes()
    )

    coarsening_protocol = (
        create_coarsening_protocol()
    )

    acceptance_gates = (
        create_acceptance_gates()
    )

    test_sealing_policy = (
        create_test_sealing_policy()
    )

    terminal_revision_policy = (
        create_terminal_revision_policy()
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
        / "carrier_split_v4.csv",
        carrier_split_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "carrier_parameter_schema_v4.csv",
        carrier_parameter_schema,
    )

    write_csv(
        OUTPUT_DIR
        / "state_morphology_regimes_v4.csv",
        state_regimes,
    )

    write_csv(
        OUTPUT_DIR
        / "model_family_registry_v4.csv",
        model_registry,
    )

    write_json(
        OUTPUT_DIR
        / "coarsening_protocol_v4.json",
        coarsening_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "acceptance_gates_v4.json",
        acceptance_gates,
    )

    write_json(
        OUTPUT_DIR
        / "test_sealing_policy.json",
        test_sealing_policy,
    )

    write_json(
        OUTPUT_DIR
        / "terminal_revision_policy.json",
        terminal_revision_policy,
    )

    write_json(
        OUTPUT_DIR
        / "training_protocol_v4.json",
        training_protocol,
    )

    write_json(
        OUTPUT_DIR
        / "hypothesis_registry_v4.json",
        hypothesis_registry,
    )

    write_json(
        OUTPUT_DIR
        / "phase4_v4_plan.json",
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
                "4A-R3 Tier C v4 final "
                "revision protocol freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "scientific_scope":
            (
                "Final controlled revision of the "
                "dimensionless physics-informed synthetic "
                "spatial bridge. This is not real microscopy "
                "and is not a calibrated alloy model."
            ),

        "revision_reason": {
            "tier_c_v3_interface_failure_systematic":
                True,

            "v3_interface_gate_failure_fraction":
                prior_results[
                    "v3_diagnosis"
                ][
                    "overall_interface_gate_failure_fraction"
                ],

            "v3_fraction_carriers_all_pairs_failing":
                prior_results[
                    "v3_diagnosis"
                ][
                    "fraction_carriers_with_all_four_pairs_failing"
                ],

            "v3_length_interface_spearman":
                prior_results[
                    "v3_diagnosis"
                ][
                    "spearman_measured_length_ratio_vs_interface_ratio"
                ],

            "targeted_change":
                (
                    "Replace spectral-band rank transport "
                    "with volume-preserving diffusion-threshold "
                    "coarsening."
                ),
        },

        "carrier_split": {
            "counts":
                carrier_counts,

            "validation_source":
                "unopened Tier C v2 test IDs",

            "test_source":
                "unopened Tier C v3 test IDs",

            "overlap_summary":
                split_overlap_summary,
        },

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "carrier_parameter_dimension":
            PHYSICAL_CARRIER_DIMENSION,

        "transformation_split_counts":
            transformation_counts,

        "coarsening_protocol":
            coarsening_protocol,

        "acceptance_gates":
            acceptance_gates,

        "test_sealing_policy":
            test_sealing_policy,

        "terminal_revision_policy":
            terminal_revision_policy,

        "training_protocol":
            training_protocol,

        "hypothesis_registry":
            hypothesis_registry,

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

        "diagnostic_decoder_fitted":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "prior_outputs_modified":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "tier_c_v4_protocol.json",
        protocol,
    )

    summary = {
        "phase":
            (
                "4A-R3 Tier C v4 final "
                "revision protocol freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "tier_c_v2_rejection_frozen":
            True,

        "tier_c_v3_rejection_frozen":
            True,

        "tier_c_v3_diagnosis_passed":
            True,

        "revision_scope":
            "coarsening construction only",

        "coarsening_construction":
            (
                "volume-preserving diffusion-threshold "
                "with exact rank transport"
            ),

        "carrier_base_length_controls_final_coarsening":
            False,

        "internal_target_interface_ratio":
            INTERNAL_TARGET_INTERFACE_RATIO,

        "internal_target_length_ratio":
            INTERNAL_TARGET_LENGTH_RATIO,

        "minimum_coarse_iterations":
            MINIMUM_COARSE_ITERATIONS,

        "maximum_coarse_iterations":
            MAXIMUM_COARSE_ITERATIONS,

        "acceptance_maximum_mean_interface_ratio":
            ACCEPTANCE_MAXIMUM_MEAN_INTERFACE_RATIO,

        "acceptance_maximum_pair_interface_ratio":
            ACCEPTANCE_MAXIMUM_PAIR_INTERFACE_RATIO,

        "acceptance_minimum_pair_pass_fraction":
            ACCEPTANCE_MINIMUM_PAIR_PASS_FRACTION,

        "acceptance_minimum_mean_length_ratio":
            ACCEPTANCE_MINIMUM_MEAN_LENGTH_RATIO,

        "acceptance_minimum_worst_length_ratio":
            ACCEPTANCE_MINIMUM_WORST_LENGTH_RATIO,

        "carrier_count":
            CARRIER_COUNT,

        "carrier_split_counts":
            carrier_counts,

        "validation_carrier_source":
            "unopened Tier C v2 test IDs",

        "test_carrier_source":
            "unopened Tier C v3 test IDs",

        "validation_test_overlap_count":
            0,

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

        "minimum_mean_validation_state_accuracy":
            MINIMUM_MEAN_VALIDATION_STATE_ACCURACY,

        "minimum_worst_validation_state_accuracy":
            MINIMUM_WORST_VALIDATION_STATE_ACCURACY,

        "minimum_mean_validation_coarsening_accuracy":
            MINIMUM_MEAN_VALIDATION_COARSENING_ACCURACY,

        "minimum_worst_validation_coarsening_accuracy":
            MINIMUM_WORST_VALIDATION_COARSENING_ACCURACY,

        "tuning_configuration_count":
            training_protocol[
                "total_tuning_configuration_count"
            ],

        "final_fit_count":
            training_protocol[
                "total_final_fit_count"
            ],

        "tier_c_v4_is_final_revision":
            True,

        "tier_c_v5_authorized":
            False,

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

        "phase4ar3_status":
            "protocol_frozen",
    }

    write_json(
        OUTPUT_DIR
        / "phase4ar3_summary.json",
        summary,
    )

    print(
        "Phase 4A-R3 Tier C v4 final "
        "revision protocol frozen."
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
