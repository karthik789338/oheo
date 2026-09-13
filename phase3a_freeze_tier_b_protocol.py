from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PHASE1D_DIR = Path(
    "outputs/phase1d_compositional_splits"
)

PHASE2L_DIR = Path(
    "outputs/phase2l_phase2_synthesis"
)

OUTPUT_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)


STATE_COUNT = 8
PRIMITIVE_OPERATION_COUNT = 6
SEMIGROUP_SIZE = 104
INFORMATION_CLASS_COUNT = 9

CARRIER_DIMENSION = 6
OBSERVATION_DIMENSION = 32
HIDDEN_DIMENSION = 96

CARRIER_COUNT = 160
CARRIER_SPLIT_COUNTS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

CARRIER_SPLIT_SEED = 31027
OBSERVATION_GENERATOR_SEED = 31028
NOISE_GENERATOR_SEED = 31029

FROZEN_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

FROZEN_NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE = 0.25

EXPECTED_TRANSFORMATION_SPLIT_COUNTS = {
    "train": 86,
    "val": 9,
    "test": 9,
}

EXPECTED_SEQUENCE_SPLIT_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

TRAJECTORIES_PER_SEQUENCE = 8

OCM_TUNING_CONFIGURATION_COUNT = 24

BASELINE_TUNING_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


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

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
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


def file_sha256(
    path: Path,
):
    digest = hashlib.sha256()

    with path.open(
        "rb",
    ) as handle:
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


def validate_phase2_freeze():
    summary = load_json(
        PHASE2L_DIR
        / "phase2l_summary.json"
    )

    if (
        summary[
            "phase2l_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 2L has not passed."
        )

    if (
        summary[
            "phase2_results_frozen"
        ]
        is not True
    ):
        raise AssertionError(
            "Phase 2 is not frozen."
        )

    if (
        summary[
            "affine_shortcut_confirmed"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier A affine shortcut was not confirmed."
        )

    if (
        summary[
            "training_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Unexpected Phase 2L training flag."
        )

    return summary


def validate_transformation_split():
    rows = load_csv(
        PHASE1D_DIR
        / "transformation_split.csv"
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for row in rows:
        split = row[
            "split"
        ]

        if split not in counts:
            raise AssertionError(
                f"Unexpected transformation split: {split}"
            )

        counts[
            split
        ] += 1

    if (
        counts
        != EXPECTED_TRANSFORMATION_SPLIT_COUNTS
    ):
        raise AssertionError(
            "Frozen transformation split changed: "
            f"{counts}"
        )

    if len(
        rows
    ) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Semigroup size changed."
        )

    return counts


def create_carrier_split():
    """
    Freeze carrier identities and their train/validation/test split.

    The actual continuous carrier vectors are generated in Phase 3B.
    """

    generator = np.random.default_rng(
        CARRIER_SPLIT_SEED
    )

    carrier_ids = np.arange(
        CARRIER_COUNT,
        dtype=np.int64,
    )

    generator.shuffle(
        carrier_ids
    )

    rows = []

    cursor = 0

    for split in (
        "train",
        "val",
        "test",
    ):
        count = CARRIER_SPLIT_COUNTS[
            split
        ]

        selected = carrier_ids[
            cursor:
            cursor + count
        ]

        for position, carrier_id in enumerate(
            selected
        ):
            rows.append(
                {
                    "carrier_id":
                        int(
                            carrier_id
                        ),

                    "split":
                        split,

                    "split_position":
                        position,

                    "carrier_dimension":
                        CARRIER_DIMENSION,

                    "carrier_vector_generated":
                        False,
                }
            )

        cursor += count

    if cursor != CARRIER_COUNT:
        raise AssertionError(
            "Carrier split does not cover all carriers."
        )

    observed_counts = {
        split:
            sum(
                row[
                    "split"
                ]
                == split
                for row in rows
            )
        for split in CARRIER_SPLIT_COUNTS
    }

    if observed_counts != CARRIER_SPLIT_COUNTS:
        raise AssertionError(
            "Carrier split counts changed."
        )

    if len(
        {
            row[
                "carrier_id"
            ]
            for row in rows
        }
    ) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier IDs are not unique."
        )

    return rows


def create_evaluation_cells():
    """
    Freeze the factorial composition/carrier evaluation design.

    train_probe denotes a deterministic 144-sequence subset selected
    from the 1,376 training sequences in Phase 3B.
    """

    cells = [
        {
            "cell_id":
                "train_joint",

            "sequence_source":
                "train",

            "carrier_split":
                "train",

            "sequence_count":
                1376,

            "trajectory_count":
                1376
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "training",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Training compositions paired with training carriers."
            ),
        },

        {
            "cell_id":
                "val_composition",

            "sequence_source":
                "val",

            "carrier_split":
                "train",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "validation_diagnostic",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Held-out validation compositions with seen carriers."
            ),
        },

        {
            "cell_id":
                "val_carrier",

            "sequence_source":
                "train_probe",

            "carrier_split":
                "val",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "validation_diagnostic",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Seen transformations with held-out validation carriers."
            ),
        },

        {
            "cell_id":
                "val_joint",

            "sequence_source":
                "val",

            "carrier_split":
                "val",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "validation_selection",

            "checkpoint_selection_allowed":
                True,

            "description": (
                "Joint held-out validation compositions and carriers."
            ),
        },

        {
            "cell_id":
                "test_iid_pairing",

            "sequence_source":
                "train_probe",

            "carrier_split":
                "train",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "final_test",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Seen transformation and carrier distributions with "
                "new sequence-carrier pairings."
            ),
        },

        {
            "cell_id":
                "test_composition",

            "sequence_source":
                "test",

            "carrier_split":
                "train",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "final_test",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Held-out test compositions with seen carriers."
            ),
        },

        {
            "cell_id":
                "test_carrier",

            "sequence_source":
                "train_probe",

            "carrier_split":
                "test",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "final_test",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Seen transformations with unseen test carriers."
            ),
        },

        {
            "cell_id":
                "test_joint",

            "sequence_source":
                "test",

            "carrier_split":
                "test",

            "sequence_count":
                144,

            "trajectory_count":
                144
                * TRAJECTORIES_PER_SEQUENCE,

            "role":
                "final_test",

            "checkpoint_selection_allowed":
                False,

            "description": (
                "Joint unseen test compositions and unseen carriers."
            ),
        },
    ]

    selection_cells = [
        row[
            "cell_id"
        ]
        for row in cells
        if row[
            "checkpoint_selection_allowed"
        ]
    ]

    if selection_cells != [
        "val_joint"
    ]:
        raise AssertionError(
            "Exactly val_joint must control checkpoint selection."
        )

    return cells


def create_final_fit_matrix():
    rows = []

    # OCM final runs.
    for seed in FROZEN_SEEDS:
        for noise in FROZEN_NOISE_LEVELS:
            rows.append(
                {
                    "model_id":
                        "OCM",

                    "seed":
                        seed,

                    "noise_fraction":
                        noise,

                    "fit_type":
                        "neural",
                }
            )

    # B1 is deterministic.
    for noise in FROZEN_NOISE_LEVELS:
        rows.append(
            {
                "model_id":
                    "B1",

                "seed":
                    "deterministic",

                "noise_fraction":
                    noise,

                "fit_type":
                    "deterministic",
            }
        )

    # Neural baselines.
    for model_id in (
        "B2",
        "B3",
        "B4",
        "B5",
    ):
        for seed in FROZEN_SEEDS:
            for noise in FROZEN_NOISE_LEVELS:
                rows.append(
                    {
                        "model_id":
                            model_id,

                        "seed":
                            seed,

                        "noise_fraction":
                            noise,

                        "fit_type":
                            "neural",
                    }
                )

    if len(
        rows
    ) != 130:
        raise AssertionError(
            "Expected 130 final Tier B fits."
        )

    deterministic_count = sum(
        row[
            "fit_type"
        ]
        == "deterministic"
        for row in rows
    )

    neural_count = sum(
        row[
            "fit_type"
        ]
        == "neural"
        for row in rows
    )

    if (
        deterministic_count != 5
        or neural_count != 125
    ):
        raise AssertionError(
            "Tier B final-fit counts changed."
        )

    return rows


def build_observation_protocol():
    return {
        "tier":
            "Tier B distributed nonlinear observation manifold",

        "state_count":
            STATE_COUNT,

        "primitive_operation_count":
            PRIMITIVE_OPERATION_COUNT,

        "semigroup_size":
            SEMIGROUP_SIZE,

        "information_class_count":
            INFORMATION_CLASS_COUNT,

        "carrier": {
            "dimension":
                CARRIER_DIMENSION,

            "count":
                CARRIER_COUNT,

            "split_counts":
                CARRIER_SPLIT_COUNTS,

            "distribution": (
                "Independent coordinates sampled uniformly from "
                "[-1, 1], followed by deterministic low-discrepancy "
                "coverage auditing."
            ),

            "trajectory_rule": (
                "The carrier is fixed throughout one process history."
            ),

            "interpretation": (
                "Continuous material-instance variation such as "
                "composition, morphology, texture, and defect context."
            ),
        },

        "observation": {
            "dimension":
                OBSERVATION_DIMENSION,

            "hidden_dimension":
                HIDDEN_DIMENSION,

            "generator_seed":
                OBSERVATION_GENERATOR_SEED,

            "definition":
                "x_clean = g(state, carrier)",

            "decoder_family": (
                "Frozen nonlinear random-feature decoder with "
                "state embeddings, polynomial carrier features, "
                "trigonometric carrier features, pairwise carrier "
                "interactions, and state-carrier interaction terms."
            ),

            "descriptor_blocks": {
                "phase_like":
                    [
                        0,
                        7,
                    ],

                "texture_like":
                    [
                        8,
                        15,
                    ],

                "morphology_like":
                    [
                        16,
                        23,
                    ],

                "defect_topology_like":
                    [
                        24,
                        31,
                    ],
            },

            "normalization": (
                "Per-coordinate mean and standard deviation estimated "
                "only from clean training-carrier observations."
            ),

            "single_state_prototype_used":
                False,

            "state_distribution_used":
                True,
        },

        "noise": {
            "levels":
                list(
                    FROZEN_NOISE_LEVELS
                ),

            "generator_seed":
                NOISE_GENERATOR_SEED,

            "definition": (
                "x_noisy = x_clean + sigma * epsilon, "
                "where epsilon is independent standard Gaussian noise "
                "after train-only coordinate normalization."
            ),

            "independent_across_time":
                True,

            "independent_across_trajectories":
                True,
        },

        "latent_dynamics": {
            "discrete_state_transition": (
                "Exactly the frozen Phase 1 primitive transformation."
            ),

            "carrier_transition":
                "identity",

            "operation_sequence_rule": (
                "Primitive operations modify the discrete history state "
                "while preserving the trajectory carrier."
            ),
        },
    }


def build_training_protocol():
    total_baseline_tuning = sum(
        BASELINE_TUNING_CONFIGURATION_COUNTS.values()
    )

    return {
        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE,

        "checkpoint_selection_cell":
            "val_joint",

        "selection_metric":
            "val_joint noisy-target final rollout MSE only",

        "clean_targets_used_for_training":
            False,

        "clean_targets_used_for_selection":
            False,

        "structural_labels_used_for_training":
            False,

        "test_cells_opened_during_tuning":
            False,

        "models": [
            "OCM",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
        ],

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_counts":
            BASELINE_TUNING_CONFIGURATION_COUNTS,

        "total_tuning_configuration_count":
            (
                OCM_TUNING_CONFIGURATION_COUNT
                + total_baseline_tuning
            ),

        "final_fit_count":
            130,

        "final_deterministic_fit_count":
            5,

        "final_neural_fit_count":
            125,

        "model_architecture_policy": (
            "Retain the Phase 2 model families but resize input and "
            "output layers for 32-dimensional Tier B observations. "
            "All architectures are retuned using the frozen Tier B "
            "validation protocol."
        ),
    }


def build_required_audits():
    return {
        "carrier_audits": [
            "All carrier IDs are unique.",
            "Train, validation, and test carrier sets are disjoint.",
            "Carrier-coordinate coverage is reported by split.",
            "No exact carrier vector occurs in more than one split.",
        ],

        "observation_audits": [
            "All generated observations are finite.",
            "Every state has nonzero within-state carrier variation.",
            "All 32 observation coordinates have nonzero train variance.",
            "Normalization statistics use training carriers only.",
            "The same carrier produces different observations across states.",
        ],

        "affine_shortcut_audits": [
            (
                "Fit one affine observation-space operator per primitive "
                "operation using training carriers only."
            ),
            (
                "Evaluate those operators on validation and test carriers "
                "without refitting."
            ),
            (
                "Report one-step MSE, rollout MSE, exact transformation "
                "rate, partition accuracy, and relation accuracy."
            ),
            (
                "Tier B shortcut removal is confirmed only if exact "
                "held-out-carrier interpolation fails above numerical "
                "precision."
            ),
        ],

        "privileged_reference_audits": [
            (
                "Report clean state classification using a privileged "
                "reference decoder trained only on training carriers."
            ),
            (
                "Report classification separately for train, validation, "
                "and test carriers."
            ),
            (
                "The privileged decoder is diagnostic and may not be "
                "used by predictive models."
            ),
        ],

        "split_audits": [
            "The frozen 86/9/9 transformation split remains unchanged.",
            "Only val_joint controls checkpoint selection.",
            "No final test cell is opened during tuning.",
            (
                "Report composition-only, carrier-only, joint-OOD, "
                "and IID-pairing results separately."
            ),
        ],
    }


def build_claim_gates():
    return {
        "affine_shortcut_removed": {
            "required_for_tier_b_acceptance":
                True,

            "criteria": [
                (
                    "Held-out-carrier affine prediction error exceeds "
                    "numerical interpolation tolerance."
                ),
                (
                    "At least one primitive operation is not recovered "
                    "exactly on every held-out carrier."
                ),
                (
                    "The affine baseline cannot recover all 104 "
                    "transformations exactly on joint-OOD carriers."
                ),
            ],

            "numerical_interpolation_tolerance":
                1e-10,
        },

        "state_manifold_valid": {
            "required_for_tier_b_acceptance":
                True,

            "criteria": [
                "Every state has positive within-state variance.",
                "No observation coordinate is constant on training data.",
                (
                    "A privileged reference decoder performs above "
                    "chance on clean validation and test carriers."
                ),
            ],

            "chance_accuracy":
                1.0
                / STATE_COUNT,
        },

        "rq_claim_policy": {
            "prediction_superiority": (
                "No model-superiority claim may be made until all "
                "frozen Tier B baselines complete."
            ),

            "structural_identifiability": (
                "Prediction, behavioral recovery, and internal channel "
                "recovery must be reported separately."
            ),

            "causal_language": (
                "Causal interpretation is limited to the controlled "
                "interventional simulator."
            ),
        },
    }


def save_input_hashes():
    inputs = {
        "phase2l_summary":
            PHASE2L_DIR
            / "phase2l_summary.json",

        "phase2_claim_registry":
            PHASE2L_DIR
            / "phase2_claim_registry.json",

        "phase2_do_not_claim":
            PHASE2L_DIR
            / "phase2_do_not_claim.json",

        "phase1d_transformation_split":
            PHASE1D_DIR
            / "transformation_split.csv",
    }

    hashes = {}

    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        hashes[
            name
        ] = {
            "path":
                str(
                    path
                ),

            "sha256":
                file_sha256(
                    path
                ),
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

    phase2_summary = (
        validate_phase2_freeze()
    )

    transformation_counts = (
        validate_transformation_split()
    )

    carrier_rows = (
        create_carrier_split()
    )

    evaluation_cells = (
        create_evaluation_cells()
    )

    final_fit_matrix = (
        create_final_fit_matrix()
    )

    observation_protocol = (
        build_observation_protocol()
    )

    training_protocol = (
        build_training_protocol()
    )

    required_audits = (
        build_required_audits()
    )

    claim_gates = (
        build_claim_gates()
    )

    save_input_hashes()

    write_csv(
        OUTPUT_DIR
        / "carrier_split.csv",
        carrier_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "evaluation_cells.csv",
        evaluation_cells,
    )

    write_csv(
        OUTPUT_DIR
        / "final_fit_matrix.csv",
        final_fit_matrix,
    )

    protocol = {
        "phase":
            "3A Tier B protocol freeze",

        "motivation": (
            "Remove Tier A's exact affine interpolation shortcut and "
            "test whether predictive and structural conclusions survive "
            "continuous within-state material-instance variation."
        ),

        "phase2_results_modified":
            False,

        "phase2_claims_inherited":
            True,

        "observation_protocol":
            observation_protocol,

        "training_protocol":
            training_protocol,

        "evaluation_cells": {
            row[
                "cell_id"
            ]:
                row
            for row in evaluation_cells
        },

        "required_audits":
            required_audits,

        "claim_gates":
            claim_gates,
    }

    write_json(
        OUTPUT_DIR
        / "tier_b_protocol.json",
        protocol,
    )

    phase_plan = {
        "3A": (
            "Freeze Tier B protocol, carrier split, evaluation cells, "
            "and run counts."
        ),

        "3B": (
            "Generate nonlinear carrier manifolds and run data-quality, "
            "state-separability, leakage, and affine-shortcut audits."
        ),

        "3C": (
            "Run OCM and baseline smoke tests on Tier B without opening "
            "final test cells."
        ),

        "3D": (
            "Perform leakage-safe Tier B tuning using val_joint only."
        ),

        "3E": (
            "Run all 130 frozen final fits across five noise levels."
        ),

        "3F": (
            "Evaluate prediction across IID-pairing, composition-only, "
            "carrier-only, and joint-OOD cells."
        ),

        "3G": (
            "Evaluate behavioral and internal structural recovery."
        ),

        "3H": (
            "Freeze Tier B claims and compare Tier A with Tier B."
        ),
    }

    write_json(
        OUTPUT_DIR
        / "phase3_plan.json",
        phase_plan,
    )

    total_tuning_count = (
        OCM_TUNING_CONFIGURATION_COUNT
        + sum(
            BASELINE_TUNING_CONFIGURATION_COUNTS.values()
        )
    )

    summary = {
        "phase":
            "3A Tier B protocol freeze",

        "phase2l_status":
            phase2_summary[
                "phase2l_status"
            ],

        "phase2_results_frozen":
            phase2_summary[
                "phase2_results_frozen"
            ],

        "state_count":
            STATE_COUNT,

        "carrier_dimension":
            CARRIER_DIMENSION,

        "carrier_count":
            CARRIER_COUNT,

        "carrier_split_counts":
            CARRIER_SPLIT_COUNTS,

        "observation_dimension":
            OBSERVATION_DIMENSION,

        "transformation_split_counts":
            transformation_counts,

        "evaluation_cell_count":
            len(
                evaluation_cells
            ),

        "checkpoint_selection_cell":
            "val_joint",

        "tuning_configuration_count":
            total_tuning_count,

        "ocm_tuning_configuration_count":
            OCM_TUNING_CONFIGURATION_COUNT,

        "baseline_tuning_configuration_count":
            sum(
                BASELINE_TUNING_CONFIGURATION_COUNTS.values()
            ),

        "final_fit_count":
            len(
                final_fit_matrix
            ),

        "final_deterministic_fit_count":
            5,

        "final_neural_fit_count":
            125,

        "affine_shortcut_removal_required":
            True,

        "training_performed":
            False,

        "observations_generated":
            False,

        "test_cells_opened":
            False,

        "phase2_outputs_modified":
            False,

        "status":
            "protocol_frozen",

        "sanity_checks":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase3a_summary.json",
        summary,
    )

    print(
        "Phase 3A Tier B protocol frozen successfully."
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
