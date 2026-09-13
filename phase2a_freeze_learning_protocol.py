from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PHASE1A_DIR = Path("outputs/phase1a_ground_truth")
PHASE1C_DIR = Path("outputs/phase1c_continuous_observations")
PHASE1D_DIR = Path("outputs/phase1d_compositional_splits")
PHASE1F_DIR = Path("outputs/phase1f_final_audit")

OUTPUT_DIR = Path("outputs/phase2a_learning_protocol")


EXPECTED_OOD_BALANCED_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

EXPECTED_IID_BALANCED_COUNTS = {
    "train": 1248,
    "val": 208,
    "test": 208,
}

EXPECTED_SEQUENCE_COUNT = 3144
EXPECTED_TRAJECTORY_COUNT = 25152

PRIMARY_NOISE_FRACTION = 0.25

FINAL_RANDOM_SEEDS = [
    11,
    23,
    37,
    53,
    71,
]


def parse_bool(value: str) -> bool:
    value = value.strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise ValueError(
        f"Cannot parse boolean value: {value}"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def save_input_hashes() -> None:
    """
    Record the exact frozen Phase 1 artifacts used by Phase 2.
    """

    files = {
        "phase1a_generators":
            PHASE1A_DIR / "generators.json",

        "phase1a_relations":
            PHASE1A_DIR / "blackwell_relations.csv",

        "phase1a_summary":
            PHASE1A_DIR / "phase1a_summary.json",

        "phase1c_observations":
            PHASE1C_DIR / "continuous_observations.npz",

        "phase1c_sequence_index":
            PHASE1C_DIR / "sequence_index.csv",

        "phase1c_trajectory_index":
            PHASE1C_DIR / "trajectory_index.csv",

        "phase1c_summary":
            PHASE1C_DIR / "phase1c_summary.json",

        "phase1d_composition_split":
            PHASE1D_DIR / "composition_sequence_split.csv",

        "phase1d_iid_split":
            PHASE1D_DIR / "iid_balanced_sequence_split.csv",

        "phase1d_summary":
            PHASE1D_DIR / "phase1d_summary.json",

        "phase1f_summary":
            PHASE1F_DIR / "phase1f_summary.json",
    }

    hashes = {}

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen input: {path}"
            )

        hashes[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    output_path = (
        OUTPUT_DIR
        / "phase1_input_hashes.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            hashes,
            handle,
            indent=2,
        )


def load_generators():
    rows = load_json(
        PHASE1A_DIR
        / "generators.json"
    )

    return [
        row["name"]
        for row in rows
    ]


def load_phase1c_sequence_index():
    rows = load_csv(
        PHASE1C_DIR
        / "sequence_index.csv"
    )

    for row in rows:

        row["sequence_index"] = int(
            row["sequence_index"]
        )

        row["sequence_length"] = int(
            row["sequence_length"]
        )

    rows.sort(
        key=lambda row:
            row["sequence_index"]
    )

    return rows


def load_phase1c_trajectory_index():
    rows = load_csv(
        PHASE1C_DIR
        / "trajectory_index.csv"
    )

    for row in rows:

        row["trajectory_index"] = int(
            row["trajectory_index"]
        )

        row["sequence_index"] = int(
            row["sequence_index"]
        )

        row["point_start"] = int(
            row["point_start"]
        )

        row["point_count"] = int(
            row["point_count"]
        )

    rows.sort(
        key=lambda row:
            row["trajectory_index"]
    )

    return rows


def build_visible_trajectory_index(
    trajectory_rows,
):
    """
    Strip symbolic state labels from the Phase 1C trajectory table.

    The original table contains initial_state and final_state.
    Phase 2 optimization code must never need those fields.
    """

    visible_rows = []

    for row in trajectory_rows:

        visible_rows.append(
            {
                "trajectory_index":
                    row["trajectory_index"],

                "trajectory_id":
                    row["trajectory_id"],

                "sequence_index":
                    row["sequence_index"],

                "sequence_id":
                    row["sequence_id"],

                "point_start":
                    row["point_start"],

                "point_count":
                    row["point_count"],
            }
        )

    return visible_rows


def write_visible_trajectory_index(
    rows,
):
    path = (
        OUTPUT_DIR
        / "visible_trajectory_index.csv"
    )

    fieldnames = [
        "trajectory_index",
        "trajectory_id",
        "sequence_index",
        "sequence_id",
        "point_start",
        "point_count",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_ood_balanced_manifest(
    sequence_index_rows,
):
    split_rows = load_csv(
        PHASE1D_DIR
        / "composition_sequence_split.csv"
    )

    split_lookup = {}

    for row in split_rows:

        if not parse_bool(
            row["balanced_variant"]
        ):
            continue

        split_lookup[
            row["sequence_id"]
        ] = row["composition_split"]

    output_rows = []

    for row in sequence_index_rows:

        sequence_id = row["sequence_id"]

        if sequence_id not in split_lookup:
            continue

        output_rows.append(
            {
                "sequence_index":
                    row["sequence_index"],

                "sequence_id":
                    sequence_id,

                "split":
                    split_lookup[
                        sequence_id
                    ],

                "operation_sequence":
                    row[
                        "operation_sequence"
                    ],

                "sequence_length":
                    row[
                        "sequence_length"
                    ],
            }
        )

    return output_rows


def build_iid_balanced_manifest(
    sequence_index_rows,
):
    iid_rows = load_csv(
        PHASE1D_DIR
        / "iid_balanced_sequence_split.csv"
    )

    iid_lookup = {
        row["sequence_id"]:
            row["iid_split"]

        for row in iid_rows
    }

    output_rows = []

    for row in sequence_index_rows:

        sequence_id = row["sequence_id"]

        if sequence_id not in iid_lookup:
            continue

        output_rows.append(
            {
                "sequence_index":
                    row["sequence_index"],

                "sequence_id":
                    sequence_id,

                "split":
                    iid_lookup[
                        sequence_id
                    ],

                "operation_sequence":
                    row[
                        "operation_sequence"
                    ],

                "sequence_length":
                    row[
                        "sequence_length"
                    ],
            }
        )

    return output_rows


def write_manifest(
    filename,
    rows,
):
    path = OUTPUT_DIR / filename

    fieldnames = [
        "sequence_index",
        "sequence_id",
        "split",
        "operation_sequence",
        "sequence_length",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def build_evaluation_only_labels():
    """
    Keep structural ground truth in a clearly separated file.

    Training code must never read this file.
    """

    rows = load_csv(
        PHASE1C_DIR
        / "sequence_index.csv"
    )

    output_rows = []

    for row in rows:

        output_rows.append(
            {
                "sequence_index":
                    row["sequence_index"],

                "sequence_id":
                    row["sequence_id"],

                "result_element_id":
                    row["result_element_id"],

                "information_class":
                    row["information_class"],

                "rank":
                    row["rank"],
            }
        )

    return output_rows


def write_evaluation_only_labels(
    rows,
):
    path = (
        OUTPUT_DIR
        / "evaluation_only_sequence_labels.csv"
    )

    fieldnames = [
        "sequence_index",
        "sequence_id",
        "result_element_id",
        "information_class",
        "rank",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_operation_vocabulary(
    generator_names,
):
    vocabulary = {
        name: index
        for index, name
        in enumerate(generator_names)
    }

    with (
        OUTPUT_DIR
        / "operation_vocabulary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            vocabulary,
            handle,
            indent=2,
        )


def write_model_protocol():
    protocol = {
        "working_model_name":
            "Operation Channel Model",

        "abbreviation":
            "OCM",

        "primary_benchmark": {
            "observation_dimension":
                12,

            "latent_state_count":
                8,

            "primary_noise_fraction":
                PRIMARY_NOISE_FRACTION,

            "all_noise_fractions": [
                0.0,
                0.1,
                0.25,
                0.5,
                1.0,
            ],
        },

        "model_interface": {
            "encoder": (
                "q_theta(s|x): continuous observation "
                "to categorical latent distribution"
            ),

            "operation_channels": (
                "One learned row-stochastic KxK "
                "transition matrix per primitive operation"
            ),

            "composition": (
                "Sequential matrix multiplication "
                "of operation channels"
            ),

            "decoder": (
                "Categorical latent distribution "
                "to continuous observation"
            ),
        },

        "primary_architecture": {
            "encoder_hidden_sizes": [
                64,
                64,
            ],

            "encoder_activation":
                "GELU",

            "decoder_type":
                "learned latent codebook",

            "latent_state_count":
                8,

            "operation_channel_parameterization":
                "row-softmax logits",

            "initial_softmax_temperature":
                1.0,

            "final_softmax_temperature":
                0.25,
        },

        "training_visibility": {
            "allowed": [
                "continuous observation vectors",
                "operation labels",
                "sequence order",
                "sequence length",
                "trajectory grouping",
                "train/validation split",
            ],

            "forbidden": [
                "symbolic_state",
                "initial_state",
                "final_state",
                "result_element_id",
                "information_class",
                "rank",
                "true transformation mapping",
                "Blackwell relation labels",
                "Hasse edges",
            ],
        },

        "test_inference": {
            "inputs": [
                "initial continuous observation",
                "operation sequence",
            ],

            "not_visible_during_rollout": [
                "intermediate target observations",
                "final target observation",
                "symbolic states",
            ],

            "scoring_after_prediction": [
                "final target observation",
                "evaluation-only symbolic labels",
            ],
        },

        "losses": {
            "reconstruction": {
                "name":
                    "L_rec",

                "weight":
                    1.0,
            },

            "one_step_observation_prediction": {
                "name":
                    "L_step",

                "weight":
                    1.0,
            },

            "latent_transition_consistency": {
                "name":
                    "L_trans",

                "distance":
                    "Jensen-Shannon divergence",

                "candidate_weights": [
                    0.1,
                    0.25,
                ],
            },

            "full_sequence_rollout": {
                "name":
                    "L_roll",

                "candidate_weights": [
                    0.25,
                    0.5,
                    1.0,
                ],
            },

            "operation_row_entropy": {
                "name":
                    "L_det",

                "candidate_weights": [
                    0.001,
                    0.01,
                ],

                "interpretation": (
                    "Encourages deterministic primitive "
                    "channels without using structural labels"
                ),
            },

            "blackwell_supervision":
                "forbidden",
        },

        "hyperparameter_selection": {
            "development_noise_fraction":
                PRIMARY_NOISE_FRACTION,

            "selection_metric":
                "compositional validation final-observation MSE",

            "structural_metrics_used_for_selection":
                False,

            "test_metrics_used_for_selection":
                False,

            "learning_rates": [
                0.001,
                0.0003,
            ],

            "maximum_epochs":
                200,

            "early_stopping_patience":
                25,

            "early_stopping_min_delta":
                1e-5,
        },

        "final_random_seeds":
            FINAL_RANDOM_SEEDS,

        "primary_structural_estimator": {
            "procedure": [
                "Harden each learned operation channel by row argmax",
                "Compose hardened primitive transformations",
                "Build induced source partitions",
                "Classify ordered pairs as equivalent, more informative, less informative, or incomparable",
            ],

            "soft_blackwell_deficiency":
                "secondary diagnostic only",
        },

        "checkpoint_rule": (
            "Select checkpoints only by validation "
            "observation-space rollout error."
        ),
    }

    path = (
        OUTPUT_DIR
        / "model_protocol.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            protocol,
            handle,
            indent=2,
        )


def write_baseline_registry():
    baselines = {
        "B0_persistence": {
            "name":
                "Persistence",

            "description":
                "Predict final observation as the initial observation.",

            "structured_operation_channels":
                False,
        },

        "B1_linear_operation": {
            "name":
                "Per-operation linear model",

            "description": (
                "One affine continuous-state transformation "
                "for every primitive operation."
            ),

            "structured_operation_channels":
                False,
        },

        "B2_operation_mlp": {
            "name":
                "Operation-conditioned MLP",

            "description": (
                "Predict the next continuous observation from "
                "the current observation and operation embedding."
            ),

            "structured_operation_channels":
                False,
        },

        "B3_gru": {
            "name":
                "GRU sequence predictor",

            "description": (
                "Sequence model over the initial observation "
                "and primitive operation tokens."
            ),

            "structured_operation_channels":
                False,
        },

        "B4_transformer": {
            "name":
                "Transformer sequence predictor",

            "description": (
                "Transformer over operation tokens conditioned "
                "on the initial continuous observation."
            ),

            "structured_operation_channels":
                False,
        },

        "B5_continuous_latent_operator": {
            "name":
                "Continuous latent operator",

            "description": (
                "Encoder and decoder with unrestricted "
                "operation-specific nonlinear latent maps."
            ),

            "structured_operation_channels":
                False,

            "importance": (
                "Main capacity-matched comparison for testing "
                "whether explicit stochastic-channel structure helps."
            ),
        },
    }

    path = (
        OUTPUT_DIR
        / "baseline_registry.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            baselines,
            handle,
            indent=2,
        )


def write_metric_registry():
    metrics = {
        "prediction": [
            "final observation MSE",
            "final clean-state accuracy",
            "rollout error by sequence length",
        ],

        "structural_recovery": [
            "four-way relation accuracy",
            "balanced relation accuracy",
            "macro F1",
            "per-relation precision",
            "per-relation recall",
            "per-relation F1",
            "confusion matrix",
            "majority-class baseline",
        ],

        "transformation_recovery": [
            "primitive mapping accuracy",
            "composite mapping accuracy",
            "exact transformation recovery",
        ],

        "generalization": [
            "compositional OOD performance",
            "IID interpolation performance",
        ],

        "evaluation_only_note": (
            "Symbolic state alignment and ground-truth structural "
            "labels may be used only after model training for scoring."
        ),
    }

    path = (
        OUTPUT_DIR
        / "metric_registry.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metrics,
            handle,
            indent=2,
        )


def split_counts(rows):
    return dict(
        Counter(
            row["split"]
            for row in rows
        )
    )


def run_sanity_checks(
    phase1f_summary,
    sequence_index_rows,
    trajectory_rows,
    ood_rows,
    iid_rows,
    evaluation_rows,
):
    if (
        phase1f_summary[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 1F is not frozen as passed."
        )

    if (
        len(sequence_index_rows)
        != EXPECTED_SEQUENCE_COUNT
    ):
        raise AssertionError(
            "Phase 1 sequence count changed."
        )

    if (
        len(trajectory_rows)
        != EXPECTED_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Phase 1 trajectory count changed."
        )

    ood_counts = split_counts(
        ood_rows
    )

    if (
        ood_counts
        != EXPECTED_OOD_BALANCED_COUNTS
    ):
        raise AssertionError(
            "OOD balanced split changed: "
            f"{ood_counts}"
        )

    iid_counts = split_counts(
        iid_rows
    )

    if (
        iid_counts
        != EXPECTED_IID_BALANCED_COUNTS
    ):
        raise AssertionError(
            "IID balanced split changed: "
            f"{iid_counts}"
        )

    if (
        len(evaluation_rows)
        != EXPECTED_SEQUENCE_COUNT
    ):
        raise AssertionError(
            "Evaluation-label count changed."
        )

    forbidden_fields = {
        "result_element_id",
        "information_class",
        "rank",
        "symbolic_state",
        "initial_state",
        "final_state",
    }

    for row in ood_rows + iid_rows:
        overlap = (
            forbidden_fields
            & set(row)
        )

        if overlap:
            raise AssertionError(
                "Training manifest contains "
                f"forbidden fields: {overlap}"
            )


def write_summary(
    generator_names,
    ood_rows,
    iid_rows,
):
    summary = {
        "phase":
            "2A learning protocol freeze",

        "working_model":
            "Operation Channel Model",

        "primitive_operation_count":
            len(generator_names),

        "primitive_operations":
            generator_names,

        "primary_latent_state_count":
            8,

        "primary_noise_fraction":
            PRIMARY_NOISE_FRACTION,

        "ood_balanced_sequence_counts":
            split_counts(
                ood_rows
            ),

        "iid_balanced_sequence_counts":
            split_counts(
                iid_rows
            ),

        "final_random_seeds":
            FINAL_RANDOM_SEEDS,

        "training_uses_structural_ground_truth":
            False,

        "blackwell_labels_used_in_loss":
            False,

        "checkpoint_uses_structural_metrics":
            False,

        "test_intermediate_observations_visible":
            False,

        "status":
            "protocol_frozen",

        "sanity_checks":
            "passed",
    }

    path = (
        OUTPUT_DIR
        / "phase2a_summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_input_hashes()

    phase1f_summary = load_json(
        PHASE1F_DIR
        / "phase1f_summary.json"
    )

    generator_names = (
        load_generators()
    )

    sequence_index_rows = (
        load_phase1c_sequence_index()
    )

    trajectory_rows = (
        load_phase1c_trajectory_index()
    )

    visible_trajectory_rows = (
        build_visible_trajectory_index(
            trajectory_rows
        )
    )

    write_visible_trajectory_index(
        visible_trajectory_rows
    )

    ood_rows = (
        build_ood_balanced_manifest(
            sequence_index_rows
        )
    )

    write_manifest(
        "ood_balanced_sequence_manifest.csv",
        ood_rows,
    )

    iid_rows = (
        build_iid_balanced_manifest(
            sequence_index_rows
        )
    )

    write_manifest(
        "iid_balanced_sequence_manifest.csv",
        iid_rows,
    )

    evaluation_rows = (
        build_evaluation_only_labels()
    )

    write_evaluation_only_labels(
        evaluation_rows
    )

    write_operation_vocabulary(
        generator_names
    )

    write_model_protocol()
    write_baseline_registry()
    write_metric_registry()

    run_sanity_checks(
        phase1f_summary=phase1f_summary,
        sequence_index_rows=(
            sequence_index_rows
        ),
        trajectory_rows=trajectory_rows,
        ood_rows=ood_rows,
        iid_rows=iid_rows,
        evaluation_rows=evaluation_rows,
    )

    summary = write_summary(
        generator_names=generator_names,
        ood_rows=ood_rows,
        iid_rows=iid_rows,
    )

    print(
        "Phase 2A learning protocol "
        "frozen successfully."
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
