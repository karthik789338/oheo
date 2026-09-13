from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PHASE1A_DIR = Path(
    "outputs/phase1a_ground_truth"
)

PHASE2A_DIR = Path(
    "outputs/phase2a_learning_protocol"
)

PHASE3A_DIR = Path(
    "outputs/phase3a_tier_b_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase3b_tier_b_data"
)

VISIBLE_DIR = OUTPUT_DIR / "visible"
PRIVILEGED_DIR = OUTPUT_DIR / "privileged"


STATE_COUNT = 8
CARRIER_DIMENSION = 6
CARRIER_COUNT = 160
OBSERVATION_DIMENSION = 32
HIDDEN_DIMENSION = 96

CARRIER_GENERATOR_SEED = 31027
OBSERVATION_GENERATOR_SEED = 31028
NOISE_GENERATOR_SEED = 31029
TRAIN_PROBE_SEED = 31030

NOISE_LEVELS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

EXPECTED_SEQUENCE_COUNTS = {
    "train": 1376,
    "val": 144,
    "test": 144,
}

EXPECTED_CARRIER_COUNTS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

EXPECTED_CELL_COUNTS = {
    "train_joint": 1376,
    "val_composition": 144,
    "val_carrier": 144,
    "val_joint": 144,
    "test_iid_pairing": 144,
    "test_composition": 144,
    "test_carrier": 144,
    "test_joint": 144,
}

TRAJECTORIES_PER_SEQUENCE = 8


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
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path: Path, rows) -> None:
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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def stable_seed(label: str) -> int:
    payload = (
        f"{NOISE_GENERATOR_SEED}|{label}"
        .encode("utf-8")
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def validate_phase3a():
    summary = load_json(
        PHASE3A_DIR
        / "phase3a_summary.json"
    )

    if summary["status"] != "protocol_frozen":
        raise AssertionError(
            "Phase 3A is not frozen."
        )

    if summary["sanity_checks"] != "passed":
        raise AssertionError(
            "Phase 3A sanity checks failed."
        )

    if summary["phase2_results_frozen"] is not True:
        raise AssertionError(
            "Phase 2 is not frozen."
        )

    if summary["observations_generated"] is not False:
        raise AssertionError(
            "Phase 3A unexpectedly reports generated observations."
        )

    return summary


def radical_inverse(
    index: int,
    base: int,
) -> float:
    result = 0.0
    factor = 1.0 / base
    current = index

    while current > 0:
        digit = current % base
        result += digit * factor
        current //= base
        factor /= base

    return result


def generate_carrier_vectors():
    carrier_split_rows = load_csv(
        PHASE3A_DIR
        / "carrier_split.csv"
    )

    if len(carrier_split_rows) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier count changed."
        )

    split_counts = {
        split: sum(
            row["split"] == split
            for row in carrier_split_rows
        )
        for split in EXPECTED_CARRIER_COUNTS
    }

    if split_counts != EXPECTED_CARRIER_COUNTS:
        raise AssertionError(
            f"Carrier split changed: {split_counts}"
        )

    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
    )

    generator = np.random.default_rng(
        CARRIER_GENERATOR_SEED
    )

    shift = generator.uniform(
        0.0,
        1.0,
        size=CARRIER_DIMENSION,
    )

    vectors = np.zeros(
        (
            CARRIER_COUNT,
            CARRIER_DIMENSION,
        ),
        dtype=np.float64,
    )

    for carrier_id in range(
        CARRIER_COUNT
    ):
        point = np.asarray(
            [
                radical_inverse(
                    carrier_id + 1,
                    base,
                )
                for base in primes
            ],
            dtype=np.float64,
        )

        point = (
            point + shift
        ) % 1.0

        vectors[
            carrier_id
        ] = 2.0 * point - 1.0

    output_rows = []

    for row in carrier_split_rows:
        carrier_id = int(
            row["carrier_id"]
        )

        output_row = {
            "carrier_id":
                carrier_id,

            "split":
                row["split"],

            "split_position":
                int(row["split_position"]),
        }

        for dimension in range(
            CARRIER_DIMENSION
        ):
            output_row[
                f"u{dimension}"
            ] = float(
                vectors[
                    carrier_id,
                    dimension,
                ]
            )

        output_rows.append(
            output_row
        )

    output_rows.sort(
        key=lambda item:
            item["carrier_id"]
    )

    write_csv(
        OUTPUT_DIR
        / "carriers.csv",
        output_rows,
    )

    return vectors, output_rows


def carrier_features(
    carriers: np.ndarray,
) -> np.ndarray:
    if (
        carriers.ndim != 2
        or carriers.shape[1]
        != CARRIER_DIMENSION
    ):
        raise ValueError(
            "Carrier array has an invalid shape."
        )

    blocks = [
        carriers,
        carriers**2,
        carriers**3,
        np.sin(
            np.pi * carriers
        ),
        np.cos(
            np.pi * carriers
        ),
    ]

    pairwise = []

    for first in range(
        CARRIER_DIMENSION
    ):
        for second in range(
            first + 1,
            CARRIER_DIMENSION,
        ):
            pairwise.append(
                (
                    carriers[:, first]
                    * carriers[:, second]
                )[:, None]
            )

    blocks.append(
        np.concatenate(
            pairwise,
            axis=1,
        )
    )

    features = np.concatenate(
        blocks,
        axis=1,
    )

    if features.shape[1] != 45:
        raise AssertionError(
            "Carrier feature dimension changed."
        )

    return features


def generate_observation_parameters():
    generator = np.random.default_rng(
        OBSERVATION_GENERATOR_SEED
    )

    feature_dimension = 45

    shared_hidden_weight = (
        generator.normal(
            size=(
                feature_dimension,
                HIDDEN_DIMENSION,
            )
        )
        / np.sqrt(
            feature_dimension
        )
    )

    shared_hidden_bias = (
        generator.normal(
            scale=0.25,
            size=HIDDEN_DIMENSION,
        )
    )

    state_hidden_scale = (
        0.75
        + 0.50
        * generator.uniform(
            size=(
                STATE_COUNT,
                HIDDEN_DIMENSION,
            )
        )
    )

    state_hidden_shift = (
        generator.normal(
            scale=0.60,
            size=(
                STATE_COUNT,
                HIDDEN_DIMENSION,
            )
        )
    )

    state_output_weight = (
        generator.normal(
            size=(
                STATE_COUNT,
                HIDDEN_DIMENSION,
                OBSERVATION_DIMENSION,
            )
        )
        / np.sqrt(
            HIDDEN_DIMENSION
        )
    )

    state_output_bias = (
        generator.normal(
            scale=0.20,
            size=(
                STATE_COUNT,
                OBSERVATION_DIMENSION,
            )
        )
    )

    shared_carrier_weight = (
        generator.normal(
            size=(
                feature_dimension,
                OBSERVATION_DIMENSION,
            )
        )
        / np.sqrt(
            feature_dimension
        )
    )

    anchor_source = generator.normal(
        size=(
            OBSERVATION_DIMENSION,
            STATE_COUNT,
        )
    )

    orthogonal, _ = np.linalg.qr(
        anchor_source
    )

    state_anchors = (
        orthogonal[
            :,
            :STATE_COUNT,
        ].T
        * np.sqrt(
            OBSERVATION_DIMENSION
        )
    )

    parameters = {
        "shared_hidden_weight":
            shared_hidden_weight,

        "shared_hidden_bias":
            shared_hidden_bias,

        "state_hidden_scale":
            state_hidden_scale,

        "state_hidden_shift":
            state_hidden_shift,

        "state_output_weight":
            state_output_weight,

        "state_output_bias":
            state_output_bias,

        "shared_carrier_weight":
            shared_carrier_weight,

        "state_anchors":
            state_anchors,
    }

    np.savez_compressed(
        OUTPUT_DIR
        / "observation_generator_parameters.npz",
        **parameters,
    )

    return parameters


def generate_raw_observations(
    carriers: np.ndarray,
    parameters,
) -> np.ndarray:
    features = carrier_features(
        carriers
    )

    base_hidden = (
        features
        @ parameters[
            "shared_hidden_weight"
        ]
        + parameters[
            "shared_hidden_bias"
        ]
    )

    shared_carrier_component = np.tanh(
        features
        @ parameters[
            "shared_carrier_weight"
        ]
    )

    observations = np.empty(
        (
            len(carriers),
            STATE_COUNT,
            OBSERVATION_DIMENSION,
        ),
        dtype=np.float64,
    )

    for state in range(
        STATE_COUNT
    ):
        hidden = np.tanh(
            base_hidden
            * parameters[
                "state_hidden_scale"
            ][state]
            + parameters[
                "state_hidden_shift"
            ][state]
        )

        state_component = np.tanh(
            hidden
            @ parameters[
                "state_output_weight"
            ][state]
            + parameters[
                "state_output_bias"
            ][state]
        )

        observations[
            :,
            state,
            :,
        ] = (
            state_component
            + 0.18
            * parameters[
                "state_anchors"
            ][state]
            + 0.12
            * shared_carrier_component
        )

    return observations


def normalize_observations(
    raw_observations: np.ndarray,
    carrier_rows,
):
    train_carrier_ids = np.asarray(
        [
            int(row["carrier_id"])
            for row in carrier_rows
            if row["split"] == "train"
        ],
        dtype=np.int64,
    )

    training_values = raw_observations[
        train_carrier_ids
    ].reshape(
        -1,
        OBSERVATION_DIMENSION,
    )

    mean = training_values.mean(
        axis=0
    )

    standard_deviation = training_values.std(
        axis=0
    )

    if np.any(
        standard_deviation < 1e-8
    ):
        raise AssertionError(
            "At least one observation coordinate "
            "has effectively zero training variance."
        )

    normalized = (
        raw_observations
        - mean[
            None,
            None,
            :,
        ]
    ) / standard_deviation[
        None,
        None,
        :,
    ]

    write_json(
        OUTPUT_DIR
        / "normalization.json",
        {
            "estimated_from":
                "clean training-carrier observations only",

            "training_carrier_count":
                int(
                    len(
                        train_carrier_ids
                    )
                ),

            "training_observation_count":
                int(
                    len(
                        training_values
                    )
                ),

            "mean":
                mean.tolist(),

            "standard_deviation":
                standard_deviation.tolist(),
        },
    )

    return (
        normalized.astype(
            np.float32
        ),
        mean,
        standard_deviation,
    )


def load_generators():
    raw = load_json(
        PHASE1A_DIR
        / "generators.json"
    )

    generators = {}

    if isinstance(
        raw,
        dict,
    ):
        for name, value in raw.items():
            if isinstance(
                value,
                dict,
            ):
                mapping = value[
                    "mapping"
                ]
            else:
                mapping = value

            generators[
                str(name)
            ] = tuple(
                int(item)
                for item in mapping
            )

    elif isinstance(
        raw,
        list,
    ):
        for row in raw:
            generators[
                row["name"]
            ] = tuple(
                int(item)
                for item in row["mapping"]
            )

    else:
        raise TypeError(
            "Unsupported generator format."
        )

    for name, mapping in generators.items():
        if len(mapping) != STATE_COUNT:
            raise AssertionError(
                f"Generator {name} has an invalid mapping."
            )

    return generators


def parse_operation_sequence(
    value: str,
):
    value = value.strip()

    try:
        parsed = json.loads(
            value
        )

        if isinstance(
            parsed,
            list,
        ):
            return tuple(
                str(item)
                for item in parsed
            )

    except json.JSONDecodeError:
        pass

    if "->" in value:
        return tuple(
            part.strip()
            for part in value.split("->")
            if part.strip()
        )

    return tuple(
        part.strip()
        for part in value.split(",")
        if part.strip()
    )


def load_sequence_manifest(
    generators,
):
    rows = load_csv(
        PHASE2A_DIR
        / "ood_balanced_sequence_manifest.csv"
    )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    parsed_rows = []

    for row in rows:
        split = row["split"]

        if split not in counts:
            raise AssertionError(
                f"Unexpected sequence split: {split}"
            )

        operations = parse_operation_sequence(
            row["operation_sequence"]
        )

        unknown = [
            operation
            for operation in operations
            if operation not in generators
        ]

        if unknown:
            raise AssertionError(
                "Unknown operations in sequence: "
                + ", ".join(unknown)
            )

        sequence_length = int(
            row["sequence_length"]
        )

        if sequence_length != len(
            operations
        ):
            raise AssertionError(
                "Sequence length does not match "
                "the operation sequence."
            )

        parsed_rows.append(
            {
                "sequence_index":
                    int(
                        row[
                            "sequence_index"
                        ]
                    ),

                "split":
                    split,

                "sequence_length":
                    sequence_length,

                "operations":
                    operations,
            }
        )

        counts[
            split
        ] += 1

    if counts != EXPECTED_SEQUENCE_COUNTS:
        raise AssertionError(
            f"Sequence counts changed: {counts}"
        )

    return parsed_rows


def create_train_probe(
    train_rows,
):
    generator = np.random.default_rng(
        TRAIN_PROBE_SEED
    )

    indices = np.arange(
        len(train_rows),
        dtype=np.int64,
    )

    generator.shuffle(
        indices
    )

    selected = [
        train_rows[
            int(index)
        ]
        for index in indices[:144]
    ]

    selected.sort(
        key=lambda row:
            row[
                "sequence_index"
            ]
    )

    if len(selected) != 144:
        raise AssertionError(
            "Train probe must contain 144 sequences."
        )

    write_csv(
        OUTPUT_DIR
        / "train_probe_sequences.csv",
        [
            {
                "sequence_index":
                    row[
                        "sequence_index"
                    ],

                "sequence_length":
                    row[
                        "sequence_length"
                    ],

                "operation_sequence":
                    json.dumps(
                        row[
                            "operations"
                        ]
                    ),
            }
            for row in selected
        ],
    )

    return selected


def state_path_for_sequence(
    initial_state: int,
    operations,
    generators,
):
    states = [
        int(
            initial_state
        )
    ]

    current = int(
        initial_state
    )

    for operation in operations:
        current = generators[
            operation
        ][current]

        states.append(
            int(
                current
            )
        )

    return tuple(states)


def load_evaluation_cells():
    rows = load_csv(
        PHASE3A_DIR
        / "evaluation_cells.csv"
    )

    if len(rows) != 8:
        raise AssertionError(
            "Evaluation cell count changed."
        )

    observed = {
        row["cell_id"]:
            int(
                row[
                    "sequence_count"
                ]
            )
        for row in rows
    }

    if observed != EXPECTED_CELL_COUNTS:
        raise AssertionError(
            f"Evaluation cells changed: {observed}"
        )

    return rows


def carrier_ids_by_split(
    carrier_rows,
):
    output = {}

    for split in (
        "train",
        "val",
        "test",
    ):
        rows = [
            row
            for row in carrier_rows
            if row["split"] == split
        ]

        rows.sort(
            key=lambda row:
                int(
                    row[
                        "split_position"
                    ]
                )
        )

        output[
            split
        ] = [
            int(
                row[
                    "carrier_id"
                ]
            )
            for row in rows
        ]

    return output


def carrier_for_trajectory(
    carrier_ids,
    sequence_position: int,
    initial_state: int,
    cell_offset: int,
):
    position = (
        17 * sequence_position
        + 23 * initial_state
        + cell_offset
    ) % len(
        carrier_ids
    )

    return carrier_ids[
        position
    ]


def sequence_rows_for_source(
    source: str,
    sequence_rows,
    train_probe,
):
    if source == "train_probe":
        return train_probe

    selected = [
        row
        for row in sequence_rows
        if row["split"] == source
    ]

    selected.sort(
        key=lambda row:
            row[
                "sequence_index"
            ]
    )

    return selected


def generate_cell(
    cell_row,
    sequence_rows,
    train_probe,
    carrier_ids,
    observations,
    generators,
    cell_offset,
    forbidden_pairings=None,
):
    cell_id = cell_row[
        "cell_id"
    ]

    sequence_source = cell_row[
        "sequence_source"
    ]

    carrier_split = cell_row[
        "carrier_split"
    ]

    selected_sequences = (
        sequence_rows_for_source(
            sequence_source,
            sequence_rows,
            train_probe,
        )
    )

    expected_sequence_count = int(
        cell_row[
            "sequence_count"
        ]
    )

    if (
        len(selected_sequences)
        != expected_sequence_count
    ):
        raise AssertionError(
            f"{cell_id} has the wrong sequence count."
        )

    allowed_carriers = carrier_ids[
        carrier_split
    ]

    clean_chunks = []
    state_chunks = []

    visible_trajectory_rows = []
    privileged_trajectory_rows = []
    visible_sequence_rows = []

    point_start = 0
    trajectory_index = 0

    pairing_lookup = {}

    for sequence_position, sequence in enumerate(
        selected_sequences
    ):
        visible_sequence_rows.append(
            {
                "cell_id":
                    cell_id,

                "sequence_index":
                    sequence[
                        "sequence_index"
                    ],

                "sequence_source":
                    sequence_source,

                "sequence_length":
                    sequence[
                        "sequence_length"
                    ],

                "operation_sequence":
                    json.dumps(
                        sequence[
                            "operations"
                        ]
                    ),
            }
        )

        for initial_state in range(
            STATE_COUNT
        ):
            pairing_key = (
                sequence[
                    "sequence_index"
                ],
                initial_state,
            )

            carrier_id = carrier_for_trajectory(
                carrier_ids=allowed_carriers,
                sequence_position=(
                    sequence_position
                ),
                initial_state=initial_state,
                cell_offset=cell_offset,
            )

            # test_iid_pairing must use a new sequence-state-carrier
            # pairing. When the deterministic formula collides with the
            # train_joint pairing, move to the next allowed train carrier.
            if forbidden_pairings is not None:
                forbidden_carrier = (
                    forbidden_pairings.get(
                        pairing_key
                    )
                )

                if (
                    forbidden_carrier
                    is not None
                    and carrier_id
                    == forbidden_carrier
                ):
                    current_position = (
                        allowed_carriers.index(
                            carrier_id
                        )
                    )

                    carrier_id = allowed_carriers[
                        (
                            current_position
                            + 1
                        )
                        % len(
                            allowed_carriers
                        )
                    ]

                    if carrier_id == forbidden_carrier:
                        raise AssertionError(
                            "Unable to construct a distinct "
                            "IID sequence-state-carrier pairing."
                        )

            states = state_path_for_sequence(
                initial_state=initial_state,
                operations=sequence[
                    "operations"
                ],
                generators=generators,
            )

            clean = observations[
                carrier_id,
                np.asarray(
                    states,
                    dtype=np.int64,
                ),
                :,
            ]

            clean_chunks.append(
                clean
            )

            state_chunks.append(
                np.asarray(
                    states,
                    dtype=np.int16,
                )
            )

            point_count = len(
                states
            )

            visible_trajectory_rows.append(
                {
                    "cell_id":
                        cell_id,

                    "trajectory_index":
                        trajectory_index,

                    "sequence_index":
                        sequence[
                            "sequence_index"
                        ],

                    "point_start":
                        point_start,

                    "point_count":
                        point_count,
                }
            )

            privileged_trajectory_rows.append(
                {
                    "cell_id":
                        cell_id,

                    "trajectory_index":
                        trajectory_index,

                    "sequence_index":
                        sequence[
                            "sequence_index"
                        ],

                    "initial_state":
                        initial_state,

                    "final_state":
                        states[-1],

                    "carrier_id":
                        carrier_id,

                    "carrier_split":
                        carrier_split,

                    "state_path":
                        json.dumps(
                            states
                        ),
                }
            )

            pairing_lookup[
                pairing_key
            ] = carrier_id

            point_start += (
                point_count
            )

            trajectory_index += 1

    clean_vectors = np.concatenate(
        clean_chunks,
        axis=0,
    ).astype(
        np.float32
    )

    state_ids = np.concatenate(
        state_chunks,
        axis=0,
    )

    expected_trajectory_count = (
        expected_sequence_count
        * TRAJECTORIES_PER_SEQUENCE
    )

    if (
        trajectory_index
        != expected_trajectory_count
    ):
        raise AssertionError(
            f"{cell_id} trajectory count changed."
        )

    noisy_vectors = np.empty(
        (
            len(
                NOISE_LEVELS
            ),
            len(
                clean_vectors
            ),
            OBSERVATION_DIMENSION,
        ),
        dtype=np.float32,
    )

    for noise_index, noise in enumerate(
        NOISE_LEVELS
    ):
        if noise == 0.0:
            noisy_vectors[
                noise_index
            ] = clean_vectors

        else:
            generator = np.random.default_rng(
                stable_seed(
                    f"{cell_id}|{noise}"
                )
            )

            epsilon = generator.normal(
                size=clean_vectors.shape,
            ).astype(
                np.float32
            )

            noisy_vectors[
                noise_index
            ] = (
                clean_vectors
                + float(
                    noise
                )
                * epsilon
            )

    np.savez_compressed(
        VISIBLE_DIR
        / f"{cell_id}_observations.npz",

        noise_fractions=np.asarray(
            NOISE_LEVELS,
            dtype=np.float32,
        ),

        vectors=noisy_vectors,
    )

    np.save(
        PRIVILEGED_DIR
        / f"{cell_id}_clean_vectors.npy",
        clean_vectors,
    )

    np.save(
        PRIVILEGED_DIR
        / f"{cell_id}_state_ids.npy",
        state_ids,
    )

    write_csv(
        VISIBLE_DIR
        / f"{cell_id}_trajectory_index.csv",
        visible_trajectory_rows,
    )

    write_csv(
        VISIBLE_DIR
        / f"{cell_id}_sequence_manifest.csv",
        visible_sequence_rows,
    )

    write_csv(
        PRIVILEGED_DIR
        / f"{cell_id}_trajectory_metadata.csv",
        privileged_trajectory_rows,
    )

    return {
        "cell_id":
            cell_id,

        "sequence_source":
            sequence_source,

        "carrier_split":
            carrier_split,

        "sequence_count":
            expected_sequence_count,

        "trajectory_count":
            trajectory_index,

        "point_count":
            int(
                len(
                    clean_vectors
                )
            ),

        "observation_dimension":
            OBSERVATION_DIMENSION,

        "noise_condition_count":
            len(
                NOISE_LEVELS
            ),

        "pairing_lookup":
            pairing_lookup,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VISIBLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRIVILEGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase3a = validate_phase3a()

    carriers, carrier_rows = (
        generate_carrier_vectors()
    )

    parameters = (
        generate_observation_parameters()
    )

    raw_observations = (
        generate_raw_observations(
            carriers=carriers,
            parameters=parameters,
        )
    )

    (
        normalized_observations,
        normalization_mean,
        normalization_standard_deviation,
    ) = normalize_observations(
        raw_observations=raw_observations,
        carrier_rows=carrier_rows,
    )

    carrier_splits = np.asarray(
        [
            row["split"]
            for row in carrier_rows
        ]
    )

    np.savez_compressed(
        PRIVILEGED_DIR
        / "carrier_state_observations.npz",

        carrier_ids=np.arange(
            CARRIER_COUNT,
            dtype=np.int64,
        ),

        carrier_splits=carrier_splits,

        carrier_vectors=carriers.astype(
            np.float32
        ),

        raw_observations=raw_observations.astype(
            np.float32
        ),

        normalized_observations=(
            normalized_observations
        ),

        normalization_mean=(
            normalization_mean.astype(
                np.float32
            )
        ),

        normalization_standard_deviation=(
            normalization_standard_deviation.astype(
                np.float32
            )
        ),
    )

    generators = load_generators()

    sequence_rows = load_sequence_manifest(
        generators
    )

    train_rows = [
        row
        for row in sequence_rows
        if row["split"] == "train"
    ]

    train_probe = create_train_probe(
        train_rows
    )

    evaluation_cells = (
        load_evaluation_cells()
    )

    carriers_by_split = (
        carrier_ids_by_split(
            carrier_rows
        )
    )

    cell_summaries = []
    pairing_maps = {}

    for cell_position, cell_row in enumerate(
        evaluation_cells
    ):
        if (
            cell_row[
                "cell_id"
            ]
            == "test_iid_pairing"
        ):
            if "train_joint" not in pairing_maps:
                raise AssertionError(
                    "train_joint must be generated before "
                    "test_iid_pairing."
                )

            forbidden_pairings = pairing_maps[
                "train_joint"
            ]

        else:
            forbidden_pairings = None

        cell_summary = generate_cell(
            cell_row=cell_row,
            sequence_rows=sequence_rows,
            train_probe=train_probe,
            carrier_ids=carriers_by_split,
            observations=normalized_observations,
            generators=generators,
            cell_offset=(
                7 * cell_position
                + 3
            ),
            forbidden_pairings=(
                forbidden_pairings
            ),
        )

        pairing_maps[
            cell_summary[
                "cell_id"
            ]
        ] = cell_summary.pop(
            "pairing_lookup"
        )

        cell_summaries.append(
            cell_summary
        )

        print(
            f"Generated {cell_summary['cell_id']} | "
            f"trajectories={cell_summary['trajectory_count']} | "
            f"points={cell_summary['point_count']}"
        )

    train_pairings = pairing_maps[
        "train_joint"
    ]

    iid_pairings = pairing_maps[
        "test_iid_pairing"
    ]

    repeated_pairings = [
        key
        for key, carrier_id
        in iid_pairings.items()
        if (
            key in train_pairings
            and train_pairings[
                key
            ] == carrier_id
        )
    ]

    if repeated_pairings:
        raise AssertionError(
            "test_iid_pairing repeated a "
            "train_joint sequence-state-carrier pairing."
        )

    write_csv(
        OUTPUT_DIR
        / "cell_generation_summary.csv",
        cell_summaries,
    )

    hashes = {
        "phase3a_summary":
            sha256_file(
                PHASE3A_DIR
                / "phase3a_summary.json"
            ),

        "carrier_split":
            sha256_file(
                PHASE3A_DIR
                / "carrier_split.csv"
            ),

        "evaluation_cells":
            sha256_file(
                PHASE3A_DIR
                / "evaluation_cells.csv"
            ),

        "observation_generator_parameters":
            sha256_file(
                OUTPUT_DIR
                / "observation_generator_parameters.npz"
            ),

        "carrier_state_observations":
            sha256_file(
                PRIVILEGED_DIR
                / "carrier_state_observations.npz"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "generation_hashes.json",
        hashes,
    )

    total_trajectories = sum(
        row[
            "trajectory_count"
        ]
        for row in cell_summaries
    )

    total_points = sum(
        row[
            "point_count"
        ]
        for row in cell_summaries
    )

    summary = {
        "phase":
            "3B Tier B nonlinear data generation",

        "phase3a_status":
            phase3a[
                "status"
            ],

        "carrier_count":
            CARRIER_COUNT,

        "carrier_dimension":
            CARRIER_DIMENSION,

        "observation_dimension":
            OBSERVATION_DIMENSION,

        "state_count":
            STATE_COUNT,

        "noise_levels":
            list(
                NOISE_LEVELS
            ),

        "evaluation_cell_count":
            len(
                cell_summaries
            ),

        "total_trajectory_count":
            total_trajectories,

        "total_point_count":
            total_points,

        "cell_summaries":
            {
                row[
                    "cell_id"
                ]:
                    row
                for row in cell_summaries
            },

        "train_probe_sequence_count":
            len(
                train_probe
            ),

        "test_iid_pairings_reused_from_training":
            0,

        "predictive_model_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "test_metrics_computed":
            False,

        "phase2_outputs_modified":
            False,

        "generation_status":
            "completed",
    }

    write_json(
        OUTPUT_DIR
        / "phase3b_generation_summary.json",
        summary,
    )

    print()
    print(
        "Phase 3B Tier B data generation completed."
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
