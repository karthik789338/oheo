from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


PHASE1B_DIR = Path("outputs/phase1b_symbolic_histories")
OUTPUT_DIR = Path("outputs/phase1c_continuous_observations")

STATE_COUNT = 8
OBSERVATION_DIM = 12

EXPECTED_SEQUENCE_COUNT = 3144
EXPECTED_TRAJECTORY_COUNT = 25152
EXPECTED_SYMBOLIC_POINT_COUNT = 254680

# Noise is expressed relative to the median pairwise distance
# between clean state prototypes.
NOISE_FRACTIONS = np.array(
    [0.0, 0.10, 0.25, 0.50, 1.00],
    dtype=np.float64,
)

PROTOTYPE_SEED = 271828
OBSERVATION_SEED = 314159
DIAGNOSTIC_SEED = 161803

DIAGNOSTIC_SAMPLES_PER_STATE = 5000


def file_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_phase1b_hashes() -> None:
    """Record exactly which frozen Phase 1B files were used."""

    expected_files = [
        "phase1a_input_hashes.json",
        "sequence_catalog.csv",
        "symbolic_trajectories.csv",
        "transformation_history_coverage.csv",
        "phase1b_summary.json",
    ]

    hashes = {}

    for filename in expected_files:
        path = PHASE1B_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Phase 1B file: {path}"
            )

        hashes[filename] = file_sha256(path)

    output_path = OUTPUT_DIR / "phase1b_input_hashes.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            hashes,
            handle,
            indent=2,
        )


def load_sequence_catalog():
    """Load the frozen symbolic process-sequence catalog."""

    path = PHASE1B_DIR / "sequence_catalog.csv"

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            row["sequence_length"] = int(
                row["sequence_length"]
            )

            row["rank"] = int(
                row["rank"]
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["sequence_id"]
    )

    return rows


def load_symbolic_trajectories(
    sequence_lookup,
):
    """Load and validate every exact symbolic trajectory."""

    path = PHASE1B_DIR / "symbolic_trajectories.csv"

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            row["initial_state"] = int(
                row["initial_state"]
            )

            row["final_state"] = int(
                row["final_state"]
            )

            row["rank"] = int(
                row["rank"]
            )

            row["preimage_size"] = int(
                row["preimage_size"]
            )

            row["state_path_list"] = [
                int(value)
                for value in json.loads(
                    row["state_path"]
                )
            ]

            sequence_id = row["sequence_id"]

            if sequence_id not in sequence_lookup:
                raise ValueError(
                    f"Unknown sequence ID: {sequence_id}"
                )

            expected_length = (
                sequence_lookup[
                    sequence_id
                ]["sequence_length"]
                + 1
            )

            if (
                len(row["state_path_list"])
                != expected_length
            ):
                raise AssertionError(
                    f"{row['trajectory_id']} has "
                    "an invalid state-path length."
                )

            if (
                row["state_path_list"][0]
                != row["initial_state"]
            ):
                raise AssertionError(
                    f"{row['trajectory_id']} does not "
                    "start at its recorded initial state."
                )

            if (
                row["state_path_list"][-1]
                != row["final_state"]
            ):
                raise AssertionError(
                    f"{row['trajectory_id']} does not "
                    "end at its recorded final state."
                )

            rows.append(row)

    rows.sort(
        key=lambda row: row["trajectory_id"]
    )

    return rows


def pairwise_distances(
    prototypes: np.ndarray,
) -> np.ndarray:
    """Return all unique pairwise Euclidean distances."""

    distances = []

    for first in range(len(prototypes)):
        for second in range(
            first + 1,
            len(prototypes),
        ):
            distance = np.linalg.norm(
                prototypes[first]
                - prototypes[second]
            )

            distances.append(
                float(distance)
            )

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def build_state_prototypes():
    """
    Construct a fixed continuous embedding for the eight symbolic states.

    The prototype geometry is random but completely reproducible.
    We normalize it so that the median pairwise state distance is 1.
    """

    rng = np.random.default_rng(
        PROTOTYPE_SEED
    )

    prototypes = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            STATE_COUNT,
            OBSERVATION_DIM,
        ),
    )

    # Remove a global offset. This does not change pairwise distances.
    prototypes = (
        prototypes
        - prototypes.mean(
            axis=0,
            keepdims=True,
        )
    )

    distances = pairwise_distances(
        prototypes
    )

    median_distance = float(
        np.median(distances)
    )

    if median_distance <= 0.0:
        raise RuntimeError(
            "Prototype geometry is degenerate."
        )

    prototypes = (
        prototypes
        / median_distance
    )

    normalized_distances = (
        pairwise_distances(
            prototypes
        )
    )

    return (
        prototypes.astype(
            np.float32
        ),
        normalized_distances,
    )


def write_state_prototypes(
    prototypes,
) -> None:
    """Write the clean continuous representation of each state."""

    path = OUTPUT_DIR / "state_prototypes.csv"

    fieldnames = [
        "state"
    ] + [
        f"z{dimension}"
        for dimension in range(
            OBSERVATION_DIM
        )
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

        for state in range(
            STATE_COUNT
        ):

            row = {
                "state": state
            }

            for dimension in range(
                OBSERVATION_DIM
            ):
                row[
                    f"z{dimension}"
                ] = float(
                    prototypes[
                        state,
                        dimension,
                    ]
                )

            writer.writerow(row)


def build_index_arrays(
    sequence_rows,
    trajectory_rows,
):
    """
    Flatten the symbolic trajectories into observation points.

    Metadata is stored once. The continuous vectors will later have
    shape:

        noise condition x symbolic point x observation dimension
    """

    sequence_index_lookup = {
        row["sequence_id"]: index
        for index, row
        in enumerate(sequence_rows)
    }

    trajectory_index_values = []
    sequence_index_values = []
    timestep_values = []
    symbolic_state_values = []
    initial_state_values = []
    is_final_values = []

    trajectory_point_start = []
    trajectory_point_count = []

    current_point = 0

    for trajectory_index, row in enumerate(
        trajectory_rows
    ):

        state_path = row[
            "state_path_list"
        ]

        sequence_index = (
            sequence_index_lookup[
                row["sequence_id"]
            ]
        )

        trajectory_point_start.append(
            current_point
        )

        trajectory_point_count.append(
            len(state_path)
        )

        for timestep, state in enumerate(
            state_path
        ):

            if (
                state < 0
                or state >= STATE_COUNT
            ):
                raise AssertionError(
                    "State outside frozen state space."
                )

            trajectory_index_values.append(
                trajectory_index
            )

            sequence_index_values.append(
                sequence_index
            )

            timestep_values.append(
                timestep
            )

            symbolic_state_values.append(
                state
            )

            initial_state_values.append(
                row["initial_state"]
            )

            is_final_values.append(
                timestep
                == len(state_path) - 1
            )

            current_point += 1

    arrays = {
        "trajectory_index":
            np.asarray(
                trajectory_index_values,
                dtype=np.int32,
            ),

        "sequence_index":
            np.asarray(
                sequence_index_values,
                dtype=np.int32,
            ),

        "timestep":
            np.asarray(
                timestep_values,
                dtype=np.int16,
            ),

        "symbolic_state":
            np.asarray(
                symbolic_state_values,
                dtype=np.int8,
            ),

        "initial_state":
            np.asarray(
                initial_state_values,
                dtype=np.int8,
            ),

        "is_final":
            np.asarray(
                is_final_values,
                dtype=np.bool_,
            ),

        "trajectory_point_start":
            np.asarray(
                trajectory_point_start,
                dtype=np.int64,
            ),

        "trajectory_point_count":
            np.asarray(
                trajectory_point_count,
                dtype=np.int16,
            ),
    }

    return arrays


def write_sequence_index(
    sequence_rows,
) -> None:

    path = OUTPUT_DIR / "sequence_index.csv"

    fieldnames = [
        "sequence_index",
        "sequence_id",
        "operation_sequence",
        "sequence_length",
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

        for index, row in enumerate(
            sequence_rows
        ):

            writer.writerow(
                {
                    "sequence_index":
                        index,

                    "sequence_id":
                        row[
                            "sequence_id"
                        ],

                    "operation_sequence":
                        row[
                            "operation_sequence"
                        ],

                    "sequence_length":
                        row[
                            "sequence_length"
                        ],

                    "result_element_id":
                        row[
                            "result_element_id"
                        ],

                    "information_class":
                        row[
                            "information_class"
                        ],

                    "rank":
                        row["rank"],
                }
            )


def write_trajectory_index(
    trajectory_rows,
    sequence_rows,
    index_arrays,
) -> None:

    sequence_index_lookup = {
        row["sequence_id"]: index
        for index, row
        in enumerate(sequence_rows)
    }

    path = OUTPUT_DIR / "trajectory_index.csv"

    fieldnames = [
        "trajectory_index",
        "trajectory_id",
        "sequence_index",
        "sequence_id",
        "initial_state",
        "final_state",
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

        for index, row in enumerate(
            trajectory_rows
        ):

            writer.writerow(
                {
                    "trajectory_index":
                        index,

                    "trajectory_id":
                        row[
                            "trajectory_id"
                        ],

                    "sequence_index":
                        sequence_index_lookup[
                            row[
                                "sequence_id"
                            ]
                        ],

                    "sequence_id":
                        row[
                            "sequence_id"
                        ],

                    "initial_state":
                        row[
                            "initial_state"
                        ],

                    "final_state":
                        row[
                            "final_state"
                        ],

                    "point_start":
                        int(
                            index_arrays[
                                "trajectory_point_start"
                            ][index]
                        ),

                    "point_count":
                        int(
                            index_arrays[
                                "trajectory_point_count"
                            ][index]
                        ),
                }
            )


def build_continuous_vectors(
    prototypes,
    symbolic_states,
    noise_sigmas,
):
    """
    Generate one continuous measurement at every symbolic observation
    point for every noise condition.

    Noise is isotropic Gaussian measurement noise.
    """

    base_vectors = prototypes[
        symbolic_states
    ].astype(
        np.float32,
        copy=True,
    )

    vector_count = len(
        symbolic_states
    )

    vectors = np.empty(
        (
            len(noise_sigmas),
            vector_count,
            OBSERVATION_DIM,
        ),
        dtype=np.float32,
    )

    rng = np.random.default_rng(
        OBSERVATION_SEED
    )

    for noise_index, sigma in enumerate(
        noise_sigmas
    ):

        if sigma == 0.0:
            vectors[
                noise_index
            ] = base_vectors

            continue

        standard_noise = (
            rng.standard_normal(
                size=base_vectors.shape
            )
            .astype(
                np.float32
            )
        )

        vectors[
            noise_index
        ] = (
            base_vectors
            + np.float32(sigma)
            * standard_noise
        )

    return vectors


def nearest_prototype_accuracy(
    prototypes,
    sigma,
    rng,
):
    """
    Measure latent-state distinguishability under one noise condition.

    This is a diagnostic only. It is not a learning result.
    """

    correct = 0
    total = 0

    prototypes_64 = (
        prototypes.astype(
            np.float64
        )
    )

    for state in range(
        STATE_COUNT
    ):

        samples = np.repeat(
            prototypes_64[
                state
            ][None, :],
            DIAGNOSTIC_SAMPLES_PER_STATE,
            axis=0,
        )

        if sigma > 0.0:
            samples += rng.normal(
                loc=0.0,
                scale=sigma,
                size=samples.shape,
            )

        differences = (
            samples[:, None, :]
            - prototypes_64[
                None,
                :,
                :
            ]
        )

        squared_distances = (
            differences
            * differences
        ).sum(
            axis=2
        )

        predictions = (
            np.argmin(
                squared_distances,
                axis=1,
            )
        )

        correct += int(
            np.sum(
                predictions == state
            )
        )

        total += len(
            predictions
        )

    return correct / total


def build_diagnostics(
    prototypes,
    noise_fractions,
    noise_sigmas,
):
    """Measure observation difficulty independently of trajectory skew."""

    rng = np.random.default_rng(
        DIAGNOSTIC_SEED
    )

    diagnostics = []

    for index, (
        fraction,
        sigma,
    ) in enumerate(
        zip(
            noise_fractions,
            noise_sigmas,
        )
    ):

        accuracy = (
            nearest_prototype_accuracy(
                prototypes=prototypes,
                sigma=float(sigma),
                rng=rng,
            )
        )

        diagnostics.append(
            {
                "noise_index":
                    index,

                "noise_fraction":
                    float(fraction),

                "sigma":
                    float(sigma),

                "nearest_prototype_accuracy":
                    float(accuracy),
            }
        )

    return diagnostics


def write_diagnostics(
    diagnostics,
) -> None:

    path = (
        OUTPUT_DIR
        / "observation_diagnostics.csv"
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "noise_index",
                "noise_fraction",
                "sigma",
                "nearest_prototype_accuracy",
            ],
        )

        writer.writeheader()
        writer.writerows(diagnostics)


def write_noise_conditions(
    noise_fractions,
    noise_sigmas,
) -> None:

    conditions = []

    for index, (
        fraction,
        sigma,
    ) in enumerate(
        zip(
            noise_fractions,
            noise_sigmas,
        )
    ):

        conditions.append(
            {
                "noise_index":
                    index,

                "noise_fraction":
                    float(fraction),

                "sigma":
                    float(sigma),

                "reference":
                    "median_clean_prototype_distance",
            }
        )

    path = (
        OUTPUT_DIR
        / "noise_conditions.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            conditions,
            handle,
            indent=2,
        )


def save_observation_arrays(
    vectors,
    prototypes,
    noise_fractions,
    noise_sigmas,
    index_arrays,
) -> None:
    """
    Save the numerical benchmark.

    Metadata is stored once for each symbolic point, while vectors are
    stored separately for every noise condition.
    """

    path = (
        OUTPUT_DIR
        / "continuous_observations.npz"
    )

    np.savez(
        path,
        vectors=vectors,

        prototypes=prototypes,

        noise_fractions=(
            noise_fractions.astype(
                np.float32
            )
        ),

        noise_sigmas=(
            noise_sigmas.astype(
                np.float32
            )
        ),

        trajectory_index=(
            index_arrays[
                "trajectory_index"
            ]
        ),

        sequence_index=(
            index_arrays[
                "sequence_index"
            ]
        ),

        timestep=(
            index_arrays[
                "timestep"
            ]
        ),

        symbolic_state=(
            index_arrays[
                "symbolic_state"
            ]
        ),

        initial_state=(
            index_arrays[
                "initial_state"
            ]
        ),

        is_final=(
            index_arrays[
                "is_final"
            ]
        ),

        trajectory_point_start=(
            index_arrays[
                "trajectory_point_start"
            ]
        ),

        trajectory_point_count=(
            index_arrays[
                "trajectory_point_count"
            ]
        ),
    )


def run_sanity_checks(
    phase1b_summary,
    sequence_rows,
    trajectory_rows,
    prototypes,
    prototype_distances,
    vectors,
    index_arrays,
    diagnostics,
):
    """Fail immediately if the frozen benchmark has changed."""

    if (
        phase1b_summary[
            "sanity_checks"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 1B was not marked as passed."
        )

    if len(sequence_rows) != EXPECTED_SEQUENCE_COUNT:
        raise AssertionError(
            "Frozen Phase 1B sequence count changed: "
            f"expected {EXPECTED_SEQUENCE_COUNT}, "
            f"found {len(sequence_rows)}."
        )

    if (
        len(trajectory_rows)
        != EXPECTED_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Frozen trajectory count changed: "
            f"expected {EXPECTED_TRAJECTORY_COUNT}, "
            f"found {len(trajectory_rows)}."
        )

    point_count = len(
        index_arrays[
            "symbolic_state"
        ]
    )

    if (
        point_count
        != EXPECTED_SYMBOLIC_POINT_COUNT
    ):
        raise AssertionError(
            "Unexpected symbolic observation count: "
            f"expected {EXPECTED_SYMBOLIC_POINT_COUNT}, "
            f"found {point_count}."
        )

    if prototypes.shape != (
        STATE_COUNT,
        OBSERVATION_DIM,
    ):
        raise AssertionError(
            "Prototype array has the wrong shape."
        )

    median_distance = float(
        np.median(
            prototype_distances
        )
    )

    if not np.isclose(
        median_distance,
        1.0,
        atol=1e-6,
    ):
        raise AssertionError(
            "Prototype median distance is not normalized to 1."
        )

    expected_vector_shape = (
        len(NOISE_FRACTIONS),
        EXPECTED_SYMBOLIC_POINT_COUNT,
        OBSERVATION_DIM,
    )

    if (
        vectors.shape
        != expected_vector_shape
    ):
        raise AssertionError(
            "Continuous observation array has "
            "the wrong shape."
        )

    if not np.isfinite(
        vectors
    ).all():
        raise AssertionError(
            "Continuous observations contain NaN or Inf."
        )

    clean_expected = prototypes[
        index_arrays[
            "symbolic_state"
        ]
    ]

    if not np.array_equal(
        vectors[0],
        clean_expected,
    ):
        raise AssertionError(
            "Zero-noise observations do not exactly "
            "match state prototypes."
        )

    states_present = set(
        int(value)
        for value in np.unique(
            index_arrays[
                "symbolic_state"
            ]
        )
    )

    if states_present != set(
        range(STATE_COUNT)
    ):
        raise AssertionError(
            "Not all symbolic states occur in the benchmark."
        )

    final_count = int(
        np.sum(
            index_arrays[
                "is_final"
            ]
        )
    )

    if (
        final_count
        != EXPECTED_TRAJECTORY_COUNT
    ):
        raise AssertionError(
            "Every trajectory must have exactly one final point."
        )

    if (
        diagnostics[0][
            "nearest_prototype_accuracy"
        ]
        != 1.0
    ):
        raise AssertionError(
            "Clean observations must have perfect "
            "nearest-prototype recovery."
        )


def write_summary(
    prototypes,
    prototype_distances,
    vectors,
    index_arrays,
    sequence_rows,
    trajectory_rows,
    noise_fractions,
    noise_sigmas,
    diagnostics,
):
    state_counts = Counter(
        int(state)
        for state in index_arrays[
            "symbolic_state"
        ]
    )

    diagnostic_summary = {
        str(item["noise_fraction"]):
            item[
                "nearest_prototype_accuracy"
            ]
        for item in diagnostics
    }

    summary = {
        "state_count":
            STATE_COUNT,

        "observation_dimension":
            OBSERVATION_DIM,

        "sequence_count":
            len(sequence_rows),

        "trajectory_count":
            len(trajectory_rows),

        "symbolic_observation_point_count":
            len(
                index_arrays[
                    "symbolic_state"
                ]
            ),

        "noise_condition_count":
            len(
                noise_fractions
            ),

        "continuous_vector_count":
            int(
                vectors.shape[0]
                * vectors.shape[1]
            ),

        "vector_array_shape":
            [
                int(value)
                for value
                in vectors.shape
            ],

        "prototype_pairwise_distance": {
            "minimum":
                float(
                    np.min(
                        prototype_distances
                    )
                ),

            "median":
                float(
                    np.median(
                        prototype_distances
                    )
                ),

            "maximum":
                float(
                    np.max(
                        prototype_distances
                    )
                ),
        },

        "noise_conditions": [
            {
                "noise_fraction":
                    float(fraction),

                "sigma":
                    float(sigma),
            }
            for fraction, sigma
            in zip(
                noise_fractions,
                noise_sigmas,
            )
        ],

        "nearest_prototype_accuracy": (
            diagnostic_summary
        ),

        "symbolic_state_point_distribution": {
            str(state):
                state_counts[state]
            for state in range(
                STATE_COUNT
            )
        },

        "prototype_seed":
            PROTOTYPE_SEED,

        "observation_seed":
            OBSERVATION_SEED,

        "diagnostic_seed":
            DIAGNOSTIC_SEED,

        "diagnostic_samples_per_state":
            DIAGNOSTIC_SAMPLES_PER_STATE,

        "ground_truth_interpretation":
            (
                "Phase 1A information relations are latent "
                "structural targets. They are not assumed to be "
                "the exact Blackwell order of the noisy continuous "
                "observation channels."
            ),

        "sanity_checks":
            "passed",
    }

    path = (
        OUTPUT_DIR
        / "phase1c_summary.json"
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

    phase1b_summary = load_json(
        PHASE1B_DIR
        / "phase1b_summary.json"
    )

    sequence_rows = (
        load_sequence_catalog()
    )

    sequence_lookup = {
        row["sequence_id"]: row
        for row in sequence_rows
    }

    trajectory_rows = (
        load_symbolic_trajectories(
            sequence_lookup
        )
    )

    save_phase1b_hashes()

    (
        prototypes,
        prototype_distances,
    ) = build_state_prototypes()

    write_state_prototypes(
        prototypes
    )

    median_distance = float(
        np.median(
            prototype_distances
        )
    )

    noise_sigmas = (
        NOISE_FRACTIONS
        * median_distance
    )

    write_noise_conditions(
        NOISE_FRACTIONS,
        noise_sigmas,
    )

    index_arrays = (
        build_index_arrays(
            sequence_rows,
            trajectory_rows,
        )
    )

    write_sequence_index(
        sequence_rows
    )

    write_trajectory_index(
        trajectory_rows,
        sequence_rows,
        index_arrays,
    )

    vectors = (
        build_continuous_vectors(
            prototypes=prototypes,
            symbolic_states=(
                index_arrays[
                    "symbolic_state"
                ]
            ),
            noise_sigmas=noise_sigmas,
        )
    )

    diagnostics = (
        build_diagnostics(
            prototypes=prototypes,
            noise_fractions=(
                NOISE_FRACTIONS
            ),
            noise_sigmas=noise_sigmas,
        )
    )

    write_diagnostics(
        diagnostics
    )

    save_observation_arrays(
        vectors=vectors,
        prototypes=prototypes,
        noise_fractions=(
            NOISE_FRACTIONS
        ),
        noise_sigmas=noise_sigmas,
        index_arrays=index_arrays,
    )

    run_sanity_checks(
        phase1b_summary=(
            phase1b_summary
        ),
        sequence_rows=sequence_rows,
        trajectory_rows=trajectory_rows,
        prototypes=prototypes,
        prototype_distances=(
            prototype_distances
        ),
        vectors=vectors,
        index_arrays=index_arrays,
        diagnostics=diagnostics,
    )

    summary = write_summary(
        prototypes=prototypes,
        prototype_distances=(
            prototype_distances
        ),
        vectors=vectors,
        index_arrays=index_arrays,
        sequence_rows=sequence_rows,
        trajectory_rows=trajectory_rows,
        noise_fractions=(
            NOISE_FRACTIONS
        ),
        noise_sigmas=noise_sigmas,
        diagnostics=diagnostics,
    )

    print(
        "Phase 1C continuous observation benchmark "
        "created successfully."
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
