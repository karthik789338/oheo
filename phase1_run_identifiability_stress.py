from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np


PHASE1A_DIR = Path("outputs/phase1a_ground_truth")
PHASE1C_DIR = Path("outputs/phase1c_continuous_observations")
PHASE1D_DIR = Path("outputs/phase1d_compositional_splits")

OUTPUT_DIR = Path("outputs/phase1e_identifiability_stress")


STATE_COUNT = 8
EXPECTED_TRANSFORMATION_COUNT = 104
EXPECTED_INFORMATION_CLASS_COUNT = 9

HISTORY_COUNTS = (2, 4, 6, 8)
REPETITIONS = (1, 4, 16)

EXPECTED_NOISE_FRACTIONS = (
    0.0,
    0.10,
    0.25,
    0.50,
    1.00,
)

BASE_RANDOM_SEED = 8675309


RELATION_NAMES = {
    0: "equivalent",
    1: "a_more_informative",
    2: "a_less_informative",
    3: "incomparable",
}


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


def save_input_hashes() -> None:
    """
    Record the frozen benchmark files used by Phase 1E.

    Phase 1D is included as a checkpoint even though the
    identifiability stress calculation does not use the train/test
    split itself.
    """

    files = {
        "phase1a_generators":
            PHASE1A_DIR / "generators.json",

        "phase1a_semigroup":
            PHASE1A_DIR / "semigroup_elements.csv",

        "phase1a_relations":
            PHASE1A_DIR / "blackwell_relations.csv",

        "phase1a_summary":
            PHASE1A_DIR / "phase1a_summary.json",

        "phase1c_prototypes":
            PHASE1C_DIR / "state_prototypes.csv",

        "phase1c_noise":
            PHASE1C_DIR / "noise_conditions.json",

        "phase1c_summary":
            PHASE1C_DIR / "phase1c_summary.json",

        "phase1d_summary":
            PHASE1D_DIR / "phase1d_summary.json",
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

    output_path = OUTPUT_DIR / "input_hashes.json"

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
    path = PHASE1A_DIR / "generators.json"

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        rows = json.load(handle)

    generators = {}

    for row in rows:
        generators[row["name"]] = tuple(
            int(value)
            for value in row["mapping"]
        )

    return generators


def load_semigroup_elements():
    path = PHASE1A_DIR / "semigroup_elements.csv"

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            row["mapping_tuple"] = tuple(
                int(value)
                for value in json.loads(
                    row["mapping"]
                )
            )

            row["rank"] = int(
                row["rank"]
            )

            row["word_length"] = int(
                row["word_length"]
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["element_id"]
    )

    return rows


def load_prototypes():
    path = PHASE1C_DIR / "state_prototypes.csv"

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(row)

    rows.sort(
        key=lambda row: int(row["state"])
    )

    dimension_names = [
        name
        for name in rows[0]
        if name.startswith("z")
    ]

    dimension_names.sort(
        key=lambda name:
            int(name[1:])
    )

    prototypes = np.asarray(
        [
            [
                float(row[name])
                for name in dimension_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    return prototypes


def load_noise_conditions():
    path = PHASE1C_DIR / "noise_conditions.json"

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        rows = json.load(handle)

    rows.sort(
        key=lambda row:
            row["noise_index"]
    )

    return rows


def compose(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    """Apply first, then second."""

    return tuple(
        second[first[state]]
        for state in range(
            len(first)
        )
    )


def build_closure(
    generator_mappings,
):
    """
    Enumerate the transformation semigroup generated by the
    supplied operation subset.
    """

    identity = tuple(
        range(STATE_COUNT)
    )

    seen = {
        identity
    }

    queue = deque(
        [identity]
    )

    while queue:

        current = queue.popleft()

        for generator in generator_mappings:

            result = compose(
                current,
                generator,
            )

            if result not in seen:
                seen.add(result)
                queue.append(result)

    return seen


def partition_of(
    mapping: tuple[int, ...],
):
    """Return the full information partition of a transformation."""

    blocks = defaultdict(list)

    for source_state, output_state in enumerate(
        mapping
    ):
        blocks[output_state].append(
            source_state
        )

    return tuple(
        sorted(
            tuple(block)
            for block in blocks.values()
        )
    )


def equality_mask_from_outputs(
    outputs,
) -> int:
    """
    Encode the information partition as a bit mask.

    A bit is set whenever two observed source histories are mapped
    to the same output state.

    This makes pairwise partial-order comparisons inexpensive.
    """

    outputs = list(
        int(value)
        for value in outputs
    )

    mask = 0
    bit_index = 0

    for first in range(
        len(outputs)
    ):

        for second in range(
            first + 1,
            len(outputs),
        ):

            if (
                outputs[first]
                == outputs[second]
            ):
                mask |= (
                    1 << bit_index
                )

            bit_index += 1

    return mask


def relation_from_masks(
    mask_a: int,
    mask_b: int,
) -> int:
    """
    Compare two deterministic information partitions.

    Fewer equality pairs means the channel keeps more source
    distinctions and is therefore more informative.
    """

    if mask_a == mask_b:
        return 0

    # Every equality made by A is also made by B.
    # A therefore has the finer partition.
    if (
        mask_a
        & ~mask_b
    ) == 0:
        return 1

    if (
        mask_b
        & ~mask_a
    ) == 0:
        return 2

    return 3


def build_relation_vector(
    masks,
):
    """
    Build relations for all ordered pairs of distinct transformations.

    Diagonal self-comparisons are deliberately excluded.
    """

    relations = []

    for first in range(
        len(masks)
    ):

        for second in range(
            len(masks)
        ):

            if first == second:
                continue

            relations.append(
                relation_from_masks(
                    int(masks[first]),
                    int(masks[second]),
                )
            )

    return np.asarray(
        relations,
        dtype=np.int8,
    )


def history_subset_mask(
    subset,
) -> int:
    """Encode a source-state subset as an integer bit mask."""

    mask = 0

    for state in subset:
        mask |= (
            1 << int(state)
        )

    return mask


def build_full_ground_truth(
    semigroup_rows,
):
    mappings = [
        row["mapping_tuple"]
        for row in semigroup_rows
    ]

    full_masks = np.asarray(
        [
            equality_mask_from_outputs(
                mapping
            )
            for mapping in mappings
        ],
        dtype=np.int64,
    )

    full_relations = build_relation_vector(
        full_masks
    )

    return (
        mappings,
        full_masks,
        full_relations,
    )


def calculate_history_subset_identifiability(
    mappings,
    full_relations,
):
    """
    Measure exactly how much structural information is available
    when only a subset of source histories is observed.

    No noise and no estimator are involved here.
    """

    rows = []

    for history_count in HISTORY_COUNTS:

        for subset in combinations(
            range(STATE_COUNT),
            history_count,
        ):

            signatures = [
                tuple(
                    mapping[state]
                    for state in subset
                )
                for mapping in mappings
            ]

            signature_counter = Counter(
                signatures
            )

            distinct_signatures = len(
                signature_counter
            )

            collision_pairs = sum(
                count * (count - 1) // 2
                for count
                in signature_counter.values()
            )

            restricted_masks = np.asarray(
                [
                    equality_mask_from_outputs(
                        signature
                    )
                    for signature in signatures
                ],
                dtype=np.int64,
            )

            distinct_information_patterns = len(
                set(
                    int(value)
                    for value
                    in restricted_masks
                )
            )

            restricted_relations = (
                build_relation_vector(
                    restricted_masks
                )
            )

            relation_accuracy = float(
                np.mean(
                    restricted_relations
                    == full_relations
                )
            )

            rows.append(
                {
                    "history_subset_mask":
                        history_subset_mask(
                            subset
                        ),

                    "history_count":
                        history_count,

                    "history_states":
                        json.dumps(
                            list(subset),
                            separators=(",", ":"),
                        ),

                    "distinct_transformation_signatures":
                        distinct_signatures,

                    "transformation_signature_fraction":
                        (
                            distinct_signatures
                            / EXPECTED_TRANSFORMATION_COUNT
                        ),

                    "collision_pair_count":
                        collision_pairs,

                    "distinct_information_patterns":
                        distinct_information_patterns,

                    "full_relation_accuracy_from_restricted_view":
                        relation_accuracy,
                }
            )

    return rows


def write_history_subset_identifiability(
    rows,
):
    path = (
        OUTPUT_DIR
        / "history_subset_identifiability.csv"
    )

    fieldnames = [
        "history_subset_mask",
        "history_count",
        "history_states",
        "distinct_transformation_signatures",
        "transformation_signature_fraction",
        "collision_pair_count",
        "distinct_information_patterns",
        "full_relation_accuracy_from_restricted_view",
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


def calculate_alphabet_coverage(
    generators,
):
    """
    Enumerate all 63 non-empty primitive-operation subsets.

    This measures how much of the full transformation structure
    is reachable under incomplete operation coverage.
    """

    generator_names = tuple(
        generators.keys()
    )

    rows = []

    for alphabet_size in range(
        1,
        len(generator_names) + 1,
    ):

        for subset in combinations(
            generator_names,
            alphabet_size,
        ):

            closure = build_closure(
                [
                    generators[name]
                    for name in subset
                ]
            )

            information_classes = {
                partition_of(mapping)
                for mapping in closure
            }

            rows.append(
                {
                    "alphabet_size":
                        alphabet_size,

                    "generator_subset":
                        json.dumps(
                            list(subset),
                            separators=(",", ":"),
                        ),

                    "reachable_transformation_count":
                        len(closure),

                    "reachable_transformation_fraction":
                        (
                            len(closure)
                            / EXPECTED_TRANSFORMATION_COUNT
                        ),

                    "reachable_information_class_count":
                        len(
                            information_classes
                        ),

                    "reachable_information_class_fraction":
                        (
                            len(information_classes)
                            / EXPECTED_INFORMATION_CLASS_COUNT
                        ),
                }
            )

    return rows


def write_alphabet_coverage(
    rows,
):
    path = (
        OUTPUT_DIR
        / "alphabet_coverage.csv"
    )

    fieldnames = [
        "alphabet_size",
        "generator_subset",
        "reachable_transformation_count",
        "reachable_transformation_fraction",
        "reachable_information_class_count",
        "reachable_information_class_fraction",
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


def make_condition_rng(
    subset_mask,
    noise_index,
):
    """
    Create a deterministic random generator for one stress condition.

    Repetition levels share the same generated observations and use
    prefixes of length 1, 4, and 16.
    """

    seed_sequence = np.random.SeedSequence(
        [
            BASE_RANDOM_SEED,
            int(subset_mask),
            int(noise_index),
        ]
    )

    return np.random.default_rng(
        seed_sequence
    )


def decode_nearest_prototype(
    observations,
    prototypes,
):
    """
    Decode continuous measurements using the known clean prototypes.

    This is deliberately an oracle reference estimator, not the
    proposed learning method.
    """

    differences = (
        observations[..., None, :]
        - prototypes[
            None,
            None,
            None,
            :,
            :
        ]
    )

    squared_distances = (
        differences
        * differences
    ).sum(
        axis=-1
    )

    return np.argmin(
        squared_distances,
        axis=-1,
    ).astype(
        np.int8
    )


def majority_vote(
    decoded,
    repetition_count,
):
    """
    Majority vote over the first `repetition_count` measurements.

    Ties are resolved deterministically by the smallest state label
    through NumPy's argmax behavior.
    """

    prefix = decoded[
        :,
        :,
        :repetition_count,
    ]

    transformation_count = (
        prefix.shape[0]
    )

    history_count = (
        prefix.shape[1]
    )

    result = np.empty(
        (
            transformation_count,
            history_count,
        ),
        dtype=np.int8,
    )

    for transformation_index in range(
        transformation_count
    ):

        for history_index in range(
            history_count
        ):

            counts = np.bincount(
                prefix[
                    transformation_index,
                    history_index,
                ],
                minlength=STATE_COUNT,
            )

            result[
                transformation_index,
                history_index,
            ] = int(
                np.argmax(
                    counts
                )
            )

    return result


def calculate_noisy_recovery(
    mappings,
    prototypes,
    noise_conditions,
    full_relations,
):
    """
    Stress the privileged reference estimator across history coverage,
    measurement noise, and repeated observations.
    """

    transformation_count = len(
        mappings
    )

    rows = []

    maximum_repetitions = max(
        REPETITIONS
    )

    for history_count in HISTORY_COUNTS:

        for subset in combinations(
            range(STATE_COUNT),
            history_count,
        ):

            subset = tuple(subset)

            subset_mask = (
                history_subset_mask(
                    subset
                )
            )

            true_outputs = np.asarray(
                [
                    [
                        mapping[state]
                        for state in subset
                    ]
                    for mapping in mappings
                ],
                dtype=np.int8,
            )

            true_restricted_masks = (
                np.asarray(
                    [
                        equality_mask_from_outputs(
                            outputs
                        )
                        for outputs
                        in true_outputs
                    ],
                    dtype=np.int64,
                )
            )

            true_restricted_relations = (
                build_relation_vector(
                    true_restricted_masks
                )
            )

            clean_output_vectors = (
                prototypes[
                    true_outputs
                ]
            )

            for noise_condition in noise_conditions:

                noise_index = int(
                    noise_condition[
                        "noise_index"
                    ]
                )

                noise_fraction = float(
                    noise_condition[
                        "noise_fraction"
                    ]
                )

                sigma = float(
                    noise_condition[
                        "sigma"
                    ]
                )

                rng = make_condition_rng(
                    subset_mask,
                    noise_index,
                )

                observations = np.repeat(
                    clean_output_vectors[
                        :,
                        :,
                        None,
                        :,
                    ],
                    maximum_repetitions,
                    axis=2,
                )

                if sigma > 0.0:

                    observations = (
                        observations
                        + rng.normal(
                            loc=0.0,
                            scale=sigma,
                            size=observations.shape,
                        )
                    )

                decoded = (
                    decode_nearest_prototype(
                        observations,
                        prototypes,
                    )
                )

                for repetition_count in REPETITIONS:

                    estimated_outputs = (
                        majority_vote(
                            decoded,
                            repetition_count,
                        )
                    )

                    mapping_accuracy = float(
                        np.mean(
                            estimated_outputs
                            == true_outputs
                        )
                    )

                    exact_recovery = np.all(
                        estimated_outputs
                        == true_outputs,
                        axis=1,
                    )

                    exact_recovery_rate = float(
                        np.mean(
                            exact_recovery
                        )
                    )

                    estimated_masks = (
                        np.asarray(
                            [
                                equality_mask_from_outputs(
                                    outputs
                                )
                                for outputs
                                in estimated_outputs
                            ],
                            dtype=np.int64,
                        )
                    )

                    estimated_relations = (
                        build_relation_vector(
                            estimated_masks
                        )
                    )

                    relation_accuracy_restricted = float(
                        np.mean(
                            estimated_relations
                            == true_restricted_relations
                        )
                    )

                    relation_accuracy_full = float(
                        np.mean(
                            estimated_relations
                            == full_relations
                        )
                    )

                    rows.append(
                        {
                            "history_subset_mask":
                                subset_mask,

                            "history_count":
                                history_count,

                            "history_states":
                                json.dumps(
                                    list(subset),
                                    separators=(",", ":"),
                                ),

                            "noise_index":
                                noise_index,

                            "noise_fraction":
                                noise_fraction,

                            "sigma":
                                sigma,

                            "repetitions":
                                repetition_count,

                            "restricted_mapping_accuracy":
                                mapping_accuracy,

                            "restricted_exact_transformation_recovery":
                                exact_recovery_rate,

                            "relation_accuracy_vs_restricted_oracle":
                                relation_accuracy_restricted,

                            "relation_accuracy_vs_full_ground_truth":
                                relation_accuracy_full,

                            "transformation_count":
                                transformation_count,

                            "ordered_distinct_relation_pair_count":
                                (
                                    transformation_count
                                    * (
                                        transformation_count
                                        - 1
                                    )
                                ),
                        }
                    )

    return rows


def write_noisy_recovery(
    rows,
):
    path = (
        OUTPUT_DIR
        / "noisy_oracle_recovery.csv"
    )

    fieldnames = [
        "history_subset_mask",
        "history_count",
        "history_states",
        "noise_index",
        "noise_fraction",
        "sigma",
        "repetitions",
        "restricted_mapping_accuracy",
        "restricted_exact_transformation_recovery",
        "relation_accuracy_vs_restricted_oracle",
        "relation_accuracy_vs_full_ground_truth",
        "transformation_count",
        "ordered_distinct_relation_pair_count",
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


def summarize_numeric(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "minimum":
            float(
                np.min(values)
            ),

        "mean":
            float(
                np.mean(values)
            ),

        "maximum":
            float(
                np.max(values)
            ),
    }


def summarize_history_coverage(
    rows,
):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["history_count"]
        ].append(row)

    summary = {}

    for history_count in sorted(grouped):

        group = grouped[
            history_count
        ]

        summary[
            str(history_count)
        ] = {
            "subset_count":
                len(group),

            "distinct_transformation_signatures":
                summarize_numeric(
                    [
                        row[
                            "distinct_transformation_signatures"
                        ]
                        for row in group
                    ]
                ),

            "full_relation_accuracy_from_restricted_view":
                summarize_numeric(
                    [
                        row[
                            "full_relation_accuracy_from_restricted_view"
                        ]
                        for row in group
                    ]
                ),
        }

    return summary


def summarize_alphabet_coverage(
    rows,
):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            row["alphabet_size"]
        ].append(row)

    summary = {}

    for alphabet_size in sorted(grouped):

        group = grouped[
            alphabet_size
        ]

        summary[
            str(alphabet_size)
        ] = {
            "subset_count":
                len(group),

            "reachable_transformations":
                summarize_numeric(
                    [
                        row[
                            "reachable_transformation_count"
                        ]
                        for row in group
                    ]
                ),

            "reachable_information_classes":
                summarize_numeric(
                    [
                        row[
                            "reachable_information_class_count"
                        ]
                        for row in group
                    ]
                ),
        }

    return summary


def summarize_noisy_recovery(
    rows,
):
    """
    Aggregate across all source-state subsets having the same
    history count, noise level, and repetition count.
    """

    grouped = defaultdict(list)

    for row in rows:

        key = (
            row["history_count"],
            row["noise_fraction"],
            row["repetitions"],
        )

        grouped[key].append(row)

    summary = {}

    for key in sorted(grouped):

        history_count, noise_fraction, repetitions = key

        group = grouped[key]

        label = (
            f"h{history_count}"
            f"_noise{noise_fraction}"
            f"_r{repetitions}"
        )

        summary[label] = {
            "history_count":
                history_count,

            "noise_fraction":
                noise_fraction,

            "repetitions":
                repetitions,

            "subset_count":
                len(group),

            "restricted_mapping_accuracy":
                summarize_numeric(
                    [
                        row[
                            "restricted_mapping_accuracy"
                        ]
                        for row in group
                    ]
                ),

            "restricted_exact_transformation_recovery":
                summarize_numeric(
                    [
                        row[
                            "restricted_exact_transformation_recovery"
                        ]
                        for row in group
                    ]
                ),

            "relation_accuracy_vs_restricted_oracle":
                summarize_numeric(
                    [
                        row[
                            "relation_accuracy_vs_restricted_oracle"
                        ]
                        for row in group
                    ]
                ),

            "relation_accuracy_vs_full_ground_truth":
                summarize_numeric(
                    [
                        row[
                            "relation_accuracy_vs_full_ground_truth"
                        ]
                        for row in group
                    ]
                ),
        }

    return summary


def run_sanity_checks(
    generators,
    semigroup_rows,
    prototypes,
    noise_conditions,
    history_rows,
    alphabet_rows,
    noisy_rows,
):
    if len(generators) != 6:
        raise AssertionError(
            "Expected six frozen primitive generators."
        )

    if (
        len(semigroup_rows)
        != EXPECTED_TRANSFORMATION_COUNT
    ):
        raise AssertionError(
            "Frozen semigroup size changed."
        )

    information_classes = {
        row["information_class"]
        for row in semigroup_rows
    }

    if (
        len(information_classes)
        != EXPECTED_INFORMATION_CLASS_COUNT
    ):
        raise AssertionError(
            "Frozen information-class count changed."
        )

    if prototypes.shape[0] != STATE_COUNT:
        raise AssertionError(
            "Prototype state count changed."
        )

    observed_noise_fractions = tuple(
        float(
            row["noise_fraction"]
        )
        for row in noise_conditions
    )

    if observed_noise_fractions != EXPECTED_NOISE_FRACTIONS:
        raise AssertionError(
            "Frozen noise conditions changed: "
            f"{observed_noise_fractions}"
        )

    expected_history_subset_count = sum(
        1
        for history_count
        in HISTORY_COUNTS
        for _ in combinations(
            range(STATE_COUNT),
            history_count,
        )
    )

    if (
        expected_history_subset_count
        != 127
    ):
        raise AssertionError(
            "Internal history-subset count is wrong."
        )

    if (
        len(history_rows)
        != 127
    ):
        raise AssertionError(
            "Expected 127 history coverage conditions."
        )

    if len(alphabet_rows) != 63:
        raise AssertionError(
            "Expected 63 non-empty operation alphabets."
        )

    expected_noisy_rows = (
        127
        * len(EXPECTED_NOISE_FRACTIONS)
        * len(REPETITIONS)
    )

    if expected_noisy_rows != 1905:
        raise AssertionError(
            "Internal noisy-condition count is wrong."
        )

    if (
        len(noisy_rows)
        != expected_noisy_rows
    ):
        raise AssertionError(
            f"Expected {expected_noisy_rows} noisy stress rows, "
            f"found {len(noisy_rows)}."
        )

    full_history_rows = [
        row
        for row in history_rows
        if row["history_count"] == STATE_COUNT
    ]

    if len(full_history_rows) != 1:
        raise AssertionError(
            "Expected exactly one full-history condition."
        )

    full_history = full_history_rows[0]

    if (
        full_history[
            "distinct_transformation_signatures"
        ]
        != EXPECTED_TRANSFORMATION_COUNT
    ):
        raise AssertionError(
            "All 104 transformations must be distinguishable "
            "when all eight source histories are observed."
        )

    if not np.isclose(
        full_history[
            "full_relation_accuracy_from_restricted_view"
        ],
        1.0,
    ):
        raise AssertionError(
            "Full history coverage must reproduce the exact "
            "Phase 1A relation structure."
        )

    full_alphabet_rows = [
        row
        for row in alphabet_rows
        if row["alphabet_size"]
        == len(generators)
    ]

    if len(full_alphabet_rows) != 1:
        raise AssertionError(
            "Expected one complete alphabet condition."
        )

    full_alphabet = full_alphabet_rows[0]

    if (
        full_alphabet[
            "reachable_transformation_count"
        ]
        != EXPECTED_TRANSFORMATION_COUNT
    ):
        raise AssertionError(
            "Complete generator alphabet must recover "
            "the 104-element semigroup."
        )

    if (
        full_alphabet[
            "reachable_information_class_count"
        ]
        != EXPECTED_INFORMATION_CLASS_COUNT
    ):
        raise AssertionError(
            "Complete alphabet must reach all nine "
            "information classes."
        )

    clean_full_rows = [
        row
        for row in noisy_rows
        if (
            row["history_count"] == STATE_COUNT
            and row["noise_fraction"] == 0.0
        )
    ]

    if len(clean_full_rows) != len(REPETITIONS):
        raise AssertionError(
            "Missing clean full-history stress conditions."
        )

    for row in clean_full_rows:

        for metric_name in [
            "restricted_mapping_accuracy",
            "restricted_exact_transformation_recovery",
            "relation_accuracy_vs_restricted_oracle",
            "relation_accuracy_vs_full_ground_truth",
        ]:

            if not np.isclose(
                row[metric_name],
                1.0,
            ):
                raise AssertionError(
                    f"Clean full-history {metric_name} "
                    "must equal 1."
                )


def write_summary(
    history_rows,
    alphabet_rows,
    noisy_rows,
):
    summary = {
        "transformation_count":
            EXPECTED_TRANSFORMATION_COUNT,

        "information_class_count":
            EXPECTED_INFORMATION_CLASS_COUNT,

        "history_counts":
            list(HISTORY_COUNTS),

        "history_subset_condition_count":
            len(history_rows),

        "alphabet_subset_condition_count":
            len(alphabet_rows),

        "noise_fractions":
            list(
                EXPECTED_NOISE_FRACTIONS
            ),

        "repetition_counts":
            list(REPETITIONS),

        "noisy_stress_condition_count":
            len(noisy_rows),

        "ordered_distinct_relation_pair_count":
            (
                EXPECTED_TRANSFORMATION_COUNT
                * (
                    EXPECTED_TRANSFORMATION_COUNT
                    - 1
                )
            ),

        "history_coverage_summary":
            summarize_history_coverage(
                history_rows
            ),

        "alphabet_coverage_summary":
            summarize_alphabet_coverage(
                alphabet_rows
            ),

        "noise_recovery_summary":
            summarize_noisy_recovery(
                noisy_rows
            ),

        "reference_estimator":
            (
                "Oracle nearest-clean-prototype decoder with "
                "majority vote over repeated observations. "
                "This is an identifiability reference ceiling, "
                "not the proposed neural method."
            ),

        "interpretation":
            (
                "Relation accuracy versus the restricted oracle "
                "isolates measurement noise. Relation accuracy "
                "versus full Phase 1A ground truth combines "
                "coverage ambiguity and measurement noise."
            ),

        "random_seed":
            BASE_RANDOM_SEED,

        "sanity_checks":
            "passed",
    }

    path = OUTPUT_DIR / "phase1e_summary.json"

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

    generators = load_generators()

    semigroup_rows = (
        load_semigroup_elements()
    )

    prototypes = load_prototypes()

    noise_conditions = (
        load_noise_conditions()
    )

    (
        mappings,
        _full_masks,
        full_relations,
    ) = build_full_ground_truth(
        semigroup_rows
    )

    history_rows = (
        calculate_history_subset_identifiability(
            mappings=mappings,
            full_relations=full_relations,
        )
    )

    write_history_subset_identifiability(
        history_rows
    )

    alphabet_rows = (
        calculate_alphabet_coverage(
            generators
        )
    )

    write_alphabet_coverage(
        alphabet_rows
    )

    print(
        "Exact coverage calculations complete."
    )

    noisy_rows = calculate_noisy_recovery(
        mappings=mappings,
        prototypes=prototypes,
        noise_conditions=noise_conditions,
        full_relations=full_relations,
    )

    write_noisy_recovery(
        noisy_rows
    )

    run_sanity_checks(
        generators=generators,
        semigroup_rows=semigroup_rows,
        prototypes=prototypes,
        noise_conditions=noise_conditions,
        history_rows=history_rows,
        alphabet_rows=alphabet_rows,
        noisy_rows=noisy_rows,
    )

    summary = write_summary(
        history_rows=history_rows,
        alphabet_rows=alphabet_rows,
        noisy_rows=noisy_rows,
    )

    print(
        "Phase 1E identifiability stress benchmark "
        "completed successfully."
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
