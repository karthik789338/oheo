from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path


PHASE1A_DIR = Path("outputs/phase1a_ground_truth")
OUTPUT_DIR = Path("outputs/phase1b_symbolic_histories")

RANDOM_SEED = 1729

VARIANTS_PER_TRANSFORMATION = 16
EXHAUSTIVE_MAX_LENGTH = 4

# Neutral variants can contain a few explicit identity operations.
MAX_IDENTITY_INSERTS = 6


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def save_phase1a_hashes() -> None:
    """
    Record the Phase 1A files used by this phase.

    We do not alter the Phase 1A directory. These hashes give us a
    reproducibility record showing exactly which symbolic benchmark
    Phase 1B was generated from.
    """

    expected_files = [
        "generators.json",
        "semigroup_elements.csv",
        "composition_table.csv",
        "blackwell_relations.csv",
        "information_classes.csv",
        "information_hasse_edges.csv",
        "phase1a_summary.json",
    ]

    hashes = {}

    for filename in expected_files:
        path = PHASE1A_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Phase 1A file: {path}"
            )

        hashes[filename] = file_sha256(path)

    output_path = OUTPUT_DIR / "phase1a_input_hashes.json"

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
    """Load the frozen primitive operations from Phase 1A."""

    path = PHASE1A_DIR / "generators.json"
    rows = load_json(path)

    generators = {}

    for row in rows:
        name = row["name"]
        mapping = tuple(row["mapping"])
        generators[name] = mapping

    return generators


def load_semigroup_elements():
    """Load all 104 frozen transformations from Phase 1A."""

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
                json.loads(row["mapping"])
            )

            row["rank"] = int(row["rank"])
            row["word_length"] = int(
                row["word_length"]
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["element_id"]
    )

    return rows


def compose(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Apply `first`, followed by `second`.
    """

    return tuple(
        second[first[state]]
        for state in range(len(first))
    )


def apply_operation_sequence(
    sequence: tuple[str, ...],
    generators: dict[str, tuple[int, ...]],
    state_count: int,
) -> tuple[int, ...]:
    """
    Return the complete transformation represented by an
    operation sequence.
    """

    mapping = tuple(range(state_count))

    for operation_name in sequence:
        mapping = compose(
            mapping,
            generators[operation_name],
        )

    return mapping


def build_state_path(
    initial_state: int,
    sequence: tuple[str, ...],
    generators: dict[str, tuple[int, ...]],
) -> list[int]:
    """
    Apply a process sequence to one starting state and record every
    intermediate symbolic state.
    """

    current_state = initial_state
    path = [current_state]

    for operation_name in sequence:
        current_state = (
            generators[operation_name][current_state]
        )

        path.append(current_state)

    return path


def parse_shortest_word(row) -> tuple[str, ...]:
    """
    Recover the generator sequence stored by Phase 1A.

    The transformation identity has shortest length zero even though
    the text field says "identity".
    """

    if row["word_length"] == 0:
        return tuple()

    return tuple(
        row["shortest_word"].split(" -> ")
    )


def permutation_order(
    mapping: tuple[int, ...],
) -> int | None:
    """
    Return the order of a permutation.

    For a non-bijective transformation, return None.
    """

    state_count = len(mapping)

    if len(set(mapping)) != state_count:
        return None

    identity = tuple(range(state_count))
    current = identity

    # A finite permutation must eventually return to identity.
    # This bound is deliberately generous for our eight-state system.
    max_steps = state_count * state_count

    for order in range(1, max_steps + 1):
        current = compose(
            current,
            mapping,
        )

        if current == identity:
            return order

    raise RuntimeError(
        "Could not determine permutation order."
    )


def find_neutral_blocks(
    generators: dict[str, tuple[int, ...]],
):
    """
    Find generator repetitions whose total transformation is identity.

    For the current benchmark:
        identity^1 = identity
        cycle^8    = identity

    We discover this from the frozen transformations instead of
    hard-coding it.
    """

    neutral_blocks = []

    for name, mapping in generators.items():

        order = permutation_order(mapping)

        if order is None:
            continue

        neutral_blocks.append(
            tuple(
                name
                for _ in range(order)
            )
        )

    return neutral_blocks


def insert_block(
    sequence: tuple[str, ...],
    block: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, ...]:
    """Insert an operation block at a random position."""

    sequence_list = list(sequence)

    position = rng.randint(
        0,
        len(sequence_list),
    )

    new_sequence = (
        sequence_list[:position]
        + list(block)
        + sequence_list[position:]
    )

    return tuple(new_sequence)


def make_neutral_variant(
    base_sequence: tuple[str, ...],
    target_mapping: tuple[int, ...],
    generators,
    neutral_blocks,
    state_count,
    rng,
):
    """
    Create a syntactically different process history that has exactly
    the same total transformation.

    We only use operations whose added composition is identity.
    Every candidate is checked again against the true mapping.
    """

    sequence = tuple(base_sequence)

    identity_insertions = rng.randint(
        0,
        MAX_IDENTITY_INSERTS,
    )

    for _ in range(identity_insertions):
        sequence = insert_block(
            sequence,
            ("identity",),
            rng,
        )

    nontrivial_blocks = [
        block
        for block in neutral_blocks
        if len(block) > 1
    ]

    # The current benchmark has cycle^8 as a nontrivial identity block.
    # Inserting it gives us alternative histories without changing
    # the overall transformation.
    if nontrivial_blocks and rng.random() < 0.65:

        block = rng.choice(
            nontrivial_blocks
        )

        sequence = insert_block(
            sequence,
            block,
            rng,
        )

    result = apply_operation_sequence(
        sequence,
        generators,
        state_count,
    )

    if result != target_mapping:
        raise AssertionError(
            "Neutral rewrite changed the transformation."
        )

    return sequence


def build_sequence_catalog(
    generators,
    semigroup_rows,
):
    """
    Build two complementary sequence cohorts.

    Cohort 1:
        Equal number of histories for every semigroup transformation.

    Cohort 2:
        Exhaustive generator words of length 1 through 4.
    """

    state_count = len(
        next(iter(generators.values()))
    )

    mapping_lookup = {
        row["mapping_tuple"]: row
        for row in semigroup_rows
    }

    neutral_blocks = find_neutral_blocks(
        generators
    )

    rng = random.Random(
        RANDOM_SEED
    )

    catalog = {}

    # -------------------------------------------------------------
    # Balanced transformation cohort
    # -------------------------------------------------------------

    for row in semigroup_rows:

        target_mapping = row["mapping_tuple"]

        base_sequence = parse_shortest_word(
            row
        )

        base_result = apply_operation_sequence(
            base_sequence,
            generators,
            state_count,
        )

        if base_result != target_mapping:
            raise AssertionError(
                f"Shortest word does not reproduce "
                f"{row['element_id']}."
            )

        variants = {
            base_sequence
        }

        attempts = 0
        max_attempts = 50000

        while (
            len(variants)
            < VARIANTS_PER_TRANSFORMATION
        ):

            attempts += 1

            if attempts > max_attempts:
                raise RuntimeError(
                    "Could not generate enough distinct "
                    f"neutral variants for {row['element_id']}."
                )

            candidate = make_neutral_variant(
                base_sequence=base_sequence,
                target_mapping=target_mapping,
                generators=generators,
                neutral_blocks=neutral_blocks,
                state_count=state_count,
                rng=rng,
            )

            variants.add(candidate)

        for sequence in variants:

            result_mapping = (
                apply_operation_sequence(
                    sequence,
                    generators,
                    state_count,
                )
            )

            result_row = mapping_lookup[
                result_mapping
            ]

            if (
                result_row["element_id"]
                != row["element_id"]
            ):
                raise AssertionError(
                    "Balanced history maps to the "
                    "wrong transformation."
                )

            entry = catalog.setdefault(
                sequence,
                {
                    "balanced_variant": False,
                    "exhaustive_short": False,
                    "canonical_shortest": False,
                },
            )

            entry["balanced_variant"] = True

            if sequence == base_sequence:
                entry["canonical_shortest"] = True

    # -------------------------------------------------------------
    # Exhaustive short-composition cohort
    # -------------------------------------------------------------

    generator_names = tuple(
        generators.keys()
    )

    for length in range(
        1,
        EXHAUSTIVE_MAX_LENGTH + 1,
    ):

        for sequence in product(
            generator_names,
            repeat=length,
        ):

            sequence = tuple(sequence)

            entry = catalog.setdefault(
                sequence,
                {
                    "balanced_variant": False,
                    "exhaustive_short": False,
                    "canonical_shortest": False,
                },
            )

            entry["exhaustive_short"] = True

    # -------------------------------------------------------------
    # Attach exact ground truth
    # -------------------------------------------------------------

    rows = []

    ordered_sequences = sorted(
        catalog,
        key=lambda sequence: (
            len(sequence),
            sequence,
        ),
    )

    for index, sequence in enumerate(
        ordered_sequences
    ):

        result_mapping = apply_operation_sequence(
            sequence,
            generators,
            state_count,
        )

        if result_mapping not in mapping_lookup:
            raise AssertionError(
                "Generated sequence produced a transformation "
                "outside the frozen semigroup."
            )

        result_row = mapping_lookup[
            result_mapping
        ]

        metadata = catalog[sequence]

        rows.append(
            {
                "sequence_id":
                    f"s{index:05d}",

                "operation_sequence":
                    json.dumps(
                        list(sequence),
                        separators=(",", ":"),
                    ),

                "sequence_length":
                    len(sequence),

                "result_element_id":
                    result_row["element_id"],

                "information_class":
                    result_row[
                        "information_class"
                    ],

                "rank":
                    result_row["rank"],

                "balanced_variant":
                    metadata[
                        "balanced_variant"
                    ],

                "exhaustive_short":
                    metadata[
                        "exhaustive_short"
                    ],

                "canonical_shortest":
                    metadata[
                        "canonical_shortest"
                    ],
            }
        )

    return rows, mapping_lookup, neutral_blocks


def write_sequence_catalog(
    sequence_rows,
):
    path = (
        OUTPUT_DIR
        / "sequence_catalog.csv"
    )

    fieldnames = [
        "sequence_id",
        "operation_sequence",
        "sequence_length",
        "result_element_id",
        "information_class",
        "rank",
        "balanced_variant",
        "exhaustive_short",
        "canonical_shortest",
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
        writer.writerows(sequence_rows)


def write_symbolic_trajectories(
    sequence_rows,
    generators,
    semigroup_rows,
):
    """
    Generate one exact trajectory for every sequence and every
    possible initial state.
    """

    path = (
        OUTPUT_DIR
        / "symbolic_trajectories.csv"
    )

    state_count = len(
        next(iter(generators.values()))
    )

    element_lookup = {
        row["element_id"]: row
        for row in semigroup_rows
    }

    fieldnames = [
        "trajectory_id",
        "sequence_id",
        "initial_state",
        "final_state",
        "state_path",
        "result_element_id",
        "information_class",
        "rank",
        "preimage_size",
    ]

    trajectory_index = 0

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

        for sequence_row in sequence_rows:

            sequence = tuple(
                json.loads(
                    sequence_row[
                        "operation_sequence"
                    ]
                )
            )

            result_element_id = (
                sequence_row[
                    "result_element_id"
                ]
            )

            result_mapping = (
                element_lookup[
                    result_element_id
                ]["mapping_tuple"]
            )

            for initial_state in range(
                state_count
            ):

                state_path = build_state_path(
                    initial_state,
                    sequence,
                    generators,
                )

                final_state = state_path[-1]

                expected_final = (
                    result_mapping[
                        initial_state
                    ]
                )

                if final_state != expected_final:
                    raise AssertionError(
                        "Trajectory disagrees with composed "
                        "transformation."
                    )

                preimage_size = sum(
                    1
                    for output_state
                    in result_mapping
                    if output_state == final_state
                )

                writer.writerow(
                    {
                        "trajectory_id":
                            f"r{trajectory_index:07d}",

                        "sequence_id":
                            sequence_row[
                                "sequence_id"
                            ],

                        "initial_state":
                            initial_state,

                        "final_state":
                            final_state,

                        "state_path":
                            json.dumps(
                                state_path,
                                separators=(",", ":"),
                            ),

                        "result_element_id":
                            result_element_id,

                        "information_class":
                            sequence_row[
                                "information_class"
                            ],

                        "rank":
                            sequence_row[
                                "rank"
                            ],

                        "preimage_size":
                            preimage_size,
                    }
                )

                trajectory_index += 1

    return trajectory_index


def write_transformation_coverage(
    sequence_rows,
    semigroup_rows,
):
    """
    Show how many different process histories represent each of the
    104 transformations.
    """

    coverage = defaultdict(
        lambda: {
            "total": 0,
            "balanced": 0,
            "exhaustive": 0,
            "canonical": 0,
            "lengths": [],
        }
    )

    for row in sequence_rows:

        element_id = row[
            "result_element_id"
        ]

        item = coverage[element_id]

        item["total"] += 1

        item["lengths"].append(
            row["sequence_length"]
        )

        if row["balanced_variant"]:
            item["balanced"] += 1

        if row["exhaustive_short"]:
            item["exhaustive"] += 1

        if row["canonical_shortest"]:
            item["canonical"] += 1

    path = (
        OUTPUT_DIR
        / "transformation_history_coverage.csv"
    )

    fieldnames = [
        "element_id",
        "information_class",
        "rank",
        "total_sequence_count",
        "balanced_variant_count",
        "exhaustive_short_count",
        "canonical_shortest_count",
        "minimum_sequence_length",
        "maximum_sequence_length",
    ]

    row_lookup = {
        row["element_id"]: row
        for row in semigroup_rows
    }

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

        for element_id in sorted(
            row_lookup
        ):

            source = row_lookup[element_id]
            item = coverage[element_id]

            writer.writerow(
                {
                    "element_id":
                        element_id,

                    "information_class":
                        source[
                            "information_class"
                        ],

                    "rank":
                        source["rank"],

                    "total_sequence_count":
                        item["total"],

                    "balanced_variant_count":
                        item["balanced"],

                    "exhaustive_short_count":
                        item["exhaustive"],

                    "canonical_shortest_count":
                        item["canonical"],

                    "minimum_sequence_length":
                        min(item["lengths"]),

                    "maximum_sequence_length":
                        max(item["lengths"]),
                }
            )

    return coverage


def run_sanity_checks(
    generators,
    semigroup_rows,
    sequence_rows,
    coverage,
    neutral_blocks,
):
    state_count = len(
        next(iter(generators.values()))
    )

    if state_count != 8:
        raise AssertionError(
            f"Expected 8 states, found {state_count}."
        )

    if len(generators) != 6:
        raise AssertionError(
            f"Expected 6 generators, found {len(generators)}."
        )

    if len(semigroup_rows) != 104:
        raise AssertionError(
            "Phase 1A semigroup size changed."
        )

    information_classes = {
        row["information_class"]
        for row in semigroup_rows
    }

    if len(information_classes) != 9:
        raise AssertionError(
            "Expected 9 information classes."
        )

    covered_elements = {
        row["result_element_id"]
        for row in sequence_rows
    }

    if len(covered_elements) != 104:
        raise AssertionError(
            "Not all 104 transformations are represented."
        )

    for element_id in covered_elements:

        balanced_count = (
            coverage[element_id][
                "balanced"
            ]
        )

        if (
            balanced_count
            != VARIANTS_PER_TRANSFORMATION
        ):
            raise AssertionError(
                f"{element_id} has "
                f"{balanced_count} balanced histories; "
                f"expected "
                f"{VARIANTS_PER_TRANSFORMATION}."
            )

    exhaustive_expected = sum(
        len(generators) ** length
        for length in range(
            1,
            EXHAUSTIVE_MAX_LENGTH + 1,
        )
    )

    exhaustive_observed = sum(
        1
        for row in sequence_rows
        if row["exhaustive_short"]
    )

    if (
        exhaustive_observed
        != exhaustive_expected
    ):
        raise AssertionError(
            "Exhaustive short-sequence count is wrong: "
            f"expected {exhaustive_expected}, "
            f"found {exhaustive_observed}."
        )

    canonical_count = sum(
        1
        for row in sequence_rows
        if row["canonical_shortest"]
    )

    if canonical_count != 104:
        raise AssertionError(
            "Expected one canonical shortest history "
            "for every transformation."
        )

    if not any(
        len(block) > 1
        for block in neutral_blocks
    ):
        raise AssertionError(
            "No nontrivial neutral permutation block "
            "was discovered."
        )


def write_summary(
    generators,
    semigroup_rows,
    sequence_rows,
    trajectory_count,
    coverage,
    neutral_blocks,
):
    length_counts = Counter(
        row["sequence_length"]
        for row in sequence_rows
    )

    rank_counts = Counter(
        row["rank"]
        for row in sequence_rows
    )

    covered_classes = {
        row["information_class"]
        for row in sequence_rows
    }

    balanced_count = sum(
        1
        for row in sequence_rows
        if row["balanced_variant"]
    )

    exhaustive_count = sum(
        1
        for row in sequence_rows
        if row["exhaustive_short"]
    )

    canonical_count = sum(
        1
        for row in sequence_rows
        if row["canonical_shortest"]
    )

    minimum_balanced_coverage = min(
        item["balanced"]
        for item in coverage.values()
    )

    maximum_balanced_coverage = max(
        item["balanced"]
        for item in coverage.values()
    )

    summary = {
        "state_count":
            len(
                next(
                    iter(
                        generators.values()
                    )
                )
            ),

        "generator_count":
            len(generators),

        "semigroup_element_count":
            len(semigroup_rows),

        "information_classes_covered":
            len(covered_classes),

        "sequence_count":
            len(sequence_rows),

        "trajectory_count":
            trajectory_count,

        "balanced_variant_sequence_count":
            balanced_count,

        "exhaustive_short_sequence_count":
            exhaustive_count,

        "canonical_shortest_sequence_count":
            canonical_count,

        "balanced_sequences_per_transformation": {
            "minimum":
                minimum_balanced_coverage,

            "maximum":
                maximum_balanced_coverage,
        },

        "sequence_length_distribution": {
            str(length): count
            for length, count
            in sorted(
                length_counts.items()
            )
        },

        "sequence_rank_distribution": {
            str(rank): count
            for rank, count
            in sorted(
                rank_counts.items()
            )
        },

        "neutral_identity_blocks": [
            list(block)
            for block in neutral_blocks
        ],

        "random_seed":
            RANDOM_SEED,

        "sanity_checks":
            "passed",
    }

    path = (
        OUTPUT_DIR
        / "phase1b_summary.json"
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


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generators = load_generators()
    semigroup_rows = (
        load_semigroup_elements()
    )

    save_phase1a_hashes()

    (
        sequence_rows,
        _mapping_lookup,
        neutral_blocks,
    ) = build_sequence_catalog(
        generators,
        semigroup_rows,
    )

    write_sequence_catalog(
        sequence_rows
    )

    trajectory_count = (
        write_symbolic_trajectories(
            sequence_rows,
            generators,
            semigroup_rows,
        )
    )

    coverage = (
        write_transformation_coverage(
            sequence_rows,
            semigroup_rows,
        )
    )

    run_sanity_checks(
        generators=generators,
        semigroup_rows=semigroup_rows,
        sequence_rows=sequence_rows,
        coverage=coverage,
        neutral_blocks=neutral_blocks,
    )

    summary = write_summary(
        generators=generators,
        semigroup_rows=semigroup_rows,
        sequence_rows=sequence_rows,
        trajectory_count=trajectory_count,
        coverage=coverage,
        neutral_blocks=neutral_blocks,
    )

    print(
        "Phase 1B symbolic histories "
        "created successfully."
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        f"Outputs written to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
