from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from pathlib import Path


STATE_COUNT = 8
OUTPUT_DIR = Path("outputs/phase1a_ground_truth")


# Each tuple gives the output state for source states 0 through 7.
#
# These operations were chosen to give us:
#   - fully information-preserving transformations,
#   - partial history collapse,
#   - incomparable information partitions,
#   - and complete reset behavior.
GENERATORS = {
    "identity": (0, 1, 2, 3, 4, 5, 6, 7),
    "cycle": (1, 2, 3, 4, 5, 6, 7, 0),
    "pair_collapse": (0, 0, 2, 2, 4, 4, 6, 6),
    "half_collapse": (0, 0, 0, 0, 4, 4, 4, 4),
    "parity_collapse": (0, 1, 0, 1, 0, 1, 0, 1),
    "reset": (0, 0, 0, 0, 0, 0, 0, 0),
}


def validate_transformation(
    name: str,
    mapping: tuple[int, ...],
) -> None:
    """Check that a transformation is valid on our finite state space."""

    if len(mapping) != STATE_COUNT:
        raise ValueError(
            f"{name}: expected {STATE_COUNT} outputs, "
            f"found {len(mapping)}"
        )

    invalid = [
        value
        for value in mapping
        if value < 0 or value >= STATE_COUNT
    ]

    if invalid:
        raise ValueError(
            f"{name}: outputs outside the state space: {invalid}"
        )


def compose(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    """
    Apply `first`, then apply `second`.

    If the starting state is s, the final state is:

        second[first[s]]
    """

    return tuple(
        second[first[state]]
        for state in range(STATE_COUNT)
    )


def partition_of(
    mapping: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """
    Return the source-state partition induced by an operation.

    Source states belong to the same block when the operation maps
    them to the same output state.
    """

    blocks: dict[int, list[int]] = defaultdict(list)

    for source_state, output_state in enumerate(mapping):
        blocks[output_state].append(source_state)

    return tuple(
        sorted(
            tuple(block)
            for block in blocks.values()
        )
    )


def partition_refines(
    finer: tuple[tuple[int, ...], ...],
    coarser: tuple[tuple[int, ...], ...],
) -> bool:
    """
    Return True when every block in `finer` lies inside a block
    in `coarser`.
    """

    coarse_sets = [
        set(block)
        for block in coarser
    ]

    for fine_block in finer:
        fine_set = set(fine_block)

        contained = any(
            fine_set.issubset(coarse_block)
            for coarse_block in coarse_sets
        )

        if not contained:
            return False

    return True


def compare_information(
    a: tuple[int, ...],
    b: tuple[int, ...],
) -> str:
    """
    Compare two deterministic operations by retained source information.

    For deterministic channels with a common source:

    A is at least as informative as B when the partition induced
    by A refines the partition induced by B.
    """

    partition_a = partition_of(a)
    partition_b = partition_of(b)

    a_refines_b = partition_refines(
        partition_a,
        partition_b,
    )

    b_refines_a = partition_refines(
        partition_b,
        partition_a,
    )

    if a_refines_b and b_refines_a:
        return "equivalent"

    if a_refines_b:
        return "a_more_informative"

    if b_refines_a:
        return "a_less_informative"

    return "incomparable"


def build_semigroup(
    generators: dict[str, tuple[int, ...]],
) -> dict[tuple[int, ...], list[str]]:
    """
    Enumerate every transformation reachable by composing generators.

    Breadth-first search also records one shortest generator sequence
    that produces each transformation.
    """

    identity = tuple(range(STATE_COUNT))

    queue = deque([identity])

    shortest_words: dict[
        tuple[int, ...],
        list[str],
    ] = {
        identity: []
    }

    while queue:
        current = queue.popleft()

        for generator_name, generator in generators.items():
            result = compose(
                current,
                generator,
            )

            if result not in shortest_words:
                shortest_words[result] = (
                    shortest_words[current]
                    + [generator_name]
                )

                queue.append(result)

    return shortest_words


def make_element_ids(
    elements: dict[tuple[int, ...], list[str]],
) -> dict[tuple[int, ...], str]:

    ordered = sorted(elements)

    return {
        mapping: f"t{index:03d}"
        for index, mapping in enumerate(ordered)
    }


def make_information_classes(
    element_ids: dict[tuple[int, ...], str],
):
    """
    Group transformations that induce the same information partition.

    Two operations can perform different state transformations while
    preserving exactly the same amount of source-state distinguishability.
    """

    members = defaultdict(list)

    for mapping, element_id in element_ids.items():
        partition = partition_of(mapping)
        members[partition].append(element_id)

    ordered_partitions = sorted(
        members,
        key=lambda partition: (
            -len(partition),
            partition,
        ),
    )

    class_ids = {
        partition: f"i{index:03d}"
        for index, partition in enumerate(ordered_partitions)
    }

    return class_ids, members


def find_hasse_edges(
    class_ids,
):
    """
    Find cover relations in the information partial order.

    Edges point from more informative classes toward less
    informative classes.
    """

    partitions = list(class_ids)
    edges = []

    for finer in partitions:
        for coarser in partitions:

            if finer == coarser:
                continue

            if not partition_refines(
                finer,
                coarser,
            ):
                continue

            has_intermediate = False

            for middle in partitions:

                if middle == finer or middle == coarser:
                    continue

                if (
                    partition_refines(finer, middle)
                    and partition_refines(middle, coarser)
                ):
                    has_intermediate = True
                    break

            if not has_intermediate:
                edges.append(
                    (
                        class_ids[finer],
                        class_ids[coarser],
                    )
                )

    return sorted(edges)


def partition_to_json(
    partition,
) -> str:

    return json.dumps(
        [
            list(block)
            for block in partition
        ],
        separators=(",", ":"),
    )


def run_sanity_checks(
    generators,
    semigroup,
    information_classes,
) -> None:
    """
    These checks encode relationships that we know analytically.

    If any one fails, Phase 1A should stop immediately.
    """

    assert compare_information(
        generators["identity"],
        generators["cycle"],
    ) == "equivalent"

    assert compare_information(
        generators["pair_collapse"],
        generators["half_collapse"],
    ) == "a_more_informative"

    assert compare_information(
        generators["pair_collapse"],
        generators["parity_collapse"],
    ) == "incomparable"

    assert compare_information(
        generators["half_collapse"],
        generators["parity_collapse"],
    ) == "incomparable"

    assert compare_information(
        generators["reset"],
        generators["half_collapse"],
    ) == "a_less_informative"

    assert len(set(generators["identity"])) == 8
    assert len(set(generators["cycle"])) == 8

    assert len(
        set(generators["pair_collapse"])
    ) == 4

    assert len(
        set(generators["half_collapse"])
    ) == 2

    assert len(
        set(generators["parity_collapse"])
    ) == 2

    assert len(
        set(generators["reset"])
    ) == 1

    # These values are frozen for this exact generator set.
    assert len(semigroup) == 104, (
        "Expected 104 semigroup elements, "
        f"found {len(semigroup)}."
    )

    assert len(information_classes) == 9, (
        "Expected 9 information classes, "
        f"found {len(information_classes)}."
    )


def write_generators(
    generators,
) -> None:

    rows = []

    for name, mapping in generators.items():

        rows.append(
            {
                "name": name,
                "mapping": list(mapping),
                "rank": len(set(mapping)),
                "bijective": (
                    len(set(mapping))
                    == STATE_COUNT
                ),
                "partition": [
                    list(block)
                    for block in partition_of(mapping)
                ],
            }
        )

    output_path = OUTPUT_DIR / "generators.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            rows,
            handle,
            indent=2,
        )


def write_semigroup_elements(
    shortest_words,
    element_ids,
    class_ids,
) -> None:

    output_path = (
        OUTPUT_DIR
        / "semigroup_elements.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "element_id",
                "mapping",
                "rank",
                "bijective",
                "information_class",
                "partition",
                "shortest_word",
                "word_length",
            ],
        )

        writer.writeheader()

        for mapping in sorted(
            element_ids,
            key=lambda item: element_ids[item],
        ):

            word = shortest_words[mapping]
            partition = partition_of(mapping)

            writer.writerow(
                {
                    "element_id":
                        element_ids[mapping],

                    "mapping":
                        json.dumps(
                            list(mapping),
                            separators=(",", ":"),
                        ),

                    "rank":
                        len(set(mapping)),

                    "bijective":
                        len(set(mapping))
                        == STATE_COUNT,

                    "information_class":
                        class_ids[partition],

                    "partition":
                        partition_to_json(
                            partition
                        ),

                    "shortest_word":
                        " -> ".join(word)
                        if word
                        else "identity",

                    "word_length":
                        len(word),
                }
            )


def write_composition_table(
    element_ids,
) -> None:

    reverse_ids = {
        element_id: mapping
        for mapping, element_id
        in element_ids.items()
    }

    output_path = (
        OUTPUT_DIR
        / "composition_table.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "first_id",
                "second_id",
                "result_id",
            ],
        )

        writer.writeheader()

        for first_id in sorted(reverse_ids):

            first = reverse_ids[first_id]

            for second_id in sorted(reverse_ids):

                second = reverse_ids[second_id]

                result = compose(
                    first,
                    second,
                )

                writer.writerow(
                    {
                        "first_id":
                            first_id,

                        "second_id":
                            second_id,

                        "result_id":
                            element_ids[result],
                    }
                )


def write_blackwell_relations(
    element_ids,
):

    reverse_ids = {
        element_id: mapping
        for mapping, element_id
        in element_ids.items()
    }

    relation_counts = Counter()

    output_path = (
        OUTPUT_DIR
        / "blackwell_relations.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "a_id",
                "b_id",
                "relation",
            ],
        )

        writer.writeheader()

        for a_id in sorted(reverse_ids):

            a = reverse_ids[a_id]

            for b_id in sorted(reverse_ids):

                b = reverse_ids[b_id]

                relation = compare_information(
                    a,
                    b,
                )

                relation_counts[
                    relation
                ] += 1

                writer.writerow(
                    {
                        "a_id": a_id,
                        "b_id": b_id,
                        "relation": relation,
                    }
                )

    return relation_counts


def write_information_classes(
    class_ids,
    members,
) -> None:

    output_path = (
        OUTPUT_DIR
        / "information_classes.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "information_class",
                "num_blocks",
                "partition",
                "member_count",
                "member_ids",
            ],
        )

        writer.writeheader()

        for partition, class_id in sorted(
            class_ids.items(),
            key=lambda item: item[1],
        ):

            member_ids = sorted(
                members[partition]
            )

            writer.writerow(
                {
                    "information_class":
                        class_id,

                    "num_blocks":
                        len(partition),

                    "partition":
                        partition_to_json(
                            partition
                        ),

                    "member_count":
                        len(member_ids),

                    "member_ids":
                        " ".join(
                            member_ids
                        ),
                }
            )


def write_hasse_edges(
    class_ids,
):

    edges = find_hasse_edges(
        class_ids
    )

    output_path = (
        OUTPUT_DIR
        / "information_hasse_edges.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "more_informative_class",
                "less_informative_class",
            ],
        )

        writer.writeheader()

        for source, target in edges:

            writer.writerow(
                {
                    "more_informative_class":
                        source,

                    "less_informative_class":
                        target,
                }
            )

    return edges


def write_summary(
    semigroup,
    information_classes,
    relation_counts,
    hasse_edges,
):

    rank_counts = Counter(
        len(set(mapping))
        for mapping in semigroup
    )

    summary = {
        "state_count":
            STATE_COUNT,

        "generator_count":
            len(GENERATORS),

        "semigroup_size":
            len(semigroup),

        "information_class_count":
            len(information_classes),

        "rank_distribution": {
            str(rank): count
            for rank, count
            in sorted(rank_counts.items())
        },

        "ordered_pair_relation_counts":
            dict(
                sorted(
                    relation_counts.items()
                )
            ),

        "hasse_edge_count":
            len(hasse_edges),

        "sanity_checks":
            "passed",
    }

    output_path = (
        OUTPUT_DIR
        / "phase1a_summary.json"
    )

    with output_path.open(
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

    for name, mapping in GENERATORS.items():
        validate_transformation(
            name,
            mapping,
        )

    semigroup = build_semigroup(
        GENERATORS
    )

    element_ids = make_element_ids(
        semigroup
    )

    class_ids, members = (
        make_information_classes(
            element_ids
        )
    )

    run_sanity_checks(
        generators=GENERATORS,
        semigroup=semigroup,
        information_classes=class_ids,
    )

    write_generators(
        GENERATORS
    )

    write_semigroup_elements(
        semigroup,
        element_ids,
        class_ids,
    )

    write_composition_table(
        element_ids
    )

    relation_counts = (
        write_blackwell_relations(
            element_ids
        )
    )

    write_information_classes(
        class_ids,
        members,
    )

    hasse_edges = write_hasse_edges(
        class_ids
    )

    summary = write_summary(
        semigroup=semigroup,
        information_classes=class_ids,
        relation_counts=relation_counts,
        hasse_edges=hasse_edges,
    )

    print(
        "Phase 1A ground-truth benchmark "
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
