from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE1A_DIR = Path("outputs/phase1a_ground_truth")
PHASE1B_DIR = Path("outputs/phase1b_symbolic_histories")
PHASE1C_DIR = Path("outputs/phase1c_continuous_observations")

OUTPUT_DIR = Path("outputs/phase1d_compositional_splits")


EXPECTED_TRANSFORMATIONS = 104
EXPECTED_INFORMATION_CLASSES = 9
EXPECTED_SEQUENCES = 3144
EXPECTED_TRAJECTORIES = 25152
EXPECTED_POINTS = 254680

BALANCED_VARIANTS_PER_TRANSFORMATION = 16

# Only genuinely composed transformations can be held out.
MIN_HOLDOUT_WORD_LENGTH = 2

# Test transformations use a moderate but clearly compositional
# shortest representation rather than arbitrarily long words.
MAX_HOLDOUT_WORD_LENGTH = 5

IID_SPLIT_SEED = 5150


SPLIT_TO_CODE = {
    "train": 0,
    "val": 1,
    "test": 2,
}

CODE_TO_SPLIT = {
    0: "train",
    1: "val",
    2: "test",
}


def parse_bool(value: str) -> bool:
    value = value.strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise ValueError(
        f"Could not parse boolean value: {value}"
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


def save_input_hashes() -> None:
    """
    Record the exact frozen inputs used to construct the splits.
    """

    files = {
        "phase1a_generators":
            PHASE1A_DIR / "generators.json",

        "phase1a_semigroup_elements":
            PHASE1A_DIR / "semigroup_elements.csv",

        "phase1a_summary":
            PHASE1A_DIR / "phase1a_summary.json",

        "phase1b_sequence_catalog":
            PHASE1B_DIR / "sequence_catalog.csv",

        "phase1b_summary":
            PHASE1B_DIR / "phase1b_summary.json",

        "phase1c_sequence_index":
            PHASE1C_DIR / "sequence_index.csv",

        "phase1c_trajectory_index":
            PHASE1C_DIR / "trajectory_index.csv",

        "phase1c_observations":
            PHASE1C_DIR / "continuous_observations.npz",

        "phase1c_summary":
            PHASE1C_DIR / "phase1c_summary.json",
    }

    hashes = {}

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}"
            )

        hashes[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }

    path = OUTPUT_DIR / "input_hashes.json"

    with path.open(
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

    return {
        row["name"]:
            tuple(
                int(value)
                for value in row["mapping"]
            )

        for row in rows
    }


def load_semigroup_elements():
    path = (
        PHASE1A_DIR
        / "semigroup_elements.csv"
    )

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


def load_sequence_catalog():
    path = (
        PHASE1B_DIR
        / "sequence_catalog.csv"
    )

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

            row["balanced_variant"] = (
                parse_bool(
                    row["balanced_variant"]
                )
            )

            row["exhaustive_short"] = (
                parse_bool(
                    row["exhaustive_short"]
                )
            )

            row["canonical_shortest"] = (
                parse_bool(
                    row["canonical_shortest"]
                )
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["sequence_id"]
    )

    return rows


def load_phase1c_sequence_index():
    path = (
        PHASE1C_DIR
        / "sequence_index.csv"
    )

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            row["sequence_index"] = int(
                row["sequence_index"]
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["sequence_index"]
    )

    return rows


def load_phase1c_trajectory_index():
    path = (
        PHASE1C_DIR
        / "trajectory_index.csv"
    )

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:

        reader = csv.DictReader(handle)

        for row in reader:

            row["trajectory_index"] = int(
                row["trajectory_index"]
            )

            row["sequence_index"] = int(
                row["sequence_index"]
            )

            rows.append(row)

    rows.sort(
        key=lambda row: row["trajectory_index"]
    )

    return rows


def find_protected_element_ids(
    generators,
    semigroup_rows,
):
    """
    Primitive generator transformations must always remain in training.
    """

    mapping_to_element = {
        row["mapping_tuple"]:
            row["element_id"]

        for row in semigroup_rows
    }

    protected = {}

    for generator_name, mapping in generators.items():

        if mapping not in mapping_to_element:
            raise AssertionError(
                f"Generator {generator_name} "
                "is missing from the semigroup."
            )

        protected[
            generator_name
        ] = mapping_to_element[mapping]

    return protected


def select_composition_holdouts(
    semigroup_rows,
    protected_element_ids,
):
    """
    Select one validation and one test transformation from each
    information class.

    Validation:
        shortest eligible nonprimitive composition.

    Test:
        longest eligible composition within the frozen length-5
        window.

    Ties are broken by element ID, making the selection completely
    deterministic.
    """

    protected_ids = set(
        protected_element_ids.values()
    )

    candidates_by_class = defaultdict(list)

    for row in semigroup_rows:

        if (
            row["element_id"]
            in protected_ids
        ):
            continue

        if (
            row["word_length"]
            < MIN_HOLDOUT_WORD_LENGTH
        ):
            continue

        if (
            row["word_length"]
            > MAX_HOLDOUT_WORD_LENGTH
        ):
            continue

        candidates_by_class[
            row["information_class"]
        ].append(row)

    information_classes = sorted(
        {
            row["information_class"]
            for row in semigroup_rows
        }
    )

    if (
        len(information_classes)
        != EXPECTED_INFORMATION_CLASSES
    ):
        raise AssertionError(
            "Information-class count changed."
        )

    validation_ids = set()
    test_ids = set()

    selections = []

    for information_class in information_classes:

        candidates = candidates_by_class[
            information_class
        ]

        if len(candidates) < 2:
            raise AssertionError(
                f"{information_class} does not "
                "have enough holdout candidates."
            )

        validation = min(
            candidates,
            key=lambda row: (
                row["word_length"],
                row["element_id"],
            ),
        )

        longest_length = max(
            row["word_length"]
            for row in candidates
        )

        longest_candidates = [
            row
            for row in candidates
            if row["word_length"]
            == longest_length
        ]

        test = min(
            longest_candidates,
            key=lambda row:
                row["element_id"],
        )

        if (
            validation["element_id"]
            == test["element_id"]
        ):
            raise AssertionError(
                f"{information_class} selected "
                "the same transformation for "
                "validation and test."
            )

        validation_ids.add(
            validation["element_id"]
        )

        test_ids.add(
            test["element_id"]
        )

        selections.append(
            {
                "information_class":
                    information_class,

                "validation_element_id":
                    validation[
                        "element_id"
                    ],

                "validation_word_length":
                    validation[
                        "word_length"
                    ],

                "validation_shortest_word":
                    validation[
                        "shortest_word"
                    ],

                "test_element_id":
                    test[
                        "element_id"
                    ],

                "test_word_length":
                    test[
                        "word_length"
                    ],

                "test_shortest_word":
                    test[
                        "shortest_word"
                    ],
            }
        )

    if validation_ids & test_ids:
        raise AssertionError(
            "Validation and test transformations overlap."
        )

    return (
        validation_ids,
        test_ids,
        selections,
    )


def assign_transformation_splits(
    semigroup_rows,
    validation_ids,
    test_ids,
):
    split_lookup = {}

    for row in semigroup_rows:

        element_id = row[
            "element_id"
        ]

        if element_id in validation_ids:
            split = "val"

        elif element_id in test_ids:
            split = "test"

        else:
            split = "train"

        split_lookup[element_id] = split

    return split_lookup


def write_holdout_selection(
    selections,
) -> None:

    path = (
        OUTPUT_DIR
        / "holdout_selection.csv"
    )

    fieldnames = [
        "information_class",
        "validation_element_id",
        "validation_word_length",
        "validation_shortest_word",
        "test_element_id",
        "test_word_length",
        "test_shortest_word",
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
        writer.writerows(selections)


def write_transformation_split(
    semigroup_rows,
    split_lookup,
    protected_element_ids,
) -> None:

    protected_ids = set(
        protected_element_ids.values()
    )

    path = (
        OUTPUT_DIR
        / "transformation_split.csv"
    )

    fieldnames = [
        "element_id",
        "split",
        "information_class",
        "rank",
        "word_length",
        "shortest_word",
        "protected_primitive",
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

        for row in semigroup_rows:

            writer.writerow(
                {
                    "element_id":
                        row["element_id"],

                    "split":
                        split_lookup[
                            row["element_id"]
                        ],

                    "information_class":
                        row[
                            "information_class"
                        ],

                    "rank":
                        row["rank"],

                    "word_length":
                        row[
                            "word_length"
                        ],

                    "shortest_word":
                        row[
                            "shortest_word"
                        ],

                    "protected_primitive":
                        (
                            row["element_id"]
                            in protected_ids
                        ),
                }
            )


def assign_sequence_splits(
    sequence_rows,
    transformation_split_lookup,
):
    rows = []

    for row in sequence_rows:

        split = (
            transformation_split_lookup[
                row["result_element_id"]
            ]
        )

        rows.append(
            {
                **row,
                "composition_split":
                    split,
            }
        )

    return rows


def write_sequence_split(
    rows,
) -> None:

    path = (
        OUTPUT_DIR
        / "composition_sequence_split.csv"
    )

    fieldnames = [
        "sequence_id",
        "composition_split",
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

        for row in rows:

            writer.writerow(
                {
                    "sequence_id":
                        row["sequence_id"],

                    "composition_split":
                        row[
                            "composition_split"
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

                    "balanced_variant":
                        row[
                            "balanced_variant"
                        ],

                    "exhaustive_short":
                        row[
                            "exhaustive_short"
                        ],

                    "canonical_shortest":
                        row[
                            "canonical_shortest"
                        ],
                }
            )


def stable_sequence_hash(
    element_id: str,
    operation_sequence: str,
) -> str:
    """
    Stable ordering for the IID control split.
    """

    sequence = json.loads(
        operation_sequence
    )

    normalized = json.dumps(
        sequence,
        separators=(",", ":"),
    )

    payload = (
        f"{IID_SPLIT_SEED}|"
        f"{element_id}|"
        f"{normalized}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_iid_balanced_split(
    sequence_rows,
):
    """
    For every transformation:

        12 balanced histories -> train
         2 balanced histories -> validation
         2 balanced histories -> test

    The canonical shortest history is always kept in training.
    """

    grouped = defaultdict(list)

    for row in sequence_rows:

        if row["balanced_variant"]:
            grouped[
                row["result_element_id"]
            ].append(row)

    output_rows = []
    split_lookup = {}

    for element_id in sorted(grouped):

        group = grouped[
            element_id
        ]

        if (
            len(group)
            != BALANCED_VARIANTS_PER_TRANSFORMATION
        ):
            raise AssertionError(
                f"{element_id} has "
                f"{len(group)} balanced variants."
            )

        canonical = [
            row
            for row in group
            if row["canonical_shortest"]
        ]

        if len(canonical) != 1:
            raise AssertionError(
                f"{element_id} does not have exactly "
                "one canonical shortest sequence."
            )

        canonical_id = canonical[0][
            "sequence_id"
        ]

        remaining = [
            row
            for row in group
            if row["sequence_id"]
            != canonical_id
        ]

        remaining.sort(
            key=lambda row:
                stable_sequence_hash(
                    element_id,
                    row[
                        "operation_sequence"
                    ],
                )
        )

        validation_ids = {
            row["sequence_id"]
            for row in remaining[:2]
        }

        test_ids = {
            row["sequence_id"]
            for row in remaining[2:4]
        }

        for row in group:

            sequence_id = row[
                "sequence_id"
            ]

            if sequence_id in validation_ids:
                split = "val"

            elif sequence_id in test_ids:
                split = "test"

            else:
                split = "train"

            split_lookup[
                sequence_id
            ] = split

            output_rows.append(
                {
                    "sequence_id":
                        sequence_id,

                    "iid_split":
                        split,

                    "result_element_id":
                        element_id,

                    "information_class":
                        row[
                            "information_class"
                        ],

                    "operation_sequence":
                        row[
                            "operation_sequence"
                        ],

                    "sequence_length":
                        row[
                            "sequence_length"
                        ],

                    "canonical_shortest":
                        row[
                            "canonical_shortest"
                        ],
                }
            )

    output_rows.sort(
        key=lambda row: row["sequence_id"]
    )

    return output_rows, split_lookup


def write_iid_balanced_split(
    rows,
) -> None:

    path = (
        OUTPUT_DIR
        / "iid_balanced_sequence_split.csv"
    )

    fieldnames = [
        "sequence_id",
        "iid_split",
        "result_element_id",
        "information_class",
        "operation_sequence",
        "sequence_length",
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
        writer.writerows(rows)


def build_split_arrays(
    phase1c_sequence_rows,
    phase1c_trajectory_rows,
    composition_sequence_rows,
    iid_lookup,
):
    """
    Build integer split arrays aligned exactly with Phase 1C.

    Codes:
        -1 = excluded from this protocol
         0 = train
         1 = validation
         2 = test
    """

    sequence_count = len(
        phase1c_sequence_rows
    )

    composition_sequence_codes = np.full(
        sequence_count,
        -1,
        dtype=np.int8,
    )

    iid_sequence_codes = np.full(
        sequence_count,
        -1,
        dtype=np.int8,
    )

    composition_lookup = {
        row["sequence_id"]:
            row["composition_split"]

        for row in composition_sequence_rows
    }

    for row in phase1c_sequence_rows:

        index = row[
            "sequence_index"
        ]

        sequence_id = row[
            "sequence_id"
        ]

        composition_split = (
            composition_lookup[
                sequence_id
            ]
        )

        composition_sequence_codes[
            index
        ] = SPLIT_TO_CODE[
            composition_split
        ]

        if sequence_id in iid_lookup:

            iid_sequence_codes[
                index
            ] = SPLIT_TO_CODE[
                iid_lookup[
                    sequence_id
                ]
            ]

    trajectory_count = len(
        phase1c_trajectory_rows
    )

    composition_trajectory_codes = (
        np.full(
            trajectory_count,
            -1,
            dtype=np.int8,
        )
    )

    iid_trajectory_codes = np.full(
        trajectory_count,
        -1,
        dtype=np.int8,
    )

    for row in phase1c_trajectory_rows:

        trajectory_index = row[
            "trajectory_index"
        ]

        sequence_index = row[
            "sequence_index"
        ]

        composition_trajectory_codes[
            trajectory_index
        ] = composition_sequence_codes[
            sequence_index
        ]

        iid_trajectory_codes[
            trajectory_index
        ] = iid_sequence_codes[
            sequence_index
        ]

    observation_path = (
        PHASE1C_DIR
        / "continuous_observations.npz"
    )

    with np.load(
        observation_path
    ) as data:

        point_sequence_index = np.array(
            data["sequence_index"],
            dtype=np.int32,
        )

    composition_point_codes = (
        composition_sequence_codes[
            point_sequence_index
        ]
    )

    iid_point_codes = (
        iid_sequence_codes[
            point_sequence_index
        ]
    )

    return {
        "composition_sequence_split":
            composition_sequence_codes,

        "composition_trajectory_split":
            composition_trajectory_codes,

        "composition_point_split":
            composition_point_codes,

        "iid_balanced_sequence_split":
            iid_sequence_codes,

        "iid_balanced_trajectory_split":
            iid_trajectory_codes,

        "iid_balanced_point_split":
            iid_point_codes,
    }


def save_split_arrays(
    arrays,
) -> None:

    path = (
        OUTPUT_DIR
        / "split_indices.npz"
    )

    np.savez(
        path,
        **arrays,
    )


def count_codes(
    array,
    include_excluded=False,
):
    counts = Counter(
        int(value)
        for value in array
    )

    result = {}

    if include_excluded:
        result["excluded"] = counts[-1]

    for code in [
        0,
        1,
        2,
    ]:
        result[
            CODE_TO_SPLIT[code]
        ] = counts[code]

    return result


def run_sanity_checks(
    semigroup_rows,
    protected_element_ids,
    transformation_split_lookup,
    selections,
    sequence_rows,
    composition_sequence_rows,
    iid_rows,
    arrays,
):
    if (
        len(semigroup_rows)
        != EXPECTED_TRANSFORMATIONS
    ):
        raise AssertionError(
            "Transformation count changed."
        )

    classes = {
        row["information_class"]
        for row in semigroup_rows
    }

    if (
        len(classes)
        != EXPECTED_INFORMATION_CLASSES
    ):
        raise AssertionError(
            "Information-class count changed."
        )

    transformation_counts = Counter(
        transformation_split_lookup.values()
    )

    expected_transformation_counts = {
        "train": 86,
        "val": 9,
        "test": 9,
    }

    if (
        dict(transformation_counts)
        != expected_transformation_counts
    ):
        raise AssertionError(
            "Unexpected transformation split: "
            f"{dict(transformation_counts)}"
        )

    for generator_name, element_id in (
        protected_element_ids.items()
    ):

        if (
            transformation_split_lookup[
                element_id
            ]
            != "train"
        ):
            raise AssertionError(
                f"Primitive {generator_name} "
                "was held out."
            )

    if len(selections) != 9:
        raise AssertionError(
            "Expected one validation/test pair "
            "for every information class."
        )

    for selection in selections:

        if not (
            MIN_HOLDOUT_WORD_LENGTH
            <= selection[
                "validation_word_length"
            ]
            <= MAX_HOLDOUT_WORD_LENGTH
        ):
            raise AssertionError(
                "Invalid validation composition length."
            )

        if (
            selection[
                "test_word_length"
            ]
            != MAX_HOLDOUT_WORD_LENGTH
        ):
            raise AssertionError(
                "Each test transformation should "
                "have shortest word length 5."
            )

    if len(sequence_rows) != EXPECTED_SEQUENCES:
        raise AssertionError(
            "Sequence count changed."
        )

    composition_counts = Counter(
        row["composition_split"]
        for row in composition_sequence_rows
    )

    expected_composition_counts = {
        "train": 2554,
        "val": 446,
        "test": 144,
    }

    if (
        dict(composition_counts)
        != expected_composition_counts
    ):
        raise AssertionError(
            "Unexpected all-sequence composition split: "
            f"{dict(composition_counts)}"
        )

    balanced_counts = Counter()

    canonical_counts = Counter()

    for row in composition_sequence_rows:

        if row["balanced_variant"]:
            balanced_counts[
                row["composition_split"]
            ] += 1

        if row["canonical_shortest"]:
            canonical_counts[
                row["composition_split"]
            ] += 1

    expected_balanced_counts = {
        "train": 1376,
        "val": 144,
        "test": 144,
    }

    if (
        dict(balanced_counts)
        != expected_balanced_counts
    ):
        raise AssertionError(
            "Balanced OOD split changed: "
            f"{dict(balanced_counts)}"
        )

    expected_canonical_counts = {
        "train": 86,
        "val": 9,
        "test": 9,
    }

    if (
        dict(canonical_counts)
        != expected_canonical_counts
    ):
        raise AssertionError(
            "Canonical OOD split changed."
        )

    iid_counts = Counter(
        row["iid_split"]
        for row in iid_rows
    )

    expected_iid_counts = {
        "train": 1248,
        "val": 208,
        "test": 208,
    }

    if (
        dict(iid_counts)
        != expected_iid_counts
    ):
        raise AssertionError(
            "IID control split changed: "
            f"{dict(iid_counts)}"
        )

    composition_trajectory_counts = (
        count_codes(
            arrays[
                "composition_trajectory_split"
            ]
        )
    )

    if composition_trajectory_counts != {
        "train": 20432,
        "val": 3568,
        "test": 1152,
    }:
        raise AssertionError(
            "Composition trajectory split changed."
        )

    composition_point_counts = count_codes(
        arrays[
            "composition_point_split"
        ]
    )

    if composition_point_counts != {
        "train": 213584,
        "val": 24904,
        "test": 16192,
    }:
        raise AssertionError(
            "Composition point split changed."
        )

    iid_sequence_counts = count_codes(
        arrays[
            "iid_balanced_sequence_split"
        ],
        include_excluded=True,
    )

    if iid_sequence_counts != {
        "excluded": 1480,
        "train": 1248,
        "val": 208,
        "test": 208,
    }:
        raise AssertionError(
            "IID sequence-index split changed."
        )

    iid_trajectory_counts = count_codes(
        arrays[
            "iid_balanced_trajectory_split"
        ],
        include_excluded=True,
    )

    if iid_trajectory_counts != {
        "excluded": 11840,
        "train": 9984,
        "val": 1664,
        "test": 1664,
    }:
        raise AssertionError(
            "IID trajectory split changed."
        )

    iid_point_counts = count_codes(
        arrays[
            "iid_balanced_point_split"
        ],
        include_excluded=True,
    )

    if iid_point_counts != {
        "excluded": 57224,
        "train": 146848,
        "val": 25616,
        "test": 24992,
    }:
        raise AssertionError(
            "IID point split changed."
        )


def write_summary(
    protected_element_ids,
    selections,
    transformation_split_lookup,
    composition_sequence_rows,
    iid_rows,
    arrays,
):
    transformation_counts = Counter(
        transformation_split_lookup.values()
    )

    sequence_counts = Counter(
        row["composition_split"]
        for row in composition_sequence_rows
    )

    balanced_counts = Counter(
        row["composition_split"]
        for row in composition_sequence_rows
        if row["balanced_variant"]
    )

    canonical_counts = Counter(
        row["composition_split"]
        for row in composition_sequence_rows
        if row["canonical_shortest"]
    )

    iid_counts = Counter(
        row["iid_split"]
        for row in iid_rows
    )

    summary = {
        "composition_ood": {
            "selection_rule": (
                "One validation and one test transformation "
                "per information class. Primitive generators are "
                "protected in training. Validation uses the shortest "
                "eligible nonprimitive transformation. Test uses "
                "the longest eligible transformation within the "
                "frozen shortest-word range 2 through 5."
            ),

            "transformation_counts":
                dict(transformation_counts),

            "all_sequence_counts":
                dict(sequence_counts),

            "balanced_sequence_counts":
                dict(balanced_counts),

            "canonical_sequence_counts":
                dict(canonical_counts),

            "trajectory_counts":
                count_codes(
                    arrays[
                        "composition_trajectory_split"
                    ]
                ),

            "point_counts":
                count_codes(
                    arrays[
                        "composition_point_split"
                    ]
                ),

            "protected_primitives":
                protected_element_ids,

            "holdouts":
                selections,
        },

        "iid_balanced_control": {
            "rule": (
                "For each transformation, retain 12 balanced "
                "histories for training, 2 for validation, and "
                "2 for testing. The canonical shortest history "
                "is always retained in training."
            ),

            "sequence_counts":
                dict(iid_counts),

            "trajectory_counts":
                count_codes(
                    arrays[
                        "iid_balanced_trajectory_split"
                    ],
                    include_excluded=True,
                ),

            "point_counts":
                count_codes(
                    arrays[
                        "iid_balanced_point_split"
                    ],
                    include_excluded=True,
                ),

            "seed":
                IID_SPLIT_SEED,
        },

        "split_codes": {
            "-1": "excluded",
            "0": "train",
            "1": "val",
            "2": "test",
        },

        "sanity_checks":
            "passed",
    }

    path = (
        OUTPUT_DIR
        / "phase1d_summary.json"
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

    generators = load_generators()

    semigroup_rows = (
        load_semigroup_elements()
    )

    sequence_rows = (
        load_sequence_catalog()
    )

    phase1c_sequence_rows = (
        load_phase1c_sequence_index()
    )

    phase1c_trajectory_rows = (
        load_phase1c_trajectory_index()
    )

    protected_element_ids = (
        find_protected_element_ids(
            generators,
            semigroup_rows,
        )
    )

    (
        validation_ids,
        test_ids,
        selections,
    ) = select_composition_holdouts(
        semigroup_rows,
        protected_element_ids,
    )

    transformation_split_lookup = (
        assign_transformation_splits(
            semigroup_rows,
            validation_ids,
            test_ids,
        )
    )

    write_holdout_selection(
        selections
    )

    write_transformation_split(
        semigroup_rows,
        transformation_split_lookup,
        protected_element_ids,
    )

    composition_sequence_rows = (
        assign_sequence_splits(
            sequence_rows,
            transformation_split_lookup,
        )
    )

    write_sequence_split(
        composition_sequence_rows
    )

    (
        iid_rows,
        iid_lookup,
    ) = build_iid_balanced_split(
        sequence_rows
    )

    write_iid_balanced_split(
        iid_rows
    )

    arrays = build_split_arrays(
        phase1c_sequence_rows=(
            phase1c_sequence_rows
        ),
        phase1c_trajectory_rows=(
            phase1c_trajectory_rows
        ),
        composition_sequence_rows=(
            composition_sequence_rows
        ),
        iid_lookup=iid_lookup,
    )

    save_split_arrays(
        arrays
    )

    run_sanity_checks(
        semigroup_rows=semigroup_rows,
        protected_element_ids=(
            protected_element_ids
        ),
        transformation_split_lookup=(
            transformation_split_lookup
        ),
        selections=selections,
        sequence_rows=sequence_rows,
        composition_sequence_rows=(
            composition_sequence_rows
        ),
        iid_rows=iid_rows,
        arrays=arrays,
    )

    summary = write_summary(
        protected_element_ids=(
            protected_element_ids
        ),
        selections=selections,
        transformation_split_lookup=(
            transformation_split_lookup
        ),
        composition_sequence_rows=(
            composition_sequence_rows
        ),
        iid_rows=iid_rows,
        arrays=arrays,
    )

    print(
        "Phase 1D compositional splits "
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
