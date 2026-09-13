from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import deque
from pathlib import Path

import numpy as np


STATE_COUNT = 8
PRIMITIVE_COUNT = 6
EXPECTED_SEMIGROUP_SIZE = 104

OUTPUT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit/"
    "recovered_transition_table"
)

SEARCH_ROOTS = (
    Path("outputs"),
)

SOURCE_ALIASES = (
    "source_state_id",
    "source_state",
    "input_state_id",
    "input_state",
    "from_state_id",
    "from_state",
    "state_id",
    "state",
)

TARGET_ALIASES = (
    "target_state_id",
    "target_state",
    "next_state_id",
    "next_state",
    "output_state_id",
    "output_state",
    "to_state_id",
    "to_state",
)

PRIMITIVE_ALIASES = (
    "primitive_id",
    "primitive",
    "primitive_name",
    "primitive_operation_id",
    "operation_id",
    "operation",
    "generator_id",
    "generator",
    "action_id",
    "action",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def natural_key(value):
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            str(value),
        )
    )


def parse_state(value) -> int:
    if isinstance(value, (int, np.integer)):
        state = int(value)

    elif isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(value)

        state = int(value)

    else:
        text = str(value).strip()

        try:
            numeric = float(text)

            if not numeric.is_integer():
                raise ValueError(value)

            state = int(numeric)

        except ValueError:
            match = re.search(
                r"(?:^|[^0-9])([0-7])$",
                text,
            )

            if match is None:
                raise ValueError(value)

            state = int(match.group(1))

    if state < 0 or state >= STATE_COUNT:
        raise ValueError(value)

    return state


def compose(current, primitive):
    """
    Apply current first and primitive second.
    """

    return tuple(
        primitive[current[state]]
        for state in range(STATE_COUNT)
    )


def semigroup_size(primitive_maps) -> int:
    identity = tuple(
        range(STATE_COUNT)
    )

    discovered = {
        identity
    }

    queue = deque(
        [identity]
    )

    while queue:
        current = queue.popleft()

        for primitive in primitive_maps:
            candidate = compose(
                current,
                primitive,
            )

            if candidate not in discovered:
                discovered.add(candidate)
                queue.append(candidate)

    return len(discovered)


def validate_candidate(
    source_path,
    source_format,
    labels,
    maps,
):
    if len(labels) != PRIMITIVE_COUNT:
        return None

    if len(maps) != PRIMITIVE_COUNT:
        return None

    normalized_maps = []

    for transformation in maps:
        if len(transformation) != STATE_COUNT:
            return None

        try:
            normalized = tuple(
                parse_state(value)
                for value in transformation
            )

        except ValueError:
            return None

        normalized_maps.append(normalized)

    if len(set(normalized_maps)) != (
        PRIMITIVE_COUNT
    ):
        return None

    closure_size = semigroup_size(
        normalized_maps
    )

    return {
        "source_path":
            Path(source_path),

        "source_format":
            source_format,

        "primitive_labels":
            tuple(str(label) for label in labels),

        "primitive_maps":
            tuple(normalized_maps),

        "semigroup_size":
            closure_size,
    }


def first_alias(fieldnames, aliases):
    available = {
        str(name).strip().lower():
            name
        for name in fieldnames
    }

    for alias in aliases:
        if alias in available:
            return available[alias]

    return None


def detect_long_csv(path, rows):
    if not rows:
        return []

    fieldnames = list(rows[0])

    source_column = first_alias(
        fieldnames,
        SOURCE_ALIASES,
    )

    target_column = first_alias(
        fieldnames,
        TARGET_ALIASES,
    )

    primitive_column = first_alias(
        fieldnames,
        PRIMITIVE_ALIASES,
    )

    if (
        source_column is None
        or target_column is None
        or primitive_column is None
    ):
        return []

    transitions = {}

    for row in rows:
        label = str(
            row[primitive_column]
        ).strip()

        if not label:
            continue

        try:
            source = parse_state(
                row[source_column]
            )

            target = parse_state(
                row[target_column]
            )

        except ValueError:
            return []

        key = (
            label,
            source,
        )

        if (
            key in transitions
            and transitions[key] != target
        ):
            return []

        transitions[key] = target

    labels = sorted(
        {
            label
            for label, _ in transitions
        },
        key=natural_key,
    )

    if len(labels) != PRIMITIVE_COUNT:
        return []

    maps = []

    for label in labels:
        try:
            transformation = tuple(
                transitions[
                    (
                        label,
                        state,
                    )
                ]
                for state in range(
                    STATE_COUNT
                )
            )

        except KeyError:
            return []

        maps.append(transformation)

    candidate = validate_candidate(
        source_path=path,
        source_format="csv_long",
        labels=labels,
        maps=maps,
    )

    return (
        [candidate]
        if candidate is not None
        else []
    )


def detect_state_rows_wide_csv(
    path,
    rows,
):
    """
    Layout:

    state_id, primitive_0, primitive_1, ..., primitive_5
    0, ...
    ...
    7, ...
    """

    if len(rows) != STATE_COUNT:
        return []

    fieldnames = list(rows[0])

    state_column = first_alias(
        fieldnames,
        SOURCE_ALIASES,
    )

    if state_column is None:
        return []

    primitive_columns = [
        column
        for column in fieldnames
        if column != state_column
    ]

    if len(primitive_columns) != (
        PRIMITIVE_COUNT
    ):
        return []

    state_rows = {}

    for row in rows:
        try:
            state = parse_state(
                row[state_column]
            )

        except ValueError:
            return []

        state_rows[state] = row

    if set(state_rows) != set(
        range(STATE_COUNT)
    ):
        return []

    labels = sorted(
        primitive_columns,
        key=natural_key,
    )

    maps = []

    for label in labels:
        try:
            transformation = tuple(
                parse_state(
                    state_rows[state][label]
                )
                for state in range(
                    STATE_COUNT
                )
            )

        except ValueError:
            return []

        maps.append(transformation)

    candidate = validate_candidate(
        source_path=path,
        source_format="csv_wide_state_rows",
        labels=labels,
        maps=maps,
    )

    return (
        [candidate]
        if candidate is not None
        else []
    )


def detect_primitive_rows_wide_csv(
    path,
    rows,
):
    """
    Layout:

    primitive_id, state_0, state_1, ..., state_7
    primitive_0, ...
    ...
    primitive_5, ...
    """

    if len(rows) != PRIMITIVE_COUNT:
        return []

    fieldnames = list(rows[0])

    primitive_column = first_alias(
        fieldnames,
        PRIMITIVE_ALIASES,
    )

    if primitive_column is None:
        return []

    possible_state_columns = [
        column
        for column in fieldnames
        if column != primitive_column
    ]

    state_columns = {}

    for column in possible_state_columns:
        try:
            state = parse_state(column)

        except ValueError:
            continue

        state_columns[state] = column

    if set(state_columns) != set(
        range(STATE_COUNT)
    ):
        return []

    ordered_rows = sorted(
        rows,
        key=lambda row:
            natural_key(
                row[primitive_column]
            ),
    )

    labels = [
        str(row[primitive_column]).strip()
        for row in ordered_rows
    ]

    maps = []

    for row in ordered_rows:
        try:
            transformation = tuple(
                parse_state(
                    row[
                        state_columns[state]
                    ]
                )
                for state in range(
                    STATE_COUNT
                )
            )

        except ValueError:
            return []

        maps.append(transformation)

    candidate = validate_candidate(
        source_path=path,
        source_format="csv_wide_primitive_rows",
        labels=labels,
        maps=maps,
    )

    return (
        [candidate]
        if candidate is not None
        else []
    )


def scan_csv(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

    except (
        UnicodeDecodeError,
        csv.Error,
        OSError,
    ):
        return []

    candidates = []

    candidates.extend(
        detect_long_csv(
            path,
            rows,
        )
    )

    candidates.extend(
        detect_state_rows_wide_csv(
            path,
            rows,
        )
    )

    candidates.extend(
        detect_primitive_rows_wide_csv(
            path,
            rows,
        )
    )

    return candidates


def recursively_find_json_candidates(
    path,
    value,
    location="root",
):
    candidates = []

    if isinstance(value, dict):
        sequence_entries = []

        for key, child in value.items():
            if (
                isinstance(child, (list, tuple))
                and len(child) == STATE_COUNT
            ):
                try:
                    transformation = tuple(
                        parse_state(item)
                        for item in child
                    )

                except ValueError:
                    continue

                sequence_entries.append(
                    (
                        str(key),
                        transformation,
                    )
                )

        if len(sequence_entries) == (
            PRIMITIVE_COUNT
        ):
            sequence_entries.sort(
                key=lambda item:
                    natural_key(item[0])
            )

            candidate = validate_candidate(
                source_path=path,
                source_format=(
                    "json_mapping:"
                    f"{location}"
                ),
                labels=[
                    item[0]
                    for item
                    in sequence_entries
                ],
                maps=[
                    item[1]
                    for item
                    in sequence_entries
                ],
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

        for key, child in value.items():
            candidates.extend(
                recursively_find_json_candidates(
                    path=path,
                    value=child,
                    location=(
                        f"{location}.{key}"
                    ),
                )
            )

    elif isinstance(value, list):
        array = None

        try:
            array = np.asarray(value)

        except Exception:
            array = None

        if (
            array is not None
            and array.ndim == 2
            and array.shape in {
                (
                    PRIMITIVE_COUNT,
                    STATE_COUNT,
                ),
                (
                    STATE_COUNT,
                    PRIMITIVE_COUNT,
                ),
            }
        ):
            if array.shape == (
                STATE_COUNT,
                PRIMITIVE_COUNT,
            ):
                array = array.T

            candidate = validate_candidate(
                source_path=path,
                source_format=(
                    "json_matrix:"
                    f"{location}"
                ),
                labels=[
                    f"primitive_{index}"
                    for index in range(
                        PRIMITIVE_COUNT
                    )
                ],
                maps=array.tolist(),
            )

            if candidate is not None:
                candidates.append(
                    candidate
                )

        for index, child in enumerate(value):
            candidates.extend(
                recursively_find_json_candidates(
                    path=path,
                    value=child,
                    location=(
                        f"{location}[{index}]"
                    ),
                )
            )

    return candidates


def scan_json(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            value = json.load(handle)

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        OSError,
    ):
        return []

    return recursively_find_json_candidates(
        path=path,
        value=value,
    )


def scan_numpy(path):
    candidates = []

    try:
        loaded = np.load(
            path,
            allow_pickle=False,
        )

    except Exception:
        return []

    arrays = []

    if isinstance(
        loaded,
        np.lib.npyio.NpzFile,
    ):
        try:
            for name in loaded.files:
                try:
                    array = loaded[name]

                except (
                    ValueError,
                    OSError,
                    EOFError,
                ):
                    # Some unrelated archives contain object arrays.
                    # Keep allow_pickle=False and skip those members.
                    continue

                arrays.append(
                    (
                        name,
                        array,
                    )
                )

        finally:
            loaded.close()

    else:
        arrays.append(
            (
                "array",
                loaded,
            )
        )

    for name, array in arrays:
        if array.ndim != 2:
            continue

        if array.shape == (
            PRIMITIVE_COUNT,
            STATE_COUNT,
        ):
            matrix = array

        elif array.shape == (
            STATE_COUNT,
            PRIMITIVE_COUNT,
        ):
            matrix = array.T

        else:
            continue

        candidate = validate_candidate(
            source_path=path,
            source_format=(
                f"{path.suffix[1:]}:{name}"
            ),
            labels=[
                f"primitive_{index}"
                for index in range(
                    PRIMITIVE_COUNT
                )
            ],
            maps=matrix.tolist(),
        )

        if candidate is not None:
            candidates.append(
                candidate
            )

    return candidates


def source_score(candidate):
    text = str(
        candidate["source_path"]
    ).lower()

    score = 0

    if "phase1" in text:
        score += 20

    if "primitive" in text:
        score += 10

    if "transition" in text:
        score += 10

    if "semigroup" in text:
        score += 8

    if "registry" in text:
        score += 4

    if "phase4" in text:
        score -= 5

    return score


def signature(candidate):
    """
    Ignore primitive labels when checking whether candidate
    artifacts define the same six transformations.
    """

    return tuple(
        sorted(
            candidate[
                "primitive_maps"
            ]
        )
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = []

    scanned_file_count = 0

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if OUTPUT_DIR in path.parents:
                continue

            if path.stat().st_size > (
                50 * 1024 * 1024
            ):
                continue

            suffix = path.suffix.lower()

            if suffix not in {
                ".csv",
                ".json",
                ".npy",
                ".npz",
            }:
                continue

            scanned_file_count += 1

            if suffix == ".csv":
                candidates.extend(
                    scan_csv(path)
                )

            elif suffix == ".json":
                candidates.extend(
                    scan_json(path)
                )

            else:
                candidates.extend(
                    scan_numpy(path)
                )

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["semigroup_size"]
        == EXPECTED_SEMIGROUP_SIZE
    ]

    print(
        f"Scanned files: {scanned_file_count}"
    )

    print(
        "Recognized six-primitive/eight-state "
        f"candidates: {len(candidates)}"
    )

    print(
        "Candidates producing the frozen "
        f"104-element semigroup: "
        f"{len(valid_candidates)}"
    )

    for candidate in valid_candidates:
        print()
        print(
            f"PATH:   {candidate['source_path']}"
        )

        print(
            f"FORMAT: {candidate['source_format']}"
        )

        print(
            "LABELS: "
            + ", ".join(
                candidate[
                    "primitive_labels"
                ]
            )
        )

    if not valid_candidates:
        raise SystemExit(
            "No recognized frozen transition artifact "
            "generated a 104-element semigroup."
        )

    groups = {}

    for candidate in valid_candidates:
        groups.setdefault(
            signature(candidate),
            [],
        ).append(candidate)

    if len(groups) != 1:
        report = []

        for group_candidates in (
            groups.values()
        ):
            report.append(
                [
                    str(
                        candidate[
                            "source_path"
                        ]
                    )
                    for candidate
                    in group_candidates
                ]
            )

        raise SystemExit(
            "Conflicting 104-element primitive systems "
            "were found. No normalized table was written.\n"
            + json.dumps(
                report,
                indent=2,
            )
        )

    equivalent_candidates = next(
        iter(groups.values())
    )

    equivalent_candidates.sort(
        key=lambda candidate: (
            -source_score(candidate),
            str(
                candidate[
                    "source_path"
                ]
            ),
        )
    )

    selected = equivalent_candidates[0]

    normalized_path = (
        OUTPUT_DIR
        / "primitive_transition_table_long.csv"
    )

    rows = []

    for primitive_index, (
        primitive_label,
        transformation,
    ) in enumerate(
        zip(
            selected[
                "primitive_labels"
            ],
            selected[
                "primitive_maps"
            ],
        )
    ):
        for source_state, target_state in enumerate(
            transformation
        ):
            rows.append(
                {
                    "primitive_id":
                        primitive_label,

                    "primitive_index":
                        primitive_index,

                    "source_state_id":
                        source_state,

                    "target_state_id":
                        target_state,
                }
            )

    with normalized_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "primitive_id",
                "primitive_index",
                "source_state_id",
                "target_state_id",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "purpose":
            (
                "Normalize an already-frozen primitive "
                "transition artifact for the Tier C v4 "
                "affine diagnostic."
            ),

        "selected_source_path":
            str(
                selected[
                    "source_path"
                ]
            ),

        "selected_source_format":
            selected[
                "source_format"
            ],

        "selected_source_sha256":
            sha256_file(
                selected[
                    "source_path"
                ]
            ),

        "primitive_labels":
            list(
                selected[
                    "primitive_labels"
                ]
            ),

        "primitive_count":
            PRIMITIVE_COUNT,

        "state_count":
            STATE_COUNT,

        "normalized_row_count":
            len(rows),

        "verified_semigroup_size":
            selected[
                "semigroup_size"
            ],

        "equivalent_candidate_count":
            len(
                equivalent_candidates
            ),

        "equivalent_candidates": [
            {
                "path":
                    str(
                        candidate[
                            "source_path"
                        ]
                    ),

                "format":
                    candidate[
                        "source_format"
                    ],

                "sha256":
                    sha256_file(
                        candidate[
                            "source_path"
                        ]
                    ),
            }
            for candidate
            in equivalent_candidates
        ],

        "normalized_path":
            str(normalized_path),

        "normalized_sha256":
            sha256_file(
                normalized_path
            ),

        "source_artifact_modified":
            False,

        "scientific_threshold_modified":
            False,

        "transition_semantics_modified":
            False,
    }

    provenance_path = (
        OUTPUT_DIR
        / "transition_recovery_provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            provenance,
            handle,
            indent=2,
        )

    print()
    print(
        "Recovered one unambiguous frozen "
        "primitive-transition system."
    )

    print(
        f"Selected source: {selected['source_path']}"
    )

    print(
        f"Normalized table: {normalized_path}"
    )

    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
