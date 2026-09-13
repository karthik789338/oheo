from __future__ import annotations

import csv
import hashlib
import json
from collections import deque
from pathlib import Path


PHASE1_DIR = Path(
    "outputs/phase1a_ground_truth"
)

GENERATOR_PATH = (
    PHASE1_DIR
    / "generators.json"
)

SEMIGROUP_PATH = (
    PHASE1_DIR
    / "semigroup_elements.csv"
)

SUMMARY_PATH = (
    PHASE1_DIR
    / "phase1a_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit/"
    "recovered_transition_table"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "primitive_transition_table_long.csv"
)

PROVENANCE_PATH = (
    OUTPUT_DIR
    / "transition_recovery_provenance.json"
)


STATE_COUNT = 8
EXPECTED_GENERATOR_COUNT = 6
EXPECTED_SEMIGROUP_SIZE = 104

EXPECTED_GENERATOR_IDS = {
    "reset": "t000",
    "half_collapse": "t001",
    "pair_collapse": "t003",
    "parity_collapse": "t005",
    "identity": "t006",
    "cycle": "t020",
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
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


def load_generators():
    raw = load_json(
        GENERATOR_PATH
    )

    if isinstance(raw, dict):
        generators = {}

        for name, payload in raw.items():
            mapping = (
                payload["mapping"]
                if isinstance(payload, dict)
                else payload
            )

            generators[str(name)] = tuple(
                int(value)
                for value in mapping
            )

    elif isinstance(raw, list):
        generators = {}

        for row in raw:
            if not isinstance(row, dict):
                raise TypeError(
                    "Generator list contains a "
                    "non-dictionary record."
                )

            name = str(
                row["name"]
            )

            mapping = tuple(
                int(value)
                for value in row["mapping"]
            )

            if name in generators:
                raise AssertionError(
                    f"Duplicate generator: {name}"
                )

            generators[name] = mapping

    else:
        raise TypeError(
            "Unsupported generators.json structure."
        )

    if len(generators) != (
        EXPECTED_GENERATOR_COUNT
    ):
        raise AssertionError(
            "Expected six generators, found "
            f"{len(generators)}."
        )

    if set(generators) != set(
        EXPECTED_GENERATOR_IDS
    ):
        raise AssertionError(
            "Frozen generator names changed. "
            f"Observed: {sorted(generators)}"
        )

    for name, mapping in generators.items():
        if len(mapping) != STATE_COUNT:
            raise AssertionError(
                f"{name} has {len(mapping)} outputs, "
                "not eight."
            )

        if any(
            target < 0
            or target >= STATE_COUNT
            for target in mapping
        ):
            raise AssertionError(
                f"{name} contains an invalid state."
            )

    if len(
        set(generators.values())
    ) != EXPECTED_GENERATOR_COUNT:
        raise AssertionError(
            "Generator mappings are not unique."
        )

    return generators


def load_semigroup_elements():
    rows = []

    with SEMIGROUP_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            rows.append(
                {
                    "element_id":
                        str(
                            row["element_id"]
                        ),

                    "mapping":
                        tuple(
                            int(value)
                            for value in json.loads(
                                row["mapping"]
                            )
                        ),

                    "shortest_word":
                        str(
                            row["shortest_word"]
                        ),

                    "word_length":
                        int(
                            row["word_length"]
                        ),
                }
            )

    if len(rows) != EXPECTED_SEMIGROUP_SIZE:
        raise AssertionError(
            "Expected 104 frozen semigroup "
            f"elements, found {len(rows)}."
        )

    element_ids = [
        row["element_id"]
        for row in rows
    ]

    mappings = [
        row["mapping"]
        for row in rows
    ]

    if len(set(element_ids)) != (
        EXPECTED_SEMIGROUP_SIZE
    ):
        raise AssertionError(
            "Semigroup element IDs are not unique."
        )

    if len(set(mappings)) != (
        EXPECTED_SEMIGROUP_SIZE
    ):
        raise AssertionError(
            "Semigroup mappings are not unique."
        )

    return rows


def compose(
    current,
    generator,
):
    """
    Apply current first, then generator.

    This matches the Phase 1 word convention:
    cycle -> half_collapse means
    half_collapse(cycle(state)).
    """

    return tuple(
        generator[
            current[state]
        ]
        for state in range(
            STATE_COUNT
        )
    )


def reconstruct_semigroup(
    generators,
):
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

        for generator in generators.values():
            candidate = compose(
                current=current,
                generator=generator,
            )

            if candidate not in discovered:
                discovered.add(candidate)
                queue.append(candidate)

    return discovered


def verify_generator_element_ids(
    generators,
    semigroup_rows,
):
    mapping_to_element = {
        row["mapping"]:
            row["element_id"]
        for row in semigroup_rows
    }

    results = []

    for name, expected_id in (
        EXPECTED_GENERATOR_IDS.items()
    ):
        mapping = generators[name]

        if mapping not in mapping_to_element:
            raise AssertionError(
                f"Generator {name} is absent from "
                "semigroup_elements.csv."
            )

        observed_id = mapping_to_element[
            mapping
        ]

        if observed_id != expected_id:
            raise AssertionError(
                f"Generator {name} expected "
                f"{expected_id}, found {observed_id}."
            )

        results.append(
            {
                "generator_name":
                    name,

                "expected_element_id":
                    expected_id,

                "observed_element_id":
                    observed_id,

                "mapping":
                    list(mapping),

                "element_id_verified":
                    True,
            }
        )

    return results


def write_transition_table(
    generators,
    generator_verification,
):
    element_id_by_name = {
        row["generator_name"]:
            row["observed_element_id"]
        for row in generator_verification
    }

    rows = []

    for primitive_index, (
        name,
        mapping,
    ) in enumerate(
        generators.items()
    ):
        element_id = (
            element_id_by_name[name]
        )

        for source_state, target_state in enumerate(
            mapping
        ):
            rows.append(
                {
                    "primitive_id":
                        name,

                    "primitive_index":
                        primitive_index,

                    "element_id":
                        element_id,

                    "source_state_id":
                        source_state,

                    "target_state_id":
                        target_state,
                }
            )

    if len(rows) != (
        EXPECTED_GENERATOR_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Expected 48 normalized transition rows."
        )

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "primitive_id",
                "primitive_index",
                "element_id",
                "source_state_id",
                "target_state_id",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    phase1_summary = load_json(
        SUMMARY_PATH
    )

    if int(
        phase1_summary[
            "generator_count"
        ]
    ) != EXPECTED_GENERATOR_COUNT:
        raise AssertionError(
            "Phase 1 generator count changed."
        )

    if int(
        phase1_summary[
            "semigroup_size"
        ]
    ) != EXPECTED_SEMIGROUP_SIZE:
        raise AssertionError(
            "Phase 1 semigroup size changed."
        )

    generators = load_generators()

    semigroup_rows = (
        load_semigroup_elements()
    )

    generator_verification = (
        verify_generator_element_ids(
            generators=generators,
            semigroup_rows=semigroup_rows,
        )
    )

    reconstructed = (
        reconstruct_semigroup(
            generators
        )
    )

    if len(reconstructed) != (
        EXPECTED_SEMIGROUP_SIZE
    ):
        raise AssertionError(
            "The frozen generators reconstructed "
            f"{len(reconstructed)} elements instead "
            "of 104."
        )

    frozen_mappings = {
        row["mapping"]
        for row in semigroup_rows
    }

    if reconstructed != frozen_mappings:
        missing = (
            frozen_mappings
            - reconstructed
        )

        extra = (
            reconstructed
            - frozen_mappings
        )

        raise AssertionError(
            "Reconstructed semigroup differs from "
            "the frozen Phase 1 registry. "
            f"Missing={len(missing)}, "
            f"extra={len(extra)}."
        )

    normalized_rows = (
        write_transition_table(
            generators=generators,
            generator_verification=(
                generator_verification
            ),
        )
    )

    provenance = {
        "phase":
            (
                "Phase 4C-R3C frozen primitive "
                "transition normalization"
            ),

        "purpose":
            (
                "Normalize the original Phase 1A "
                "generator registry into the long-form "
                "transition table required by the "
                "Tier C v4 affine diagnostic."
            ),

        "source_generator_path":
            str(GENERATOR_PATH),

        "source_generator_sha256":
            sha256_file(
                GENERATOR_PATH
            ),

        "source_semigroup_path":
            str(SEMIGROUP_PATH),

        "source_semigroup_sha256":
            sha256_file(
                SEMIGROUP_PATH
            ),

        "source_summary_path":
            str(SUMMARY_PATH),

        "source_summary_sha256":
            sha256_file(
                SUMMARY_PATH
            ),

        "generator_count":
            len(generators),

        "generator_names":
            list(
                generators.keys()
            ),

        "generator_element_verification":
            generator_verification,

        "normalized_transition_row_count":
            len(normalized_rows),

        "reconstructed_semigroup_size":
            len(reconstructed),

        "frozen_semigroup_size":
            len(frozen_mappings),

        "reconstructed_semigroup_matches_frozen_registry":
            reconstructed
            == frozen_mappings,

        "normalized_transition_path":
            str(OUTPUT_CSV),

        "normalized_transition_sha256":
            sha256_file(
                OUTPUT_CSV
            ),

        "source_artifacts_modified":
            False,

        "transition_mappings_modified":
            False,

        "acceptance_thresholds_modified":
            False,

        "test_artifacts_accessed":
            False,

        "status":
            "verified_and_normalized",
    }

    with PROVENANCE_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            provenance,
            handle,
            indent=2,
        )

    print(
        "Frozen primitive transitions "
        "verified and normalized."
    )

    print(
        json.dumps(
            provenance,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
