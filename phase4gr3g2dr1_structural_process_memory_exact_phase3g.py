from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4fr3b_run_tier_c_v4_tuning as runtime
import phase4gr3g2b_primary_confirmatory_test as primary_eval

import phase2e_evaluate_structural_recovery as structural
import phase2f_audit_ood_structural_generalization as structural_ood
import phase3g_evaluate_structural_recovery as phase3g_structural


PROTOCOL_VERSION = "tier_c_v4"

STATE_COUNT = 8
OPERATION_COUNT = 6
TRANSFORMATION_COUNT = 104
INFORMATION_CLASS_COUNT = 9

EXPECTED_SEEDS = [
    11,
    23,
    37,
    53,
    71,
]

NOISE_LEVELS = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
]

NOISE_TO_INDEX = {
    0.0: 0,
    0.1: 1,
    0.25: 2,
    0.5: 3,
    1.0: 4,
}

TEST_CELLS = [
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
]

EXPECTED_SEQUENCE_COUNT = 144
EXPECTED_TRAJECTORY_COUNT = 1152
EXPECTED_TRAJECTORIES_PER_SEQUENCE = 8

EXPECTED_ALL_RELATION_PAIRS = 10712
EXPECTED_TEST_INVOLVED_PAIRS = 1782
EXPECTED_TEST_TEST_PAIRS = 72

EVALUATION_BATCH_SIZE = 16
ALIGNMENT_BATCH_SIZE = 32


# Frozen primitive operation -> semigroup element binding.
#
# These are Phase-1 symbolic identities, not selected from
# Phase-4 test performance.
PRIMITIVE_ELEMENT_IDS = {
    0: "t006",  # identity
    1: "t020",  # cycle
    2: "t003",  # pair_collapse
    3: "t001",  # half_collapse
    4: "t005",  # parity_collapse
    5: "t000",  # reset
}


G1_ROOT = Path(
    "outputs/"
    "phase4gr3g1_tier_c_v4_sealed_test"
)

G1_SUMMARY = (
    G1_ROOT
    / "phase4gr3g1_sealed_test_generation_summary.json"
)

G2B_RESULT = Path(
    "outputs/"
    "phase4gr3g2b_tier_c_v4_primary_confirmatory/"
    "primary_confirmatory_result.json"
)

G2C_SUMMARY = Path(
    "outputs/"
    "phase4gr3g2c_tier_c_v4_full_predictive/"
    "phase4gr3g2c_full_predictive_summary.json"
)

FINAL_REGISTRY = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

METRIC_REGISTRY = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_metric_registry.csv"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g2dr1_tier_c_v4_exact_phase3g_structural"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoint_results"
)

STATE_PATH_ROWS_PATH = (
    OUTPUT_DIR
    / "state_path_sequence_metrics.csv"
)

TRANSFORMATION_ROWS_PATH = (
    OUTPUT_DIR
    / "transformation_metrics.csv"
)

CHECKPOINT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "ocm_checkpoint_structural_summary.csv"
)

NOISE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "ocm_noise_structural_summary.csv"
)

STATE_PATH_SUMMARY_PATH = (
    OUTPUT_DIR
    / "state_path_condition_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase4gr3g2d_structural_summary.json"
)

HASHES_PATH = (
    OUTPUT_DIR
    / "phase4gr3g2d_result_hashes.json"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(rows),
        f"No rows supplied for {path}.",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
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


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
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


def parse_json_int_list(
    value: str,
) -> tuple[int, ...] | None:
    try:
        parsed = json.loads(
            value
        )
    except Exception:
        return None

    if not isinstance(
        parsed,
        list,
    ):
        return None

    try:
        return tuple(
            int(item)
            for item in parsed
        )
    except Exception:
        return None


def verify_frozen_state() -> dict[str, str]:
    print(
        "[1/9] Verifying frozen Phase-4 state"
    )

    for path in (
        G1_SUMMARY,
        G2B_RESULT,
        G2C_SUMMARY,
        FINAL_REGISTRY,
        METRIC_REGISTRY,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    g1 = load_json(
        G1_SUMMARY
    )

    require(
        g1[
            "phase4gr3g1_status"
        ] == "sealed_test_generation_frozen",
        "G1 is not frozen.",
    )

    require(
        g1[
            "test_open_count"
        ] == 1,
        "Test-open count changed.",
    )

    require(
        g1[
            "test_regeneration_authorized"
        ] is False,
        "Test regeneration became authorized.",
    )

    g2b = load_json(
        G2B_RESULT
    )

    require(
        g2b[
            "phase4gr3g2b_status"
        ] == "primary_confirmatory_result_frozen",
        "G2B is not frozen.",
    )

    g2c = load_json(
        G2C_SUMMARY
    )

    require(
        g2c[
            "phase4gr3g2c_status"
        ] == "full_predictive_results_frozen",
        "G2C is not frozen.",
    )

    require(
        g2c[
            "primary_predictive_superiority_supported"
        ] is False,
        "Stored primary result changed.",
    )

    metric_rows = load_csv(
        METRIC_REGISTRY
    )

    metric_ids = {
        row[
            "metric_id"
        ]
        for row in metric_rows
    }

    required_metrics = {
        "state_path_accuracy",
        "exact_transformation_rate",
        "information_class_accuracy",
        "blackwell_relation_balanced_accuracy",
    }

    require(
        required_metrics.issubset(
            metric_ids
        ),
        (
            "Frozen structural metric "
            "registry changed."
        ),
    )

    return {
        "g1_summary_sha256":
            sha256_file(
                G1_SUMMARY
            ),

        "g2b_result_sha256":
            sha256_file(
                G2B_RESULT
            ),

        "g2c_summary_sha256":
            sha256_file(
                G2C_SUMMARY
            ),

        "final_registry_sha256":
            sha256_file(
                FINAL_REGISTRY
            ),

        "metric_registry_sha256":
            sha256_file(
                METRIC_REGISTRY
            ),

        "phase2e_source_sha256":
            sha256_file(
                Path(
                    structural.__file__
                )
            ),

        "phase2f_source_sha256":
            sha256_file(
                Path(
                    structural_ood.__file__
                )
            ),
    }


def load_ocm_registry(
) -> list[dict[str, str]]:
    print(
        "[2/9] Loading 25 frozen OCM checkpoints"
    )

    rows = load_csv(
        FINAL_REGISTRY
    )

    ocm_rows = [
        row
        for row in rows
        if row[
            "model_id"
        ] == "OCM"
    ]

    require(
        len(
            ocm_rows
        ) == 25,
        (
            "Expected 25 frozen OCM "
            f"checkpoints; found {len(ocm_rows)}."
        ),
    )

    grouped = defaultdict(list)

    for row in ocm_rows:
        noise = float(
            row[
                "noise_fraction"
            ]
        )

        seed = int(
            row[
                "effective_seed"
            ]
        )

        grouped[
            noise
        ].append(
            seed
        )

        checkpoint = Path(
            row[
                "best_checkpoint_path"
            ]
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                checkpoint
            )

        require(
            sha256_file(
                checkpoint
            )
            == row[
                "best_checkpoint_sha256"
            ],
            (
                "Frozen OCM checkpoint "
                f"changed: {checkpoint}"
            ),
        )

    require(
        sorted(
            grouped
        )
        == NOISE_LEVELS,
        "OCM noise grid changed.",
    )

    for noise in NOISE_LEVELS:
        require(
            sorted(
                grouped[
                    noise
                ]
            )
            == EXPECTED_SEEDS,
            (
                f"OCM seed grid changed "
                f"at noise={noise}."
            ),
        )

    ocm_rows.sort(
        key=lambda row: (
            float(
                row[
                    "noise_fraction"
                ]
            ),
            int(
                row[
                    "effective_seed"
                ]
            ),
        )
    )

    return ocm_rows


def candidate_phase1_csvs(
    prefix: str,
) -> list[Path]:
    candidates = []

    for root in Path(
        "outputs"
    ).glob(
        prefix + "*"
    ):
        if not root.is_dir():
            continue

        candidates.extend(
            root.rglob(
                "*.csv"
            )
        )

    return sorted(
        set(
            candidates
        )
    )


def discover_true_transformation_table(
) -> tuple[
    Path,
    str,
    list[dict[str, str]],
]:
    candidates = (
        candidate_phase1_csvs(
            "phase1a"
        )
    )

    if not candidates:
        candidates = list(
            Path(
                "outputs"
            ).rglob(
                "*.csv"
            )
        )

    matches = []

    for path in candidates:
        try:
            rows = load_csv(
                path
            )
        except Exception:
            continue

        if len(rows) != TRANSFORMATION_COUNT:
            continue

        if not rows:
            continue

        if "element_id" not in rows[0]:
            continue

        columns = list(
            rows[0].keys()
        )

        preferred = [
            column
            for column in columns
            if (
                "mapping"
                in column.lower()
                or "transformation"
                in column.lower()
            )
        ]

        for column in (
            preferred
            + [
                column
                for column in columns
                if column
                not in preferred
            ]
        ):
            parsed = [
                parse_json_int_list(
                    row[
                        column
                    ]
                )
                for row in rows
            ]

            if all(
                value is not None
                and len(value)
                == STATE_COUNT
                and all(
                    0 <= item
                    < STATE_COUNT
                    for item in value
                )
                for value in parsed
            ):
                matches.append(
                    (
                        path,
                        column,
                        rows,
                    )
                )

                break

    require(
        len(
            matches
        ) >= 1,
        (
            "Could not discover the frozen "
            "104-element transformation table."
        ),
    )

    matches.sort(
        key=lambda item: (
            0
            if (
                "semigroup"
                in item[0].name.lower()
                or "transformation"
                in item[0].name.lower()
            )
            else 1,
            str(
                item[0]
            ),
        )
    )

    chosen = matches[0]

    print(
        "  transformation table:",
        chosen[0],
    )

    print(
        "  transformation mapping column:",
        chosen[1],
    )

    return chosen


def try_parse_operation_word(
    value: str,
) -> tuple[int, ...] | None:
    """
    Parse the exact frozen Phase-1B operation sequence.

    Accepted symbolic operations are the six frozen primitive
    generators used throughout Tier C v4.
    """

    symbolic_to_id = {
        "identity":
            0,

        "cycle":
            1,

        "pair_collapse":
            2,

        "half_collapse":
            3,

        "parity_collapse":
            4,

        "reset":
            5,

        # Frozen semigroup element aliases.
        "t006":
            0,

        "t020":
            1,

        "t003":
            2,

        "t001":
            3,

        "t005":
            4,

        "t000":
            5,
    }

    raw = str(
        value
    ).strip()

    if raw in {
        "",
        "[]",
        "()",
    }:
        return tuple()

    # First try JSON because Phase-1B stores operation_sequence
    # as a serialized list.
    try:
        parsed = json.loads(
            raw
        )
    except Exception:
        parsed = None

    if isinstance(
        parsed,
        list,
    ):
        result = []

        for item in parsed:
            if isinstance(
                item,
                int,
            ):
                operation_id = int(
                    item
                )

                if not (
                    0
                    <= operation_id
                    < OPERATION_COUNT
                ):
                    return None

                result.append(
                    operation_id
                )

                continue

            token = str(
                item
            ).strip()

            if token not in symbolic_to_id:
                return None

            result.append(
                symbolic_to_id[
                    token
                ]
            )

        return tuple(
            result
        )

    # Fallback for simple comma/space separated text.
    cleaned = (
        raw
        .replace(
            "[",
            " ",
        )
        .replace(
            "]",
            " ",
        )
        .replace(
            "(",
            " ",
        )
        .replace(
            ")",
            " ",
        )
        .replace(
            ",",
            " ",
        )
    )

    tokens = [
        token.strip(
            " '\""
        )
        for token in cleaned.split()
        if token.strip(
            " '\""
        )
    ]

    result = []

    for token in tokens:
        if token in symbolic_to_id:
            result.append(
                symbolic_to_id[
                    token
                ]
            )

            continue

        try:
            operation_id = int(
                token
            )
        except Exception:
            return None

        if not (
            0
            <= operation_id
            < OPERATION_COUNT
        ):
            return None

        result.append(
            operation_id
        )

    return tuple(
        result
    )

def discover_canonical_word_table(
) -> tuple[
    Path,
    str,
    list[dict[str, str]],
]:
    """
    Load the exact frozen Phase-1B canonical-shortest
    representatives.

    Phase 1B stores all symbolic histories in
    sequence_catalog.csv. The canonical representative for
    each of the 104 semigroup elements is identified by
    canonical_shortest == true.

    The transformation identifier is result_element_id and
    the corresponding generator word is operation_sequence.
    """

    path = Path(
        "outputs/"
        "phase1b_symbolic_histories/"
        "sequence_catalog.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    rows = load_csv(
        path
    )

    require(
        len(rows) == 3144,
        (
            "Frozen Phase-1B sequence catalog "
            f"changed: expected 3144 rows, "
            f"found {len(rows)}."
        ),
    )

    required_columns = {
        "sequence_id",
        "operation_sequence",
        "sequence_length",
        "result_element_id",
        "canonical_shortest",
    }

    require(
        required_columns.issubset(
            set(
                rows[0].keys()
            )
        ),
        (
            "Frozen Phase-1B sequence-catalog "
            "schema changed."
        ),
    )

    def parse_bool(value: str) -> bool:
        normalized = str(
            value
        ).strip().lower()

        if normalized in {
            "true",
            "1",
            "yes",
        }:
            return True

        if normalized in {
            "false",
            "0",
            "no",
        }:
            return False

        raise ValueError(
            "Unexpected canonical_shortest "
            f"value: {value!r}"
        )

    canonical = [
        row
        for row in rows
        if parse_bool(
            row[
                "canonical_shortest"
            ]
        )
    ]

    require(
        len(canonical)
        == TRANSFORMATION_COUNT,
        (
            "Expected exactly 104 frozen "
            "canonical-shortest histories; "
            f"found {len(canonical)}."
        ),
    )

    element_ids = [
        row[
            "result_element_id"
        ]
        for row in canonical
    ]

    require(
        len(
            set(
                element_ids
            )
        )
        == TRANSFORMATION_COUNT,
        (
            "Canonical-shortest histories do "
            "not cover 104 unique transformations."
        ),
    )

    normalized_rows = []

    for row in canonical:
        normalized_rows.append(
            {
                **row,

                # Normalize to the interface used by G2D.
                "element_id":
                    row[
                        "result_element_id"
                    ],
            }
        )

    normalized_rows.sort(
        key=lambda row:
            row[
                "element_id"
            ]
    )

    print(
        "  canonical-word table:",
        path,
    )

    print(
        "  canonical rows:",
        len(
            normalized_rows
        ),
    )

    print(
        "  canonical ID column:",
        "result_element_id",
    )

    print(
        "  canonical-word column:",
        "operation_sequence",
    )

    return (
        path,
        "operation_sequence",
        normalized_rows,
    )

def compose_mapping_word(
    primitive_mappings: dict[
        int,
        tuple[int, ...],
    ],
    word: tuple[int, ...],
    *,
    reverse_word: bool = False,
) -> tuple[int, ...]:
    result = list(
        range(
            STATE_COUNT
        )
    )

    operations = (
        tuple(
            reversed(
                word
            )
        )
        if reverse_word
        else word
    )

    for operation_id in operations:
        primitive = (
            primitive_mappings[
                int(
                    operation_id
                )
            ]
        )

        result = [
            primitive[
                state
            ]
            for state in result
        ]

    return tuple(
        int(value)
        for value in result
    )


def load_symbolic_ground_truth(
) -> dict[str, Any]:
    print(
        "[3/9] Loading and validating frozen symbolic ground truth"
    )

    (
        transformation_path,
        mapping_column,
        transformation_rows,
    ) = (
        discover_true_transformation_table()
    )

    (
        word_path,
        word_column,
        word_rows,
    ) = (
        discover_canonical_word_table()
    )

    true_mappings = {
        row[
            "element_id"
        ]:
            tuple(
                parse_json_int_list(
                    row[
                        mapping_column
                    ]
                )
            )
        for row in transformation_rows
    }

    require(
        len(
            true_mappings
        )
        == TRANSFORMATION_COUNT,
        (
            "True transformation ID "
            "count changed."
        ),
    )

    words = {
        row[
            "element_id"
        ]:
            tuple(
                try_parse_operation_word(
                    row[
                        word_column
                    ]
                )
            )
        for row in word_rows
    }

    require(
        set(
            words
        )
        == set(
            true_mappings
        ),
        (
            "Canonical-word and "
            "transformation IDs differ."
        ),
    )

    true_primitives = {}

    for operation_id, element_id in (
        PRIMITIVE_ELEMENT_IDS.items()
    ):
        require(
            element_id
            in true_mappings,
            (
                "Frozen primitive element "
                f"{element_id} missing."
            ),
        )

        true_primitives[
            operation_id
        ] = true_mappings[
            element_id
        ]

    forward_correct = sum(
        compose_mapping_word(
            true_primitives,
            words[
                element_id
            ],
            reverse_word=False,
        )
        == true_mappings[
            element_id
        ]
        for element_id in true_mappings
    )

    reverse_correct = sum(
        compose_mapping_word(
            true_primitives,
            words[
                element_id
            ],
            reverse_word=True,
        )
        == true_mappings[
            element_id
        ]
        for element_id in true_mappings
    )

    require(
        max(
            forward_correct,
            reverse_correct,
        )
        == TRANSFORMATION_COUNT,
        (
            "Canonical-word composition does "
            "not reproduce all 104 frozen "
            "transformations. "
            f"forward={forward_correct}, "
            f"reverse={reverse_correct}"
        ),
    )

    require(
        not (
            forward_correct
            == TRANSFORMATION_COUNT
            and reverse_correct
            == TRANSFORMATION_COUNT
        )
        or all(
            len(
                word
            ) <= 1
            for word in words.values()
        )
        is False,
        (
            "Canonical-word composition "
            "orientation is unexpectedly ambiguous."
        ),
    )

    reverse_word = (
        reverse_correct
        == TRANSFORMATION_COUNT
        and forward_correct
        != TRANSFORMATION_COUNT
    )

    print(
        "  canonical composition:",
        (
            "reverse-word"
            if reverse_word
            else "forward-word"
        ),
    )

    split_lookup = (
        structural_ood
        .load_transformation_splits()
    )

    require(
        len(
            split_lookup
        )
        == TRANSFORMATION_COUNT,
        (
            "Frozen transformation split "
            "count changed."
        ),
    )

    split_counts = Counter(
        split_lookup.values()
    )

    require(
        dict(
            split_counts
        )
        == {
            "train": 86,
            "val": 9,
            "test": 9,
        },
        (
            "Frozen transformation "
            f"splits changed: {dict(split_counts)}"
        ),
    )

    true_information_masks = {
        element_id:
            np.asarray(
                structural.equality_mask(
                    mapping
                )
            )
        for element_id, mapping
        in true_mappings.items()
    }

    information_signatures = {
        mask.tobytes()
        for mask
        in true_information_masks.values()
    }

    require(
        len(
            information_signatures
        )
        == INFORMATION_CLASS_COUNT,
        (
            "Frozen information-class "
            f"count changed: "
            f"{len(information_signatures)}"
        ),
    )

    return {
        "true_mappings":
            true_mappings,

        "words":
            words,

        "split_lookup":
            split_lookup,

        "true_information_masks":
            true_information_masks,

        "reverse_word":
            reverse_word,

        "transformation_table_path":
            str(
                transformation_path
            ),

        "transformation_table_sha256":
            sha256_file(
                transformation_path
            ),

        "word_table_path":
            str(
                word_path
            ),

        "word_table_sha256":
            sha256_file(
                word_path
            ),
    }


def discover_development_state_bank(
) -> tuple[
    Path,
    np.ndarray,
]:
    print(
        "[4/9] Discovering non-test development state-field bank"
    )

    roots = list(
        Path(
            "outputs"
        ).glob(
            "phase4br3_tier_c_v4_development_fields*"
        )
    )

    require(
        bool(
            roots
        ),
        (
            "Could not locate Phase-4 "
            "development field directory."
        ),
    )

    candidates = []

    for root in roots:
        for path in root.rglob(
            "*.npy"
        ):
            name = (
                path.name.lower()
            )

            if (
                "normalized"
                not in name
                or "state"
                not in name
                or "field"
                not in name
            ):
                continue

            try:
                array = np.load(
                    path,
                    mmap_mode="r",
                )
            except Exception:
                continue

            if (
                array.ndim == 5
                and array.shape[1]
                == STATE_COUNT
                and tuple(
                    array.shape[
                        2:
                    ]
                )
                == (
                    4,
                    32,
                    32,
                )
            ):
                candidates.append(
                    (
                        path,
                        array,
                    )
                )

    require(
        len(
            candidates
        ) >= 1,
        (
            "Could not discover normalized "
            "development carrier/state fields."
        ),
    )

    candidates.sort(
        key=lambda item: (
            0
            if "carrier_state_fields_normalized"
            in item[0].name.lower()
            else 1,
            str(
                item[0]
            ),
        )
    )

    path, array = (
        candidates[0]
    )

    require(
        array.shape[0] >= 32,
        (
            "Development state field bank "
            "contains too few carriers."
        ),
    )

    print(
        "  development state bank:",
        path,
    )

    print(
        "  shape:",
        tuple(
            array.shape
        ),
    )

    return (
        path,
        array,
    )


def load_development_carrier_splits(
) -> tuple[
    Path,
    np.ndarray,
]:
    """
    Load the frozen Tier-C-v4 development carrier split.

    The pre-existing Phase-3G hard estimator learns the
    state-label permutation from TRAIN carriers only.
    """

    path = Path(
        "outputs/"
        "phase4br3_tier_c_v4_development_fields/"
        "privileged/"
        "development_carrier_splits.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    splits = np.load(
        path
    ).astype(
        str
    )

    require(
        splits.shape == (128,),
        (
            "Development carrier-split shape "
            f"changed: {splits.shape}"
        ),
    )

    counts = Counter(
        splits.tolist()
    )

    require(
        dict(
            counts
        )
        == {
            "train": 96,
            "val": 32,
        },
        (
            "Frozen development carrier splits "
            f"changed: {dict(counts)}"
        ),
    )

    print(
        "  development carrier splits:",
        path,
    )

    print(
        "  split counts:",
        dict(
            counts
        ),
    )

    return (
        path,
        splits,
    )


def best_true_to_latent_alignment(
    score_matrix: np.ndarray,
) -> tuple[
    tuple[int, ...],
    float,
]:
    require(
        score_matrix.shape
        == (
            STATE_COUNT,
            STATE_COUNT,
        ),
        (
            "Alignment score matrix "
            "must be 8x8."
        ),
    )

    best_score = None
    best_permutation = None

    for permutation in (
        itertools.permutations(
            range(
                STATE_COUNT
            )
        )
    ):
        score = sum(
            float(
                score_matrix[
                    true_state,
                    permutation[
                        true_state
                    ],
                ]
            )
            for true_state
            in range(
                STATE_COUNT
            )
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score
            best_permutation = (
                permutation
            )

    require(
        best_permutation is not None,
        "Latent alignment failed.",
    )

    mean_probability = float(
        best_score
        / STATE_COUNT
    )

    return (
        tuple(
            int(value)
            for value
            in best_permutation
        ),
        mean_probability,
    )


@torch.no_grad()
def determine_alignment(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    configuration: dict[str, Any],
    state_bank: np.ndarray,
    carrier_splits: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """
    Tier-C-v4 adaptation of the pre-existing Phase-3G
    align_ocm_states() estimator.

    Method is frozen by Phase 3G:
      1. use TRAIN carriers only for label assignment;
      2. hard argmax the OCM categorical state;
      3. build an 8x8 integer confusion matrix;
      4. solve the permutation assignment;
      5. evaluate the fixed assignment on train and val.

    The only Tier-C adaptation is the frozen predictive codec
    preceding model.encode(), because Tier-C-v4 OCM receives
    learned codec observations rather than raw Phase-3 vectors.
    """

    temperature = float(
        configuration[
            "temperature"
        ]
    )

    latent_state_count = int(
        configuration[
            "latent_state_count"
        ]
    )

    require(
        latent_state_count
        == STATE_COUNT,
        (
            "Frozen OCM latent-state count "
            "is not eight."
        ),
    )

    require(
        state_bank.shape
        == (
            128,
            STATE_COUNT,
            4,
            32,
            32,
        ),
        (
            "Development state-bank shape "
            f"changed: {state_bank.shape}"
        ),
    )

    require(
        carrier_splits.shape
        == (
            128,
        ),
        (
            "Development carrier-split "
            "shape changed."
        ),
    )

    train_indices = np.where(
        carrier_splits
        == "train"
    )[0]

    require(
        len(
            train_indices
        )
        == 96,
        (
            "Frozen training-carrier "
            "count changed."
        ),
    )

    # Copy prevents the read-only mmap warning.
    train_fields = np.array(
        state_bank[
            train_indices
        ],
        dtype=np.float32,
        copy=True,
    )

    train_tensor = torch.from_numpy(
        train_fields
    ).to(
        device=device,
        dtype=torch.float32,
    )

    train_encoded = (
        runtime.accepted_impl
        .encode_all_fields(
            codec,
            train_tensor,
        )
    )

    require(
        train_encoded.shape[
            :2
        ]
        == (
            96,
            STATE_COUNT,
        ),
        (
            "Unexpected encoded training "
            "state-manifold shape."
        ),
    )

    train_distribution = (
        model.encode(
            train_encoded.reshape(
                -1,
                runtime.OBSERVATION_DIMENSION,
            ),
            temperature=temperature,
        )
        .reshape(
            len(
                train_indices
            ),
            STATE_COUNT,
            STATE_COUNT,
        )
    )

    predicted_latent = (
        torch.argmax(
            train_distribution,
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    confusion = np.zeros(
        (
            STATE_COUNT,
            STATE_COUNT,
        ),
        dtype=np.int64,
    )

    for true_state in range(
        STATE_COUNT
    ):
        counts = np.bincount(
            predicted_latent[
                :,
                true_state,
            ],
            minlength=STATE_COUNT,
        )

        confusion[
            true_state
        ] = counts

    (
        true_to_latent,
        assignment_score,
    ) = (
        phase3g_structural
        .best_state_assignment(
            confusion
        )
    )

    latent_to_true = np.empty(
        STATE_COUNT,
        dtype=np.int64,
    )

    for true_state, latent_state in enumerate(
        true_to_latent
    ):
        latent_to_true[
            latent_state
        ] = true_state

    split_accuracy = {}

    for split in (
        "train",
        "val",
    ):
        indices = np.where(
            carrier_splits
            == split
        )[0]

        values = np.array(
            state_bank[
                indices
            ],
            dtype=np.float32,
            copy=True,
        )

        tensor = torch.from_numpy(
            values
        ).to(
            device=device,
            dtype=torch.float32,
        )

        encoded = (
            runtime.accepted_impl
            .encode_all_fields(
                codec,
                tensor,
            )
        )

        distributions = (
            model.encode(
                encoded.reshape(
                    -1,
                    runtime.OBSERVATION_DIMENSION,
                ),
                temperature=temperature,
            )
            .reshape(
                len(
                    indices
                ),
                STATE_COUNT,
                STATE_COUNT,
            )
        )

        latent_predictions = (
            torch.argmax(
                distributions,
                dim=-1,
            )
            .detach()
            .cpu()
            .numpy()
        )

        decoded_true = (
            latent_to_true[
                latent_predictions
            ]
        )

        true_labels = np.broadcast_to(
            np.arange(
                STATE_COUNT,
                dtype=np.int64,
            )[
                None,
                :
            ],
            decoded_true.shape,
        )

        split_accuracy[
            split
        ] = float(
            np.mean(
                decoded_true
                == true_labels
            )
        )

    training_assignment_accuracy = float(
        assignment_score
        / (
            len(
                train_indices
            )
            * STATE_COUNT
        )
    )

    unique_top1_count = int(
        len(
            np.unique(
                predicted_latent
            )
        )
    )

    return {
        "true_to_latent":
            tuple(
                int(value)
                for value
                in true_to_latent
            ),

        "latent_to_true":
            tuple(
                int(value)
                for value
                in latent_to_true
            ),

        "confusion":
            confusion,

        "training_assignment_score":
            int(
                assignment_score
            ),

        "training_assignment_accuracy":
            training_assignment_accuracy,

        "split_alignment_accuracy":
            split_accuracy,

        # Preserve existing G2D field names for output
        # compatibility, but now they represent the exact
        # hard Phase-3G assignment diagnostic.
        "alignment_mean_probability":
            training_assignment_accuracy,

        "alignment_unique_top1_count":
            unique_top1_count,

        "estimator":
            (
                "Phase-3G hard argmax confusion "
                "assignment on training carriers"
            ),
    }

@torch.no_grad()
def derive_hard_primitive_mappings(
    model: torch.nn.Module,
    configuration: dict[str, Any],
    alignment: dict[str, Any],
    device: torch.device,
) -> dict[
    int,
    tuple[int, ...],
]:
    """
    Exact Phase-3G hard-channel estimator.

    Do not push artificial one-hot distributions through
    apply_operation(). Read the frozen OCM operation-channel
    logits directly, softmax at the frozen temperature, and
    hard-argmax each transition row.
    """

    temperature = float(
        configuration[
            "temperature"
        ]
    )

    _, logits = (
        phase3g_structural
        .find_channel_logits(
            model=model,
            operation_count=(
                OPERATION_COUNT
            ),
        )
    )

    require(
        logits is not None,
        (
            "Could not locate frozen "
            "OCM channel logits."
        ),
    )

    probabilities = torch.softmax(
        logits
        / temperature,
        dim=-1,
    )

    maximum_row_sum_error = float(
        (
            probabilities.sum(
                dim=-1
            )
            - 1.0
        )
        .abs()
        .max()
        .item()
    )

    require(
        maximum_row_sum_error
        < 1e-6,
        (
            "OCM channel rows are "
            "not stochastic."
        ),
    )

    latent_operation_mappings = (
        torch.argmax(
            probabilities,
            dim=-1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    require(
        latent_operation_mappings.shape
        == (
            OPERATION_COUNT,
            STATE_COUNT,
        ),
        (
            "Unexpected hard OCM "
            "channel shape: "
            f"{latent_operation_mappings.shape}"
        ),
    )

    true_to_latent = np.asarray(
        alignment[
            "true_to_latent"
        ],
        dtype=np.int64,
    )

    latent_to_true = np.asarray(
        alignment[
            "latent_to_true"
        ],
        dtype=np.int64,
    )

    primitive_mappings = {}

    for operation_id in range(
        OPERATION_COUNT
    ):
        mapping = np.empty(
            STATE_COUNT,
            dtype=np.int64,
        )

        for true_state in range(
            STATE_COUNT
        ):
            current_latent = int(
                true_to_latent[
                    true_state
                ]
            )

            next_latent = int(
                latent_operation_mappings[
                    operation_id,
                    current_latent,
                ]
            )

            mapping[
                true_state
            ] = int(
                latent_to_true[
                    next_latent
                ]
            )

        primitive_mappings[
            operation_id
        ] = tuple(
            int(value)
            for value
            in mapping
        )

    alignment[
        "maximum_channel_row_sum_error"
    ] = maximum_row_sum_error

    return primitive_mappings

def evaluate_transformations(
    primitive_mappings: dict[
        int,
        tuple[int, ...],
    ],
    ground_truth: dict[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    true_mappings = (
        ground_truth[
            "true_mappings"
        ]
    )

    words = (
        ground_truth[
            "words"
        ]
    )

    split_lookup = (
        ground_truth[
            "split_lookup"
        ]
    )

    reverse_word = bool(
        ground_truth[
            "reverse_word"
        ]
    )

    predicted_mappings = {}

    rows = []

    for element_id in sorted(
        true_mappings
    ):
        truth = (
            true_mappings[
                element_id
            ]
        )

        prediction = (
            compose_mapping_word(
                primitive_mappings,
                words[
                    element_id
                ],
                reverse_word=(
                    reverse_word
                ),
            )
        )

        predicted_mappings[
            element_id
        ] = prediction

        exact = (
            prediction
            == truth
        )

        true_mask = np.asarray(
            structural.equality_mask(
                truth
            )
        )

        predicted_mask = np.asarray(
            structural.equality_mask(
                prediction
            )
        )

        information_correct = bool(
            np.array_equal(
                true_mask,
                predicted_mask,
            )
        )

        rows.append(
            {
                "element_id":
                    element_id,

                "split":
                    split_lookup[
                        element_id
                    ],

                "exact_mapping":
                    exact,

                "information_class_correct":
                    information_correct,

                "true_mapping":
                    json.dumps(
                        list(
                            truth
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),

                "predicted_mapping":
                    json.dumps(
                        list(
                            prediction
                        ),
                        separators=(
                            ",",
                            ":",
                        ),
                    ),
            }
        )

    require(
        len(rows)
        == TRANSFORMATION_COUNT,
        (
            "Transformation result "
            "count changed."
        ),
    )

    all_exact = float(
        np.mean(
            [
                row[
                    "exact_mapping"
                ]
                for row in rows
            ]
        )
    )

    all_information = float(
        np.mean(
            [
                row[
                    "information_class_correct"
                ]
                for row in rows
            ]
        )
    )

    test_rows = [
        row
        for row in rows
        if row[
            "split"
        ] == "test"
    ]

    require(
        len(
            test_rows
        ) == 9,
        (
            "Frozen test transformation "
            "count changed."
        ),
    )

    test_exact = float(
        np.mean(
            [
                row[
                    "exact_mapping"
                ]
                for row in test_rows
            ]
        )
    )

    test_information = float(
        np.mean(
            [
                row[
                    "information_class_correct"
                ]
                for row in test_rows
            ]
        )
    )

    truth_by_scope = defaultdict(
        list
    )

    prediction_by_scope = defaultdict(
        list
    )

    element_ids = sorted(
        true_mappings
    )

    for first_id in element_ids:
        first_true_mask = (
            structural.equality_mask(
                true_mappings[
                    first_id
                ]
            )
        )

        first_prediction_mask = (
            structural.equality_mask(
                predicted_mappings[
                    first_id
                ]
            )
        )

        first_split = (
            split_lookup[
                first_id
            ]
        )

        for second_id in element_ids:
            if (
                first_id
                == second_id
            ):
                continue

            second_true_mask = (
                structural.equality_mask(
                    true_mappings[
                        second_id
                    ]
                )
            )

            second_prediction_mask = (
                structural.equality_mask(
                    predicted_mappings[
                        second_id
                    ]
                )
            )

            second_split = (
                split_lookup[
                    second_id
                ]
            )

            true_relation = (
                structural
                .relation_from_masks(
                    first_true_mask,
                    second_true_mask,
                )
            )

            predicted_relation = (
                structural
                .relation_from_masks(
                    first_prediction_mask,
                    second_prediction_mask,
                )
            )

            scopes = (
                structural_ood
                .relation_scopes(
                    first_split,
                    second_split,
                )
            )

            for scope in scopes:
                truth_by_scope[
                    scope
                ].append(
                    true_relation
                )

                prediction_by_scope[
                    scope
                ].append(
                    predicted_relation
                )

    require(
        len(
            truth_by_scope[
                "all"
            ]
        )
        == EXPECTED_ALL_RELATION_PAIRS,
        (
            "All relation pair "
            "count changed."
        ),
    )

    require(
        len(
            truth_by_scope[
                "test_involved"
            ]
        )
        == EXPECTED_TEST_INVOLVED_PAIRS,
        (
            "Test-involved relation "
            "pair count changed."
        ),
    )

    require(
        len(
            truth_by_scope[
                "test_test"
            ]
        )
        == EXPECTED_TEST_TEST_PAIRS,
        (
            "Test-test relation pair "
            "count changed."
        ),
    )

    relation_metrics = {}

    for scope in (
        "all",
        "test_involved",
        "test_test",
    ):
        relation_metrics[
            scope
        ] = (
            structural_ood
            .supported_relation_metrics(
                truth_by_scope[
                    scope
                ],
                prediction_by_scope[
                    scope
                ],
            )
        )

    summary = {
        "exact_transformation_rate":
            all_exact,

        "test_exact_transformation_rate":
            test_exact,

        "information_class_accuracy":
            all_information,

        "test_information_class_accuracy":
            test_information,

        "blackwell_relation_accuracy":
            float(
                relation_metrics[
                    "all"
                ][
                    "accuracy"
                ]
            ),

        "blackwell_relation_balanced_accuracy":
            float(
                relation_metrics[
                    "all"
                ][
                    "balanced_accuracy"
                ]
            ),

        "blackwell_relation_macro_f1":
            float(
                relation_metrics[
                    "all"
                ][
                    "macro_f1"
                ]
            ),

        "blackwell_relation_majority_baseline":
            float(
                relation_metrics[
                    "all"
                ][
                    "majority_baseline"
                ]
            ),

        "test_involved_blackwell_relation_balanced_accuracy":
            float(
                relation_metrics[
                    "test_involved"
                ][
                    "balanced_accuracy"
                ]
            ),

        "test_test_blackwell_relation_balanced_accuracy":
            float(
                relation_metrics[
                    "test_test"
                ][
                    "balanced_accuracy"
                ]
            ),
    }

    return (
        summary,
        rows,
    )


def load_true_state_paths(
    cell_id: str,
    cell: primary_eval.SealedTestCell,
) -> dict[
    int,
    tuple[int, ...],
]:
    path = (
        G1_ROOT
        / "privileged"
        / (
            f"{cell_id}_"
            "trajectory_metadata.csv"
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    rows = load_csv(
        path
    )

    require(
        len(rows)
        == EXPECTED_TRAJECTORY_COUNT,
        (
            f"{cell_id} privileged "
            "trajectory count changed."
        ),
    )

    row_by_trajectory = {}

    for index, row in enumerate(
        rows
    ):
        trajectory_index = int(
            row.get(
                "trajectory_index",
                index,
            )
        )

        row_by_trajectory[
            trajectory_index
        ] = row

    preferred_columns = [
        "state_path",
        "state_ids",
        "symbolic_state_path",
        "true_state_path",
        "latent_state_path",
    ]

    columns = list(
        rows[0].keys()
    )

    ordered_columns = (
        [
            column
            for column
            in preferred_columns
            if column in columns
        ]
        + [
            column
            for column in columns
            if column
            not in preferred_columns
        ]
    )

    valid_columns = []

    for column in ordered_columns:
        good = True

        for trajectory_row in (
            cell.rows
        ):
            trajectory_index = int(
                trajectory_row[
                    "trajectory_index"
                ]
            )

            metadata = (
                row_by_trajectory[
                    trajectory_index
                ]
            )

            parsed = (
                parse_json_int_list(
                    metadata.get(
                        column,
                        "",
                    )
                )
            )

            expected_length = (
                int(
                    trajectory_row[
                        "sequence_length"
                    ]
                )
                + 1
            )

            if (
                parsed is None
                or len(parsed)
                != expected_length
                or any(
                    state < 0
                    or state >= STATE_COUNT
                    for state in parsed
                )
            ):
                good = False
                break

        if good:
            valid_columns.append(
                column
            )

    require(
        len(
            valid_columns
        ) == 1,
        (
            f"{cell_id}: could not uniquely "
            "identify the frozen state-path "
            f"column; candidates={valid_columns}"
        ),
    )

    state_column = (
        valid_columns[0]
    )

    print(
        f"  {cell_id} state-path column:",
        state_column,
    )

    state_paths = {}

    for trajectory_row in (
        cell.rows
    ):
        trajectory_index = int(
            trajectory_row[
                "trajectory_index"
            ]
        )

        state_paths[
            trajectory_index
        ] = tuple(
            parse_json_int_list(
                row_by_trajectory[
                    trajectory_index
                ][
                    state_column
                ]
            )
        )

    return state_paths


@torch.no_grad()
def evaluate_state_path_condition(
    model: torch.nn.Module,
    codec: torch.nn.Module,
    configuration: dict[str, Any],
    alignment: dict[str, Any],
    cell_id: str,
    cell: primary_eval.SealedTestCell,
    state_paths: dict[
        int,
        tuple[int, ...],
    ],
    noise: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    noise_index = (
        NOISE_TO_INDEX[
            noise
        ]
    )

    temperature = float(
        configuration[
            "temperature"
        ]
    )

    latent_to_true = np.asarray(
        alignment[
            "latent_to_true"
        ],
        dtype=np.int64,
    )

    trajectory_results = []

    for start in range(
        0,
        EXPECTED_TRAJECTORY_COUNT,
        EVALUATION_BATCH_SIZE,
    ):
        indices = list(
            range(
                start,
                min(
                    start
                    + EVALUATION_BATCH_SIZE,
                    EXPECTED_TRAJECTORY_COUNT,
                ),
            )
        )

        batch = cell.make_batch(
            trajectory_indices=indices,
            noise_index=noise_index,
            device=device,
        )

        observations = (
            batch[
                "observations"
            ]
        )

        point_mask = (
            batch[
                "point_mask"
            ]
        )

        encoded = (
            runtime.accepted_impl
            .encode_all_fields(
                codec,
                observations,
            )
        )

        distributions = (
            model.encode(
                encoded.reshape(
                    -1,
                    runtime.OBSERVATION_DIMENSION,
                ),
                temperature=temperature,
            )
            .reshape(
                encoded.shape[0],
                encoded.shape[1],
                STATE_COUNT,
            )
        )

        predicted_latent = (
            distributions.argmax(
                dim=2
            )
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        masks = (
            point_mask
            .detach()
            .cpu()
            .numpy()
            .astype(
                bool
            )
        )

        for local_index in range(
            len(indices)
        ):
            trajectory_index = int(
                batch[
                    "trajectory_indices"
                ][
                    local_index
                ]
            )

            sequence_index = int(
                batch[
                    "cell_sequence_indices"
                ][
                    local_index
                ]
            )

            source_sequence_id = str(
                batch[
                    "source_sequence_ids"
                ][
                    local_index
                ]
            )

            valid_count = int(
                masks[
                    local_index
                ].sum()
            )

            truth = np.asarray(
                state_paths[
                    trajectory_index
                ],
                dtype=np.int64,
            )

            require(
                len(
                    truth
                ) == valid_count,
                (
                    "True state-path length "
                    "does not match point mask."
                ),
            )

            predicted_true = (
                latent_to_true[
                    predicted_latent[
                        local_index,
                        :valid_count,
                    ]
                ]
            )

            accuracy = float(
                np.mean(
                    predicted_true
                    == truth
                )
            )

            trajectory_results.append(
                {
                    "trajectory_index":
                        trajectory_index,

                    "cell_sequence_index":
                        sequence_index,

                    "source_sequence_id":
                        source_sequence_id,

                    "state_path_accuracy":
                        accuracy,
                }
            )

    require(
        len(
            trajectory_results
        )
        == EXPECTED_TRAJECTORY_COUNT,
        (
            "State-path trajectory "
            "count changed."
        ),
    )

    grouped = defaultdict(list)

    for row in trajectory_results:
        grouped[
            (
                row[
                    "cell_sequence_index"
                ],
                row[
                    "source_sequence_id"
                ],
            )
        ].append(
            row
        )

    require(
        len(
            grouped
        )
        == EXPECTED_SEQUENCE_COUNT,
        (
            "State-path sequence "
            "count changed."
        ),
    )

    output = []

    for (
        sequence_index,
        source_sequence_id,
    ), rows in sorted(
        grouped.items()
    ):
        require(
            len(rows)
            == EXPECTED_TRAJECTORIES_PER_SEQUENCE,
            (
                "Expected eight state-path "
                "trajectories per sequence."
            ),
        )

        output.append(
            {
                "cell_id":
                    cell_id,

                "noise_fraction":
                    noise,

                "cell_sequence_index":
                    sequence_index,

                "source_sequence_id":
                    source_sequence_id,

                "trajectory_count":
                    len(
                        rows
                    ),

                "state_path_accuracy":
                    float(
                        np.mean(
                            [
                                row[
                                    "state_path_accuracy"
                                ]
                                for row
                                in rows
                            ]
                        )
                    ),
            }
        )

    return output


def configuration_map(
) -> dict[
    str,
    dict[str, Any],
]:
    records = (
        runtime.load_configurations()
    )

    return {
        str(
            record[
                "configuration_id"
            ]
        ):
            record[
                "configuration"
            ]
        for record in records
    }


def architecture_maximum_length(
) -> int:
    train = runtime.TrajectoryCell(
        "train_joint",
        11008,
    )

    val = runtime.TrajectoryCell(
        "val_joint",
        1152,
    )

    value = max(
        int(
            train.maximum_sequence_length
        ),
        int(
            val.maximum_sequence_length
        ),
    )

    del train
    del val

    return value


def summarize_values(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                array.mean()
            ),

        "std":
            float(
                array.std(
                    ddof=1
                )
            )
            if len(array) > 1
            else 0.0,

        "minimum":
            float(
                array.min()
            ),

        "maximum":
            float(
                array.max()
            ),
    }


def main() -> None:
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_signature = (
        verify_frozen_state()
    )

    ocm_rows = (
        load_ocm_registry()
    )

    ground_truth = (
        load_symbolic_ground_truth()
    )

    (
        development_state_bank_path,
        development_state_bank,
    ) = (
        discover_development_state_bank()
    )

    (
        development_carrier_splits_path,
        development_carrier_splits,
    ) = (
        load_development_carrier_splits()
    )

    print(
        "[5/9] Loading sealed cells and frozen state paths"
    )

    cells = {}

    state_paths = {}

    for cell_id in TEST_CELLS:
        cell = (
            primary_eval
            .SealedTestCell(
                cell_id
            )
        )

        cells[
            cell_id
        ] = cell

        state_paths[
            cell_id
        ] = (
            load_true_state_paths(
                cell_id,
                cell,
            )
        )

    configurations = (
        configuration_map()
    )

    maximum_length = (
        architecture_maximum_length()
    )

    require(
        all(
            cell.maximum_sequence_length
            <= maximum_length
            for cell in cells.values()
        ),
        (
            "Sealed state-path sequence "
            "exceeds frozen architecture length."
        ),
    )

    device = (
        runtime.resolve_device(
            "cuda"
        )
    )

    print(
        "Device:",
        device,
    )

    print(
        "[6/9] Evaluating 25 frozen OCM structural representations"
    )

    checkpoint_summary_rows = []
    transformation_rows_all = []
    state_path_rows_all = []

    for checkpoint_index, row in enumerate(
        ocm_rows,
        start=1,
    ):
        noise = float(
            row[
                "noise_fraction"
            ]
        )

        seed = int(
            row[
                "effective_seed"
            ]
        )

        configuration_id = (
            row[
                "configuration_id"
            ]
        )

        require(
            configuration_id
            in configurations,
            (
                "Frozen OCM configuration "
                f"missing: {configuration_id}"
            ),
        )

        configuration = (
            configurations[
                configuration_id
            ]
        )

        checkpoint_path = Path(
            row[
                "best_checkpoint_path"
            ]
        )

        print()
        print(
            f"[OCM {checkpoint_index}/25] "
            f"seed={seed} noise={noise}"
        )

        if hasattr(
            runtime,
            "set_global_determinism",
        ):
            runtime.set_global_determinism(
                seed
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        codec, model, optimizer = (
            runtime.build_model_bundle(
                model_id="OCM",
                configuration=configuration,
                maximum_sequence_length=(
                    maximum_length
                ),
                device=device,
            )
        )

        del optimizer

        codec.load_state_dict(
            checkpoint[
                "codec_state_dict"
            ]
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        codec.eval()
        model.eval()

        alignment = (
            determine_alignment(
                model=model,
                codec=codec,
                configuration=configuration,
                state_bank=(
                    development_state_bank
                ),
                carrier_splits=(
                    development_carrier_splits
                ),
                device=device,
            )
        )

        print(
            "  alignment mean probability:",
            (
                f"{alignment['alignment_mean_probability']:.6f}"
            ),
        )

        print(
            "  alignment unique top1:",
            alignment[
                "alignment_unique_top1_count"
            ],
        )

        primitive_mappings = (
            derive_hard_primitive_mappings(
                model=model,
                configuration=configuration,
                alignment=alignment,
                device=device,
            )
        )

        (
            transformation_summary,
            transformation_rows,
        ) = (
            evaluate_transformations(
                primitive_mappings=(
                    primitive_mappings
                ),
                ground_truth=(
                    ground_truth
                ),
            )
        )

        for transformation_row in (
            transformation_rows
        ):
            transformation_rows_all.append(
                {
                    "effective_seed":
                        seed,

                    "noise_fraction":
                        noise,

                    **transformation_row,
                }
            )

        state_condition_values = {}

        for cell_id in TEST_CELLS:
            sequence_rows = (
                evaluate_state_path_condition(
                    model=model,
                    codec=codec,
                    configuration=configuration,
                    alignment=alignment,
                    cell_id=cell_id,
                    cell=cells[
                        cell_id
                    ],
                    state_paths=(
                        state_paths[
                            cell_id
                        ]
                    ),
                    noise=noise,
                    device=device,
                )
            )

            for sequence_row in (
                sequence_rows
            ):
                state_path_rows_all.append(
                    {
                        "effective_seed":
                            seed,

                        **sequence_row,
                    }
                )

            state_condition_values[
                cell_id
            ] = float(
                np.mean(
                    [
                        sequence_row[
                            "state_path_accuracy"
                        ]
                        for sequence_row
                        in sequence_rows
                    ]
                )
            )

        checkpoint_summary = {
            "effective_seed":
                seed,

            "noise_fraction":
                noise,

            "configuration_id":
                configuration_id,

            "checkpoint_path":
                str(
                    checkpoint_path
                ),

            "checkpoint_sha256":
                row[
                    "best_checkpoint_sha256"
                ],

            "alignment_mean_probability":
                alignment[
                    "alignment_mean_probability"
                ],

            "alignment_unique_top1_count":
                alignment[
                    "alignment_unique_top1_count"
                ],

            **transformation_summary,

            **{
                (
                    "state_path_accuracy_"
                    + cell_id
                ):
                    state_condition_values[
                        cell_id
                    ]
                for cell_id in TEST_CELLS
            },
        }

        checkpoint_summary_rows.append(
            checkpoint_summary
        )

        checkpoint_result_path = (
            CHECKPOINT_DIR
            / (
                f"OCM_seed_{seed}"
                f"_noise_"
                + str(
                    noise
                ).replace(
                    ".",
                    "p",
                )
                + ".json"
            )
        )

        write_json(
            checkpoint_result_path,
            checkpoint_summary,
        )

        print(
            "  exact transformation rate:",
            (
                f"{transformation_summary['exact_transformation_rate']:.6f}"
            ),
        )

        print(
            "  test exact transformation rate:",
            (
                f"{transformation_summary['test_exact_transformation_rate']:.6f}"
            ),
        )

        print(
            "  information-class accuracy:",
            (
                f"{transformation_summary['information_class_accuracy']:.6f}"
            ),
        )

        print(
            "  Blackwell balanced accuracy:",
            (
                f"{transformation_summary['blackwell_relation_balanced_accuracy']:.6f}"
            ),
        )

        del checkpoint
        del codec
        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    require(
        len(
            checkpoint_summary_rows
        ) == 25,
        (
            "Structural checkpoint "
            "summary count changed."
        ),
    )

    require(
        len(
            transformation_rows_all
        )
        == (
            25
            * TRANSFORMATION_COUNT
        ),
        (
            "Transformation row count changed."
        ),
    )

    require(
        len(
            state_path_rows_all
        )
        == (
            25
            * len(
                TEST_CELLS
            )
            * EXPECTED_SEQUENCE_COUNT
        ),
        (
            "State-path sequence "
            "row count changed."
        ),
    )

    print(
        "[7/9] Aggregating five-seed structural results"
    )

    noise_summary_rows = []

    transformation_metric_fields = [
        "alignment_mean_probability",
        "alignment_unique_top1_count",
        "exact_transformation_rate",
        "test_exact_transformation_rate",
        "information_class_accuracy",
        "test_information_class_accuracy",
        "blackwell_relation_accuracy",
        "blackwell_relation_balanced_accuracy",
        "blackwell_relation_macro_f1",
        "blackwell_relation_majority_baseline",
        "test_involved_blackwell_relation_balanced_accuracy",
        "test_test_blackwell_relation_balanced_accuracy",
    ]

    for noise in NOISE_LEVELS:
        rows = [
            row
            for row
            in checkpoint_summary_rows
            if abs(
                float(
                    row[
                        "noise_fraction"
                    ]
                )
                - noise
            ) < 1e-12
        ]

        require(
            len(rows) == 5,
            (
                f"Noise={noise} does not "
                "contain five OCM seeds."
            ),
        )

        record = {
            "noise_fraction":
                noise,

            "seed_count":
                5,
        }

        for metric in (
            transformation_metric_fields
        ):
            summary = summarize_values(
                [
                    float(
                        row[
                            metric
                        ]
                    )
                    for row
                    in rows
                ]
            )

            for statistic, value in (
                summary.items()
            ):
                record[
                    metric
                    + "_"
                    + statistic
                ] = value

        noise_summary_rows.append(
            record
        )

    state_path_summary_rows = []

    for noise in NOISE_LEVELS:
        for cell_id in TEST_CELLS:
            per_seed = []

            for seed in EXPECTED_SEEDS:
                rows = [
                    row
                    for row
                    in state_path_rows_all
                    if (
                        int(
                            row[
                                "effective_seed"
                            ]
                        )
                        == seed
                        and abs(
                            float(
                                row[
                                    "noise_fraction"
                                ]
                            )
                            - noise
                        ) < 1e-12
                        and row[
                            "cell_id"
                        ]
                        == cell_id
                    )
                ]

                require(
                    len(rows)
                    == EXPECTED_SEQUENCE_COUNT,
                    (
                        "State-path condition "
                        "does not contain 144 "
                        "sequence rows."
                    ),
                )

                per_seed.append(
                    float(
                        np.mean(
                            [
                                float(
                                    row[
                                        "state_path_accuracy"
                                    ]
                                )
                                for row
                                in rows
                            ]
                        )
                    )
                )

            summary = (
                summarize_values(
                    per_seed
                )
            )

            state_path_summary_rows.append(
                {
                    "noise_fraction":
                        noise,

                    "cell_id":
                        cell_id,

                    "seed_count":
                        5,

                    "state_path_accuracy_mean":
                        summary[
                            "mean"
                        ],

                    "state_path_accuracy_std":
                        summary[
                            "std"
                        ],

                    "state_path_accuracy_minimum":
                        summary[
                            "minimum"
                        ],

                    "state_path_accuracy_maximum":
                        summary[
                            "maximum"
                        ],
                }
            )

    print(
        "[8/9] Writing immutable G2D result tables"
    )

    write_csv(
        CHECKPOINT_SUMMARY_PATH,
        checkpoint_summary_rows,
    )

    write_csv(
        TRANSFORMATION_ROWS_PATH,
        transformation_rows_all,
    )

    write_csv(
        STATE_PATH_ROWS_PATH,
        state_path_rows_all,
    )

    write_csv(
        NOISE_SUMMARY_PATH,
        noise_summary_rows,
    )

    write_csv(
        STATE_PATH_SUMMARY_PATH,
        state_path_summary_rows,
    )

    print(
        "[9/9] Freezing G2D summary and hashes"
    )

    summary = {
        "phase":
            (
                "4G-R3G2D-R1 Tier C v4 "
                "exact Phase-3G structural/process-memory evaluation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "test_open_count":
            1,

        "additional_training_performed":
            False,

        "additional_tuning_performed":
            False,

        "checkpoint_reselection_performed":
            False,

        "structural_labels_used_only_after_predictive_checkpoints_frozen":
            True,

        "alignment_source":
            (
                "Non-test normalized development "
                "carrier/state fields only."
            ),

        "alignment_changes_model_parameters":
            False,

        "evaluated_model":
            "OCM",

        "structural_estimator_lineage":
            (
                "Phase 3G pre-existing hard OCM estimator: "
                "training-carrier argmax confusion alignment "
                "plus direct operation-channel-logit argmax."
            ),

        "tier_c_adaptation":
            (
                "Frozen Tier-C-v4 codec is applied before "
                "OCM state encoding because the predictive "
                "OCM consumes codec observations."
            ),

        "supersedes_g2d_v0_for_registered_structural_metrics":
            True,

        "g2d_v0_status":
            (
                "preserved diagnostic; estimator implementation "
                "did not match pre-existing Phase-3G semantics"
            ),

        "development_carrier_splits_path":
            str(
                development_carrier_splits_path
            ),

        "development_carrier_splits_sha256":
            sha256_file(
                development_carrier_splits_path
            ),

        "phase3g_structural_source_path":
            str(
                Path(
                    phase3g_structural.__file__
                )
            ),

        "phase3g_structural_source_sha256":
            sha256_file(
                Path(
                    phase3g_structural.__file__
                )
            ),

        "ocm_checkpoint_count":
            25,

        "noise_levels":
            NOISE_LEVELS,

        "seeds":
            EXPECTED_SEEDS,

        "sealed_test_cells":
            TEST_CELLS,

        "transformation_count":
            TRANSFORMATION_COUNT,

        "information_class_count":
            INFORMATION_CLASS_COUNT,

        "all_ordered_distinct_relation_pair_count":
            EXPECTED_ALL_RELATION_PAIRS,

        "test_involved_relation_pair_count":
            EXPECTED_TEST_INVOLVED_PAIRS,

        "test_test_relation_pair_count":
            EXPECTED_TEST_TEST_PAIRS,

        "checkpoint_summary_path":
            str(
                CHECKPOINT_SUMMARY_PATH
            ),

        "transformation_metrics_path":
            str(
                TRANSFORMATION_ROWS_PATH
            ),

        "state_path_metrics_path":
            str(
                STATE_PATH_ROWS_PATH
            ),

        "noise_summary_path":
            str(
                NOISE_SUMMARY_PATH
            ),

        "state_path_summary_path":
            str(
                STATE_PATH_SUMMARY_PATH
            ),

        "development_state_bank_path":
            str(
                development_state_bank_path
            ),

        "development_state_bank_sha256":
            sha256_file(
                development_state_bank_path
            ),

        "symbolic_ground_truth":
            {
                key:
                    value
                for key, value
                in ground_truth.items()
                if key
                in {
                    "reverse_word",
                    "transformation_table_path",
                    "transformation_table_sha256",
                    "word_table_path",
                    "word_table_sha256",
                }
            },

        "source_signature":
            source_signature,

        "elapsed_seconds":
            float(
                time.time()
                - start_time
            ),

        "phase4gr3g2d_status":
            "exact_phase3g_structural_results_frozen",

        "next_phase":
            (
                "4G-R3G2E final statistical "
                "and hypothesis freeze"
            ),
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    artifacts = [
        CHECKPOINT_SUMMARY_PATH,
        TRANSFORMATION_ROWS_PATH,
        STATE_PATH_ROWS_PATH,
        NOISE_SUMMARY_PATH,
        STATE_PATH_SUMMARY_PATH,
        SUMMARY_PATH,
    ]

    write_json(
        HASHES_PATH,
        {
            str(path):
                sha256_file(
                    path
                )
            for path in artifacts
        },
    )

    print()
    print(
        "=" * 90
    )

    print(
        "PHASE 4G-R3G2D-R1 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "OCM checkpoints:",
        len(
            checkpoint_summary_rows
        ),
    )

    print()
    print(
        "Five-seed structural summary:"
    )

    for row in noise_summary_rows:
        print(
            "noise=",
            row[
                "noise_fraction"
            ],
            "| exact=",
            round(
                row[
                    "exact_transformation_rate_mean"
                ],
                6,
            ),
            "| test_exact=",
            round(
                row[
                    "test_exact_transformation_rate_mean"
                ],
                6,
            ),
            "| info=",
            round(
                row[
                    "information_class_accuracy_mean"
                ],
                6,
            ),
            "| Blackwell BA=",
            round(
                row[
                    "blackwell_relation_balanced_accuracy_mean"
                ],
                6,
            ),
        )

    print()
    print(
        "Status:",
        summary[
            "phase4gr3g2d_status"
        ],
    )


if __name__ == "__main__":
    main()
