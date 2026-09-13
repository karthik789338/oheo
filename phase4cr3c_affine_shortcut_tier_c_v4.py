from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

FIELD_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

AUDIT_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_audit"
)

OUTPUT_DIR = AUDIT_DIR


PROTOCOL_VERSION = "tier_c_v4"

STATE_COUNT = 8
PRIMITIVE_COUNT = 6
SEMIGROUP_SIZE = 104

FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32
FLATTENED_FIELD_DIMENSION = (
    FIELD_CHANNEL_COUNT
    * GRID_HEIGHT
    * GRID_WIDTH
)

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32

TRAIN_SAMPLE_COUNT = (
    TRAIN_CARRIER_COUNT
    * STATE_COUNT
)

VALIDATION_SAMPLE_COUNT = (
    VALIDATION_CARRIER_COUNT
    * STATE_COUNT
)

PCA_COMPONENT_COUNT = 128
PCA_RANDOM_SEED = 72061
PCA_ITERATED_POWER = 7

RIDGE_ALPHA = 1e-6
RIDGE_SOLVER = "svd"

MINIMUM_VALIDATION_CONTINUOUS_MSE = 1e-8

MAXIMUM_EXACT_PRIMITIVE_RECOVERY_RATE = 1.0

MAXIMUM_FRACTION_VALIDATION_CARRIERS_ALL_104_EXACT = 1.0

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

SOURCE_STATE_ALIASES = (
    "source_state_id",
    "source_state",
    "input_state_id",
    "input_state",
    "from_state_id",
    "from_state",
    "state_id",
)

TARGET_STATE_ALIASES = (
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
    "primitive_operation_id",
    "operation_id",
    "operation",
    "generator_id",
    "generator",
    "action_id",
    "action",
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transition-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit primitive transition CSV. "
            "When omitted, the script searches outputs/ "
            "for an unambiguous frozen 6-by-8 transition table."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def write_json(path, value):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def natural_sort_key(value):
    text = str(value)

    parts = re.split(
        r"(\d+)",
        text,
    )

    return tuple(
        int(part)
        if part.isdigit()
        else part.lower()
        for part in parts
    )


def parse_state_identifier(value):
    text = str(value).strip()

    try:
        parsed = int(text)

    except ValueError:
        try:
            numeric = float(text)

            if not numeric.is_integer():
                raise ValueError

            parsed = int(numeric)

        except ValueError:
            match = re.fullmatch(
                r"[A-Za-z_\-]*([0-7])",
                text,
            )

            if match is None:
                raise ValueError(
                    f"Could not parse state identifier: "
                    f"{value!r}"
                )

            parsed = int(
                match.group(1)
            )

    if parsed < 0 or parsed >= STATE_COUNT:
        raise ValueError(
            f"State identifier outside 0-7: {parsed}"
        )

    return parsed


def find_alias(
    fieldnames,
    aliases,
):
    available = set(fieldnames)

    for alias in aliases:
        if alias in available:
            return alias

    return None


def find_sealed_test_files():
    found = []

    prohibited_fragments = (
        *SEALED_TEST_CELLS,
        "test_carrier_parameters",
        "test_carrier_state_fields",
        "sealed_test_fields",
        "sealed_test_manifest",
    )

    for root in (
        FIELD_DIR,
        DATA_DIR,
        AUDIT_DIR,
    ):
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            lower_name = path.name.lower()

            if any(
                fragment in lower_name
                for fragment
                in prohibited_fragments
            ):
                found.append(
                    str(path)
                )

    return sorted(
        set(found)
    )


def validate_authorization():
    protocol = load_json(
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    predecoder = load_json(
        AUDIT_DIR
        / "phase4cr3_predecoder_summary.json"
    )

    state_decoder = load_json(
        AUDIT_DIR
        / "state_decoder_summary.json"
    )

    global_statistics = load_json(
        AUDIT_DIR
        / "global_statistics_summary.json"
    )

    field_summary = load_json(
        FIELD_DIR
        / "phase4br3_field_summary.json"
    )

    dataset_summary = load_json(
        DATA_DIR
        / "phase4br3_summary.json"
    )

    acceptance_gates = load_json(
        PHASE4AR3_DIR
        / "acceptance_gates_v4.json"
    )

    if (
        protocol["phase4ar3_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R3 is not frozen."
        )

    if (
        protocol["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v4 protocol version changed."
        )

    if (
        protocol["tier_c_v4_is_final_revision"]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 is not frozen as final."
        )

    if (
        protocol["tier_c_v5_authorized"]
        is not False
    ):
        raise AssertionError(
            "Tier C v5 was unexpectedly authorized."
        )

    if (
        predecoder[
            "phase4cr3_predecoder_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Pre-decoder acceptance audit did not pass."
        )

    if (
        state_decoder[
            "state_decoder_diagnostic_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "State-separability diagnostic did not pass."
        )

    if (
        state_decoder[
            "affine_shortcut_diagnostic_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Affine-shortcut diagnostic was not authorized."
        )

    if (
        global_statistics[
            "global_statistics_diagnostic_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Global-statistics diagnostic did not pass."
        )

    if (
        global_statistics[
            "global_statistics_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "Global-statistics gate failed."
        )

    if (
        global_statistics[
            "affine_shortcut_diagnostic_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Affine-shortcut diagnostic is not authorized."
        )

    if (
        global_statistics[
            "predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training was incorrectly authorized."
        )

    if (
        field_summary[
            "test_carrier_count_generated"
        ]
        != 0
    ):
        raise AssertionError(
            "Tier C v4 test carriers were generated."
        )

    if (
        dataset_summary[
            "sealed_test_files_absent"
        ]
        is not True
    ):
        raise AssertionError(
            "Sealed test files are not recorded as absent."
        )

    affine_gate = acceptance_gates[
        "affine_shortcut_control"
    ]

    if (
        int(
            affine_gate[
                "diagnostic"
            ].split("PCA-")[1].split()[0]
        )
        != PCA_COMPONENT_COUNT
    ):
        raise AssertionError(
            "Frozen PCA component count changed."
        )

    if (
        float(
            affine_gate[
                "minimum_validation_continuous_mse"
            ]
        )
        != MINIMUM_VALIDATION_CONTINUOUS_MSE
    ):
        raise AssertionError(
            "Frozen affine MSE threshold changed."
        )

    if (
        float(
            affine_gate[
                "validation_exact_primitive_rate_must_be_below"
            ]
        )
        != MAXIMUM_EXACT_PRIMITIVE_RECOVERY_RATE
    ):
        raise AssertionError(
            "Frozen exact primitive threshold changed."
        )

    if (
        float(
            affine_gate[
                "fraction_validation_carriers_all_104_exact_must_be_below"
            ]
        )
        != (
            MAXIMUM_FRACTION_VALIDATION_CARRIERS_ALL_104_EXACT
        )
    ):
        raise AssertionError(
            "Frozen all-104 threshold changed."
        )

    sealed_test_files = (
        find_sealed_test_files()
    )

    if sealed_test_files:
        raise AssertionError(
            "Sealed Tier C v4 test files exist: "
            + ", ".join(
                sealed_test_files
            )
        )

    return {
        "protocol":
            protocol,

        "predecoder":
            predecoder,

        "state_decoder":
            state_decoder,

        "global_statistics":
            global_statistics,

        "field_summary":
            field_summary,

        "dataset_summary":
            dataset_summary,

        "acceptance_gates":
            acceptance_gates,
    }


def normalize_transition_csv(path):
    rows = load_csv(path)

    if not rows:
        raise ValueError(
            f"Transition table is empty: {path}"
        )

    fieldnames = list(
        rows[0].keys()
    )

    source_column = find_alias(
        fieldnames,
        SOURCE_STATE_ALIASES,
    )

    target_column = find_alias(
        fieldnames,
        TARGET_STATE_ALIASES,
    )

    primitive_column = find_alias(
        fieldnames,
        PRIMITIVE_ALIASES,
    )

    if (
        source_column is None
        or target_column is None
        or primitive_column is None
    ):
        raise ValueError(
            "CSV does not contain recognizable "
            "primitive/source/target columns."
        )

    normalized_pairs = {}

    for row in rows:
        primitive_label = str(
            row[primitive_column]
        ).strip()

        if not primitive_label:
            continue

        source_state = parse_state_identifier(
            row[source_column]
        )

        target_state = parse_state_identifier(
            row[target_column]
        )

        key = (
            primitive_label,
            source_state,
        )

        if (
            key in normalized_pairs
            and normalized_pairs[key]
            != target_state
        ):
            raise ValueError(
                f"Conflicting transitions for {key} "
                f"in {path}."
            )

        normalized_pairs[key] = target_state

    primitive_labels = sorted(
        {
            primitive
            for primitive, _
            in normalized_pairs
        },
        key=natural_sort_key,
    )

    if len(primitive_labels) != PRIMITIVE_COUNT:
        raise ValueError(
            f"Expected six primitives, found "
            f"{len(primitive_labels)}."
        )

    primitive_maps = {}

    for primitive in primitive_labels:
        source_states = {
            source
            for candidate, source
            in normalized_pairs
            if candidate == primitive
        }

        if source_states != set(
            range(STATE_COUNT)
        ):
            raise ValueError(
                f"Primitive {primitive!r} does not "
                "define all eight source states."
            )

        primitive_maps[primitive] = tuple(
            normalized_pairs[
                (
                    primitive,
                    source_state,
                )
            ]
            for source_state in range(
                STATE_COUNT
            )
        )

    normalized_row_count = len(
        normalized_pairs
    )

    if normalized_row_count != (
        PRIMITIVE_COUNT * STATE_COUNT
    ):
        raise ValueError(
            "Expected exactly 48 unique primitive "
            "state transitions."
        )

    signature = tuple(
        (
            primitive,
            primitive_maps[primitive],
        )
        for primitive in primitive_labels
    )

    return {
        "path":
            Path(path),

        "source_column":
            source_column,

        "target_column":
            target_column,

        "primitive_column":
            primitive_column,

        "primitive_labels":
            primitive_labels,

        "primitive_maps":
            primitive_maps,

        "signature":
            signature,

        "normalized_row_count":
            normalized_row_count,
    }


def candidate_score(path):
    text = str(path).lower()

    score = 0

    if "phase1" in text:
        score += 10

    if "primitive" in text:
        score += 6

    if "transition" in text:
        score += 6

    if "semigroup" in text:
        score += 4

    if "registry" in text:
        score += 2

    return score


def discover_transition_table(
    explicit_path,
):
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(
                explicit_path
            )

        selected = normalize_transition_csv(
            explicit_path
        )

        selected[
            "discovery_mode"
        ] = "explicit"

        selected[
            "equivalent_candidate_paths"
        ] = [
            str(explicit_path)
        ]

        return selected

    candidates = []

    for path in Path("outputs").rglob("*.csv"):
        if not path.is_file():
            continue

        if OUTPUT_DIR in path.parents:
            continue

        try:
            if path.stat().st_size > (
                100 * 1024 * 1024
            ):
                continue

            normalized = (
                normalize_transition_csv(
                    path
                )
            )

            candidates.append(
                normalized
            )

        except (
            ValueError,
            KeyError,
            UnicodeDecodeError,
            csv.Error,
        ):
            continue

    if not candidates:
        raise FileNotFoundError(
            "Could not automatically locate a frozen "
            "six-primitive, eight-state transition CSV. "
            "Rerun with --transition-csv PATH."
        )

    signatures = {}

    for candidate in candidates:
        signatures.setdefault(
            candidate["signature"],
            [],
        ).append(candidate)

    if len(signatures) != 1:
        conflict_report = []

        for signature_candidates in (
            signatures.values()
        ):
            conflict_report.append(
                [
                    str(candidate["path"])
                    for candidate
                    in signature_candidates
                ]
            )

        raise RuntimeError(
            "Multiple conflicting primitive transition "
            "tables were found. Rerun with "
            "--transition-csv PATH. Candidates: "
            + json.dumps(
                conflict_report
            )
        )

    equivalent_candidates = next(
        iter(signatures.values())
    )

    equivalent_candidates.sort(
        key=lambda candidate: (
            -candidate_score(
                candidate["path"]
            ),
            str(candidate["path"]),
        )
    )

    selected = equivalent_candidates[0]

    selected[
        "discovery_mode"
    ] = "automatic"

    selected[
        "equivalent_candidate_paths"
    ] = [
        str(candidate["path"])
        for candidate
        in equivalent_candidates
    ]

    return selected


def write_transition_registry(
    transition_source,
):
    rows = []

    for primitive_index, primitive in enumerate(
        transition_source[
            "primitive_labels"
        ]
    ):
        transformation = (
            transition_source[
                "primitive_maps"
            ][primitive]
        )

        for source_state, target_state in enumerate(
            transformation
        ):
            rows.append(
                {
                    "primitive_index":
                        primitive_index,

                    "primitive_label":
                        primitive,

                    "source_state_id":
                        source_state,

                    "target_state_id":
                        target_state,
                }
            )

    write_csv(
        OUTPUT_DIR
        / "affine_primitive_transition_registry.csv",
        rows,
    )


def compose_transformation(
    current_transformation,
    primitive_transformation,
):
    return tuple(
        primitive_transformation[
            current_transformation[
                source_state
            ]
        ]
        for source_state in range(
            STATE_COUNT
        )
    )


def construct_semigroup(
    primitive_labels,
    primitive_maps,
):
    identity = tuple(
        range(STATE_COUNT)
    )

    canonical_words = {
        identity: tuple()
    }

    ordered_transformations = [
        identity
    ]

    queue = deque(
        [identity]
    )

    while queue:
        current = queue.popleft()

        current_word = (
            canonical_words[current]
        )

        for primitive in primitive_labels:
            candidate = compose_transformation(
                current_transformation=current,
                primitive_transformation=(
                    primitive_maps[
                        primitive
                    ]
                ),
            )

            if candidate in canonical_words:
                continue

            canonical_words[candidate] = (
                current_word
                + (primitive,)
            )

            ordered_transformations.append(
                candidate
            )

            queue.append(candidate)

    if len(
        ordered_transformations
    ) != SEMIGROUP_SIZE:
        raise AssertionError(
            "Primitive transition closure produced "
            f"{len(ordered_transformations)} elements, "
            f"not the frozen size {SEMIGROUP_SIZE}."
        )

    rows = []

    for element_id, transformation in enumerate(
        ordered_transformations
    ):
        word = canonical_words[
            transformation
        ]

        row = {
            "semigroup_element_id":
                element_id,

            "canonical_word_length":
                len(word),

            "canonical_word":
                json.dumps(
                    list(word)
                ),

            "is_identity":
                len(word) == 0,
        }

        for source_state, target_state in enumerate(
            transformation
        ):
            row[
                f"state_{source_state}_target"
            ] = target_state

        rows.append(row)

    write_csv(
        OUTPUT_DIR
        / "affine_semigroup_registry.csv",
        rows,
    )

    return {
        "identity":
            identity,

        "transformations":
            ordered_transformations,

        "canonical_words":
            canonical_words,

        "rows":
            rows,
    }


def load_clean_field_data():
    carrier_ids = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_ids.npy"
    ).astype(np.int64)

    carrier_splits = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_splits.npy"
    ).astype(str)

    fields = np.load(
        FIELD_DIR
        / "privileged"
        / "development_carrier_state_fields_normalized.npy",
        mmap_mode="r",
    )

    expected_shape = (
        TRAIN_CARRIER_COUNT
        + VALIDATION_CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if fields.shape != expected_shape:
        raise AssertionError(
            f"Normalized field shape changed: "
            f"{fields.shape}"
        )

    if fields.dtype != np.float32:
        raise AssertionError(
            "Normalized fields are not float32."
        )

    training_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(training_indices) != (
        TRAIN_CARRIER_COUNT
    ):
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(validation_indices) != (
        VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Validation carrier count changed."
        )

    training_ids = carrier_ids[
        training_indices
    ]

    validation_ids = carrier_ids[
        validation_indices
    ]

    if set(
        training_ids.tolist()
    ) & set(
        validation_ids.tolist()
    ):
        raise AssertionError(
            "Training and validation carriers overlap."
        )

    training_fields = np.asarray(
        fields[training_indices],
        dtype=np.float32,
    ).reshape(
        TRAIN_CARRIER_COUNT,
        STATE_COUNT,
        FLATTENED_FIELD_DIMENSION,
    ).copy()

    validation_fields = np.asarray(
        fields[validation_indices],
        dtype=np.float32,
    ).reshape(
        VALIDATION_CARRIER_COUNT,
        STATE_COUNT,
        FLATTENED_FIELD_DIMENSION,
    ).copy()

    if not np.isfinite(
        training_fields
    ).all():
        raise FloatingPointError(
            "Training fields contain NaN or Inf."
        )

    if not np.isfinite(
        validation_fields
    ).all():
        raise FloatingPointError(
            "Validation fields contain NaN or Inf."
        )

    return {
        "training_carrier_ids":
            training_ids.copy(),

        "validation_carrier_ids":
            validation_ids.copy(),

        "training_fields":
            training_fields,

        "validation_fields":
            validation_fields,
    }


def write_affine_protocol(
    transition_source,
):
    protocol = {
        "phase":
            (
                "4C-R3C Tier C v4 PCA-128 "
                "ridge affine-shortcut protocol"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "protocol_frozen_before_model_fitting":
            True,

        "transition_source_path":
            str(
                transition_source["path"]
            ),

        "transition_discovery_mode":
            transition_source[
                "discovery_mode"
            ],

        "primitive_count":
            PRIMITIVE_COUNT,

        "state_count":
            STATE_COUNT,

        "expected_semigroup_size":
            SEMIGROUP_SIZE,

        "training_fields":
            (
                "Clean normalized Tier C v4 "
                "training-carrier fields only."
            ),

        "validation_fields":
            (
                "Clean normalized Tier C v4 "
                "validation-carrier fields only."
            ),

        "pca": {
            "component_count":
                PCA_COMPONENT_COUNT,

            "solver":
                "randomized",

            "random_seed":
                PCA_RANDOM_SEED,

            "iterated_power":
                PCA_ITERATED_POWER,

            "whiten":
                False,

            "fit_scope":
                "training fields only",
        },

        "primitive_operator": {
            "model":
                "multi-output ridge affine regression",

            "ridge_alpha":
                RIDGE_ALPHA,

            "solver":
                RIDGE_SOLVER,

            "fit_intercept":
                True,

            "one_operator_per_primitive":
                True,

            "operator_tuning":
                False,
        },

        "structural_decoder": {
            "method":
                (
                    "Nearest one of the eight actual "
                    "same-carrier validation states "
                    "in PCA-128 space."
                ),

            "classifier_fitted":
                False,

            "validation_labels_used_for_operator_fitting":
                False,
        },

        "semigroup_operator_composition": {
            "canonical_words":
                (
                    "Deterministic breadth-first closure "
                    "under naturally sorted primitive labels."
                ),

            "composition_tuning":
                False,

            "expected_element_count":
                SEMIGROUP_SIZE,
        },

        "acceptance_thresholds": {
            "minimum_validation_continuous_latent_mse":
                MINIMUM_VALIDATION_CONTINUOUS_MSE,

            "validation_exact_primitive_recovery_rate_must_be_below":
                MAXIMUM_EXACT_PRIMITIVE_RECOVERY_RATE,

            "fraction_validation_carriers_all_104_exact_must_be_below":
                (
                    MAXIMUM_FRACTION_VALIDATION_CARRIERS_ALL_104_EXACT
                ),
        },

        "test_fields_used":
            False,

        "diagnostic_coefficients_eligible_for_predictive_use":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "affine_shortcut_protocol.json",
        protocol,
    )

    return protocol


def fit_training_pca(field_data):
    training_flat = (
        field_data[
            "training_fields"
        ].reshape(
            TRAIN_SAMPLE_COUNT,
            FLATTENED_FIELD_DIMENSION,
        )
    )

    validation_flat = (
        field_data[
            "validation_fields"
        ].reshape(
            VALIDATION_SAMPLE_COUNT,
            FLATTENED_FIELD_DIMENSION,
        )
    )

    pca = PCA(
        n_components=PCA_COMPONENT_COUNT,
        svd_solver="randomized",
        whiten=False,
        random_state=PCA_RANDOM_SEED,
        iterated_power=PCA_ITERATED_POWER,
        copy=True,
    )

    training_scores_flat = pca.fit_transform(
        training_flat
    )

    validation_scores_flat = pca.transform(
        validation_flat
    )

    if training_scores_flat.shape != (
        TRAIN_SAMPLE_COUNT,
        PCA_COMPONENT_COUNT,
    ):
        raise AssertionError(
            "Training PCA score shape changed."
        )

    if validation_scores_flat.shape != (
        VALIDATION_SAMPLE_COUNT,
        PCA_COMPONENT_COUNT,
    ):
        raise AssertionError(
            "Validation PCA score shape changed."
        )

    if not np.isfinite(
        training_scores_flat
    ).all():
        raise FloatingPointError(
            "Training PCA scores contain NaN or Inf."
        )

    if not np.isfinite(
        validation_scores_flat
    ).all():
        raise FloatingPointError(
            "Validation PCA scores contain NaN or Inf."
        )

    training_scores = (
        training_scores_flat.reshape(
            TRAIN_CARRIER_COUNT,
            STATE_COUNT,
            PCA_COMPONENT_COUNT,
        )
    )

    validation_scores = (
        validation_scores_flat.reshape(
            VALIDATION_CARRIER_COUNT,
            STATE_COUNT,
            PCA_COMPONENT_COUNT,
        )
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "affine_pca_model.npz",
        mean=pca.mean_,
        components=pca.components_,
        explained_variance=(
            pca.explained_variance_
        ),
        explained_variance_ratio=(
            pca.explained_variance_ratio_
        ),
        singular_values=pca.singular_values_,
    )

    result = {
        "pca":
            pca,

        "training_scores":
            training_scores,

        "validation_scores":
            validation_scores,

        "training_flat":
            training_flat,

        "validation_flat":
            validation_flat,

        "explained_variance_ratio_sum":
            float(
                pca.explained_variance_ratio_.sum()
            ),

        "training_reconstruction_mse":
            float(
                np.mean(
                    (
                        pca.inverse_transform(
                            training_scores_flat
                        )
                        - training_flat
                    )
                    ** 2
                )
            ),

        "validation_reconstruction_mse":
            float(
                np.mean(
                    (
                        pca.inverse_transform(
                            validation_scores_flat
                        )
                        - validation_flat
                    )
                    ** 2
                )
            ),
    }

    write_json(
        OUTPUT_DIR
        / "affine_pca_summary.json",
        {
            "component_count":
                PCA_COMPONENT_COUNT,

            "solver":
                "randomized",

            "random_seed":
                PCA_RANDOM_SEED,

            "iterated_power":
                PCA_ITERATED_POWER,

            "training_sample_count":
                TRAIN_SAMPLE_COUNT,

            "validation_sample_count":
                VALIDATION_SAMPLE_COUNT,

            "flattened_field_dimension":
                FLATTENED_FIELD_DIMENSION,

            "explained_variance_ratio_sum":
                result[
                    "explained_variance_ratio_sum"
                ],

            "training_reconstruction_mse":
                result[
                    "training_reconstruction_mse"
                ],

            "validation_reconstruction_mse":
                result[
                    "validation_reconstruction_mse"
                ],

            "validation_fields_used_for_pca_fitting":
                False,

            "test_fields_used":
                False,
        },
    )

    return result


def nearest_same_carrier_states(
    predicted_scores,
    candidate_scores,
):
    differences = (
        predicted_scores[:, None, :]
        - candidate_scores[None, :, :]
    )

    squared_distances = np.sum(
        differences**2,
        axis=2,
    )

    nearest_states = np.argmin(
        squared_distances,
        axis=1,
    )

    nearest_distances = squared_distances[
        np.arange(
            len(predicted_scores)
        ),
        nearest_states,
    ]

    return (
        nearest_states.astype(np.int64),
        nearest_distances,
    )


def fit_primitive_operators(
    transition_source,
    pca_result,
    field_data,
):
    primitive_labels = (
        transition_source[
            "primitive_labels"
        ]
    )

    primitive_maps = (
        transition_source[
            "primitive_maps"
        ]
    )

    training_scores = pca_result[
        "training_scores"
    ]

    validation_scores = pca_result[
        "validation_scores"
    ]

    validation_fields = field_data[
        "validation_fields"
    ]

    operators = {}

    metric_rows = []
    prediction_rows = []

    coefficient_matrices = []
    intercept_vectors = []

    all_validation_latent_squared_errors = []
    all_validation_field_squared_errors = []
    all_validation_exact_flags = []

    for primitive_index, primitive in enumerate(
        primitive_labels
    ):
        state_map = np.asarray(
            primitive_maps[primitive],
            dtype=np.int64,
        )

        training_inputs = (
            training_scores.reshape(
                TRAIN_SAMPLE_COUNT,
                PCA_COMPONENT_COUNT,
            )
        )

        training_targets = (
            training_scores[
                :,
                state_map,
                :,
            ].reshape(
                TRAIN_SAMPLE_COUNT,
                PCA_COMPONENT_COUNT,
            )
        )

        validation_inputs = (
            validation_scores.reshape(
                VALIDATION_SAMPLE_COUNT,
                PCA_COMPONENT_COUNT,
            )
        )

        validation_targets = (
            validation_scores[
                :,
                state_map,
                :,
            ].reshape(
                VALIDATION_SAMPLE_COUNT,
                PCA_COMPONENT_COUNT,
            )
        )

        ridge = Ridge(
            alpha=RIDGE_ALPHA,
            fit_intercept=True,
            solver=RIDGE_SOLVER,
        )

        ridge.fit(
            training_inputs,
            training_targets,
        )

        training_predictions = ridge.predict(
            training_inputs
        )

        validation_predictions = ridge.predict(
            validation_inputs
        )

        coefficient_matrix = (
            ridge.coef_.T.copy()
        )

        intercept_vector = (
            ridge.intercept_.copy()
        )

        operators[primitive] = {
            "matrix":
                coefficient_matrix,

            "intercept":
                intercept_vector,

            "state_map":
                state_map,
        }

        coefficient_matrices.append(
            coefficient_matrix
        )

        intercept_vectors.append(
            intercept_vector
        )

        training_latent_mse = float(
            np.mean(
                (
                    training_predictions
                    - training_targets
                )
                ** 2
            )
        )

        validation_latent_errors = (
            validation_predictions
            - validation_targets
        ) ** 2

        validation_latent_mse = float(
            validation_latent_errors.mean()
        )

        validation_predicted_fields = (
            pca_result["pca"]
            .inverse_transform(
                validation_predictions
            )
        )

        validation_target_fields = (
            validation_fields[
                :,
                state_map,
                :,
            ].reshape(
                VALIDATION_SAMPLE_COUNT,
                FLATTENED_FIELD_DIMENSION,
            )
        )

        validation_field_errors = (
            validation_predicted_fields
            - validation_target_fields
        ) ** 2

        validation_field_mse = float(
            validation_field_errors.mean()
        )

        primitive_exact_flags = []

        for carrier_index in range(
            VALIDATION_CARRIER_COUNT
        ):
            start = (
                carrier_index
                * STATE_COUNT
            )

            end = start + STATE_COUNT

            carrier_predictions = (
                validation_predictions[
                    start:end
                ]
            )

            (
                nearest_states,
                nearest_distances,
            ) = nearest_same_carrier_states(
                predicted_scores=(
                    carrier_predictions
                ),
                candidate_scores=(
                    validation_scores[
                        carrier_index
                    ]
                ),
            )

            target_states = state_map

            exact_flags = (
                nearest_states
                == target_states
            )

            primitive_exact_flags.extend(
                exact_flags.tolist()
            )

            for source_state in range(
                STATE_COUNT
            ):
                flat_index = (
                    start + source_state
                )

                latent_mse = float(
                    np.mean(
                        validation_latent_errors[
                            flat_index
                        ]
                    )
                )

                field_mse = float(
                    np.mean(
                        validation_field_errors[
                            flat_index
                        ]
                    )
                )

                prediction_rows.append(
                    {
                        "primitive_index":
                            primitive_index,

                        "primitive_label":
                            primitive,

                        "validation_carrier_id":
                            int(
                                field_data[
                                    "validation_carrier_ids"
                                ][carrier_index]
                            ),

                        "source_state_id":
                            source_state,

                        "target_state_id":
                            int(
                                target_states[
                                    source_state
                                ]
                            ),

                        "nearest_predicted_state_id":
                            int(
                                nearest_states[
                                    source_state
                                ]
                            ),

                        "structural_recovery_exact":
                            bool(
                                exact_flags[
                                    source_state
                                ]
                            ),

                        "latent_mse":
                            latent_mse,

                        "reconstructed_field_mse":
                            field_mse,

                        "nearest_state_squared_distance":
                            float(
                                nearest_distances[
                                    source_state
                                ]
                            ),
                    }
                )

        primitive_exact_flags = np.asarray(
            primitive_exact_flags,
            dtype=bool,
        )

        exact_recovery_rate = float(
            primitive_exact_flags.mean()
        )

        metric_rows.append(
            {
                "primitive_index":
                    primitive_index,

                "primitive_label":
                    primitive,

                "training_pair_count":
                    TRAIN_SAMPLE_COUNT,

                "validation_pair_count":
                    VALIDATION_SAMPLE_COUNT,

                "training_latent_mse":
                    training_latent_mse,

                "validation_latent_mse":
                    validation_latent_mse,

                "validation_reconstructed_field_mse":
                    validation_field_mse,

                "validation_exact_structural_recovery_count":
                    int(
                        primitive_exact_flags.sum()
                    ),

                "validation_exact_structural_recovery_rate":
                    exact_recovery_rate,

                "ridge_alpha":
                    RIDGE_ALPHA,
            }
        )

        all_validation_latent_squared_errors.append(
            validation_latent_errors.reshape(-1)
        )

        all_validation_field_squared_errors.append(
            validation_field_errors.reshape(-1)
        )

        all_validation_exact_flags.append(
            primitive_exact_flags
        )

    write_csv(
        OUTPUT_DIR
        / "affine_primitive_metrics.csv",
        metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "affine_primitive_validation_predictions.csv",
        prediction_rows,
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "affine_primitive_operators.npz",
        primitive_labels=np.asarray(
            primitive_labels,
            dtype=str,
        ),
        coefficient_matrices=np.stack(
            coefficient_matrices,
            axis=0,
        ),
        intercept_vectors=np.stack(
            intercept_vectors,
            axis=0,
        ),
        ridge_alpha=np.asarray(
            RIDGE_ALPHA,
            dtype=np.float64,
        ),
    )

    all_latent_errors = np.concatenate(
        all_validation_latent_squared_errors
    )

    all_field_errors = np.concatenate(
        all_validation_field_squared_errors
    )

    all_exact_flags = np.concatenate(
        all_validation_exact_flags
    )

    return {
        "operators":
            operators,

        "metric_rows":
            metric_rows,

        "prediction_rows":
            prediction_rows,

        "validation_continuous_latent_mse":
            float(
                all_latent_errors.mean()
            ),

        "validation_reconstructed_field_mse":
            float(
                all_field_errors.mean()
            ),

        "validation_exact_primitive_count":
            int(
                all_exact_flags.sum()
            ),

        "validation_primitive_application_count":
            int(
                len(all_exact_flags)
            ),

        "validation_exact_primitive_recovery_rate":
            float(
                all_exact_flags.mean()
            ),
    }


def compose_affine_word(
    word,
    operators,
):
    matrix = np.eye(
        PCA_COMPONENT_COUNT,
        dtype=np.float64,
    )

    intercept = np.zeros(
        PCA_COMPONENT_COUNT,
        dtype=np.float64,
    )

    for primitive in word:
        primitive_matrix = (
            operators[primitive][
                "matrix"
            ]
        )

        primitive_intercept = (
            operators[primitive][
                "intercept"
            ]
        )

        matrix = (
            matrix
            @ primitive_matrix
        )

        intercept = (
            intercept
            @ primitive_matrix
            + primitive_intercept
        )

    return matrix, intercept


def evaluate_semigroup_compositions(
    semigroup,
    primitive_result,
    pca_result,
    field_data,
):
    validation_scores = pca_result[
        "validation_scores"
    ]

    operators = primitive_result[
        "operators"
    ]

    element_metric_rows = []
    prediction_rows = []

    carrier_element_exact = np.zeros(
        (
            VALIDATION_CARRIER_COUNT,
            SEMIGROUP_SIZE,
        ),
        dtype=bool,
    )

    carrier_element_mse = np.empty(
        (
            VALIDATION_CARRIER_COUNT,
            SEMIGROUP_SIZE,
        ),
        dtype=np.float64,
    )

    total_exact_state_predictions = 0
    total_state_predictions = 0

    for element_id, transformation in enumerate(
        semigroup["transformations"]
    ):
        word = semigroup[
            "canonical_words"
        ][transformation]

        (
            matrix,
            intercept,
        ) = compose_affine_word(
            word=word,
            operators=operators,
        )

        target_states = np.asarray(
            transformation,
            dtype=np.int64,
        )

        element_exact_state_count = 0
        element_state_count = 0

        element_carrier_exact_count = 0
        element_squared_errors = []

        for carrier_index in range(
            VALIDATION_CARRIER_COUNT
        ):
            carrier_inputs = (
                validation_scores[
                    carrier_index
                ]
            )

            carrier_predictions = (
                carrier_inputs
                @ matrix
                + intercept
            )

            carrier_targets = (
                validation_scores[
                    carrier_index,
                    target_states,
                ]
            )

            squared_errors = (
                carrier_predictions
                - carrier_targets
            ) ** 2

            carrier_mse = float(
                squared_errors.mean()
            )

            carrier_element_mse[
                carrier_index,
                element_id,
            ] = carrier_mse

            element_squared_errors.append(
                squared_errors.reshape(-1)
            )

            (
                nearest_states,
                nearest_distances,
            ) = nearest_same_carrier_states(
                predicted_scores=(
                    carrier_predictions
                ),
                candidate_scores=(
                    validation_scores[
                        carrier_index
                    ]
                ),
            )

            exact_flags = (
                nearest_states
                == target_states
            )

            transformation_exact = bool(
                exact_flags.all()
            )

            carrier_element_exact[
                carrier_index,
                element_id,
            ] = transformation_exact

            element_exact_state_count += int(
                exact_flags.sum()
            )

            element_state_count += (
                STATE_COUNT
            )

            if transformation_exact:
                element_carrier_exact_count += 1

            for source_state in range(
                STATE_COUNT
            ):
                prediction_rows.append(
                    {
                        "semigroup_element_id":
                            element_id,

                        "canonical_word_length":
                            len(word),

                        "canonical_word":
                            json.dumps(
                                list(word)
                            ),

                        "validation_carrier_id":
                            int(
                                field_data[
                                    "validation_carrier_ids"
                                ][carrier_index]
                            ),

                        "source_state_id":
                            source_state,

                        "target_state_id":
                            int(
                                target_states[
                                    source_state
                                ]
                            ),

                        "nearest_predicted_state_id":
                            int(
                                nearest_states[
                                    source_state
                                ]
                            ),

                        "state_recovery_exact":
                            bool(
                                exact_flags[
                                    source_state
                                ]
                            ),

                        "carrier_transformation_exact":
                            transformation_exact,

                        "latent_mse":
                            float(
                                np.mean(
                                    squared_errors[
                                        source_state
                                    ]
                                )
                            ),

                        "nearest_state_squared_distance":
                            float(
                                nearest_distances[
                                    source_state
                                ]
                            ),
                    }
                )

        element_squared_errors = np.concatenate(
            element_squared_errors
        )

        element_state_accuracy = (
            element_exact_state_count
            / element_state_count
        )

        element_carrier_exact_fraction = (
            element_carrier_exact_count
            / VALIDATION_CARRIER_COUNT
        )

        element_metric_rows.append(
            {
                "semigroup_element_id":
                    element_id,

                "canonical_word_length":
                    len(word),

                "canonical_word":
                    json.dumps(
                        list(word)
                    ),

                "validation_latent_mse":
                    float(
                        element_squared_errors.mean()
                    ),

                "exact_state_recovery_count":
                    element_exact_state_count,

                "state_prediction_count":
                    element_state_count,

                "exact_state_recovery_rate":
                    element_state_accuracy,

                "exact_carrier_transformation_count":
                    element_carrier_exact_count,

                "validation_carrier_count":
                    VALIDATION_CARRIER_COUNT,

                "exact_carrier_transformation_fraction":
                    element_carrier_exact_fraction,
            }
        )

        total_exact_state_predictions += (
            element_exact_state_count
        )

        total_state_predictions += (
            element_state_count
        )

    write_csv(
        OUTPUT_DIR
        / "affine_semigroup_element_metrics.csv",
        element_metric_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "affine_semigroup_validation_predictions.csv",
        prediction_rows,
    )

    carrier_all_104_exact = (
        carrier_element_exact.all(
            axis=1
        )
    )

    carrier_rows = []

    for carrier_index, carrier_id in enumerate(
        field_data[
            "validation_carrier_ids"
        ]
    ):
        exact_element_count = int(
            carrier_element_exact[
                carrier_index
            ].sum()
        )

        carrier_rows.append(
            {
                "validation_carrier_id":
                    int(carrier_id),

                "exact_semigroup_element_count":
                    exact_element_count,

                "semigroup_element_count":
                    SEMIGROUP_SIZE,

                "exact_semigroup_element_fraction":
                    (
                        exact_element_count
                        / SEMIGROUP_SIZE
                    ),

                "all_104_semigroup_elements_exact":
                    bool(
                        carrier_all_104_exact[
                            carrier_index
                        ]
                    ),

                "mean_semigroup_latent_mse":
                    float(
                        carrier_element_mse[
                            carrier_index
                        ].mean()
                    ),

                "maximum_semigroup_latent_mse":
                    float(
                        carrier_element_mse[
                            carrier_index
                        ].max()
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "affine_semigroup_carrier_metrics.csv",
        carrier_rows,
    )

    fraction_carriers_all_104_exact = float(
        carrier_all_104_exact.mean()
    )

    return {
        "element_metric_rows":
            element_metric_rows,

        "carrier_rows":
            carrier_rows,

        "overall_exact_state_recovery_rate":
            (
                total_exact_state_predictions
                / total_state_predictions
            ),

        "validation_carriers_all_104_exact_count":
            int(
                carrier_all_104_exact.sum()
            ),

        "fraction_validation_carriers_all_104_exact":
            fraction_carriers_all_104_exact,

        "mean_semigroup_latent_mse":
            float(
                carrier_element_mse.mean()
            ),

        "maximum_carrier_element_latent_mse":
            float(
                carrier_element_mse.max()
            ),
    }


def write_input_hashes(
    transition_source,
):
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "acceptance_gates_v4":
            PHASE4AR3_DIR
            / "acceptance_gates_v4.json",

        "predecoder_summary":
            AUDIT_DIR
            / "phase4cr3_predecoder_summary.json",

        "state_decoder_summary":
            AUDIT_DIR
            / "state_decoder_summary.json",

        "global_statistics_summary":
            AUDIT_DIR
            / "global_statistics_summary.json",

        "development_carrier_ids":
            FIELD_DIR
            / "privileged"
            / "development_carrier_ids.npy",

        "development_carrier_splits":
            FIELD_DIR
            / "privileged"
            / "development_carrier_splits.npy",

        "normalized_field_bank":
            FIELD_DIR
            / "privileged"
            / "development_carrier_state_fields_normalized.npy",

        "primitive_transition_source":
            transition_source[
                "path"
            ],

        "affine_protocol":
            OUTPUT_DIR
            / "affine_shortcut_protocol.json",

        "normalized_transition_registry":
            OUTPUT_DIR
            / "affine_primitive_transition_registry.csv",

        "semigroup_registry":
            OUTPUT_DIR
            / "affine_semigroup_registry.csv",
    }

    hashes = {}

    for name, path in paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "affine_shortcut_input_hashes.json",
        hashes,
    )


def main():
    arguments = parse_arguments()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "affine_shortcut_summary.json"
    )

    if (
        summary_path.exists()
        and not arguments.force
    ):
        existing = load_json(
            summary_path
        )

        if existing.get(
            "affine_shortcut_diagnostic_status"
        ) in {
            "passed",
            "failed_terminal_gate",
        }:
            print(
                "Tier C v4 affine-shortcut diagnostic "
                "already has a frozen result."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    print(
        "[1/7] Validating affine-diagnostic authorization"
    )

    validate_authorization()

    print(
        "[2/7] Locating and validating the frozen "
        "primitive transition table"
    )

    transition_source = (
        discover_transition_table(
            arguments.transition_csv
        )
    )

    print(
        "Selected transition table: "
        f"{transition_source['path']}"
    )

    print(
        "Primitive labels: "
        + ", ".join(
            transition_source[
                "primitive_labels"
            ]
        )
    )

    write_transition_registry(
        transition_source
    )

    print(
        "[3/7] Reconstructing the frozen "
        "104-element semigroup"
    )

    semigroup = construct_semigroup(
        primitive_labels=(
            transition_source[
                "primitive_labels"
            ]
        ),
        primitive_maps=(
            transition_source[
                "primitive_maps"
            ]
        ),
    )

    print(
        f"Semigroup size: "
        f"{len(semigroup['transformations'])}"
    )

    write_affine_protocol(
        transition_source
    )

    write_input_hashes(
        transition_source
    )

    print(
        "[4/7] Fitting PCA-128 on clean "
        "training fields only"
    )

    field_data = load_clean_field_data()

    pca_result = fit_training_pca(
        field_data
    )

    print(
        "[5/7] Fitting six training-only "
        "ridge affine primitive operators"
    )

    primitive_result = (
        fit_primitive_operators(
            transition_source=(
                transition_source
            ),
            pca_result=pca_result,
            field_data=field_data,
        )
    )

    print(
        "[6/7] Evaluating composed affine "
        "operators over all 104 semigroup elements"
    )

    semigroup_result = (
        evaluate_semigroup_compositions(
            semigroup=semigroup,
            primitive_result=(
                primitive_result
            ),
            pca_result=pca_result,
            field_data=field_data,
        )
    )

    print(
        "[7/7] Freezing the affine-shortcut decision"
    )

    validation_continuous_mse = (
        primitive_result[
            "validation_continuous_latent_mse"
        ]
    )

    exact_primitive_rate = (
        primitive_result[
            "validation_exact_primitive_recovery_rate"
        ]
    )

    fraction_all_104_exact = (
        semigroup_result[
            "fraction_validation_carriers_all_104_exact"
        ]
    )

    continuous_mse_gate_passed = bool(
        validation_continuous_mse
        >= MINIMUM_VALIDATION_CONTINUOUS_MSE
    )

    exact_primitive_gate_passed = bool(
        exact_primitive_rate
        < MAXIMUM_EXACT_PRIMITIVE_RECOVERY_RATE
    )

    all_104_gate_passed = bool(
        fraction_all_104_exact
        < (
            MAXIMUM_FRACTION_VALIDATION_CARRIERS_ALL_104_EXACT
        )
    )

    semigroup_size_gate_passed = bool(
        len(
            semigroup[
                "transformations"
            ]
        )
        == SEMIGROUP_SIZE
    )

    affine_shortcut_gate_passed = bool(
        continuous_mse_gate_passed
        and exact_primitive_gate_passed
        and all_104_gate_passed
        and semigroup_size_gate_passed
    )

    summary = {
        "phase":
            (
                "4C-R3C Tier C v4 PCA-128 "
                "ridge affine-shortcut diagnostic"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_predecoder_gate_passed":
            True,

        "source_state_separability_gate_passed":
            True,

        "source_global_statistics_gate_passed":
            True,

        "transition_source_path":
            str(
                transition_source["path"]
            ),

        "transition_discovery_mode":
            transition_source[
                "discovery_mode"
            ],

        "equivalent_transition_candidate_count":
            len(
                transition_source[
                    "equivalent_candidate_paths"
                ]
            ),

        "primitive_count":
            len(
                transition_source[
                    "primitive_labels"
                ]
            ),

        "primitive_labels":
            transition_source[
                "primitive_labels"
            ],

        "semigroup_size":
            len(
                semigroup[
                    "transformations"
                ]
            ),

        "expected_semigroup_size":
            SEMIGROUP_SIZE,

        "semigroup_size_gate_passed":
            semigroup_size_gate_passed,

        "pca_component_count":
            PCA_COMPONENT_COUNT,

        "pca_solver":
            "randomized",

        "pca_random_seed":
            PCA_RANDOM_SEED,

        "pca_explained_variance_ratio_sum":
            pca_result[
                "explained_variance_ratio_sum"
            ],

        "pca_training_reconstruction_mse":
            pca_result[
                "training_reconstruction_mse"
            ],

        "pca_validation_reconstruction_mse":
            pca_result[
                "validation_reconstruction_mse"
            ],

        "ridge_alpha":
            RIDGE_ALPHA,

        "ridge_solver":
            RIDGE_SOLVER,

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "training_field_count":
            TRAIN_SAMPLE_COUNT,

        "validation_field_count":
            VALIDATION_SAMPLE_COUNT,

        "validation_primitive_application_count":
            primitive_result[
                "validation_primitive_application_count"
            ],

        "validation_continuous_latent_mse":
            validation_continuous_mse,

        "minimum_required_validation_continuous_mse":
            MINIMUM_VALIDATION_CONTINUOUS_MSE,

        "continuous_mse_gate_passed":
            continuous_mse_gate_passed,

        "validation_reconstructed_field_mse":
            primitive_result[
                "validation_reconstructed_field_mse"
            ],

        "validation_exact_primitive_recovery_count":
            primitive_result[
                "validation_exact_primitive_count"
            ],

        "validation_exact_primitive_recovery_rate":
            exact_primitive_rate,

        "required_exact_primitive_recovery_rate_below":
            MAXIMUM_EXACT_PRIMITIVE_RECOVERY_RATE,

        "exact_primitive_gate_passed":
            exact_primitive_gate_passed,

        "semigroup_overall_exact_state_recovery_rate":
            semigroup_result[
                "overall_exact_state_recovery_rate"
            ],

        "semigroup_mean_latent_mse":
            semigroup_result[
                "mean_semigroup_latent_mse"
            ],

        "validation_carriers_all_104_exact_count":
            semigroup_result[
                "validation_carriers_all_104_exact_count"
            ],

        "fraction_validation_carriers_all_104_exact":
            fraction_all_104_exact,

        "required_fraction_validation_carriers_all_104_exact_below":
            (
                MAXIMUM_FRACTION_VALIDATION_CARRIERS_ALL_104_EXACT
            ),

        "all_104_gate_passed":
            all_104_gate_passed,

        "exact_affine_shortcut_present":
            bool(
                not affine_shortcut_gate_passed
            ),

        "affine_shortcut_gate_passed":
            affine_shortcut_gate_passed,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_fields_used":
            False,

        "test_metrics_computed":
            False,

        "diagnostic_coefficients_created":
            True,

        "diagnostic_coefficients_eligible_for_predictive_use":
            False,

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "phase4cr3_final_freeze_authorized":
            affine_shortcut_gate_passed,

        "predictive_training_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "prior_outputs_modified":
            False,

        "affine_shortcut_diagnostic_status":
            (
                "passed"
                if affine_shortcut_gate_passed
                else "failed_terminal_gate"
            ),
    }

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "Phase 4C-R3C Tier C v4 affine-shortcut "
        "diagnostic completed."
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

    if not affine_shortcut_gate_passed:
        raise SystemExit(
            "Tier C v4 failed the terminal affine-"
            "shortcut gate. Do not train predictive "
            "models, do not generate test artifacts, "
            "and do not create Tier C v5."
        )


if __name__ == "__main__":
    main()
