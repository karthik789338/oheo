from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


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
CHANNEL_COUNT = 4

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32

EXPECTED_TRAIN_SAMPLE_COUNT = (
    TRAIN_CARRIER_COUNT * STATE_COUNT
)

EXPECTED_VALIDATION_SAMPLE_COUNT = (
    VALIDATION_CARRIER_COUNT * STATE_COUNT
)

MAXIMUM_VALIDATION_ACCURACY = 0.95

CLASSIFIER_RANDOM_SEED = 72051
CLASSIFIER_C = 1.0
CLASSIFIER_MAX_ITERATIONS = 10000
CLASSIFIER_TOLERANCE = 1e-10

MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE = 1e-6

CHANNEL_NAMES = (
    "phase_field",
    "orientation_cosine",
    "orientation_sine",
    "defect_density",
)

STATISTIC_NAMES = (
    "mean",
    "standard_deviation",
    "minimum",
    "quartile_25",
    "median",
    "quartile_75",
    "maximum",
)

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)


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
        return list(csv.DictReader(handle))


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


def find_sealed_test_files():
    found = []

    prohibited_name_fragments = (
        *SEALED_TEST_CELLS,
        "test_carrier_parameters",
        "test_carrier_state_fields",
        "test_fields.npy",
        "test_manifest",
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

            name = path.name.lower()

            if any(
                fragment in name
                for fragment
                in prohibited_name_fragments
            ):
                found.append(str(path))

    return sorted(set(found))


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

    field_summary = load_json(
        FIELD_DIR
        / "phase4br3_field_summary.json"
    )

    dataset_summary = load_json(
        DATA_DIR
        / "phase4br3_summary.json"
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
        float(
            protocol[
                "acceptance_maximum_mean_interface_ratio"
            ]
        )
        != 0.70
    ):
        raise AssertionError(
            "Tier C v4 acceptance protocol changed."
        )

    if (
        predecoder[
            "phase4cr3_predecoder_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "Phase 4C-R3 pre-decoder audit "
            "did not pass."
        )

    if (
        predecoder[
            "diagnostic_decoder_fitting_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Diagnostic fitting was not authorized."
        )

    if (
        state_decoder[
            "state_decoder_diagnostic_status"
        ]
        != "passed"
    ):
        raise AssertionError(
            "The Tier C v4 state decoder "
            "diagnostic did not pass."
        )

    if (
        state_decoder[
            "state_separability_gate_passed"
        ]
        is not True
    ):
        raise AssertionError(
            "State-separability gate failed."
        )

    if (
        state_decoder[
            "global_statistics_diagnostic_authorized"
        ]
        is not True
    ):
        raise AssertionError(
            "Global-statistics diagnostic "
            "is not authorized."
        )

    if (
        state_decoder[
            "predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training was incorrectly "
            "authorized."
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
            "Tier C v4 test files are not "
            "recorded as absent."
        )

    sealed_test_files = (
        find_sealed_test_files()
    )

    if sealed_test_files:
        raise AssertionError(
            "Sealed Tier C v4 test files exist: "
            + ", ".join(sealed_test_files)
        )

    return {
        "protocol":
            protocol,

        "predecoder":
            predecoder,

        "state_decoder":
            state_decoder,

        "field_summary":
            field_summary,

        "dataset_summary":
            dataset_summary,
    }


def create_feature_registry():
    rows = []
    feature_index = 0

    for channel_index, channel_name in enumerate(
        CHANNEL_NAMES
    ):
        for statistic_name in STATISTIC_NAMES:
            rows.append(
                {
                    "feature_index":
                        feature_index,

                    "feature_name":
                        (
                            f"{channel_name}__"
                            f"{statistic_name}"
                        ),

                    "channel_index":
                        channel_index,

                    "channel_name":
                        channel_name,

                    "statistic":
                        statistic_name,

                    "spatial_information_used":
                        False,

                    "training_labels_used_to_define_feature":
                        False,
                }
            )

            feature_index += 1

    if len(rows) != 28:
        raise AssertionError(
            "Global-statistics feature count "
            "must equal 28."
        )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_feature_registry.csv",
        rows,
    )

    return rows


def write_diagnostic_protocol(
    feature_registry,
):
    protocol = {
        "phase":
            (
                "4C-R3B Tier C v4 global-statistics "
                "shortcut diagnostic protocol"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "protocol_frozen_before_feature_extraction":
            True,

        "feature_count":
            len(feature_registry),

        "channel_count":
            CHANNEL_COUNT,

        "statistics_per_channel":
            len(STATISTIC_NAMES),

        "channel_names":
            list(CHANNEL_NAMES),

        "statistic_names":
            list(STATISTIC_NAMES),

        "spatial_arrangement_used":
            False,

        "training_fields":
            (
                "Clean raw Tier C v4 training-carrier "
                "fields only."
            ),

        "validation_fields":
            (
                "Clean raw Tier C v4 validation-carrier "
                "fields only."
            ),

        "standardization":
            (
                "Training-set mean and scale only."
            ),

        "classifier":
            "multinomial logistic regression",

        "classifier_solver":
            "lbfgs",

        "classifier_c":
            CLASSIFIER_C,

        "classifier_tolerance":
            CLASSIFIER_TOLERANCE,

        "classifier_max_iterations":
            CLASSIFIER_MAX_ITERATIONS,

        "classifier_random_seed":
            CLASSIFIER_RANDOM_SEED,

        "maximum_allowed_validation_accuracy":
            MAXIMUM_VALIDATION_ACCURACY,

        "within_carrier_feature_range_tolerance":
            MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE,

        "test_fields_used":
            False,

        "predictive_checkpoint_created":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "global_statistics_protocol.json",
        protocol,
    )

    return protocol


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
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    expected_shape = (
        TRAIN_CARRIER_COUNT
        + VALIDATION_CARRIER_COUNT,
        STATE_COUNT,
        CHANNEL_COUNT,
        32,
        32,
    )

    if fields.shape != expected_shape:
        raise AssertionError(
            f"Tier C v4 raw field shape changed: "
            f"{fields.shape}"
        )

    if fields.dtype != np.float32:
        raise AssertionError(
            "Tier C v4 raw fields are not float32."
        )

    if len(set(
        carrier_ids.tolist()
    )) != (
        TRAIN_CARRIER_COUNT
        + VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development carrier IDs are not unique."
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

    if set(
        carrier_ids[training_indices].tolist()
    ) & set(
        carrier_ids[validation_indices].tolist()
    ):
        raise AssertionError(
            "Training and validation carriers overlap."
        )

    labels = np.broadcast_to(
        np.arange(
            STATE_COUNT,
            dtype=np.int64,
        )[None, :],
        (
            len(carrier_ids),
            STATE_COUNT,
        ),
    )

    return {
        "carrier_ids":
            carrier_ids,

        "carrier_splits":
            carrier_splits,

        "fields":
            fields,

        "labels":
            labels,

        "training_indices":
            training_indices,

        "validation_indices":
            validation_indices,
    }


def extract_global_statistics(fields):
    values = np.asarray(
        fields,
        dtype=np.float64,
    )

    if values.ndim != 5:
        raise AssertionError(
            "Expected fields with shape "
            "[carrier, state, channel, height, width]."
        )

    if values.shape[2] != CHANNEL_COUNT:
        raise AssertionError(
            "Field channel count changed."
        )

    means = values.mean(
        axis=(-2, -1)
    )

    standard_deviations = values.std(
        axis=(-2, -1),
        ddof=0,
    )

    minima = values.min(
        axis=(-2, -1)
    )

    quartile_25 = np.quantile(
        values,
        0.25,
        axis=(-2, -1),
        method="linear",
    )

    medians = np.quantile(
        values,
        0.50,
        axis=(-2, -1),
        method="linear",
    )

    quartile_75 = np.quantile(
        values,
        0.75,
        axis=(-2, -1),
        method="linear",
    )

    maxima = values.max(
        axis=(-2, -1)
    )

    statistic_arrays = (
        means,
        standard_deviations,
        minima,
        quartile_25,
        medians,
        quartile_75,
        maxima,
    )

    feature_blocks = []

    for channel_index in range(
        CHANNEL_COUNT
    ):
        for statistic_array in statistic_arrays:
            feature_blocks.append(
                statistic_array[
                    :,
                    :,
                    channel_index,
                ]
            )

    features = np.stack(
        feature_blocks,
        axis=-1,
    )

    expected_shape = (
        values.shape[0],
        STATE_COUNT,
        28,
    )

    if features.shape != expected_shape:
        raise AssertionError(
            f"Global-statistics shape changed: "
            f"{features.shape}"
        )

    if not np.isfinite(features).all():
        raise FloatingPointError(
            "Global-statistics features contain "
            "NaN or Inf."
        )

    return features


def audit_feature_invariance(
    features,
    carrier_ids,
    carrier_splits,
):
    carrier_feature_ranges = (
        features.max(axis=1)
        - features.min(axis=1)
    )

    maximum_range_per_carrier = (
        carrier_feature_ranges.max(
            axis=1
        )
    )

    maximum_feature_range = float(
        carrier_feature_ranges.max()
    )

    mean_maximum_range = float(
        maximum_range_per_carrier.mean()
    )

    carriers_above_tolerance = int(
        np.sum(
            maximum_range_per_carrier
            > MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE
        )
    )

    rows = []

    for index, carrier_id in enumerate(
        carrier_ids
    ):
        rows.append(
            {
                "carrier_id":
                    int(carrier_id),

                "carrier_split":
                    str(
                        carrier_splits[index]
                    ),

                "maximum_within_carrier_feature_range":
                    float(
                        maximum_range_per_carrier[
                            index
                        ]
                    ),

                "feature_invariance_tolerance":
                    MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE,

                "feature_invariance_passed":
                    bool(
                        maximum_range_per_carrier[
                            index
                        ]
                        <= MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_carrier_invariance.csv",
        rows,
    )

    result = {
        "carrier_count":
            len(carrier_ids),

        "feature_count":
            features.shape[-1],

        "maximum_within_carrier_feature_range":
            maximum_feature_range,

        "mean_carrier_maximum_feature_range":
            mean_maximum_range,

        "feature_invariance_tolerance":
            MAXIMUM_WITHIN_CARRIER_FEATURE_RANGE,

        "carriers_above_tolerance":
            carriers_above_tolerance,

        "all_carriers_feature_invariant":
            carriers_above_tolerance == 0,
    }

    write_json(
        OUTPUT_DIR
        / "global_statistics_invariance_summary.json",
        result,
    )

    return result


def prepare_classification_data(
    field_data,
    features,
):
    training_indices = field_data[
        "training_indices"
    ]

    validation_indices = field_data[
        "validation_indices"
    ]

    labels = field_data["labels"]

    training_features = features[
        training_indices
    ].reshape(
        -1,
        features.shape[-1],
    )

    validation_features = features[
        validation_indices
    ].reshape(
        -1,
        features.shape[-1],
    )

    training_labels = labels[
        training_indices
    ].reshape(-1)

    validation_labels = labels[
        validation_indices
    ].reshape(-1)

    training_carrier_ids = np.repeat(
        field_data["carrier_ids"][
            training_indices
        ],
        STATE_COUNT,
    )

    validation_carrier_ids = np.repeat(
        field_data["carrier_ids"][
            validation_indices
        ],
        STATE_COUNT,
    )

    if training_features.shape != (
        EXPECTED_TRAIN_SAMPLE_COUNT,
        28,
    ):
        raise AssertionError(
            "Training global-statistics shape changed."
        )

    if validation_features.shape != (
        EXPECTED_VALIDATION_SAMPLE_COUNT,
        28,
    ):
        raise AssertionError(
            "Validation global-statistics shape changed."
        )

    expected_class_counts_train = {
        state_id: TRAIN_CARRIER_COUNT
        for state_id in range(STATE_COUNT)
    }

    expected_class_counts_validation = {
        state_id: VALIDATION_CARRIER_COUNT
        for state_id in range(STATE_COUNT)
    }

    observed_training_counts = {
        state_id:
            int(
                np.sum(
                    training_labels == state_id
                )
            )
        for state_id in range(STATE_COUNT)
    }

    observed_validation_counts = {
        state_id:
            int(
                np.sum(
                    validation_labels == state_id
                )
            )
        for state_id in range(STATE_COUNT)
    }

    if (
        observed_training_counts
        != expected_class_counts_train
    ):
        raise AssertionError(
            "Training class balance changed."
        )

    if (
        observed_validation_counts
        != expected_class_counts_validation
    ):
        raise AssertionError(
            "Validation class balance changed."
        )

    if set(
        training_carrier_ids.tolist()
    ) & set(
        validation_carrier_ids.tolist()
    ):
        raise AssertionError(
            "Classifier training and validation "
            "carriers overlap."
        )

    return {
        "training_features":
            training_features,

        "training_labels":
            training_labels,

        "training_carrier_ids":
            training_carrier_ids,

        "validation_features":
            validation_features,

        "validation_labels":
            validation_labels,

        "validation_carrier_ids":
            validation_carrier_ids,
    }


def fit_classifier(data):
    scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )

    training_features_scaled = (
        scaler.fit_transform(
            data["training_features"]
        )
    )

    validation_features_scaled = (
        scaler.transform(
            data["validation_features"]
        )
    )

    classifier = LogisticRegression(
        C=CLASSIFIER_C,
        penalty="l2",
        solver="lbfgs",
        max_iter=CLASSIFIER_MAX_ITERATIONS,
        tol=CLASSIFIER_TOLERANCE,
        fit_intercept=True,
        class_weight=None,
        random_state=CLASSIFIER_RANDOM_SEED,
    )

    classifier.fit(
        training_features_scaled,
        data["training_labels"],
    )

    training_predictions = (
        classifier.predict(
            training_features_scaled
        )
    )

    validation_predictions = (
        classifier.predict(
            validation_features_scaled
        )
    )

    training_probabilities = (
        classifier.predict_proba(
            training_features_scaled
        )
    )

    validation_probabilities = (
        classifier.predict_proba(
            validation_features_scaled
        )
    )

    training_accuracy = float(
        np.mean(
            training_predictions
            == data["training_labels"]
        )
    )

    validation_accuracy = float(
        np.mean(
            validation_predictions
            == data["validation_labels"]
        )
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "global_statistics_classifier.npz",
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classifier_classes=classifier.classes_,
        classifier_coefficients=classifier.coef_,
        classifier_intercepts=classifier.intercept_,
        classifier_iterations=classifier.n_iter_,
    )

    return {
        "scaler":
            scaler,

        "classifier":
            classifier,

        "training_predictions":
            training_predictions,

        "validation_predictions":
            validation_predictions,

        "training_probabilities":
            training_probabilities,

        "validation_probabilities":
            validation_probabilities,

        "training_accuracy":
            training_accuracy,

        "validation_accuracy":
            validation_accuracy,
    }


def create_confusion_rows(
    split,
    labels,
    predictions,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(
            STATE_COUNT,
            dtype=np.int64,
        ),
    )

    rows = []

    for true_state in range(
        STATE_COUNT
    ):
        for predicted_state in range(
            STATE_COUNT
        ):
            rows.append(
                {
                    "split":
                        split,

                    "true_state":
                        true_state,

                    "predicted_state":
                        predicted_state,

                    "count":
                        int(
                            matrix[
                                true_state,
                                predicted_state,
                            ]
                        ),
                }
            )

    return rows


def create_per_state_rows(
    split,
    labels,
    predictions,
    probabilities,
):
    rows = []

    for state_id in range(
        STATE_COUNT
    ):
        selected = labels == state_id

        sample_count = int(
            selected.sum()
        )

        correct_count = int(
            np.sum(
                predictions[selected]
                == state_id
            )
        )

        mean_true_class_probability = float(
            probabilities[
                selected,
                state_id,
            ].mean()
        )

        rows.append(
            {
                "split":
                    split,

                "state_id":
                    state_id,

                "sample_count":
                    sample_count,

                "correct_count":
                    correct_count,

                "accuracy":
                    correct_count
                    / sample_count,

                "mean_true_class_probability":
                    mean_true_class_probability,
            }
        )

    return rows


def write_feature_values(
    data,
):
    rows = []

    feature_registry = load_csv(
        OUTPUT_DIR
        / "global_statistics_feature_registry.csv"
    )

    feature_names = [
        row["feature_name"]
        for row in feature_registry
    ]

    for split in (
        "training",
        "validation",
    ):
        features = data[
            f"{split}_features"
        ]

        labels = data[
            f"{split}_labels"
        ]

        carrier_ids = data[
            f"{split}_carrier_ids"
        ]

        for sample_index in range(
            len(labels)
        ):
            row = {
                "split":
                    split,

                "sample_index":
                    sample_index,

                "carrier_id":
                    int(
                        carrier_ids[
                            sample_index
                        ]
                    ),

                "state_id":
                    int(
                        labels[
                            sample_index
                        ]
                    ),
            }

            for feature_index, feature_name in (
                enumerate(feature_names)
            ):
                row[feature_name] = float(
                    features[
                        sample_index,
                        feature_index,
                    ]
                )

            rows.append(row)

    write_csv(
        OUTPUT_DIR
        / "global_statistics_features.csv",
        rows,
    )


def write_input_hashes():
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "predecoder_summary":
            AUDIT_DIR
            / "phase4cr3_predecoder_summary.json",

        "state_decoder_summary":
            AUDIT_DIR
            / "state_decoder_summary.json",

        "development_carrier_ids":
            FIELD_DIR
            / "privileged"
            / "development_carrier_ids.npy",

        "development_carrier_splits":
            FIELD_DIR
            / "privileged"
            / "development_carrier_splits.npy",

        "raw_field_bank":
            FIELD_DIR
            / "privileged"
            / "development_carrier_state_fields_raw.npy",

        "feature_registry":
            OUTPUT_DIR
            / "global_statistics_feature_registry.csv",

        "diagnostic_protocol":
            OUTPUT_DIR
            / "global_statistics_protocol.json",
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "global_statistics_input_hashes.json",
        hashes,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_summary_path = (
        OUTPUT_DIR
        / "global_statistics_summary.json"
    )

    if final_summary_path.exists():
        existing = load_json(
            final_summary_path
        )

        if existing.get(
            "global_statistics_diagnostic_status"
        ) in {
            "passed",
            "failed_terminal_gate",
        }:
            print(
                "Tier C v4 global-statistics diagnostic "
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
        "[1/6] Validating diagnostic authorization"
    )

    validate_authorization()

    print(
        "[2/6] Freezing the 28-feature registry "
        "and classifier protocol"
    )

    feature_registry = (
        create_feature_registry()
    )

    diagnostic_protocol = (
        write_diagnostic_protocol(
            feature_registry
        )
    )

    print(
        "[3/6] Extracting clean global statistics"
    )

    field_data = load_clean_field_data()

    features = extract_global_statistics(
        field_data["fields"]
    )

    invariance_summary = (
        audit_feature_invariance(
            features=features,
            carrier_ids=(
                field_data[
                    "carrier_ids"
                ]
            ),
            carrier_splits=(
                field_data[
                    "carrier_splits"
                ]
            ),
        )
    )

    classification_data = (
        prepare_classification_data(
            field_data=field_data,
            features=features,
        )
    )

    write_feature_values(
        classification_data
    )

    write_input_hashes()

    print(
        "[4/6] Fitting the training-only "
        "global-statistics classifier"
    )

    result = fit_classifier(
        classification_data
    )

    print(
        "[5/6] Computing confusion and "
        "per-state diagnostics"
    )

    confusion_rows = []

    confusion_rows.extend(
        create_confusion_rows(
            split="training",
            labels=(
                classification_data[
                    "training_labels"
                ]
            ),
            predictions=(
                result[
                    "training_predictions"
                ]
            ),
        )
    )

    confusion_rows.extend(
        create_confusion_rows(
            split="validation",
            labels=(
                classification_data[
                    "validation_labels"
                ]
            ),
            predictions=(
                result[
                    "validation_predictions"
                ]
            ),
        )
    )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_confusion.csv",
        confusion_rows,
    )

    per_state_rows = []

    per_state_rows.extend(
        create_per_state_rows(
            split="training",
            labels=(
                classification_data[
                    "training_labels"
                ]
            ),
            predictions=(
                result[
                    "training_predictions"
                ]
            ),
            probabilities=(
                result[
                    "training_probabilities"
                ]
            ),
        )
    )

    per_state_rows.extend(
        create_per_state_rows(
            split="validation",
            labels=(
                classification_data[
                    "validation_labels"
                ]
            ),
            predictions=(
                result[
                    "validation_predictions"
                ]
            ),
            probabilities=(
                result[
                    "validation_probabilities"
                ]
            ),
        )
    )

    write_csv(
        OUTPUT_DIR
        / "global_statistics_per_state_metrics.csv",
        per_state_rows,
    )

    print(
        "[6/6] Freezing the shortcut decision"
    )

    validation_accuracy_gate_passed = bool(
        result["validation_accuracy"]
        <= MAXIMUM_VALIDATION_ACCURACY
    )

    feature_count_gate_passed = bool(
        len(feature_registry) == 28
    )

    carrier_split_gate_passed = bool(
        len(
            set(
                classification_data[
                    "training_carrier_ids"
                ].tolist()
            )
            & set(
                classification_data[
                    "validation_carrier_ids"
                ].tolist()
            )
        )
        == 0
    )

    all_values_finite = bool(
        np.isfinite(
            classification_data[
                "training_features"
            ]
        ).all()
        and np.isfinite(
            classification_data[
                "validation_features"
            ]
        ).all()
    )

    global_statistics_gate_passed = bool(
        validation_accuracy_gate_passed
        and feature_count_gate_passed
        and carrier_split_gate_passed
        and all_values_finite
    )

    summary = {
        "phase":
            (
                "4C-R3B Tier C v4 global-statistics "
                "shortcut diagnostic"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_predecoder_gate_passed":
            True,

        "source_state_separability_gate_passed":
            True,

        "feature_count":
            len(feature_registry),

        "channel_count":
            CHANNEL_COUNT,

        "statistics_per_channel":
            len(STATISTIC_NAMES),

        "feature_registry":
            [
                row["feature_name"]
                for row in feature_registry
            ],

        "classifier":
            "multinomial logistic regression",

        "classifier_solver":
            "lbfgs",

        "classifier_c":
            CLASSIFIER_C,

        "classifier_random_seed":
            CLASSIFIER_RANDOM_SEED,

        "sklearn_version":
            sklearn.__version__,

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "training_sample_count":
            EXPECTED_TRAIN_SAMPLE_COUNT,

        "validation_sample_count":
            EXPECTED_VALIDATION_SAMPLE_COUNT,

        "training_validation_carrier_overlap_count":
            0,

        "training_accuracy":
            result["training_accuracy"],

        "validation_accuracy":
            result["validation_accuracy"],

        "maximum_allowed_validation_accuracy":
            MAXIMUM_VALIDATION_ACCURACY,

        "validation_accuracy_gate_passed":
            validation_accuracy_gate_passed,

        "feature_count_gate_passed":
            feature_count_gate_passed,

        "carrier_split_gate_passed":
            carrier_split_gate_passed,

        "all_feature_values_finite":
            all_values_finite,

        "maximum_within_carrier_feature_range":
            invariance_summary[
                "maximum_within_carrier_feature_range"
            ],

        "carriers_above_feature_invariance_tolerance":
            invariance_summary[
                "carriers_above_tolerance"
            ],

        "all_carriers_feature_invariant":
            invariance_summary[
                "all_carriers_feature_invariant"
            ],

        "global_statistics_shortcut_present":
            bool(
                result["validation_accuracy"]
                > MAXIMUM_VALIDATION_ACCURACY
            ),

        "global_statistics_gate_passed":
            global_statistics_gate_passed,

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

        "global_statistics_classifier_fitted":
            True,

        "global_statistics_classifier_eligible_for_predictive_use":
            False,

        "affine_shortcut_diagnostic_authorized":
            global_statistics_gate_passed,

        "predictive_training_performed":
            False,

        "predictive_checkpoint_selection_performed":
            False,

        "predictive_training_authorized":
            False,

        "tier_c_v5_authorized":
            False,

        "prior_outputs_modified":
            False,

        "global_statistics_diagnostic_status":
            (
                "passed"
                if global_statistics_gate_passed
                else "failed_terminal_gate"
            ),
    }

    write_json(
        final_summary_path,
        summary,
    )

    print()
    print(
        "Phase 4C-R3B Tier C v4 global-statistics "
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

    if not global_statistics_gate_passed:
        raise SystemExit(
            "Tier C v4 failed the terminal global-"
            "statistics shortcut gate. Do not run the "
            "affine diagnostic, do not train predictive "
            "models, and do not create Tier C v5."
        )


if __name__ == "__main__":
    main()
