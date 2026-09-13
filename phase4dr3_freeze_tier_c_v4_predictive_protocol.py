from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

ACCEPTANCE_DIR = Path(
    "outputs/phase4cr3_tier_c_v4_acceptance_freeze"
)

DATA_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_data"
)

OUTPUT_DIR = Path(
    "outputs/phase4dr3_tier_c_v4_predictive_protocol"
)


PROTOCOL_VERSION = "tier_c_v4"

MODEL_IDS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

TUNED_MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

NEURAL_MODEL_IDS = (
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

EXPECTED_TUNING_CONFIGURATION_COUNTS = {
    "OCM": 24,
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
}

EXPECTED_TUNING_CONFIGURATION_COUNT = 43

DEVELOPMENT_SEED = 11
DEVELOPMENT_NOISE_FRACTION = 0.25

FINAL_SEEDS = (
    11,
    23,
    37,
    53,
    71,
)

NOISE_FRACTIONS = (
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
)

TRAINING_CELL = "train_joint"

VALIDATION_CELLS = (
    "val_composition",
    "val_carrier",
    "val_joint",
)

CHECKPOINT_SELECTION_CELL = "val_joint"

CHECKPOINT_SELECTION_METRIC = (
    "noisy_target_rollout_mse"
)

TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

PRIMARY_TEST_CELL = "test_joint"
PRIMARY_NOISE_FRACTION = 0.25
PRIMARY_METRIC = "noisy_target_rollout_mse"

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 73011

EXPECTED_DEVELOPMENT_CELLS = {
    "train_joint",
    "val_composition",
    "val_carrier",
    "val_joint",
}


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(
    path: Path,
    value,
):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(
    path: Path,
    rows,
):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
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


def require(
    condition,
    message,
):
    if not condition:
        raise AssertionError(message)


def normalize_model_id(value):
    return str(value).strip().upper()


def first_present_key(
    row,
    candidates,
):
    for key in candidates:
        if key in row:
            return key

    return None


def validate_acceptance_sources():
    acceptance_path = (
        ACCEPTANCE_DIR
        / "phase4cr3d_acceptance_summary.json"
    )

    gate_path = (
        ACCEPTANCE_DIR
        / "tier_c_v4_acceptance_gates.csv"
    )

    protocol_path = (
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    dataset_path = (
        DATA_DIR
        / "phase4br3_summary.json"
    )

    for path in (
        acceptance_path,
        gate_path,
        protocol_path,
        dataset_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    acceptance = load_json(
        acceptance_path
    )

    protocol = load_json(
        protocol_path
    )

    dataset = load_json(
        dataset_path
    )

    require(
        acceptance[
            "protocol_version"
        ] == PROTOCOL_VERSION,
        "Tier C v4 acceptance protocol changed.",
    )

    require(
        acceptance[
            "tier_c_v4_development_acceptance_passed"
        ] is True,
        "Tier C v4 development acceptance did not pass.",
    )

    require(
        acceptance[
            "phase4d_predictive_protocol_freeze_authorized"
        ] is True,
        "Predictive protocol freeze was not authorized.",
    )

    require(
        acceptance[
            "predictive_training_authorized"
        ] is False,
        "Predictive training was already authorized.",
    )

    require(
        acceptance[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation was already authorized.",
    )

    require(
        acceptance[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation was already authorized.",
    )

    require(
        acceptance[
            "sealed_test_files_absent"
        ] is True,
        "Sealed test files are not recorded as absent.",
    )

    require(
        acceptance[
            "tier_c_v5_authorized"
        ] is False,
        "Tier C v5 was unexpectedly authorized.",
    )

    require(
        protocol[
            "phase4ar3_status"
        ] == "protocol_frozen",
        "Phase 4A-R3 is not frozen.",
    )

    require(
        protocol[
            "tier_c_v4_is_final_revision"
        ] is True,
        "Tier C v4 is not the final revision.",
    )

    require(
        protocol[
            "tier_c_v5_authorized"
        ] is False,
        "Phase 4A-R3 unexpectedly authorizes v5.",
    )

    require(
        dataset[
            "phase4br3_status"
        ] == "completed",
        "Tier C v4 development data is incomplete.",
    )

    require(
        set(
            dataset[
                "development_cells"
            ]
        ) == EXPECTED_DEVELOPMENT_CELLS,
        "Development-cell registry changed.",
    )

    require(
        dataset[
            "test_cell_count_generated"
        ] == 0,
        "Tier C v4 test cells already exist.",
    )

    require(
        dataset[
            "test_carrier_parameters_generated"
        ] is False,
        "Test carrier parameters already exist.",
    )

    require(
        dataset[
            "test_fields_generated"
        ] is False,
        "Test fields already exist.",
    )

    require(
        dataset[
            "test_manifests_generated"
        ] is False,
        "Test manifests already exist.",
    )

    return {
        "acceptance_path":
            acceptance_path,

        "acceptance":
            acceptance,

        "gate_path":
            gate_path,

        "protocol_path":
            protocol_path,

        "protocol":
            protocol,

        "dataset_path":
            dataset_path,

        "dataset":
            dataset,
    }


def candidate_summary_matches(value):
    if not isinstance(value, dict):
        return False

    if int(
        value.get(
            "configuration_count",
            -1,
        )
    ) != EXPECTED_TUNING_CONFIGURATION_COUNT:
        return False

    observed_counts = value.get(
        "configuration_counts"
    )

    if not isinstance(
        observed_counts,
        dict,
    ):
        return False

    normalized_counts = {
        normalize_model_id(key):
            int(count)
        for key, count
        in observed_counts.items()
    }

    if normalized_counts != (
        EXPECTED_TUNING_CONFIGURATION_COUNTS
    ):
        return False

    phase_name = str(
        value.get(
            "phase",
            "",
        )
    ).lower()

    return (
        "3d" in phase_name
        and "tuning registry" in phase_name
    )


def discover_phase3d_summary():
    candidates = []

    for path in Path("outputs").rglob(
        "*.json"
    ):
        if OUTPUT_DIR in path.parents:
            continue

        if path.stat().st_size > (
            20 * 1024 * 1024
        ):
            continue

        try:
            value = load_json(path)

        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            OSError,
        ):
            continue

        if candidate_summary_matches(value):
            candidates.append(
                (
                    path,
                    value,
                )
            )

    if not candidates:
        raise FileNotFoundError(
            "Could not locate the frozen Phase 3D "
            "43-configuration tuning-registry summary."
        )

    candidates.sort(
        key=lambda item: (
            "phase3d" not in str(
                item[0]
            ).lower(),
            len(str(item[0])),
            str(item[0]),
        )
    )

    selected_path, selected_value = (
        candidates[0]
    )

    return {
        "path":
            selected_path,

        "value":
            selected_value,

        "candidate_paths": [
            str(path)
            for path, _
            in candidates
        ],
    }


def normalize_registry_rows(
    rows,
):
    if len(rows) != (
        EXPECTED_TUNING_CONFIGURATION_COUNT
    ):
        return None

    model_key = first_present_key(
        rows[0],
        (
            "model_id",
            "model",
            "model_name",
            "model_family",
            "family_id",
            "family",
            "baseline_id",
            "architecture_id",
        ),
    )

    config_key = first_present_key(
        rows[0],
        (
            "configuration_id",
            "config_id",
            "configuration_name",
            "candidate_id",
            "registry_id",
            "id",
        ),
    )

    if (
        model_key is None
        or config_key is None
    ):
        return None

    normalized = []

    for row in rows:
        model_id = normalize_model_id(
            row[model_key]
        )

        configuration_id = str(
            row[config_key]
        ).strip()

        if model_id not in (
            TUNED_MODEL_IDS
        ):
            return None

        if not configuration_id:
            return None

        normalized.append(
            {
                "model_id":
                    model_id,

                "configuration_id":
                    configuration_id,

                "source_configuration":
                    dict(row),
            }
        )

    counts = Counter(
        row["model_id"]
        for row in normalized
    )

    if dict(counts) != (
        EXPECTED_TUNING_CONFIGURATION_COUNTS
    ):
        return None

    configuration_keys = [
        (
            row["model_id"],
            row["configuration_id"],
        )
        for row in normalized
    ]

    if len(set(configuration_keys)) != len(
        configuration_keys
    ):
        raise AssertionError(
            "Phase 3D model/configuration pairs "
            "are not unique."
        )

    normalized.sort(
        key=lambda row: (
            row["model_id"],
            row["configuration_id"],
        )
    )

    return normalized


def read_csv_rows(path):
    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            return list(
                csv.DictReader(handle)
            )

    except (
        UnicodeDecodeError,
        csv.Error,
        OSError,
    ):
        return []


def discover_phase3d_registry(
    phase3d_summary,
):
    summary_value = phase3d_summary[
        "value"
    ]

    for key in (
        "configurations",
        "configuration_registry",
        "tuning_registry",
    ):
        possible_rows = summary_value.get(
            key
        )

        if isinstance(
            possible_rows,
            list,
        ):
            normalized = (
                normalize_registry_rows(
                    possible_rows
                )
            )

            if normalized is not None:
                return {
                    "path":
                        phase3d_summary[
                            "path"
                        ],

                    "rows":
                        normalized,

                    "discovery_mode":
                        (
                            "embedded_in_phase3d_summary"
                        ),
                }

    search_directories = [
        phase3d_summary[
            "path"
        ].parent,
        Path("outputs"),
    ]

    # Phase 3D registries may be stored as JSON rather
    # than as a 43-row CSV. Search JSON artifacts first.
    json_candidates = []
    json_seen = set()

    for directory in search_directories:
        for path in directory.rglob("*.json"):
            resolved = path.resolve()

            if resolved in json_seen:
                continue

            json_seen.add(resolved)

            if OUTPUT_DIR in path.parents:
                continue

            if path.stat().st_size > (
                100 * 1024 * 1024
            ):
                continue

            try:
                value = load_json(path)

            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                OSError,
            ):
                continue

            possible_registries = []

            if isinstance(value, list):
                possible_registries.append(
                    value
                )

            if isinstance(value, dict):
                for key in (
                    "configurations",
                    "configuration_registry",
                    "tuning_registry",
                    "candidates",
                ):
                    payload = value.get(key)

                    if isinstance(payload, list):
                        possible_registries.append(
                            payload
                        )

                    elif isinstance(payload, dict):
                        combined = []

                        for (
                            model_hint,
                            model_rows,
                        ) in payload.items():
                            normalized_hint = (
                                normalize_model_id(
                                    model_hint
                                )
                            )

                            if (
                                normalized_hint
                                not in TUNED_MODEL_IDS
                            ):
                                continue

                            if not isinstance(
                                model_rows,
                                list,
                            ):
                                continue

                            for model_row in model_rows:
                                if not isinstance(
                                    model_row,
                                    dict,
                                ):
                                    continue

                                copied = dict(
                                    model_row
                                )

                                copied.setdefault(
                                    "model_id",
                                    normalized_hint,
                                )

                                combined.append(
                                    copied
                                )

                        if combined:
                            possible_registries.append(
                                combined
                            )

            for possible_rows in possible_registries:
                if not isinstance(
                    possible_rows,
                    list,
                ):
                    continue

                canonical_rows = []

                for possible_row in possible_rows:
                    if not isinstance(
                        possible_row,
                        dict,
                    ):
                        canonical_rows = []
                        break

                    copied = dict(
                        possible_row
                    )

                    configuration_key = (
                        first_present_key(
                            copied,
                            (
                                "configuration_id",
                                "config_id",
                                "configuration_name",
                                "candidate_id",
                                "registry_id",
                                "id",
                            ),
                        )
                    )

                    model_key = (
                        first_present_key(
                            copied,
                            (
                                "model_id",
                                "model",
                                "model_name",
                                "model_family",
                                "family_id",
                                "family",
                                "baseline_id",
                                "architecture_id",
                            ),
                        )
                    )

                    if configuration_key is None:
                        canonical_rows = []
                        break

                    configuration_id = str(
                        copied[
                            configuration_key
                        ]
                    ).strip()

                    if not configuration_id:
                        canonical_rows = []
                        break

                    if model_key is None:
                        upper_id = (
                            configuration_id.upper()
                        )

                        inferred_model = None

                        for candidate_model in (
                            "OCM",
                            "B1",
                            "B2",
                            "B3",
                            "B4",
                            "B5",
                        ):
                            if (
                                upper_id
                                == candidate_model
                                or upper_id.startswith(
                                    candidate_model + "_"
                                )
                                or upper_id.startswith(
                                    candidate_model + "-"
                                )
                            ):
                                inferred_model = (
                                    candidate_model
                                )
                                break

                        if inferred_model is None:
                            canonical_rows = []
                            break

                        copied[
                            "model_id"
                        ] = inferred_model

                    copied[
                        "configuration_id"
                    ] = configuration_id

                    canonical_rows.append(
                        copied
                    )

                if not canonical_rows:
                    continue

                normalized = (
                    normalize_registry_rows(
                        canonical_rows
                    )
                )

                if normalized is None:
                    continue

                signature = tuple(
                    (
                        row["model_id"],
                        row["configuration_id"],
                        json.dumps(
                            row[
                                "source_configuration"
                            ],
                            sort_keys=True,
                        ),
                    )
                    for row in normalized
                )

                json_candidates.append(
                    {
                        "path":
                            path,

                        "rows":
                            normalized,

                        "signature":
                            signature,
                    }
                )

    if json_candidates:
        # Prefer a registry located beside the selected
        # Phase 3D summary and whose name explicitly says registry.
        summary_parent = (
            phase3d_summary[
                "path"
            ].parent.resolve()
        )

        json_candidates.sort(
            key=lambda item: (
                item[
                    "path"
                ].parent.resolve()
                != summary_parent,

                "registry" not in item[
                    "path"
                ].name.lower(),

                "phase3d" not in str(
                    item["path"]
                ).lower(),

                str(item["path"]),
            )
        )

        selected = json_candidates[0]

        return {
            "path":
                selected["path"],

            "rows":
                selected["rows"],

            "discovery_mode":
                "json_registry",

            "equivalent_paths": [
                str(item["path"])
                for item
                in json_candidates
                if item["signature"]
                == selected["signature"]
            ],
        }

    candidates = []
    already_seen = set()

    for directory in search_directories:
        for path in directory.rglob(
            "*.csv"
        ):
            resolved = path.resolve()

            if resolved in already_seen:
                continue

            already_seen.add(resolved)

            if OUTPUT_DIR in path.parents:
                continue

            if path.stat().st_size > (
                100 * 1024 * 1024
            ):
                continue

            rows = read_csv_rows(path)

            normalized = (
                normalize_registry_rows(
                    rows
                )
            )

            if normalized is None:
                continue

            signature = tuple(
                (
                    row["model_id"],
                    row["configuration_id"],
                    json.dumps(
                        row[
                            "source_configuration"
                        ],
                        sort_keys=True,
                    ),
                )
                for row in normalized
            )

            candidates.append(
                {
                    "path":
                        path,

                    "rows":
                        normalized,

                    "signature":
                        signature,
                }
            )

    if not candidates:
        raise FileNotFoundError(
            "The Phase 3D summary was found, but "
            "the associated 43-row configuration "
            "registry could not be located."
        )

    signatures = {}

    for candidate in candidates:
        signatures.setdefault(
            candidate["signature"],
            [],
        ).append(candidate)

    if len(signatures) != 1:
        raise RuntimeError(
            "Conflicting 43-configuration Phase 3D "
            "registries were found: "
            + json.dumps(
                [
                    [
                        str(item["path"])
                        for item in group
                    ]
                    for group
                    in signatures.values()
                ],
                indent=2,
            )
        )

    equivalent = next(
        iter(
            signatures.values()
        )
    )

    summary_parent = (
        phase3d_summary[
            "path"
        ].parent.resolve()
    )

    equivalent.sort(
        key=lambda item: (
            item["path"].parent.resolve()
            != summary_parent,
            "registry" not in item[
                "path"
            ].name.lower(),
            str(item["path"]),
        )
    )

    selected = equivalent[0]

    return {
        "path":
            selected["path"],

        "rows":
            selected["rows"],

        "discovery_mode":
            "csv_registry",

        "equivalent_paths": [
            str(item["path"])
            for item in equivalent
        ],
    }


def create_model_registry():
    rows = [
        {
            "model_id":
                "B0",

            "role":
                "persistence baseline",

            "model_class":
                "nontrainable",

            "transition_semantics":
                (
                    "Predict the current field at every "
                    "future rollout step."
                ),

            "tuned":
                False,

            "final_seed_policy":
                "deterministic",

            "primary_model":
                False,
        },
        {
            "model_id":
                "B1",

            "role":
                "affine baseline",

            "model_class":
                "deterministic fitted baseline",

            "transition_semantics":
                (
                    "Preserve the frozen Tier B B1 "
                    "transition semantics after replacing "
                    "the observation interface with the "
                    "Tier C field codec."
                ),

            "tuned":
                True,

            "final_seed_policy":
                "one deterministic fit per noise level",

            "primary_model":
                False,
        },
        {
            "model_id":
                "B2",

            "role":
                "neural baseline 2",

            "model_class":
                "neural",

            "transition_semantics":
                "Inherited from the frozen Tier B registry.",

            "tuned":
                True,

            "final_seed_policy":
                "five frozen seeds per noise level",

            "primary_model":
                False,
        },
        {
            "model_id":
                "B3",

            "role":
                "neural baseline 3",

            "model_class":
                "neural",

            "transition_semantics":
                "Inherited from the frozen Tier B registry.",

            "tuned":
                True,

            "final_seed_policy":
                "five frozen seeds per noise level",

            "primary_model":
                False,
        },
        {
            "model_id":
                "B4",

            "role":
                "neural baseline 4",

            "model_class":
                "neural",

            "transition_semantics":
                "Inherited from the frozen Tier B registry.",

            "tuned":
                True,

            "final_seed_policy":
                "five frozen seeds per noise level",

            "primary_model":
                False,
        },
        {
            "model_id":
                "B5",

            "role":
                "neural baseline 5",

            "model_class":
                "neural",

            "transition_semantics":
                "Inherited from the frozen Tier B registry.",

            "tuned":
                True,

            "final_seed_policy":
                "five frozen seeds per noise level",

            "primary_model":
                False,
        },
        {
            "model_id":
                "OCM",

            "role":
                "primary operation-channel model",

            "model_class":
                "neural",

            "transition_semantics":
                (
                    "Preserve the frozen Tier B OCM "
                    "operation-channel semantics after "
                    "replacing the observation interface "
                    "with the Tier C field codec."
                ),

            "tuned":
                True,

            "final_seed_policy":
                "five frozen seeds per noise level",

            "primary_model":
                True,
        },
    ]

    require(
        {
            row["model_id"]
            for row in rows
        } == set(MODEL_IDS),
        "Model registry is incomplete.",
    )

    return rows


def create_field_codec_registry():
    return {
        "adapter_id":
            "tier_c_v4_shared_convolutional_codec",

        "purpose":
            (
                "Provide the same trainable spatial "
                "observation interface to B1-B5 and OCM "
                "without reusing diagnostic weights."
            ),

        "input_shape":
            [
                4,
                32,
                32,
            ],

        "input_dtype_on_disk":
            "float16",

        "training_compute_dtype":
            "float32",

        "latent_dimension":
            128,

        "encoder": [
            {
                "layer":
                    "Conv2d",

                "input_channels":
                    4,

                "output_channels":
                    32,

                "kernel_size":
                    3,

                "stride":
                    1,

                "padding":
                    1,
            },
            {
                "layer":
                    "GroupNorm",

                "groups":
                    8,
            },
            {
                "layer":
                    "GELU",
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    32,
            },
            {
                "layer":
                    "ResidualDownsampleBlock",

                "input_channels":
                    32,

                "output_channels":
                    64,

                "stride":
                    2,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    64,
            },
            {
                "layer":
                    "ResidualDownsampleBlock",

                "input_channels":
                    64,

                "output_channels":
                    128,

                "stride":
                    2,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    128,
            },
            {
                "layer":
                    "ResidualDownsampleBlock",

                "input_channels":
                    128,

                "output_channels":
                    128,

                "stride":
                    2,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    128,
            },
            {
                "layer":
                    "Flatten",
            },
            {
                "layer":
                    "Linear",

                "input_features":
                    2048,

                "output_features":
                    128,
            },
        ],

        "decoder": [
            {
                "layer":
                    "Linear",

                "input_features":
                    128,

                "output_features":
                    2048,
            },
            {
                "layer":
                    "Reshape",

                "shape":
                    [
                        128,
                        4,
                        4,
                    ],
            },
            {
                "layer":
                    "ConvTranspose2d",

                "input_channels":
                    128,

                "output_channels":
                    128,

                "kernel_size":
                    4,

                "stride":
                    2,

                "padding":
                    1,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    128,
            },
            {
                "layer":
                    "ConvTranspose2d",

                "input_channels":
                    128,

                "output_channels":
                    64,

                "kernel_size":
                    4,

                "stride":
                    2,

                "padding":
                    1,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    64,
            },
            {
                "layer":
                    "ConvTranspose2d",

                "input_channels":
                    64,

                "output_channels":
                    32,

                "kernel_size":
                    4,

                "stride":
                    2,

                "padding":
                    1,
            },
            {
                "layer":
                    "ResidualBlock",

                "channels":
                    32,
            },
            {
                "layer":
                    "Conv2d",

                "input_channels":
                    32,

                "output_channels":
                    4,

                "kernel_size":
                    3,

                "stride":
                    1,

                "padding":
                    1,

                "activation":
                    "linear",
            },
        ],

        "normalization":
            (
                "Use the already-frozen normalization "
                "computed from clean training-carrier "
                "fields only."
            ),

        "data_augmentation":
            False,

        "pretraining":
            False,

        "shared_weights_across_models":
            False,

        "architecture_shared_across_models":
            True,

        "diagnostic_state_decoder_weights_reused":
            False,

        "affine_diagnostic_coefficients_reused":
            False,

        "global_statistics_classifier_reused":
            False,

        "test_data_used_to_define_adapter":
            False,
    }


def create_tuning_registry(
    source_registry,
):
    rows = []

    for source_row in source_registry[
        "rows"
    ]:
        rows.append(
            {
                "model_id":
                    source_row[
                        "model_id"
                    ],

                "configuration_id":
                    source_row[
                        "configuration_id"
                    ],

                "source_registry_path":
                    str(
                        source_registry[
                            "path"
                        ]
                    ),

                "source_configuration_json":
                    json.dumps(
                        source_row[
                            "source_configuration"
                        ],
                        sort_keys=True,
                        separators=(
                            ",",
                            ":",
                        ),
                    ),

                "observation_adapter_id":
                    (
                        "tier_c_v4_shared_"
                        "convolutional_codec"
                    ),

                "tuning_seed":
                    DEVELOPMENT_SEED,

                "tuning_noise_fraction":
                    DEVELOPMENT_NOISE_FRACTION,

                "training_cell":
                    TRAINING_CELL,

                "checkpoint_selection_cell":
                    CHECKPOINT_SELECTION_CELL,

                "checkpoint_selection_metric":
                    CHECKPOINT_SELECTION_METRIC,

                "checkpoint_selection_direction":
                    "lower_is_better",

                "test_access_allowed":
                    False,

                "status":
                    "frozen_not_executed",
            }
        )

    require(
        len(rows)
        == EXPECTED_TUNING_CONFIGURATION_COUNT,
        "Frozen tuning registry must contain 43 rows.",
    )

    counts = Counter(
        row["model_id"]
        for row in rows
    )

    require(
        dict(counts)
        == EXPECTED_TUNING_CONFIGURATION_COUNTS,
        "Frozen tuning-registry model counts changed.",
    )

    return rows


def create_final_run_registry():
    rows = []

    for noise_fraction in (
        NOISE_FRACTIONS
    ):
        rows.append(
            {
                "model_id":
                    "B0",

                "noise_fraction":
                    noise_fraction,

                "seed":
                    "deterministic",

                "run_type":
                    "evaluation_only",

                "configuration_id":
                    "persistence",

                "training_allowed":
                    False,

                "status":
                    "planned",
            }
        )

        rows.append(
            {
                "model_id":
                    "B1",

                "noise_fraction":
                    noise_fraction,

                "seed":
                    "deterministic",

                "run_type":
                    "deterministic_final_fit",

                "configuration_id":
                    "selected_after_tuning",

                "training_allowed":
                    False,

                "status":
                    "planned",
            }
        )

        for model_id in (
            NEURAL_MODEL_IDS
        ):
            for seed in FINAL_SEEDS:
                rows.append(
                    {
                        "model_id":
                            model_id,

                        "noise_fraction":
                            noise_fraction,

                        "seed":
                            seed,

                        "run_type":
                            "neural_final_fit",

                        "configuration_id":
                            "selected_after_tuning",

                        "training_allowed":
                            False,

                        "status":
                            "planned",
                    }
                )

    expected_row_count = (
        len(NOISE_FRACTIONS)
        * (
            1
            + 1
            + len(NEURAL_MODEL_IDS)
            * len(FINAL_SEEDS)
        )
    )

    require(
        len(rows) == expected_row_count,
        (
            "Final run registry should contain "
            f"{expected_row_count} rows."
        ),
    )

    return rows


def create_metric_registry():
    return [
        {
            "metric_id":
                "noisy_target_rollout_mse",

            "role":
                "primary predictive metric",

            "target":
                "stored noisy observation",

            "aggregation":
                (
                    "Mean squared error over forecast "
                    "points, channels, and pixels; then "
                    "mean within sequence."
                ),

            "checkpoint_selection_allowed":
                True,
        },
        {
            "metric_id":
                "clean_target_rollout_mse",

            "role":
                "secondary denoising metric",

            "target":
                "clean latent-state field",

            "aggregation":
                (
                    "Mean squared error over forecast "
                    "points, channels, and pixels; then "
                    "mean within sequence."
                ),

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "noisy_target_one_step_mse",

            "role":
                "secondary one-step metric",

            "target":
                "stored noisy next observation",

            "aggregation":
                "Mean within sequence.",

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "clean_target_one_step_mse",

            "role":
                "secondary one-step metric",

            "target":
                "clean next-state field",

            "aggregation":
                "Mean within sequence.",

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "state_path_accuracy",

            "role":
                "secondary structural diagnostic",

            "target":
                (
                    "Frozen symbolic state path, decoded "
                    "only after predictive checkpoints "
                    "are frozen."
                ),

            "aggregation":
                "Mean within sequence.",

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "exact_transformation_rate",

            "role":
                "secondary structural diagnostic",

            "target":
                "Frozen 104-element transformation registry.",

            "aggregation":
                "Exact transformation recovery rate.",

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "information_class_accuracy",

            "role":
                "secondary Process-Memory diagnostic",

            "target":
                "Frozen nine information classes.",

            "aggregation":
                "Accuracy over transformations.",

            "checkpoint_selection_allowed":
                False,
        },
        {
            "metric_id":
                "blackwell_relation_balanced_accuracy",

            "role":
                "secondary Process-Memory diagnostic",

            "target":
                "Frozen Blackwell relation registry.",

            "aggregation":
                "Balanced accuracy over ordered pairs.",

            "checkpoint_selection_allowed":
                False,
        },
    ]


def create_test_opening_policy():
    return {
        "sealed_test_cells":
            list(TEST_CELLS),

        "test_carrier_count":
            32,

        "test_seed_source":
            (
                "Use only the unopened test seeds and "
                "carrier IDs already frozen in the "
                "Phase 4A-R3 protocol."
            ),

        "test_generation_before_tuning":
            False,

        "test_generation_before_final_fits":
            False,

        "test_generation_authorized_now":
            False,

        "test_evaluation_authorized_now":
            False,

        "requirements_before_test_generation": [
            (
                "All 43 tuning runs are completed or "
                "formally marked invalid."
            ),
            (
                "One selected configuration per tuned "
                "model is frozen using val_joint noisy-"
                "target rollout MSE only."
            ),
            (
                "All 130 trainable final fits are "
                "completed and checksummed."
            ),
            (
                "All five B0 persistence evaluations "
                "are completed."
            ),
            (
                "No final checkpoint is selected using "
                "a test artifact."
            ),
            (
                "The validation-selected best baseline "
                "for the primary comparison is frozen."
            ),
            (
                "A one-time test-opening authorization "
                "artifact is created."
            ),
        ],

        "primary_confirmatory_condition": {
            "cell":
                PRIMARY_TEST_CELL,

            "noise_fraction":
                PRIMARY_NOISE_FRACTION,

            "metric":
                PRIMARY_METRIC,

            "primary_model":
                "OCM",

            "comparator":
                (
                    "Best B1-B5 baseline selected using "
                    "mean val_joint noisy-target rollout "
                    "MSE at noise fraction 0.25 before "
                    "test generation."
                ),

            "statistical_unit":
                "sequence",

            "neural_seed_aggregation":
                (
                    "Average each sequence's metric "
                    "across the five frozen seeds before "
                    "the paired comparison."
                ),

            "bootstrap_replicates":
                BOOTSTRAP_REPLICATES,

            "bootstrap_seed":
                BOOTSTRAP_SEED,

            "confidence_interval":
                (
                    "Two-sided 95% percentile paired "
                    "sequence bootstrap interval for "
                    "OCM minus best-baseline MSE."
                ),

            "predictive_superiority_supported_if":
                (
                    "The mean OCM-minus-baseline "
                    "difference is below zero and the "
                    "upper endpoint of the 95% interval "
                    "is also below zero."
                ),
        },

        "secondary_conditions": {
            "cells":
                list(TEST_CELLS),

            "noise_fractions":
                list(NOISE_FRACTIONS),

            "metrics": [
                "noisy_target_rollout_mse",
                "clean_target_rollout_mse",
                "noisy_target_one_step_mse",
                "clean_target_one_step_mse",
                "state_path_accuracy",
                "exact_transformation_rate",
                "information_class_accuracy",
                "blackwell_relation_balanced_accuracy",
            ],

            "confirmatory_status":
                False,

            "multiple_comparison_policy":
                (
                    "Report all conditions with Holm-"
                    "adjusted p-values where formal "
                    "pairwise tests are used."
                ),
        },

        "test_open_count":
            0,

        "maximum_test_open_count":
            1,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/7] Validating Tier C v4 "
        "development acceptance"
    )

    sources = (
        validate_acceptance_sources()
    )

    print(
        "[2/7] Locating the frozen "
        "Phase 3D tuning registry"
    )

    phase3d_summary = (
        discover_phase3d_summary()
    )

    source_registry = (
        discover_phase3d_registry(
            phase3d_summary
        )
    )

    print(
        "Phase 3D summary: "
        f"{phase3d_summary['path']}"
    )

    print(
        "Phase 3D registry: "
        f"{source_registry['path']}"
    )

    print(
        "[3/7] Freezing model and "
        "observation-adapter registries"
    )

    model_registry = (
        create_model_registry()
    )

    field_codec = (
        create_field_codec_registry()
    )

    write_csv(
        OUTPUT_DIR
        / "tier_c_v4_model_registry.csv",
        model_registry,
    )

    write_json(
        OUTPUT_DIR
        / "tier_c_v4_field_codec_registry.json",
        field_codec,
    )

    print(
        "[4/7] Freezing the 43-configuration "
        "development tuning registry"
    )

    tuning_registry = (
        create_tuning_registry(
            source_registry
        )
    )

    write_csv(
        OUTPUT_DIR
        / "tier_c_v4_tuning_registry.csv",
        tuning_registry,
    )

    print(
        "[5/7] Freezing final seeds, "
        "noise levels, and planned runs"
    )

    final_run_registry = (
        create_final_run_registry()
    )

    write_csv(
        OUTPUT_DIR
        / "tier_c_v4_final_run_registry.csv",
        final_run_registry,
    )

    print(
        "[6/7] Freezing metrics and "
        "one-time test-opening policy"
    )

    metric_registry = (
        create_metric_registry()
    )

    test_policy = (
        create_test_opening_policy()
    )

    write_csv(
        OUTPUT_DIR
        / "tier_c_v4_metric_registry.csv",
        metric_registry,
    )

    write_json(
        OUTPUT_DIR
        / "tier_c_v4_test_opening_policy.json",
        test_policy,
    )

    source_paths = {
        "phase4ar3_protocol":
            sources[
                "protocol_path"
            ],

        "phase4cr3d_acceptance":
            sources[
                "acceptance_path"
            ],

        "phase4cr3d_gates":
            sources[
                "gate_path"
            ],

        "phase4br3_dataset":
            sources[
                "dataset_path"
            ],

        "phase3d_tuning_summary":
            phase3d_summary[
                "path"
            ],

        "phase3d_tuning_registry":
            source_registry[
                "path"
            ],
    }

    input_hashes = {
        name: {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }
        for name, path
        in source_paths.items()
    }

    write_json(
        OUTPUT_DIR
        / "phase4dr3_input_hashes.json",
        input_hashes,
    )

    output_paths = {
        "model_registry":
            OUTPUT_DIR
            / "tier_c_v4_model_registry.csv",

        "field_codec_registry":
            OUTPUT_DIR
            / "tier_c_v4_field_codec_registry.json",

        "tuning_registry":
            OUTPUT_DIR
            / "tier_c_v4_tuning_registry.csv",

        "final_run_registry":
            OUTPUT_DIR
            / "tier_c_v4_final_run_registry.csv",

        "metric_registry":
            OUTPUT_DIR
            / "tier_c_v4_metric_registry.csv",

        "test_opening_policy":
            OUTPUT_DIR
            / "tier_c_v4_test_opening_policy.json",
    }

    output_hashes = {
        name: {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }
        for name, path
        in output_paths.items()
    }

    write_json(
        OUTPUT_DIR
        / "phase4dr3_output_hashes.json",
        output_hashes,
    )

    trainable_final_run_count = sum(
        row["run_type"] != "evaluation_only"
        for row in final_run_registry
    )

    persistence_evaluation_count = sum(
        row["model_id"] == "B0"
        for row in final_run_registry
    )

    print(
        "[7/7] Freezing predictive protocol decision"
    )

    summary = {
        "phase":
            (
                "4D-R3 Tier C v4 predictive-learning "
                "protocol freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_development_acceptance_passed":
            True,

        "source_phase3d_registry_discovery_mode":
            source_registry[
                "discovery_mode"
            ],

        "source_phase3d_summary_path":
            str(
                phase3d_summary[
                    "path"
                ]
            ),

        "source_phase3d_registry_path":
            str(
                source_registry[
                    "path"
                ]
            ),

        "model_ids":
            list(MODEL_IDS),

        "primary_model":
            "OCM",

        "baseline_model_ids":
            [
                "B0",
                "B1",
                "B2",
                "B3",
                "B4",
                "B5",
            ],

        "tuned_model_ids":
            list(TUNED_MODEL_IDS),

        "tuning_configuration_count":
            len(tuning_registry),

        "tuning_configuration_counts":
            dict(
                Counter(
                    row["model_id"]
                    for row
                    in tuning_registry
                )
            ),

        "development_seed":
            DEVELOPMENT_SEED,

        "development_noise_fraction":
            DEVELOPMENT_NOISE_FRACTION,

        "training_cell":
            TRAINING_CELL,

        "checkpoint_selection_cell":
            CHECKPOINT_SELECTION_CELL,

        "checkpoint_selection_metric":
            CHECKPOINT_SELECTION_METRIC,

        "checkpoint_selection_direction":
            "lower_is_better",

        "checkpoint_tie_breaking": [
            (
                "Lower val_joint noisy-target "
                "rollout MSE."
            ),
            (
                "If exactly tied, lower val_joint "
                "noisy-target one-step MSE."
            ),
            (
                "If still tied, earlier epoch."
            ),
            (
                "If still tied, lexicographically "
                "smaller checkpoint path."
            ),
        ],

        "validation_cells_for_reporting":
            list(VALIDATION_CELLS),

        "final_seeds":
            list(FINAL_SEEDS),

        "noise_fractions":
            list(NOISE_FRACTIONS),

        "planned_final_run_count":
            len(final_run_registry),

        "planned_trainable_final_fit_count":
            trainable_final_run_count,

        "planned_persistence_evaluation_count":
            persistence_evaluation_count,

        "field_codec_adapter_id":
            field_codec[
                "adapter_id"
            ],

        "field_codec_latent_dimension":
            field_codec[
                "latent_dimension"
            ],

        "field_codec_pretraining":
            False,

        "diagnostic_checkpoint_reuse":
            False,

        "primary_test_cell":
            PRIMARY_TEST_CELL,

        "primary_noise_fraction":
            PRIMARY_NOISE_FRACTION,

        "primary_metric":
            PRIMARY_METRIC,

        "primary_statistical_test":
            (
                "Paired sequence bootstrap of OCM "
                "minus validation-selected best baseline."
            ),

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "sealed_test_cell_count":
            len(TEST_CELLS),

        "sealed_test_cells":
            list(TEST_CELLS),

        "test_open_count":
            0,

        "maximum_test_open_count":
            1,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "predictive_training_performed":
            False,

        "tuning_execution_authorized":
            False,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "implementation_smoke_test_authorized":
            True,

        "tier_c_v5_authorized":
            False,

        "prior_outputs_modified":
            False,

        "phase4dr3_status":
            "protocol_frozen",

        "next_phase":
            (
                "4E-R3 Tier C v4 predictive "
                "implementation smoke test"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4dr3_predictive_protocol_summary.json",
        summary,
    )

    print(
        "Phase 4D-R3 Tier C v4 predictive "
        "protocol frozen."
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
