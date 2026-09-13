from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PHASE4D_DIR = Path(
    "outputs/phase4dr3_tier_c_v4_predictive_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4dr31_tier_c_v4_execution_contract_audit"
)

PHASE3_REGISTRY_PATH = Path(
    "outputs/phase3d_tier_b_tuning/tuning_registry.json"
)

SOURCE_PATHS = (
    Path("phase3d_freeze_tuning_registry.py"),
    Path("phase3c_run_tier_b_smoke_tests.py"),
    Path("phase3d_tune_tier_b_models.py"),
    Path("phase3e_freeze_final_fit_registry.py"),
    Path("phase3e_run_final_fits.py"),
)

PROTOCOL_SUMMARY_PATH = (
    PHASE4D_DIR
    / "phase4dr3_predictive_protocol_summary.json"
)

FIELD_CODEC_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_field_codec_registry.json"
)

TUNING_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_tuning_registry.csv"
)

MODEL_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_model_registry.csv"
)

FINAL_RUN_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_final_run_registry.csv"
)

METRIC_REGISTRY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_metric_registry.csv"
)

TEST_POLICY_PATH = (
    PHASE4D_DIR
    / "tier_c_v4_test_opening_policy.json"
)


RELEVANT_CONSTANT_PATTERNS = (
    "BATCH",
    "EPOCH",
    "PATIENCE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "TEMPERATURE",
    "LAMBDA",
    "GRADIENT",
    "CLIP",
    "SEED",
    "WORKER",
    "PIN_MEMORY",
)

SOURCE_EVIDENCE_PATTERNS = {
    "optimizer": (
        r"torch\.optim",
        r"\bAdamW?\b",
        r"\bSGD\b",
        r"\bRMSprop\b",
    ),
    "scheduler": (
        r"lr_scheduler",
        r"ReduceLROnPlateau",
        r"CosineAnnealing",
        r"OneCycleLR",
    ),
    "gradient_clipping": (
        r"clip_grad",
        r"gradient_clip",
        r"max_grad_norm",
    ),
    "automatic_mixed_precision": (
        r"autocast",
        r"GradScaler",
        r"amp\.",
    ),
    "early_stopping": (
        r"early_stopping",
        r"patience",
        r"best_epoch",
    ),
    "checkpoint_selection": (
        r"checkpoint",
        r"best_validation",
        r"best_val",
    ),
    "batching": (
        r"DataLoader",
        r"batch_size",
        r"collate_fn",
    ),
    "rollout": (
        r"rollout",
        r"teacher_forcing",
        r"autoregressive",
    ),
    "loss_weights": (
        r"lambda_transition",
        r"lambda_rollout",
        r"lambda_determinism",
    ),
    "temperature": (
        r"temperature",
    ),
    "determinism": (
        r"use_deterministic_algorithms",
        r"cudnn\.deterministic",
        r"manual_seed",
    ),
}


TIER_C_EXECUTION_FIELDS = (
    {
        "field":
            "field_codec_training_scope",
        "description":
            (
                "Whether encoder, transition model, and decoder "
                "are optimized jointly from the first epoch."
            ),
    },
    {
        "field":
            "field_codec_reconstruction_objective",
        "description":
            (
                "Exact clean/noisy target, loss function, and "
                "aggregation used to train the codec."
            ),
    },
    {
        "field":
            "field_codec_reconstruction_loss_weight",
        "description":
            "Weight assigned to codec reconstruction loss.",
    },
    {
        "field":
            "predictive_field_loss_weight",
        "description":
            (
                "Weight assigned to decoded field-space "
                "prediction loss."
            ),
    },
    {
        "field":
            "latent_transition_loss_weight",
        "description":
            (
                "Whether a direct latent transition loss is used "
                "and its weight."
            ),
    },
    {
        "field":
            "rollout_loss_application",
        "description":
            (
                "Whether inherited rollout weights apply in "
                "latent space, decoded field space, or both."
            ),
    },
    {
        "field":
            "teacher_forcing_policy",
        "description":
            (
                "Exact teacher-forcing or autoregressive rollout "
                "policy during training."
            ),
    },
    {
        "field":
            "variable_length_batching_policy",
        "description":
            (
                "Padding, masks, and per-sequence aggregation for "
                "variable-length trajectories."
            ),
    },
    {
        "field":
            "optimizer_scope",
        "description":
            (
                "Whether one optimizer jointly updates codec and "
                "transition parameters."
            ),
    },
    {
        "field":
            "learning_rate_scope",
        "description":
            (
                "Whether inherited model learning rates apply to "
                "all parameters or only transition parameters."
            ),
    },
    {
        "field":
            "weight_decay_scope",
        "description":
            (
                "Exact weight-decay application to convolutional, "
                "normalization, bias, and transition parameters."
            ),
    },
    {
        "field":
            "gradient_clipping_policy",
        "description":
            (
                "Explicit clipping value or an explicit statement "
                "that clipping is disabled."
            ),
    },
    {
        "field":
            "mixed_precision_policy",
        "description":
            (
                "Whether CUDA automatic mixed precision and loss "
                "scaling are used."
            ),
    },
    {
        "field":
            "codec_initialization_policy",
        "description":
            (
                "Initialization method and random-seed derivation "
                "for encoder and decoder."
            ),
    },
    {
        "field":
            "checkpoint_contents",
        "description":
            (
                "Exact parameters and optimizer state retained in "
                "a selected checkpoint."
            ),
    },
    {
        "field":
            "b1_reporting_name",
        "description":
            (
                "B1 must be reported as an affine latent-transition "
                "baseline, not a globally affine field model."
            ),
    },
)


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


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def serialize_value(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def extract_literal_constants(
    path: Path,
):
    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    rows = []

    for node in tree.body:
        name = None
        value_node = None

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue

            target = node.targets[0]

            if isinstance(target, ast.Name):
                name = target.id
                value_node = node.value

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                name = node.target.id
                value_node = node.value

        if (
            name is None
            or value_node is None
        ):
            continue

        if not any(
            pattern in name.upper()
            for pattern in RELEVANT_CONSTANT_PATTERNS
        ):
            continue

        try:
            value = ast.literal_eval(
                value_node
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

        rows.append(
            {
                "source_path":
                    str(path),

                "line_number":
                    int(node.lineno),

                "constant_name":
                    name,

                "constant_value_json":
                    serialize_value(value),
            }
        )

    return rows


def extract_source_evidence(
    path: Path,
):
    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    rows = []
    recorded = set()

    for category, patterns in (
        SOURCE_EVIDENCE_PATTERNS.items()
    ):
        compiled = [
            re.compile(
                pattern,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        ]

        for index, line in enumerate(lines):
            if not any(
                pattern.search(line)
                for pattern in compiled
            ):
                continue

            start = max(
                0,
                index - 2,
            )

            end = min(
                len(lines),
                index + 3,
            )

            key = (
                category,
                index + 1,
            )

            if key in recorded:
                continue

            recorded.add(key)

            rows.append(
                {
                    "source_path":
                        str(path),

                    "category":
                        category,

                    "line_number":
                        index + 1,

                    "source_excerpt":
                        "\n".join(
                            f"{line_number + 1}: "
                            f"{lines[line_number]}"
                            for line_number
                            in range(start, end)
                        ),
                }
            )

    return rows


def summarize_inherited_configurations():
    rows = read_csv(
        TUNING_REGISTRY_PATH
    )

    require(
        len(rows) == 43,
        "Expected 43 frozen Tier C tuning rows.",
    )

    configurations_by_model = (
        defaultdict(list)
    )

    flattened_rows = []

    for registry_row in rows:
        model_id = registry_row[
            "model_id"
        ]

        configuration = json.loads(
            registry_row[
                "source_configuration_json"
            ]
        )

        configurations_by_model[
            model_id
        ].append(configuration)

        for key, value in configuration.items():
            flattened_rows.append(
                {
                    "model_id":
                        model_id,

                    "configuration_id":
                        registry_row[
                            "configuration_id"
                        ],

                    "configuration_key":
                        key,

                    "configuration_value_json":
                        serialize_value(value),
                }
            )

    summary_rows = []

    for model_id in sorted(
        configurations_by_model
    ):
        configurations = (
            configurations_by_model[
                model_id
            ]
        )

        all_keys = sorted(
            set().union(
                *(
                    set(configuration)
                    for configuration
                    in configurations
                )
            )
        )

        for key in all_keys:
            values = [
                configuration[key]
                for configuration
                in configurations
                if key in configuration
            ]

            unique_values = sorted(
                {
                    serialize_value(value)
                    for value in values
                }
            )

            summary_rows.append(
                {
                    "model_id":
                        model_id,

                    "configuration_count":
                        len(configurations),

                    "configuration_key":
                        key,

                    "present_count":
                        len(values),

                    "present_in_every_configuration":
                        (
                            len(values)
                            == len(configurations)
                        ),

                    "unique_value_count":
                        len(unique_values),

                    "unique_values_json":
                        json.dumps(
                            unique_values
                        ),
                }
            )

    model_counts = Counter(
        row["model_id"]
        for row in rows
    )

    return {
        "registry_rows":
            rows,

        "model_counts":
            dict(model_counts),

        "summary_rows":
            summary_rows,

        "flattened_rows":
            flattened_rows,
    }


def flatten_json(
    value,
    prefix="root",
):
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(
                flatten_json(
                    child,
                    f"{prefix}.{key}",
                )
            )

    elif isinstance(value, list):
        if len(value) <= 50:
            rows.append(
                {
                    "json_path":
                        prefix,

                    "value_json":
                        serialize_value(value),
                }
            )

    else:
        rows.append(
            {
                "json_path":
                    prefix,

                "value_json":
                    serialize_value(value),
            }
        )

    return rows


def determine_explicitly_frozen_fields(
    protocol,
    codec,
    registry,
):
    frozen = {
        "field_codec_architecture":
            bool(
                codec.get("encoder")
                and codec.get("decoder")
            ),

        "field_codec_latent_dimension":
            (
                codec.get(
                    "latent_dimension"
                )
                == 128
            ),

        "field_codec_pretraining_disabled":
            codec.get(
                "pretraining"
            ) is False,

        "diagnostic_weights_not_reused":
            (
                codec.get(
                    "diagnostic_state_decoder_weights_reused"
                )
                is False
                and codec.get(
                    "affine_diagnostic_coefficients_reused"
                )
                is False
                and codec.get(
                    "global_statistics_classifier_reused"
                )
                is False
            ),

        "development_seed":
            protocol.get(
                "development_seed"
            )
            == 11,

        "development_noise_fraction":
            protocol.get(
                "development_noise_fraction"
            )
            == 0.25,

        "checkpoint_selection_cell":
            protocol.get(
                "checkpoint_selection_cell"
            )
            == "val_joint",

        "checkpoint_selection_metric":
            protocol.get(
                "checkpoint_selection_metric"
            )
            == "noisy_target_rollout_mse",

        "final_seeds":
            protocol.get(
                "final_seeds"
            )
            == [
                11,
                23,
                37,
                53,
                71,
            ],

        "noise_fractions":
            protocol.get(
                "noise_fractions"
            )
            == [
                0.0,
                0.1,
                0.25,
                0.5,
                1.0,
            ],

        "phase3_registry_batch_size_present":
            (
                "batch_size"
                in registry
            ),

        "phase3_registry_maximum_epochs_present":
            (
                "maximum_epochs"
                in registry
            ),

        "phase3_registry_patience_present":
            (
                "early_stopping_patience"
                in registry
            ),
    }

    return frozen


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        PROTOCOL_SUMMARY_PATH,
        FIELD_CODEC_PATH,
        TUNING_REGISTRY_PATH,
        MODEL_REGISTRY_PATH,
        FINAL_RUN_REGISTRY_PATH,
        METRIC_REGISTRY_PATH,
        TEST_POLICY_PATH,
        PHASE3_REGISTRY_PATH,
        *SOURCE_PATHS,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        "[1/6] Validating frozen Phase 4D-R3 artifacts"
    )

    protocol = load_json(
        PROTOCOL_SUMMARY_PATH
    )

    codec = load_json(
        FIELD_CODEC_PATH
    )

    phase3_registry = load_json(
        PHASE3_REGISTRY_PATH
    )

    require(
        protocol[
            "phase4dr3_status"
        ] == "protocol_frozen",
        "Phase 4D-R3 is not frozen.",
    )

    require(
        protocol[
            "implementation_smoke_test_authorized"
        ] is True,
        "Smoke-test preparation was not authorized.",
    )

    require(
        protocol[
            "tuning_execution_authorized"
        ] is False,
        "Tuning was already authorized.",
    )

    require(
        protocol[
            "test_open_count"
        ] == 0,
        "Tier C test set was opened.",
    )

    require(
        protocol[
            "test_fields_generated"
        ] is False,
        "Tier C test fields were generated.",
    )

    print(
        "[2/6] Auditing the 43 inherited configurations"
    )

    configuration_audit = (
        summarize_inherited_configurations()
    )

    write_csv(
        OUTPUT_DIR
        / "inherited_configuration_key_summary.csv",
        configuration_audit[
            "summary_rows"
        ],
    )

    write_csv(
        OUTPUT_DIR
        / "inherited_configuration_values.csv",
        configuration_audit[
            "flattened_rows"
        ],
    )

    print(
        "[3/6] Extracting Tier B registry controls"
    )

    flattened_registry = flatten_json(
        phase3_registry
    )

    write_csv(
        OUTPUT_DIR
        / "phase3_registry_flattened.csv",
        flattened_registry,
    )

    print(
        "[4/6] Extracting source constants and evidence"
    )

    constant_rows = []
    evidence_rows = []

    for source_path in SOURCE_PATHS:
        constant_rows.extend(
            extract_literal_constants(
                source_path
            )
        )

        evidence_rows.extend(
            extract_source_evidence(
                source_path
            )
        )

    write_csv(
        OUTPUT_DIR
        / "tier_b_training_constants.csv",
        constant_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "tier_b_execution_source_evidence.csv",
        evidence_rows,
    )

    print(
        "[5/6] Determining execution-contract completeness"
    )

    frozen_fields = (
        determine_explicitly_frozen_fields(
            protocol=protocol,
            codec=codec,
            registry=phase3_registry,
        )
    )

    missing_rows = []

    for item in TIER_C_EXECUTION_FIELDS:
        missing_rows.append(
            {
                "field":
                    item["field"],

                "description":
                    item["description"],

                "explicitly_frozen":
                    False,

                "required_before_smoke_execution":
                    True,
            }
        )

    write_csv(
        OUTPUT_DIR
        / "missing_tier_c_execution_fields.csv",
        missing_rows,
    )

    input_hashes = {}

    for path in required_paths:
        input_hashes[str(path)] = (
            sha256_file(path)
        )

    write_json(
        OUTPUT_DIR
        / "execution_contract_input_hashes.json",
        input_hashes,
    )

    print(
        "[6/6] Writing the audit decision"
    )

    summary = {
        "phase":
            (
                "4D-R3.1 Tier C v4 predictive "
                "execution-contract completeness audit"
            ),

        "protocol_version":
            "tier_c_v4",

        "source_phase4dr3_status":
            protocol[
                "phase4dr3_status"
            ],

        "source_tuning_configuration_count":
            len(
                configuration_audit[
                    "registry_rows"
                ]
            ),

        "source_tuning_configuration_counts":
            configuration_audit[
                "model_counts"
            ],

        "source_phase3_registry_path":
            str(
                PHASE3_REGISTRY_PATH
            ),

        "source_script_count":
            len(SOURCE_PATHS),

        "literal_training_constant_count":
            len(constant_rows),

        "source_evidence_record_count":
            len(evidence_rows),

        "explicitly_frozen_field_checks":
            frozen_fields,

        "explicitly_frozen_field_check_count":
            len(frozen_fields),

        "explicitly_frozen_field_pass_count":
            sum(
                int(value)
                for value in frozen_fields.values()
            ),

        "tier_c_execution_field_count":
            len(TIER_C_EXECUTION_FIELDS),

        "tier_c_execution_fields_missing_count":
            len(missing_rows),

        "tier_c_execution_fields_missing":
            [
                row["field"]
                for row in missing_rows
            ],

        "b1_required_reporting_name":
            (
                "affine latent-transition baseline "
                "with nonlinear field codec"
            ),

        "b1_is_global_field_space_affine_model":
            False,

        "execution_contract_complete":
            False,

        "implementation_code_written":
            False,

        "implementation_smoke_test_executed":
            False,

        "tuning_execution_performed":
            False,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "prior_protocol_modified":
            False,

        "audit_status":
            "blocked_missing_tier_c_execution_fields",

        "next_phase":
            (
                "4D-R3.2 Tier C v4 predictive "
                "execution-contract freeze"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4dr31_execution_contract_audit_summary.json",
        summary,
    )

    print(
        "Phase 4D-R3.1 execution-contract audit completed."
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
