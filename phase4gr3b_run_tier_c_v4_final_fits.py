from __future__ import annotations

import os

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4fr3b_run_tier_c_v4_tuning as tuning_runtime


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_RUN_COUNT = 130

EXPECTED_RUN_COUNTS = {
    "B1": 5,
    "B2": 25,
    "B3": 25,
    "B4": 25,
    "B5": 25,
    "OCM": 25,
}

EXPECTED_NOISE_COUNTS = {
    0.0: 26,
    0.1: 26,
    0.25: 26,
    0.5: 26,
    1.0: 26,
}

EXPECTED_SEED_COUNTS = {
    11: 30,
    23: 25,
    37: 25,
    53: 25,
    71: 25,
}

TRAINABLE_MODEL_IDS = (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
)

MAXIMUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 25

TRAIN_TRAJECTORY_COUNT = 11008
VALIDATION_TRAJECTORY_COUNT = 1152

PHYSICAL_MICROBATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 64
EFFECTIVE_BATCH_SIZE = 256
GRADIENT_CLIP_NORM = 5.0


PREFLIGHT_ROOT = Path(
    "outputs/"
    "phase4gr3_tier_c_v4_final_fit_preflight"
)

PREFLIGHT_SUMMARY_PATH = (
    PREFLIGHT_ROOT
    / "phase4gr3_final_fit_preflight_summary.json"
)

RESOLVED_REGISTRY_PATH = (
    PREFLIGHT_ROOT
    / "resolved_trainable_final_fit_registry.csv"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

SELECTED_REGISTRY_PATH = Path(
    "outputs/"
    "phase4fr3c_tier_c_v4_tuning_selection/"
    "selected_tuning_registry.json"
)

VISIBLE_DATA_ROOT = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_data/"
    "visible"
)

VISIBLE_INDEX_ROOT = Path(
    "outputs/"
    "phase4fr3ar1_tier_c_v4_visible_trajectory_indexes/"
    "visible"
)

TUNING_RUNTIME_PATH = Path(
    "phase4fr3b_run_tier_c_v4_tuning.py"
)

ACCEPTED_IMPLEMENTATION_PATH = Path(
    "phase4er3c_execute_predictive_smoke.py"
)

NORMALIZATION_ARTIFACT_PATH = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_fields/"
    "normalization_v4.json"
)

NORMALIZATION_BINDING_SUMMARY_PATH = Path(
    "outputs/"
    "phase4gr3ar2_tier_c_v4_normalization_binding/"
    "phase4gr3ar2_normalization_binding_summary.json"
)

EXPECTED_NORMALIZATION_SHA256 = (
    "3bbfd2c72e774d32efead77287778004"
    "a498051028441838970c83baf406012c"
)

OUTPUT_ROOT = Path(
    "outputs/"
    "phase4gr3b_tier_c_v4_final_fits"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def parse_bool(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

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
        f"Cannot parse Boolean value: {value!r}"
    )


def reject_privileged_path(
    path: Path,
) -> None:
    normalized = str(path).replace(
        "\\",
        "/",
    ).lower()

    require(
        "/privileged/" not in normalized,
        f"Privileged path rejected: {path}",
    )


def load_json(
    path: Path,
) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def atomic_write_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


def atomic_torch_save(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        value,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
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
            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(
    value: Any,
) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def normalize_float(
    value: Any,
) -> float:
    result = float(value)

    require(
        math.isfinite(result),
        f"Non-finite floating-point value: {value}",
    )

    return result


def find_normalization_artifact(
    contract: dict[str, Any],
) -> Path:
    for required_path in (
        NORMALIZATION_ARTIFACT_PATH,
        NORMALIZATION_BINDING_SUMMARY_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                required_path
            )

    binding = load_json(
        NORMALIZATION_BINDING_SUMMARY_PATH
    )

    require(
        binding[
            "phase4gr3ar2_status"
        ] == "frozen_normalization_artifact_bound",
        "Normalization artifact binding is not frozen.",
    )

    require(
        binding[
            "visible_fields_already_normalized"
        ] is True,
        "Visible fields are not declared normalized.",
    )

    require(
        binding[
            "normalization_applied_by_final_fit_runner"
        ] is False,
        (
            "Final-fit runner is unexpectedly "
            "authorized to normalize fields again."
        ),
    )

    require(
        binding[
            "double_normalization_prohibited"
        ] is True,
        "Double-normalization prohibition is absent.",
    )

    require(
        Path(
            binding[
                "normalization_artifact_path"
            ]
        ) == NORMALIZATION_ARTIFACT_PATH,
        "Bound normalization path changed.",
    )

    observed_sha256 = sha256_file(
        NORMALIZATION_ARTIFACT_PATH
    )

    require(
        observed_sha256
        == EXPECTED_NORMALIZATION_SHA256,
        "Normalization artifact hash changed.",
    )

    require(
        observed_sha256
        == binding[
            "normalization_artifact_sha256"
        ],
        "Binding hash does not match artifact.",
    )

    require(
        contract[
            "predictive_field_objective"
        ][
            "target"
        ] == "stored noisy normalized field",
        "Predictive target is not stored normalized field.",
    )

    reject_privileged_path(
        NORMALIZATION_ARTIFACT_PATH
    )

    return NORMALIZATION_ARTIFACT_PATH

def load_and_validate_registry() -> list[dict[str, Any]]:
    with RESOLVED_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    require(
        len(rows) == EXPECTED_RUN_COUNT,
        (
            "Resolved final-fit registry must "
            f"contain {EXPECTED_RUN_COUNT} rows."
        ),
    )

    parsed_rows = []

    for expected_index, row in enumerate(
        rows
    ):
        execution_index = int(
            row[
                "execution_index"
            ]
        )

        require(
            execution_index == expected_index,
            (
                "Execution indexes are not "
                "contiguous from zero."
            ),
        )

        model_id = row[
            "model_id"
        ]

        require(
            model_id in TRAINABLE_MODEL_IDS,
            f"Unexpected model ID: {model_id}",
        )

        noise_fraction = normalize_float(
            row[
                "noise_fraction"
            ]
        )

        noise_index = int(
            row[
                "noise_index"
            ]
        )

        effective_seed = int(
            row[
                "effective_seed"
            ]
        )

        configuration = json.loads(
            row[
                "selected_configuration_json"
            ]
        )

        configuration_id = row[
            "selected_configuration_id"
        ]

        require(
            configuration[
                "configuration_id"
            ] == configuration_id,
            (
                "Selected configuration JSON does "
                "not match selected_configuration_id."
            ),
        )

        require(
            parse_bool(
                row[
                    "initialize_from_tuning_checkpoint"
                ]
            ) is False,
            (
                "A final-fit row attempts to initialize "
                "from a tuning checkpoint."
            ),
        )

        require(
            parse_bool(
                row[
                    "fresh_seeded_initialization_required"
                ]
            ) is True,
            "Fresh initialization is not required.",
        )

        require(
            parse_bool(
                row[
                    "selected_tuning_epoch_used_as_fixed_duration"
                ]
            ) is False,
            (
                "A selected tuning epoch was incorrectly "
                "used as a fixed final-fit duration."
            ),
        )

        require(
            parse_bool(
                row[
                    "independent_early_stopping_required"
                ]
            ) is True,
            "Independent early stopping is not required.",
        )

        require(
            int(
                row[
                    "maximum_epochs"
                ]
            ) == MAXIMUM_EPOCHS,
            "Maximum epochs changed.",
        )

        require(
            int(
                row[
                    "early_stopping_patience"
                ]
            ) == EARLY_STOPPING_PATIENCE,
            "Early-stopping patience changed.",
        )

        require(
            int(
                row[
                    "physical_microbatch_size"
                ]
            ) == PHYSICAL_MICROBATCH_SIZE,
            "Physical microbatch size changed.",
        )

        require(
            int(
                row[
                    "gradient_accumulation_steps"
                ]
            ) == GRADIENT_ACCUMULATION_STEPS,
            "Gradient accumulation changed.",
        )

        require(
            int(
                row[
                    "effective_batch_size"
                ]
            ) == EFFECTIVE_BATCH_SIZE,
            "Effective batch size changed.",
        )

        require(
            normalize_float(
                row[
                    "gradient_clip_norm"
                ]
            ) == GRADIENT_CLIP_NORM,
            "Gradient clipping changed.",
        )

        require(
            parse_bool(
                row[
                    "mixed_precision"
                ]
            ) is False,
            "Mixed precision was enabled.",
        )

        require(
            row[
                "training_cell"
            ] == "train_joint",
            "Training cell changed.",
        )

        require(
            row[
                "checkpoint_selection_cell"
            ] == "val_joint",
            "Validation cell changed.",
        )

        require(
            row[
                "checkpoint_selection_metric"
            ] == "noisy_target_rollout_mse",
            "Checkpoint-selection metric changed.",
        )

        require(
            row[
                "checkpoint_selection_direction"
            ] == "minimize",
            "Checkpoint-selection direction changed.",
        )

        require(
            parse_bool(
                row[
                    "execution_authorized_by_phase4fr3c"
                ]
            ) is True,
            "Run lacks Phase 4F-R3C authorization.",
        )

        require(
            parse_bool(
                row[
                    "final_fit_training_allowed"
                ]
            ) is True,
            "Final-fit training is not authorized.",
        )

        require(
            parse_bool(
                row[
                    "privileged_directory_access_allowed"
                ]
            ) is False,
            "Privileged-directory access is allowed.",
        )

        require(
            parse_bool(
                row[
                    "clean_target_access_allowed"
                ]
            ) is False,
            "Clean-target access is allowed.",
        )

        require(
            parse_bool(
                row[
                    "test_access_allowed"
                ]
            ) is False,
            "Test access is allowed.",
        )

        selected_checkpoint_path = Path(
            row[
                "selected_tuning_checkpoint_path"
            ]
        )

        require(
            selected_checkpoint_path.exists(),
            (
                "Selected tuning checkpoint is missing: "
                f"{selected_checkpoint_path}"
            ),
        )

        require(
            sha256_file(
                selected_checkpoint_path
            )
            == row[
                "selected_tuning_checkpoint_sha256"
            ],
            (
                "Selected tuning checkpoint hash "
                f"changed for {row['run_id']}."
            ),
        )

        output_directory = Path(
            row[
                "output_directory"
            ]
        )

        reject_privileged_path(
            output_directory
        )

        expected_output_directory = (
            OUTPUT_ROOT
            / model_id
            / row[
                "run_id"
            ]
        )

        require(
            output_directory
            == expected_output_directory,
            (
                "Output directory does not match "
                f"the frozen run ID for {row['run_id']}."
            ),
        )

        parsed_rows.append(
            {
                "execution_index":
                    execution_index,

                "run_id":
                    row[
                        "run_id"
                    ],

                "model_id":
                    model_id,

                "run_type":
                    row[
                        "run_type"
                    ],

                "noise_fraction":
                    noise_fraction,

                "noise_index":
                    noise_index,

                "original_seed_token":
                    row[
                        "original_seed_token"
                    ],

                "effective_seed":
                    effective_seed,

                "configuration_id":
                    configuration_id,

                "configuration":
                    configuration,

                "selected_tuning_best_epoch":
                    int(
                        row[
                            "selected_tuning_best_epoch"
                        ]
                    ),

                "selected_tuning_validation_metric":
                    normalize_float(
                        row[
                            "selected_tuning_validation_metric"
                        ]
                    ),

                "selected_tuning_checkpoint_path":
                    str(
                        selected_checkpoint_path
                    ),

                "selected_tuning_checkpoint_sha256":
                    row[
                        "selected_tuning_checkpoint_sha256"
                    ],

                "output_directory":
                    output_directory,
            }
        )

    run_ids = [
        row[
            "run_id"
        ]
        for row in parsed_rows
    ]

    require(
        len(run_ids) == len(set(run_ids)),
        "Resolved final-fit run IDs are not unique.",
    )

    model_counts = Counter(
        row[
            "model_id"
        ]
        for row in parsed_rows
    )

    noise_counts = Counter(
        row[
            "noise_fraction"
        ]
        for row in parsed_rows
    )

    seed_counts = Counter(
        row[
            "effective_seed"
        ]
        for row in parsed_rows
    )

    require(
        dict(model_counts)
        == EXPECTED_RUN_COUNTS,
        (
            "Final-fit model counts changed: "
            f"{dict(model_counts)}"
        ),
    )

    require(
        dict(noise_counts)
        == EXPECTED_NOISE_COUNTS,
        (
            "Final-fit noise counts changed: "
            f"{dict(noise_counts)}"
        ),
    )

    require(
        dict(seed_counts)
        == EXPECTED_SEED_COUNTS,
        (
            "Final-fit seed counts changed: "
            f"{dict(seed_counts)}"
        ),
    )

    return parsed_rows


def validate_preflight() -> dict[str, Any]:
    required_paths = (
        PREFLIGHT_SUMMARY_PATH,
        RESOLVED_REGISTRY_PATH,
        EXECUTION_CONTRACT_PATH,
        SELECTED_REGISTRY_PATH,
        TUNING_RUNTIME_PATH,
        ACCEPTED_IMPLEMENTATION_PATH,
        NORMALIZATION_ARTIFACT_PATH,
        NORMALIZATION_BINDING_SUMMARY_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = load_json(
        PREFLIGHT_SUMMARY_PATH
    )

    require(
        summary[
            "phase4gr3_status"
        ] == "ready_for_130_trainable_final_fits",
        "Final-fit preflight did not pass.",
    )

    require(
        summary[
            "final_fit_preflight_passed"
        ] is True,
        "Final-fit preflight pass flag is false.",
    )

    require(
        summary[
            "resolved_trainable_final_fit_count"
        ] == EXPECTED_RUN_COUNT,
        "Preflight does not authorize 130 runs.",
    )

    require(
        summary[
            "final_fit_execution_authorized"
        ] is True,
        "Final-fit execution is not authorized.",
    )

    require(
        summary[
            "fresh_seeded_initialization_required"
        ] is True,
        "Fresh initialization is not required.",
    )

    require(
        summary[
            "selected_tuning_checkpoint_weights_reused"
        ] is False,
        "Tuning checkpoint reuse is enabled.",
    )

    require(
        summary[
            "independent_final_fit_early_stopping_required"
        ] is True,
        "Independent final-fit early stopping is absent.",
    )

    require(
        summary[
            "additional_tuning_authorized"
        ] is False,
        "Additional tuning is authorized.",
    )

    require(
        summary[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation is authorized.",
    )

    require(
        summary[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation is authorized.",
    )

    require(
        summary[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        sha256_file(
            RESOLVED_REGISTRY_PATH
        )
        == summary[
            "resolved_final_fit_registry_sha256"
        ],
        "Resolved final-fit registry hash changed.",
    )

    return {
        "summary":
            summary,

        "required_paths":
            required_paths,
    }


def capture_random_states() -> dict[str, Any]:
    return {
        "python_random_state":
            random.getstate(),

        "numpy_random_state":
            np.random.get_state(),

        "torch_cpu_random_state":
            torch.get_rng_state(),

        "torch_cuda_random_state_if_available":
            (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
    }


def restore_random_states(
    checkpoint: dict[str, Any],
) -> None:
    random.setstate(
        checkpoint[
            "python_random_state"
        ]
    )

    np.random.set_state(
        checkpoint[
            "numpy_random_state"
        ]
    )

    torch.set_rng_state(
        checkpoint[
            "torch_cpu_random_state"
        ]
    )

    cuda_state = checkpoint.get(
        "torch_cuda_random_state_if_available"
    )

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )


def build_source_signature(
    run: dict[str, Any],
    train_cell: Any,
    validation_cell: Any,
    normalization_artifact: Path,
) -> dict[str, Any]:
    signature = {
        "protocol_version":
            PROTOCOL_VERSION,

        "run_id":
            run[
                "run_id"
            ],

        "model_id":
            run[
                "model_id"
            ],

        "noise_fraction":
            run[
                "noise_fraction"
            ],

        "noise_index":
            run[
                "noise_index"
            ],

        "effective_seed":
            run[
                "effective_seed"
            ],

        "configuration_id":
            run[
                "configuration_id"
            ],

        "configuration":
            run[
                "configuration"
            ],

        "resolved_registry_sha256":
            sha256_file(
                RESOLVED_REGISTRY_PATH
            ),

        "preflight_summary_sha256":
            sha256_file(
                PREFLIGHT_SUMMARY_PATH
            ),

        "execution_contract_sha256":
            sha256_file(
                EXECUTION_CONTRACT_PATH
            ),

        "selected_registry_sha256":
            sha256_file(
                SELECTED_REGISTRY_PATH
            ),

        "normalization_artifact_path":
            str(
                normalization_artifact
            ),

        "normalization_artifact_sha256":
            sha256_file(
                normalization_artifact
            ),

        "training_field_path":
            str(
                train_cell.fields_path
            ),

        "validation_field_path":
            str(
                validation_cell.fields_path
            ),

        "training_manifest_sha256":
            sha256_file(
                train_cell.manifest_path
            ),

        "validation_manifest_sha256":
            sha256_file(
                validation_cell.manifest_path
            ),

        "training_index_sha256":
            sha256_file(
                train_cell.index_path
            ),

        "validation_index_sha256":
            sha256_file(
                validation_cell.index_path
            ),

        "accepted_tuning_runtime_sha256":
            sha256_file(
                TUNING_RUNTIME_PATH
            ),

        "accepted_implementation_sha256":
            sha256_file(
                ACCEPTED_IMPLEMENTATION_PATH
            ),

        "final_fit_runner_sha256":
            sha256_file(
                Path(__file__)
            ),
    }

    signature[
        "signature_sha256"
    ] = sha256_json(signature)

    return signature


def create_checkpoint(
    *,
    checkpoint_kind: str,
    run: dict[str, Any],
    source_signature: dict[str, Any],
    normalization_artifact: Path,
    codec: torch.nn.Module,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int | None,
    best_metric: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    random_states = capture_random_states()

    return {
        "checkpoint_kind":
            checkpoint_kind,

        "phase":
            (
                "4G-R3B Tier C v4 "
                "trainable final fit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "run_id":
            run[
                "run_id"
            ],

        "model_id":
            run[
                "model_id"
            ],

        "configuration_id":
            run[
                "configuration_id"
            ],

        "full_configuration":
            run[
                "configuration"
            ],

        "noise_fraction":
            run[
                "noise_fraction"
            ],

        "noise_index":
            run[
                "noise_index"
            ],

        "effective_seed":
            run[
                "effective_seed"
            ],

        "epoch":
            epoch,

        "best_epoch":
            best_epoch,

        "best_validation_metric":
            best_metric,

        "epochs_without_improvement":
            epochs_without_improvement,

        "history":
            history,

        "model_state_dict":
            model.state_dict(),

        "codec_state_dict":
            codec.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "execution_contract_sha256":
            sha256_file(
                EXECUTION_CONTRACT_PATH
            ),

        "normalization_artifact_path":
            str(
                normalization_artifact
            ),

        "normalization_artifact_sha256":
            sha256_file(
                normalization_artifact
            ),

        "training_manifest_sha256":
            source_signature[
                "training_manifest_sha256"
            ],

        "source_signature":
            source_signature,

        "initialize_from_tuning_checkpoint":
            False,

        "selected_tuning_checkpoint_path":
            run[
                "selected_tuning_checkpoint_path"
            ],

        "selected_tuning_checkpoint_sha256":
            run[
                "selected_tuning_checkpoint_sha256"
            ],

        "selected_tuning_checkpoint_weights_loaded":
            False,

        "test_data_read":
            False,

        **random_states,
    }


def run_final_fit(
    run: dict[str, Any],
    train_cell: Any,
    validation_cell: Any,
    normalization_artifact: Path,
    device: torch.device,
    overwrite_completed: bool,
) -> dict[str, Any]:
    run_id = run[
        "run_id"
    ]

    run_directory = run[
        "output_directory"
    ]

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        run_directory
        / "final_fit_result.json"
    )

    history_path = (
        run_directory
        / "epoch_history.json"
    )

    last_checkpoint_path = (
        run_directory
        / "last_checkpoint.pt"
    )

    best_checkpoint_path = (
        run_directory
        / "best_checkpoint.pt"
    )

    source_signature = build_source_signature(
        run,
        train_cell,
        validation_cell,
        normalization_artifact,
    )

    if (
        result_path.exists()
        and not overwrite_completed
    ):
        existing = load_json(
            result_path
        )

        require(
            existing[
                "source_signature_sha256"
            ]
            == source_signature[
                "signature_sha256"
            ],
            (
                "Completed result source signature "
                f"changed for {run_id}."
            ),
        )

        if (
            existing[
                "final_fit_status"
            ] == "completed"
        ):
            print(
                f"{run_id}: already complete"
            )

            return existing

    # Set the frozen noise condition before any batch
    # is read. TrajectoryCell.make_batch uses these
    # module-level values in the accepted runtime.
    tuning_runtime.NOISE_INDEX = int(
        run[
            "noise_index"
        ]
    )

    tuning_runtime.NOISE_FRACTION = float(
        run[
            "noise_fraction"
        ]
    )

    effective_seed = int(
        run[
            "effective_seed"
        ]
    )

    tuning_runtime.set_global_determinism(
        effective_seed
    )

    # This constructs entirely fresh codec and transition
    # weights. The selected tuning checkpoint is never loaded.
    codec, model, optimizer = (
        tuning_runtime.build_model_bundle(
            model_id=run[
                "model_id"
            ],
            configuration=run[
                "configuration"
            ],
            maximum_sequence_length=max(
                train_cell.maximum_sequence_length,
                validation_cell.maximum_sequence_length,
            ),
            device=device,
        )
    )

    start_epoch = 1
    best_epoch: int | None = None
    best_metric = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    if (
        last_checkpoint_path.exists()
        and not overwrite_completed
    ):
        checkpoint = torch.load(
            last_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        require(
            checkpoint[
                "source_signature"
            ][
                "signature_sha256"
            ]
            == source_signature[
                "signature_sha256"
            ],
            (
                "Checkpoint source signature "
                f"changed for {run_id}."
            ),
        )

        require(
            checkpoint[
                "selected_tuning_checkpoint_weights_loaded"
            ] is False,
            (
                "Resumed checkpoint claims tuning "
                "weights were loaded."
            ),
        )

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

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        restore_random_states(
            checkpoint
        )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_epoch = checkpoint[
            "best_epoch"
        ]

        best_metric = float(
            checkpoint[
                "best_validation_metric"
            ]
        )

        epochs_without_improvement = int(
            checkpoint[
                "epochs_without_improvement"
            ]
        )

        history = list(
            checkpoint[
                "history"
            ]
        )

        print(
            f"{run_id}: resuming at "
            f"epoch {start_epoch}"
        )

    run_start = time.time()
    stop_reason = "maximum_epochs_reached"

    for epoch in range(
        start_epoch,
        MAXIMUM_EPOCHS + 1,
    ):
        training_record = (
            tuning_runtime.run_training_epoch(
                model_id=run[
                    "model_id"
                ],
                model=model,
                codec=codec,
                optimizer=optimizer,
                configuration=run[
                    "configuration"
                ],
                train_cell=train_cell,
                device=device,
                epoch=epoch,
                tuning_seed=effective_seed,
            )
        )

        validation_record = (
            tuning_runtime.evaluate_validation(
                model_id=run[
                    "model_id"
                ],
                model=model,
                codec=codec,
                configuration=run[
                    "configuration"
                ],
                validation_cell=validation_cell,
                device=device,
            )
        )

        validation_metric = float(
            validation_record[
                "selection_metric"
            ]
        )

        require(
            math.isfinite(
                validation_metric
            ),
            (
                "Final-fit validation metric is "
                f"non-finite for {run_id}."
            ),
        )

        improved = (
            validation_metric
            < best_metric
        )

        if improved:
            best_metric = (
                validation_metric
            )

            best_epoch = epoch
            epochs_without_improvement = 0

            best_checkpoint = (
                create_checkpoint(
                    checkpoint_kind=(
                        "best_validation_checkpoint"
                    ),
                    run=run,
                    source_signature=(
                        source_signature
                    ),
                    normalization_artifact=(
                        normalization_artifact
                    ),
                    codec=codec,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    epochs_without_improvement=(
                        epochs_without_improvement
                    ),
                    history=history,
                )
            )

            atomic_torch_save(
                best_checkpoint_path,
                best_checkpoint,
            )

        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch":
                epoch,

            "run_id":
                run_id,

            "noise_fraction":
                run[
                    "noise_fraction"
                ],

            "noise_index":
                run[
                    "noise_index"
                ],

            "effective_seed":
                effective_seed,

            "training":
                training_record,

            "validation":
                validation_record,

            "improved":
                improved,

            "best_epoch":
                best_epoch,

            "best_validation_rollout_mse":
                best_metric,

            "epochs_without_improvement":
                epochs_without_improvement,
        }

        history.append(
            epoch_record
        )

        atomic_write_json(
            history_path,
            history,
        )

        last_checkpoint = (
            create_checkpoint(
                checkpoint_kind=(
                    "last_epoch_checkpoint"
                ),
                run=run,
                source_signature=(
                    source_signature
                ),
                normalization_artifact=(
                    normalization_artifact
                ),
                codec=codec,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                epochs_without_improvement=(
                    epochs_without_improvement
                ),
                history=history,
            )
        )

        atomic_torch_save(
            last_checkpoint_path,
            last_checkpoint,
        )

        print(
            f"{run_id} | "
            f"epoch={epoch:03d} | "
            f"train_total="
            f"{training_record['component_means']['total_loss']:.8f} | "
            f"val_rollout="
            f"{validation_metric:.8f} | "
            f"best={best_metric:.8f} | "
            f"bad_epochs="
            f"{epochs_without_improvement}"
        )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            stop_reason = (
                "early_stopping_patience_reached"
            )

            break

    require(
        best_epoch is not None,
        (
            f"{run_id} never produced a valid "
            "validation-selected checkpoint."
        ),
    )

    require(
        best_checkpoint_path.exists(),
        (
            f"{run_id} best checkpoint "
            "is missing."
        ),
    )

    total_optimizer_steps = sum(
        int(
            epoch_record[
                "training"
            ][
                "optimizer_step_count"
            ]
        )
        for epoch_record in history
    )

    result = {
        "phase":
            (
                "4G-R3B Tier C v4 "
                "trainable final fit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "execution_index":
            run[
                "execution_index"
            ],

        "run_id":
            run_id,

        "model_id":
            run[
                "model_id"
            ],

        "run_type":
            run[
                "run_type"
            ],

        "configuration_id":
            run[
                "configuration_id"
            ],

        "configuration":
            run[
                "configuration"
            ],

        "noise_fraction":
            run[
                "noise_fraction"
            ],

        "noise_index":
            run[
                "noise_index"
            ],

        "original_seed_token":
            run[
                "original_seed_token"
            ],

        "effective_seed":
            effective_seed,

        "fresh_seeded_initialization":
            True,

        "initialize_from_tuning_checkpoint":
            False,

        "selected_tuning_checkpoint_weights_loaded":
            False,

        "selected_tuning_checkpoint_path":
            run[
                "selected_tuning_checkpoint_path"
            ],

        "selected_tuning_checkpoint_sha256":
            run[
                "selected_tuning_checkpoint_sha256"
            ],

        "selected_tuning_checkpoint_usage":
            "provenance_only",

        "selected_tuning_best_epoch":
            run[
                "selected_tuning_best_epoch"
            ],

        "selected_tuning_epoch_used_as_fixed_duration":
            False,

        "independent_final_fit_early_stopping":
            True,

        "training_cell":
            "train_joint",

        "validation_cell":
            "val_joint",

        "checkpoint_selection_metric":
            (
                "val_joint noisy-target "
                "rollout field MSE"
            ),

        "checkpoint_selection_direction":
            "minimize",

        "best_epoch":
            best_epoch,

        "best_validation_rollout_mse":
            best_metric,

        "epochs_completed":
            len(history),

        "stop_reason":
            stop_reason,

        "maximum_epochs":
            MAXIMUM_EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "physical_microbatch_size":
            PHYSICAL_MICROBATCH_SIZE,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "effective_batch_size":
            EFFECTIVE_BATCH_SIZE,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "mixed_precision":
            False,

        "optimizer":
            "torch.optim.Adam",

        "total_optimizer_steps":
            total_optimizer_steps,

        "best_checkpoint_path":
            str(
                best_checkpoint_path
            ),

        "best_checkpoint_sha256":
            sha256_file(
                best_checkpoint_path
            ),

        "last_checkpoint_path":
            str(
                last_checkpoint_path
            ),

        "last_checkpoint_sha256":
            sha256_file(
                last_checkpoint_path
            ),

        "history_path":
            str(
                history_path
            ),

        "history_sha256":
            sha256_file(
                history_path
            ),

        "normalization_artifact_path":
            str(
                normalization_artifact
            ),

        "normalization_artifact_sha256":
            sha256_file(
                normalization_artifact
            ),

        "source_signature_sha256":
            source_signature[
                "signature_sha256"
            ],

        "elapsed_seconds":
            float(
                time.time()
                - run_start
            ),

        "validation_selection_performed":
            True,

        "privileged_directory_read":
            False,

        "clean_targets_read":
            False,

        "privileged_state_ids_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "scientific_test_comparison_performed":
            False,

        "final_fit_status":
            "completed",
    }

    atomic_write_json(
        result_path,
        result,
    )

    return result


def write_global_status(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = []
    incomplete = []
    interrupted = []

    completed_by_model = Counter()
    completed_by_noise = Counter()
    completed_by_seed = Counter()

    for run in runs:
        run_directory = run[
            "output_directory"
        ]

        result_path = (
            run_directory
            / "final_fit_result.json"
        )

        last_checkpoint_path = (
            run_directory
            / "last_checkpoint.pt"
        )

        if result_path.exists():
            result = load_json(
                result_path
            )

            if (
                result.get(
                    "final_fit_status"
                )
                == "completed"
            ):
                completed.append(
                    run[
                        "run_id"
                    ]
                )

                completed_by_model[
                    run[
                        "model_id"
                    ]
                ] += 1

                completed_by_noise[
                    run[
                        "noise_fraction"
                    ]
                ] += 1

                completed_by_seed[
                    run[
                        "effective_seed"
                    ]
                ] += 1

                continue

        incomplete.append(
            run[
                "run_id"
            ]
        )

        if last_checkpoint_path.exists():
            interrupted.append(
                run[
                    "run_id"
                ]
            )

    status = {
        "phase":
            (
                "4G-R3B Tier C v4 resumable "
                "trainable final-fit status"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "authorized_run_count":
            len(runs),

        "completed_run_count":
            len(completed),

        "incomplete_run_count":
            len(incomplete),

        "interrupted_resumable_run_count":
            len(interrupted),

        "completed_run_ids":
            completed,

        "incomplete_run_ids":
            incomplete,

        "interrupted_resumable_run_ids":
            interrupted,

        "completed_counts_by_model":
            dict(
                sorted(
                    completed_by_model.items()
                )
            ),

        "completed_counts_by_noise":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    completed_by_noise.items()
                )
            },

        "completed_counts_by_seed":
            {
                str(key):
                    value
                for key, value
                in sorted(
                    completed_by_seed.items()
                )
            },

        "all_trainable_final_fits_completed":
            len(incomplete) == 0,

        "fresh_seeded_initialization_required":
            True,

        "selected_tuning_checkpoint_weights_reused":
            False,

        "training_cell":
            "train_joint",

        "validation_cell":
            "val_joint",

        "privileged_directory_read":
            False,

        "clean_targets_read":
            False,

        "test_data_generated":
            False,

        "test_data_read":
            False,

        "test_metrics_computed":
            False,

        "test_open_count":
            0,

        "b0_persistence_evaluations_completed":
            False,

        "phase4gr3b_status":
            (
                "completed"
                if len(incomplete) == 0
                else "in_progress"
            ),
    }

    atomic_write_json(
        OUTPUT_ROOT
        / "phase4gr3b_final_fit_status.json",
        status,
    )

    return status


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
    )

    parser.add_argument(
        "--model-id",
        action="append",
        choices=TRAINABLE_MODEL_IDS,
        default=[],
    )

    parser.add_argument(
        "--noise-fraction",
        action="append",
        type=float,
        default=[],
    )

    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--overwrite-completed",
        action="store_true",
    )

    parser.add_argument(
        "--list-runs",
        action="store_true",
    )

    arguments = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_preflight()

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    normalization_artifact = (
        find_normalization_artifact(
            contract
        )
    )

    reject_privileged_path(
        normalization_artifact
    )

    print(
        "Normalization artifact:",
        normalization_artifact,
    )

    print(
        "Normalization artifact SHA256:",
        sha256_file(
            normalization_artifact
        ),
    )

    runs = load_and_validate_registry()

    if arguments.list_runs:
        for run in runs:
            print(
                run[
                    "execution_index"
                ],
                run[
                    "run_id"
                ],
                run[
                    "configuration_id"
                ],
                run[
                    "noise_fraction"
                ],
                run[
                    "effective_seed"
                ],
            )

        return

    device = tuning_runtime.resolve_device(
        arguments.device
    )

    require(
        device.type == "cuda",
        (
            "The frozen 130-run final-fit execution "
            "must run on CUDA."
        ),
    )

    print(
        "CUDA device:",
        torch.cuda.get_device_name(
            device
        ),
    )

    # The same visible cells and released structural
    # indexes accepted in Phase 4F-R3B are reused.
    train_cell = (
        tuning_runtime.TrajectoryCell(
            cell_id="train_joint",
            expected_trajectory_count=(
                TRAIN_TRAJECTORY_COUNT
            ),
        )
    )

    validation_cell = (
        tuning_runtime.TrajectoryCell(
            cell_id="val_joint",
            expected_trajectory_count=(
                VALIDATION_TRAJECTORY_COUNT
            ),
        )
    )

    require(
        train_cell.maximum_sequence_length
        >= validation_cell.maximum_sequence_length,
        (
            "Validation sequence length exceeds "
            "the frozen training interface."
        ),
    )

    selected = list(runs)

    if arguments.run_id:
        requested = set(
            arguments.run_id
        )

        selected = [
            run
            for run in selected
            if run[
                "run_id"
            ] in requested
        ]

        found = {
            run[
                "run_id"
            ]
            for run in selected
        }

        require(
            found == requested,
            (
                "Unknown run IDs: "
                f"{sorted(requested - found)}"
            ),
        )

    if arguments.model_id:
        requested_models = set(
            arguments.model_id
        )

        selected = [
            run
            for run in selected
            if run[
                "model_id"
            ] in requested_models
        ]

    if arguments.noise_fraction:
        requested_noises = {
            float(value)
            for value
            in arguments.noise_fraction
        }

        selected = [
            run
            for run in selected
            if run[
                "noise_fraction"
            ] in requested_noises
        ]

    if arguments.seed:
        requested_seeds = set(
            arguments.seed
        )

        selected = [
            run
            for run in selected
            if run[
                "effective_seed"
            ] in requested_seeds
        ]

    if arguments.max_runs is not None:
        require(
            arguments.max_runs >= 1,
            "--max-runs must be positive.",
        )

        selected = selected[
            :arguments.max_runs
        ]

    require(
        bool(selected),
        "No final-fit runs were selected.",
    )

    print(
        f"Selected {len(selected)} final-fit run(s)."
    )

    print(
        "Resolved final-fit registry SHA256:",
        sources[
            "summary"
        ][
            "resolved_final_fit_registry_sha256"
        ],
    )

    for position, run in enumerate(
        selected,
        start=1,
    ):
        print(
            "=" * 100
        )

        print(
            f"[{position}/{len(selected)}] "
            f"{run['run_id']} | "
            f"configuration="
            f"{run['configuration_id']} | "
            f"noise={run['noise_fraction']} | "
            f"seed={run['effective_seed']}"
        )

        result = run_final_fit(
            run=run,
            train_cell=train_cell,
            validation_cell=(
                validation_cell
            ),
            normalization_artifact=(
                normalization_artifact
            ),
            device=device,
            overwrite_completed=(
                arguments.overwrite_completed
            ),
        )

        print(
            json.dumps(
                {
                    "run_id":
                        result[
                            "run_id"
                        ],

                    "model_id":
                        result[
                            "model_id"
                        ],

                    "noise_fraction":
                        result[
                            "noise_fraction"
                        ],

                    "effective_seed":
                        result[
                            "effective_seed"
                        ],

                    "best_epoch":
                        result[
                            "best_epoch"
                        ],

                    "best_validation_rollout_mse":
                        result[
                            "best_validation_rollout_mse"
                        ],

                    "epochs_completed":
                        result[
                            "epochs_completed"
                        ],

                    "stop_reason":
                        result[
                            "stop_reason"
                        ],

                    "final_fit_status":
                        result[
                            "final_fit_status"
                        ],
                },
                indent=2,
            )
        )

        write_global_status(
            runs
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_status = write_global_status(
        runs
    )

    print(
        json.dumps(
            final_status,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
