from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_VERSION = "tier_c_v4"

EXPECTED_CONFIGURATION_COUNTS = {
    "B1": 3,
    "B2": 4,
    "B3": 4,
    "B4": 4,
    "B5": 4,
    "OCM": 24,
}

EXPECTED_CONFIGURATION_COUNT = 43

DEVELOPMENT_DATA_ROOT = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_data"
)

TUNING_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_tuning_registry.csv"
)

IMPLEMENTATION_ACCEPTANCE_PATH = Path(
    "outputs/"
    "phase4er3d_tier_c_v4_implementation_acceptance/"
    "phase4er3d_implementation_acceptance_summary.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

EXECUTION_CONTRACT_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "phase4dr32_execution_contract_summary.json"
)

CODEC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_field_codec_registry.json"
)

CODEC_MICROARCHITECTURE_PATH = Path(
    "outputs/"
    "phase4dr33_tier_c_v4_codec_microarchitecture/"
    "tier_c_v4_codec_microarchitecture.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4fr3_tier_c_v4_tuning_preflight"
)


REQUIRED_CONFIGURATION_KEYS = {
    "B1": {
        "configuration_id",
        "ridge",
    },

    "B2": {
        "configuration_id",
        "learning_rate",
        "operation_embedding_dim",
        "hidden_width",
    },

    "B3": {
        "configuration_id",
        "learning_rate",
        "operation_embedding_dim",
        "hidden_size",
    },

    "B4": {
        "configuration_id",
        "learning_rate",
        "model_dimension",
        "attention_heads",
        "layer_count",
        "feedforward_dimension",
        "dropout",
    },

    "B5": {
        "configuration_id",
        "learning_rate",
        "latent_dimension",
        "encoder_width",
        "operator_hidden_width",
    },

    "OCM": {
        "configuration_id",
        "learning_rate",
        "latent_state_count",
        "temperature",
        "lambda_transition",
        "lambda_rollout",
        "lambda_determinism",
    },
}


def load_json(path: Path) -> Any:
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
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


def find_exactly_one(
    candidates: list[Path],
    description: str,
) -> Path:
    existing = [
        path
        for path in candidates
        if path.exists()
    ]

    require(
        len(existing) == 1,
        (
            f"Expected exactly one {description}; "
            f"found {[str(path) for path in existing]}"
        ),
    )

    return existing[0]


def discover_development_files() -> dict[str, Path]:
    train_fields = find_exactly_one(
        [
            DEVELOPMENT_DATA_ROOT
            / "visible"
            / "train_joint_fields.npy",

            DEVELOPMENT_DATA_ROOT
            / "train_joint_fields.npy",
        ],
        "train_joint field array",
    )

    val_carrier_fields = find_exactly_one(
        [
            DEVELOPMENT_DATA_ROOT
            / "visible"
            / "val_carrier_fields.npy",

            DEVELOPMENT_DATA_ROOT
            / "val_carrier_fields.npy",
        ],
        "val_carrier field array",
    )

    val_composition_fields = find_exactly_one(
        [
            DEVELOPMENT_DATA_ROOT
            / "visible"
            / "val_composition_fields.npy",

            DEVELOPMENT_DATA_ROOT
            / "val_composition_fields.npy",
        ],
        "val_composition field array",
    )

    val_joint_fields = find_exactly_one(
        [
            DEVELOPMENT_DATA_ROOT
            / "visible"
            / "val_joint_fields.npy",

            DEVELOPMENT_DATA_ROOT
            / "val_joint_fields.npy",
        ],
        "val_joint field array",
    )

    noise_levels = find_exactly_one(
        [
            DEVELOPMENT_DATA_ROOT
            / "noise_levels.npy",

            DEVELOPMENT_DATA_ROOT
            / "visible"
            / "noise_levels.npy",
        ],
        "noise-level array",
    )

    sequence_candidates = sorted(
        path
        for path in DEVELOPMENT_DATA_ROOT.rglob(
            "*"
        )
        if path.is_file()
        and "train_joint" in path.name.lower()
        and "sequence" in path.name.lower()
        and path.suffix.lower()
        in {
            ".csv",
            ".json",
            ".jsonl",
            ".parquet",
        }
    )

    trajectory_candidates = sorted(
        path
        for path in DEVELOPMENT_DATA_ROOT.rglob(
            "*"
        )
        if path.is_file()
        and "train_joint" in path.name.lower()
        and "trajectory" in path.name.lower()
        and path.suffix.lower()
        in {
            ".csv",
            ".json",
            ".jsonl",
            ".parquet",
        }
    )

    require(
        len(sequence_candidates) >= 1,
        "No train_joint sequence manifest was found.",
    )

    require(
        len(trajectory_candidates) >= 1,
        "No train_joint trajectory index was found.",
    )

    return {
        "train_joint_fields":
            train_fields,

        "val_carrier_fields":
            val_carrier_fields,

        "val_composition_fields":
            val_composition_fields,

        "val_joint_fields":
            val_joint_fields,

        "noise_levels":
            noise_levels,

        "train_sequence_manifest":
            sequence_candidates[0],

        "train_trajectory_index":
            trajectory_candidates[0],
    }


def inspect_array(path: Path) -> dict[str, Any]:
    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    shape = list(array.shape)

    require(
        len(shape) == 5,
        (
            f"Expected a five-dimensional field array "
            f"at {path}; found {shape}."
        ),
    )

    require(
        shape[0] == 5,
        (
            f"Expected five noise levels at {path}; "
            f"found {shape[0]}."
        ),
    )

    require(
        shape[-3:] == [4, 32, 32],
        (
            f"Unexpected spatial field shape at {path}: "
            f"{shape[-3:]}"
        ),
    )

    require(
        str(array.dtype) == "float16",
        (
            f"Expected on-disk float16 at {path}; "
            f"found {array.dtype}."
        ),
    )

    return {
        "path":
            str(path),

        "shape":
            shape,

        "dtype":
            str(array.dtype),

        "memory_mapped":
            isinstance(
                array,
                np.memmap,
            ),

        "file_size_bytes":
            path.stat().st_size,

        "sha256":
            sha256_file(path),
    }


def load_configurations() -> tuple[
    list[dict[str, Any]],
    list[str],
]:
    rows = []

    with TUNING_REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        fieldnames = list(
            reader.fieldnames or []
        )

        require(
            "model_id" in fieldnames,
            "Tuning registry lacks model_id.",
        )

        require(
            "source_configuration_json"
            in fieldnames,
            (
                "Tuning registry lacks "
                "source_configuration_json."
            ),
        )

        for registry_index, row in enumerate(
            reader
        ):
            model_id = row["model_id"]

            require(
                model_id
                in EXPECTED_CONFIGURATION_COUNTS,
                (
                    "Unexpected model ID in tuning "
                    f"registry: {model_id}"
                ),
            )

            configuration = json.loads(
                row[
                    "source_configuration_json"
                ]
            )

            required_keys = (
                REQUIRED_CONFIGURATION_KEYS[
                    model_id
                ]
            )

            missing_keys = sorted(
                required_keys
                - set(configuration)
            )

            require(
                not missing_keys,
                (
                    f"{model_id} configuration at "
                    f"registry row {registry_index} "
                    f"lacks {missing_keys}."
                ),
            )

            configuration_id = (
                configuration[
                    "configuration_id"
                ]
            )

            rows.append(
                {
                    "registry_index":
                        registry_index,

                    "model_id":
                        model_id,

                    "configuration_id":
                        configuration_id,

                    "configuration":
                        configuration,
                }
            )

    return rows, fieldnames


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/7] Validating implementation acceptance"
    )

    required_protocol_paths = (
        IMPLEMENTATION_ACCEPTANCE_PATH,
        EXECUTION_CONTRACT_PATH,
        EXECUTION_CONTRACT_SUMMARY_PATH,
        CODEC_REGISTRY_PATH,
        CODEC_MICROARCHITECTURE_PATH,
        TUNING_REGISTRY_PATH,
    )

    for path in required_protocol_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    acceptance = load_json(
        IMPLEMENTATION_ACCEPTANCE_PATH
    )

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    contract_summary = load_json(
        EXECUTION_CONTRACT_SUMMARY_PATH
    )

    codec_registry = load_json(
        CODEC_REGISTRY_PATH
    )

    require(
        acceptance[
            "phase4er3d_status"
        ] == "implementation_accepted",
        "Implementation acceptance is not frozen.",
    )

    require(
        acceptance[
            "implementation_accepted"
        ] is True,
        "Implementation is not accepted.",
    )

    require(
        acceptance[
            "tuning_execution_authorized"
        ] is True,
        "Development tuning is not authorized.",
    )

    require(
        acceptance[
            "final_fit_execution_authorized"
        ] is False,
        "Final fitting is already authorized.",
    )

    require(
        acceptance[
            "test_artifact_generation_authorized"
        ] is False,
        "Test generation is already authorized.",
    )

    require(
        acceptance[
            "test_evaluation_authorized"
        ] is False,
        "Test evaluation is already authorized.",
    )

    require(
        acceptance[
            "test_open_count"
        ] == 0,
        "Test data was opened.",
    )

    require(
        contract_summary[
            "phase4dr32_status"
        ] == "execution_contract_frozen",
        "Execution contract is not frozen.",
    )

    require(
        contract[
            "batching"
        ][
            "physical_microbatch_size_trajectories"
        ] == 4,
        "Physical microbatch size changed.",
    )

    require(
        contract[
            "batching"
        ][
            "gradient_accumulation_steps"
        ] == 64,
        "Gradient accumulation steps changed.",
    )

    require(
        contract[
            "batching"
        ][
            "effective_batch_size_trajectories"
        ] == 256,
        "Effective batch size changed.",
    )

    require(
        contract[
            "gradient_clipping_policy"
        ][
            "maximum_norm"
        ] == 5.0,
        "Gradient clip norm changed.",
    )

    require(
        contract[
            "mixed_precision_policy"
        ][
            "enabled"
        ] is False,
        "Mixed precision was enabled.",
    )

    require(
        codec_registry[
            "latent_dimension"
        ] == 128,
        "Codec latent dimension changed.",
    )

    print(
        "[2/7] Loading and validating 43 frozen configurations"
    )

    configurations, registry_columns = (
        load_configurations()
    )

    configuration_counts = Counter(
        row["model_id"]
        for row in configurations
    )

    require(
        len(configurations)
        == EXPECTED_CONFIGURATION_COUNT,
        (
            "Unexpected tuning configuration count: "
            f"{len(configurations)}"
        ),
    )

    require(
        dict(configuration_counts)
        == EXPECTED_CONFIGURATION_COUNTS,
        (
            "Unexpected model configuration counts: "
            f"{dict(configuration_counts)}"
        ),
    )

    configuration_ids = [
        row["configuration_id"]
        for row in configurations
    ]

    require(
        len(configuration_ids)
        == len(set(configuration_ids)),
        "Tuning configuration IDs are not unique.",
    )

    print(
        "[3/7] Discovering development-only files"
    )

    development_files = (
        discover_development_files()
    )

    for label, path in (
        development_files.items()
    ):
        require(
            "test" not in str(path).lower(),
            (
                f"Development file discovery selected "
                f"a test-like path for {label}: {path}"
            ),
        )

    print(
        "[4/7] Inspecting field arrays by memory mapping"
    )

    field_arrays = {
        label:
            inspect_array(path)
        for label, path
        in development_files.items()
        if label.endswith(
            "_fields"
        )
    }

    print(
        "[5/7] Validating noise-level registry"
    )

    noise_levels = np.load(
        development_files[
            "noise_levels"
        ],
        allow_pickle=False,
    )

    require(
        noise_levels.shape == (5,),
        (
            "Unexpected noise-level shape: "
            f"{noise_levels.shape}"
        ),
    )

    require(
        np.allclose(
            noise_levels.astype(
                np.float64
            ),
            np.array(
                [
                    0.0,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                ],
                dtype=np.float64,
            ),
        ),
        (
            "Noise levels differ from the "
            "frozen five-level registry."
        ),
    )

    development_noise_index = int(
        np.where(
            np.isclose(
                noise_levels,
                0.25,
            )
        )[0][0]
    )

    print(
        "[6/7] Checking execution resources"
    )

    disk_usage = shutil.disk_usage(
        Path.cwd()
    )

    gpu_record = {
        "cuda_available":
            torch.cuda.is_available(),

        "cuda_device_count":
            torch.cuda.device_count(),

        "cuda_device_name":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "cuda_total_memory_bytes":
            (
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                if torch.cuda.is_available()
                else None
            ),

        "torch_version":
            torch.__version__,

        "torch_cuda_build":
            torch.version.cuda,
    }

    require(
        torch.cuda.is_available(),
        (
            "CUDA is unavailable. The 43-run tuning "
            "execution should not start on this node."
        ),
    )

    print(
        "[7/7] Writing tuning-preflight record"
    )

    source_hashes = {
        str(path):
            sha256_file(path)
        for path in required_protocol_paths
    }

    source_hashes.update(
        {
            str(path):
                sha256_file(path)
            for path in (
                development_files.values()
            )
        }
    )

    write_json(
        OUTPUT_DIR
        / "tuning_preflight_source_hashes.json",
        source_hashes,
    )

    configuration_plan = [
        {
            "registry_index":
                row["registry_index"],

            "model_id":
                row["model_id"],

            "configuration_id":
                row["configuration_id"],

            "execution_status":
                "pending",

            "validation_cell":
                "val_joint",

            "development_noise_fraction":
                0.25,

            "development_noise_index":
                development_noise_index,

            "test_access_authorized":
                False,
        }
        for row in configurations
    ]

    write_json(
        OUTPUT_DIR
        / "tuning_execution_plan.json",
        configuration_plan,
    )

    summary = {
        "phase":
            (
                "4F-R3 Tier C v4 predictive "
                "development-tuning preflight"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_implementation_accepted":
            True,

        "tuning_execution_authorized":
            True,

        "final_fit_execution_authorized":
            False,

        "test_artifact_generation_authorized":
            False,

        "test_evaluation_authorized":
            False,

        "test_open_count":
            0,

        "configuration_count":
            len(configurations),

        "configuration_counts":
            dict(
                sorted(
                    configuration_counts.items()
                )
            ),

        "configuration_ids_unique":
            True,

        "registry_columns":
            registry_columns,

        "development_training_cell":
            "train_joint",

        "development_validation_cell":
            "val_joint",

        "development_noise_fraction":
            0.25,

        "development_noise_index":
            development_noise_index,

        "noise_levels":
            [
                float(value)
                for value in noise_levels
            ],

        "field_arrays":
            field_arrays,

        "train_sequence_manifest_path":
            str(
                development_files[
                    "train_sequence_manifest"
                ]
            ),

        "train_trajectory_index_path":
            str(
                development_files[
                    "train_trajectory_index"
                ]
            ),

        "physical_microbatch_size":
            4,

        "gradient_accumulation_steps":
            64,

        "effective_batch_size":
            256,

        "maximum_epochs":
            200,

        "early_stopping_patience":
            25,

        "gradient_clip_norm":
            5.0,

        "mixed_precision":
            False,

        "gpu":
            gpu_record,

        "disk": {
            "total_bytes":
                disk_usage.total,

            "used_bytes":
                disk_usage.used,

            "free_bytes":
                disk_usage.free,
        },

        "model_parameters_initialized":
            False,

        "training_batches_read":
            False,

        "validation_batches_read":
            False,

        "optimizer_steps_performed":
            0,

        "checkpoints_created":
            False,

        "scientific_metrics_computed":
            False,

        "privileged_data_read":
            False,

        "test_data_read":
            False,

        "preflight_passed":
            True,

        "phase4fr3_preflight_status":
            "ready_for_development_tuning",

        "next_action":
            (
                "Execute the 43 frozen Tier C v4 "
                "development tuning configurations."
            ),
    }

    write_json(
        OUTPUT_DIR
        / "phase4fr3_tuning_preflight_summary.json",
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Phase 4F-R3 tuning preflight passed."
    )

    print(
        f"Outputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
