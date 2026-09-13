from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

NORMALIZATION_PATH = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_fields/"
    "normalization_v4.json"
)

EXPECTED_NORMALIZATION_SHA256 = (
    "3bbfd2c72e774d32efead77287778004"
    "a498051028441838970c83baf406012c"
)

NORMALIZATION_AUDIT_PATH = Path(
    "outputs/"
    "phase4gr3ar1_tier_c_v4_normalization_audit/"
    "phase4gr3ar1_normalization_audit_summary.json"
)

CODEC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_field_codec_registry.json"
)

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

DEVELOPMENT_SUMMARY_PATH = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_data/"
    "phase4br3_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3ar2_tier_c_v4_normalization_binding"
)

BINDING_PATH = (
    OUTPUT_DIR
    / "tier_c_v4_normalization_binding.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase4gr3ar2_normalization_binding_summary.json"
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(message)


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


def numeric_values_are_finite(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return True

    if isinstance(value, (int, float)):
        return math.isfinite(
            float(value)
        )

    if isinstance(value, dict):
        return all(
            numeric_values_are_finite(
                child
            )
            for child in value.values()
        )

    if isinstance(value, list):
        return all(
            numeric_values_are_finite(
                child
            )
            for child in value
        )

    return True


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        NORMALIZATION_PATH,
        NORMALIZATION_AUDIT_PATH,
        CODEC_REGISTRY_PATH,
        EXECUTION_CONTRACT_PATH,
        DEVELOPMENT_SUMMARY_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    normalization_sha256 = sha256_file(
        NORMALIZATION_PATH
    )

    require(
        normalization_sha256
        == EXPECTED_NORMALIZATION_SHA256,
        (
            "Tier C v4 normalization artifact hash "
            "does not match the audited frozen hash."
        ),
    )

    normalization = load_json(
        NORMALIZATION_PATH
    )

    audit = load_json(
        NORMALIZATION_AUDIT_PATH
    )

    codec_registry = load_json(
        CODEC_REGISTRY_PATH
    )

    execution_contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    development_summary = load_json(
        DEVELOPMENT_SUMMARY_PATH
    )

    require(
        normalization.get(
            "statistics_source"
        )
        == "clean training-carrier fields only",
        (
            "Normalization statistics source is not "
            "clean training-carrier fields only."
        ),
    )

    channel_mean = normalization.get(
        "channel_mean"
    )

    require(
        isinstance(
            channel_mean,
            list,
        ),
        "Normalization artifact lacks channel_mean.",
    )

    require(
        len(channel_mean) == 4,
        (
            "Expected four field-channel means; "
            f"found {len(channel_mean)}."
        ),
    )

    require(
        numeric_values_are_finite(
            normalization
        ),
        (
            "Normalization artifact contains "
            "non-finite numeric values."
        ),
    )

    require(
        codec_registry[
            "normalization"
        ]
        == (
            "Use the already-frozen normalization "
            "computed from clean training-carrier "
            "fields only."
        ),
        "Codec normalization policy changed.",
    )

    require(
        execution_contract[
            "predictive_field_objective"
        ][
            "target"
        ] == "stored noisy normalized field",
        "Predictive target is not stored normalized field.",
    )

    require(
        (
            "normalization_artifact_sha256"
            in execution_contract[
                "checkpoint_contents"
            ]
        ),
        (
            "Checkpoint contract does not require "
            "normalization_artifact_sha256."
        ),
    )

    require(
        development_summary[
            "normalization_source"
        ]
        == "clean training-carrier fields only",
        "Development normalization source changed.",
    )

    matching_candidates = [
        candidate
        for candidate in audit[
            "candidates"
        ]
        if candidate.get(
            "relative_path"
        ) == str(
            NORMALIZATION_PATH
        )
    ]

    require(
        len(matching_candidates) == 1,
        (
            "Expected exactly one audited Tier C v4 "
            "normalization candidate."
        ),
    )

    candidate = matching_candidates[0]

    require(
        candidate[
            "sha256"
        ] == normalization_sha256,
        "Audit candidate hash does not match artifact.",
    )

    require(
        candidate[
            "referenced_by_normalization_metadata"
        ] is True,
        (
            "Tier C v4 normalization artifact was not "
            "referenced by frozen metadata."
        ),
    )

    path_mentions = [
        mention
        for mention in audit[
            "all_normalization_mentions"
        ]
        if str(
            mention.get(
                "value"
            )
        ) == str(
            NORMALIZATION_PATH
        )
    ]

    hash_mentions = [
        mention
        for mention in audit[
            "all_normalization_mentions"
        ]
        if str(
            mention.get(
                "value"
            )
        ) == normalization_sha256
    ]

    require(
        bool(path_mentions),
        (
            "No frozen metadata reference to the "
            "normalization path was found."
        ),
    )

    require(
        bool(hash_mentions),
        (
            "No frozen metadata reference to the "
            "normalization hash was found."
        ),
    )

    binding = {
        "phase":
            (
                "4G-R3A-R2 Tier C v4 frozen "
                "normalization binding"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "normalization_artifact_path":
            str(
                NORMALIZATION_PATH
            ),

        "normalization_artifact_sha256":
            normalization_sha256,

        "statistics_source":
            normalization[
                "statistics_source"
            ],

        "field_channel_count":
            len(channel_mean),

        "artifact_referenced_by_frozen_metadata":
            True,

        "artifact_path_reference_count":
            len(path_mentions),

        "artifact_hash_reference_count":
            len(hash_mentions),

        "visible_field_storage_space":
            "already_normalized",

        "final_fit_input_transform_required":
            False,

        "apply_normalization_again":
            False,

        "double_normalization_prohibited":
            True,

        "checkpoint_usage":
            (
                "Record artifact path and SHA256 for "
                "provenance. Do not apply another "
                "normalization transform."
            ),

        "privileged_data_read":
            False,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "normalization_binding_frozen":
            True,
    }

    write_json(
        BINDING_PATH,
        binding,
    )

    summary = {
        "phase":
            (
                "4G-R3A-R2 Tier C v4 normalization "
                "artifact resolution"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "normalization_artifact_path":
            str(
                NORMALIZATION_PATH
            ),

        "normalization_artifact_sha256":
            normalization_sha256,

        "normalization_binding_path":
            str(
                BINDING_PATH
            ),

        "normalization_binding_sha256":
            sha256_file(
                BINDING_PATH
            ),

        "statistics_source":
            "clean training-carrier fields only",

        "visible_fields_already_normalized":
            True,

        "normalization_applied_by_final_fit_runner":
            False,

        "double_normalization_prohibited":
            True,

        "original_protocol_files_modified":
            False,

        "model_parameters_initialized":
            False,

        "training_batches_read":
            False,

        "validation_batches_read":
            False,

        "optimizer_steps_performed":
            0,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "phase4gr3ar2_status":
            "frozen_normalization_artifact_bound",
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
