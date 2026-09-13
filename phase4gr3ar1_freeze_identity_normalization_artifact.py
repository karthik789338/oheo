from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


PROTOCOL_VERSION = "tier_c_v4"

EXECUTION_CONTRACT_PATH = Path(
    "outputs/"
    "phase4dr32_tier_c_v4_execution_contract/"
    "tier_c_v4_execution_contract.json"
)

CODEC_REGISTRY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_field_codec_registry.json"
)

PROTOCOL_SUMMARY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "phase4dr3_predictive_protocol_summary.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3ar1_tier_c_v4_normalization_artifact"
)

ARTIFACT_PATH = (
    OUTPUT_DIR
    / "tier_c_v4_identity_normalization_artifact.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase4gr3ar1_normalization_artifact_summary.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
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
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def walk_scalars(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_scalars(
                child,
                path + (str(key),),
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_scalars(
                child,
                path + (str(index),),
            )

    else:
        yield path, value


def collect_normalization_mentions(
    source_name: str,
    value: Any,
) -> list[dict[str, Any]]:
    terms = (
        "normalization",
        "normalisation",
        "standardization",
        "standardisation",
        "zscore",
        "z_score",
    )

    mentions = []

    for path, scalar in walk_scalars(value):
        path_text = ".".join(path)
        searchable = (
            path_text
            + " "
            + str(scalar)
        ).lower()

        if any(
            term in searchable
            for term in terms
        ):
            mentions.append(
                {
                    "source":
                        source_name,

                    "path":
                        path_text,

                    "value":
                        scalar,
                }
            )

    return mentions


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_paths = (
        EXECUTION_CONTRACT_PATH,
        CODEC_REGISTRY_PATH,
        PROTOCOL_SUMMARY_PATH,
    )

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    codec_registry = load_json(
        CODEC_REGISTRY_PATH
    )

    protocol_summary = load_json(
        PROTOCOL_SUMMARY_PATH
    )

    mentions = []

    mentions.extend(
        collect_normalization_mentions(
            "execution_contract",
            contract,
        )
    )

    mentions.extend(
        collect_normalization_mentions(
            "codec_registry",
            codec_registry,
        )
    )

    mentions.extend(
        collect_normalization_mentions(
            "protocol_summary",
            protocol_summary,
        )
    )

    allowed_placeholder_mentions = []

    unexpected_mentions = []

    for mention in mentions:
        value = str(
            mention["value"]
        ).strip().lower()

        if value == "normalization_artifact_sha256":
            allowed_placeholder_mentions.append(
                mention
            )
        else:
            unexpected_mentions.append(
                mention
            )

    require(
        len(
            allowed_placeholder_mentions
        ) == 1,
        (
            "Expected exactly one checkpoint placeholder "
            "for normalization_artifact_sha256; found "
            f"{len(allowed_placeholder_mentions)}."
        ),
    )

    require(
        not unexpected_mentions,
        (
            "A normalization-related policy was found, so "
            "identity normalization cannot be inferred safely:\n"
            + json.dumps(
                unexpected_mentions,
                indent=2,
            )
        ),
    )

    require(
        protocol_summary[
            "field_codec_pretraining"
        ] is False,
        "Field-codec pretraining is unexpectedly enabled.",
    )

    artifact = {
        "phase":
            (
                "4G-R3A-R1 Tier C v4 identity "
                "normalization artifact freeze"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "artifact_type":
            "normalization_declaration",

        "normalization_mode":
            "identity",

        "normalization_applied":
            False,

        "centering_applied":
            False,

        "scaling_applied":
            False,

        "standardization_applied":
            False,

        "learned_statistics_present":
            False,

        "channel_means":
            None,

        "channel_standard_deviations":
            None,

        "minimum_maximum_scaling":
            None,

        "input_disk_dtype":
            "float16",

        "compute_dtype":
            "float32",

        "permitted_numeric_conversion":
            (
                "Cast stored float16 field values to "
                "float32 for computation without changing "
                "their numerical scale."
            ),

        "field_values_modified_before_codec":
            False,

        "training_statistics_read":
            False,

        "validation_statistics_read":
            False,

        "privileged_data_read":
            False,

        "test_data_read":
            False,

        "test_open_count":
            0,

        "reason":
            (
                "The frozen execution contract requires a "
                "normalization artifact hash in checkpoints, "
                "but the frozen protocol and codec registry "
                "specify no normalization procedure or "
                "normalization statistics."
            ),

        "source_execution_contract_path":
            str(
                EXECUTION_CONTRACT_PATH
            ),

        "source_execution_contract_sha256":
            sha256_file(
                EXECUTION_CONTRACT_PATH
            ),

        "source_codec_registry_path":
            str(
                CODEC_REGISTRY_PATH
            ),

        "source_codec_registry_sha256":
            sha256_file(
                CODEC_REGISTRY_PATH
            ),

        "source_protocol_summary_path":
            str(
                PROTOCOL_SUMMARY_PATH
            ),

        "source_protocol_summary_sha256":
            sha256_file(
                PROTOCOL_SUMMARY_PATH
            ),

        "normalization_mentions_audited":
            mentions,

        "identity_normalization_frozen":
            True,
    }

    write_json(
        ARTIFACT_PATH,
        artifact,
    )

    artifact_sha256 = sha256_file(
        ARTIFACT_PATH
    )

    summary = {
        "phase":
            (
                "4G-R3A-R1 Tier C v4 normalization "
                "contract repair"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "normalization_artifact_path":
            str(
                ARTIFACT_PATH
            ),

        "normalization_artifact_sha256":
            artifact_sha256,

        "normalization_mode":
            "identity",

        "normalization_applied":
            False,

        "learned_statistics_present":
            False,

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

        "phase4gr3ar1_status":
            "identity_normalization_artifact_frozen",
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
