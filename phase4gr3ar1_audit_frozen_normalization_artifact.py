from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "tier_c_v4"

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

SEARCH_ROOTS = (
    Path("outputs"),
    Path("."),
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3ar1_tier_c_v4_normalization_audit"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "phase4gr3ar1_normalization_audit_summary.json"
)

NORMALIZATION_TERMS = (
    "normalization",
    "normalisation",
    "normalize",
    "normalise",
    "channel_mean",
    "channel_std",
    "channel_means",
    "channel_stds",
    "training_mean",
    "training_std",
    "clean_training",
)

CANDIDATE_SUFFIXES = {
    ".json",
    ".npz",
    ".npy",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
    ".csv",
    ".yaml",
    ".yml",
}

TEXT_SUFFIXES = {
    ".json",
    ".csv",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
}


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
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def is_forbidden_path(path: Path) -> bool:
    normalized = str(path).replace(
        "\\",
        "/",
    ).lower()

    forbidden_tokens = (
        "/test_",
        "/test/",
        "/privileged/",
        "sealed_test",
    )

    return any(
        token in normalized
        for token in forbidden_tokens
    )


def walk_scalars(
    value: Any,
    key_path: tuple[str, ...] = (),
):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_scalars(
                child,
                key_path + (str(key),),
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_scalars(
                child,
                key_path + (str(index),),
            )

    else:
        yield key_path, value


def collect_json_mentions(
    source_path: Path,
    value: Any,
) -> list[dict[str, Any]]:
    mentions = []

    for key_path, scalar in walk_scalars(value):
        path_text = ".".join(key_path)

        searchable = (
            path_text
            + " "
            + str(scalar)
        ).lower()

        if any(
            term in searchable
            for term in NORMALIZATION_TERMS
        ):
            mentions.append(
                {
                    "source_path":
                        str(source_path),

                    "json_path":
                        path_text,

                    "value":
                        scalar,
                }
            )

    return mentions


def extract_existing_paths_from_scalar(
    scalar: Any,
) -> list[Path]:
    if not isinstance(scalar, str):
        return []

    candidates = []

    strings = {
        scalar.strip(),
    }

    strings.update(
        re.findall(
            r"(?:outputs|data|artifacts)/[^\s,\"']+",
            scalar,
        )
    )

    for value in strings:
        value = value.strip(
            "[](){}<>.,:;\"'"
        )

        if not value:
            continue

        path = Path(value)

        if (
            path.exists()
            and path.is_file()
        ):
            candidates.append(path)

    return candidates


def scan_json_references() -> tuple[
    list[dict[str, Any]],
    set[Path],
]:
    mentions = []
    referenced_candidates: set[Path] = set()

    json_paths = sorted(
        {
            path
            for root in SEARCH_ROOTS
            if root.exists()
            for path in root.rglob("*.json")
            if path.is_file()
            and not is_forbidden_path(path)
        }
    )

    for path in json_paths:
        try:
            value = load_json(path)
        except Exception:
            continue

        path_mentions = collect_json_mentions(
            path,
            value,
        )

        mentions.extend(
            path_mentions
        )

        for mention in path_mentions:
            for candidate in (
                extract_existing_paths_from_scalar(
                    mention["value"]
                )
            ):
                if not is_forbidden_path(candidate):
                    referenced_candidates.add(
                        candidate.resolve()
                    )

    return mentions, referenced_candidates


def scan_candidate_filenames() -> set[Path]:
    candidates: set[Path] = set()

    filename_terms = (
        "normal",
        "channel_mean",
        "channel_std",
        "statistics",
        "scaler",
    )

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if is_forbidden_path(path):
                continue

            if path.suffix.lower() not in CANDIDATE_SUFFIXES:
                continue

            name = path.name.lower()

            if any(
                term in name
                for term in filename_terms
            ):
                candidates.add(
                    path.resolve()
                )

    return candidates


def inspect_text_candidate(
    path: Path,
) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []

    if path.stat().st_size > 10 * 1024 * 1024:
        return []

    try:
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []

    matched_lines = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        lowered = line.lower()

        if any(
            term in lowered
            for term in NORMALIZATION_TERMS
        ):
            matched_lines.append(
                f"{line_number}: {line[:500]}"
            )

    return matched_lines[:50]


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        CODEC_REGISTRY_PATH,
        EXECUTION_CONTRACT_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    codec_registry = load_json(
        CODEC_REGISTRY_PATH
    )

    execution_contract = load_json(
        EXECUTION_CONTRACT_PATH
    )

    require(
        codec_registry.get(
            "normalization"
        )
        == (
            "Use the already-frozen normalization "
            "computed from clean training-carrier "
            "fields only."
        ),
        (
            "The codec normalization policy differs "
            "from the expected frozen statement."
        ),
    )

    direct_mentions = []

    direct_mentions.extend(
        collect_json_mentions(
            CODEC_REGISTRY_PATH,
            codec_registry,
        )
    )

    direct_mentions.extend(
        collect_json_mentions(
            EXECUTION_CONTRACT_PATH,
            execution_contract,
        )
    )

    all_json_mentions, referenced_candidates = (
        scan_json_references()
    )

    filename_candidates = (
        scan_candidate_filenames()
    )

    all_candidates = sorted(
        referenced_candidates
        | filename_candidates,
        key=lambda path: str(path),
    )

    candidate_records = []

    for path in all_candidates:
        candidate_records.append(
            {
                "path":
                    str(path),

                "relative_path":
                    str(
                        path.relative_to(
                            Path.cwd().resolve()
                        )
                    )
                    if path.is_relative_to(
                        Path.cwd().resolve()
                    )
                    else None,

                "suffix":
                    path.suffix.lower(),

                "size_bytes":
                    path.stat().st_size,

                "sha256":
                    sha256_file(path),

                "referenced_by_normalization_metadata":
                    path
                    in referenced_candidates,

                "matched_text_lines":
                    inspect_text_candidate(path),

                "forbidden_path":
                    is_forbidden_path(path),
            }
        )

    summary = {
        "phase":
            (
                "4G-R3A-R1 Tier C v4 frozen "
                "normalization-artifact audit"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "codec_normalization_policy":
            codec_registry[
                "normalization"
            ],

        "identity_normalization_allowed":
            False,

        "normalization_required":
            True,

        "direct_protocol_mentions":
            direct_mentions,

        "all_normalization_mentions":
            all_json_mentions,

        "candidate_count":
            len(
                candidate_records
            ),

        "candidates":
            candidate_records,

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

        "audit_status":
            (
                "single_candidate_found"
                if len(candidate_records) == 1
                else (
                    "no_candidate_found"
                    if len(candidate_records) == 0
                    else "multiple_candidates_found"
                )
            ),
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
