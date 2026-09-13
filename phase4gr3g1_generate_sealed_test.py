from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4br3_construct_tier_c_v4_development_fields as fieldgen
import phase4br3_materialize_tier_c_v4_development_dataset as materialize


PROTOCOL_VERSION = "tier_c_v4"
EXPECTED_TEST_CARRIER_COUNT = 32
EXPECTED_SEQUENCE_COUNT = 144
TRAJECTORIES_PER_SEQUENCE = 8
EXPECTED_TRAJECTORY_COUNT = EXPECTED_SEQUENCE_COUNT * TRAJECTORIES_PER_SEQUENCE
EXPECTED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)
NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)

# Factorized test execution binding.  Sequence/state paths come from the
# already-frozen Phase-3B cell of the same name.  Carrier shift is introduced
# only in the carrier and joint conditions.
CELL_CARRIER_SPLIT = {
    "test_iid_pairing": "train",
    "test_composition": "train",
    "test_carrier": "test",
    "test_joint": "test",
}

AUTH_DIR = Path("outputs/phase4gr3f_tier_c_v4_test_authorization")
AUTH_PATH = AUTH_DIR / "one_time_test_opening_authorization.json"
G0_DIR = Path("outputs/phase4gr3g0_tier_c_v4_test_execution_interfaces")
G0_SUMMARY_PATH = G0_DIR / "phase4gr3g0_interface_freeze_summary.json"
G0_SOURCE_HASHES_PATH = G0_DIR / "phase4gr3g0_source_hashes.json"
PHASE4A_DIR = Path("outputs/phase4ar3_tier_c_v4_protocol")
CARRIER_SPLIT_PATH = PHASE4A_DIR / "carrier_split_v4.csv"
DEV_FIELD_DIR = Path("outputs/phase4br3_tier_c_v4_development_fields")
DEV_DATA_DIR = Path("outputs/phase4br3_tier_c_v4_development_data")
NORMALIZATION_PATH = DEV_FIELD_DIR / "normalization_v4.json"
NORMALIZATION_BINDING_PATH = Path(
    "outputs/phase4gr3ar2_tier_c_v4_normalization_binding/"
    "phase4gr3ar2_normalization_binding_summary.json"
)

OUTPUT_DIR = Path("outputs/phase4gr3g1_tier_c_v4_sealed_test")
FIELD_DIR = OUTPUT_DIR / "field_bank"
FIELD_PRIVILEGED_DIR = FIELD_DIR / "privileged"
VISIBLE_DIR = OUTPUT_DIR / "visible"
PRIVILEGED_DIR = OUTPUT_DIR / "privileged"
MANIFEST_DIR = OUTPUT_DIR / "cell_manifests"
LEDGER_PATH = OUTPUT_DIR / "single_test_opening_ledger.json"
SUMMARY_PATH = OUTPUT_DIR / "phase4gr3g1_sealed_test_generation_summary.json"
ARTIFACT_MANIFEST_PATH = OUTPUT_DIR / "sealed_test_artifact_manifest.csv"

FIELD_SOURCE_PATH = Path("phase4br3_construct_tier_c_v4_development_fields.py")
MATERIALIZER_SOURCE_PATH = Path("phase4br3_materialize_tier_c_v4_development_dataset.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"No rows supplied for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_marker(name: str) -> Path:
    return OUTPUT_DIR / f"stage_{name}_complete.json"


def verify_file_hash(path: Path, expected: str) -> None:
    require(path.exists(), f"Missing frozen source: {path}")
    observed = sha256_file(path)
    require(observed == expected, f"Frozen source hash changed: {path}")


def verify_pre_open_state() -> tuple[dict[str, Any], dict[str, Any]]:
    require(AUTH_PATH.exists(), f"Missing authorization: {AUTH_PATH}")
    require(G0_SUMMARY_PATH.exists(), f"Missing G0 summary: {G0_SUMMARY_PATH}")
    require(G0_SOURCE_HASHES_PATH.exists(), f"Missing G0 source hashes: {G0_SOURCE_HASHES_PATH}")
    require(NORMALIZATION_PATH.exists(), f"Missing frozen normalization: {NORMALIZATION_PATH}")
    require(NORMALIZATION_BINDING_PATH.exists(), f"Missing normalization binding: {NORMALIZATION_BINDING_PATH}")

    authorization = load_json(AUTH_PATH)
    g0 = load_json(G0_SUMMARY_PATH)
    source_hashes = load_json(G0_SOURCE_HASHES_PATH)

    require(authorization["phase4gr3f_status"] == "one_time_test_opening_authorized", "Test authorization is not frozen.")
    require(authorization["one_time_test_authorization_frozen"] is True, "Test authorization is not immutable.")
    require(authorization["test_generation_authorized"] is True, "Test generation is not authorized.")
    require(authorization["test_evaluation_authorized"] is True, "Test evaluation is not authorized.")
    require(authorization["test_open_count_after_this_phase"] == 0, "Authorization already consumed the opening.")
    require(authorization["maximum_test_open_count"] == 1, "Maximum test-open count changed.")
    require(tuple(authorization["sealed_test_cells"]) == EXPECTED_TEST_CELLS, "Authorized sealed-cell set changed.")
    require(authorization["test_carrier_count"] == EXPECTED_TEST_CARRIER_COUNT, "Authorized test-carrier count changed.")

    require(g0["phase4gr3g0_status"] == "ready_for_single_sealed_test_opening", "G0 is not ready.")
    require(g0["test_open_count"] == 0, "G0 records an opened test.")
    require(g0["single_opening_consumed"] is False, "G0 says opening was consumed.")

    # Re-bind the exact implementation frozen at G0.
    for path in (FIELD_SOURCE_PATH, MATERIALIZER_SOURCE_PATH):
        key = str(path)
        require(key in source_hashes, f"G0 source hash missing for {path}.")
        verify_file_hash(path, source_hashes[key])

    normalization_binding = load_json(NORMALIZATION_BINDING_PATH)
    require(normalization_binding["phase4gr3ar2_status"] == "frozen_normalization_artifact_bound", "Normalization binding is not frozen.")
    require(normalization_binding["visible_fields_already_normalized"] is True, "Frozen development fields are not normalized as expected.")
    require(normalization_binding["double_normalization_prohibited"] is True, "Double-normalization prohibition changed.")

    expected_normalization_sha = normalization_binding.get("normalization_artifact_sha256")
    if expected_normalization_sha is not None:
        verify_file_hash(NORMALIZATION_PATH, expected_normalization_sha)

    seeds = authorization["frozen_generation_seeds"]
    require(int(seeds["fresh_carrier_parameter_seed"]) == fieldgen.CARRIER_PARAMETER_SEED, "Carrier-parameter seed changed.")
    require(int(seeds["fresh_field_initialization_seed"]) == fieldgen.FIELD_INITIALIZATION_SEED, "Field initialization seed changed.")
    require(int(seeds["fresh_solver_seed"]) == fieldgen.SOLVER_SEED, "Solver seed changed.")
    require(int(seeds["fresh_noise_seed"]) == materialize.NOISE_SEED, "Noise seed changed.")

    # We intentionally switch the manifest seed from the development seed 71032
    # to the already-frozen sealed-test manifest seed 71033.
    require(int(seeds["sealed_test_manifest_seed"]) == 71033, "Sealed-test manifest seed changed.")

    # Disk preflight uses only a conservative bound; no sealed source is read yet.
    free_bytes = shutil.disk_usage(Path(".")).free
    require(free_bytes >= 8 * 1024**3, f"At least 8 GiB free space is required before opening; found {free_bytes / 1024**3:.2f} GiB.")

    return authorization, g0


def open_or_resume_ledger(authorization: dict[str, Any]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    auth_sha = sha256_file(AUTH_PATH)
    if LEDGER_PATH.exists():
        ledger = load_json(LEDGER_PATH)
        require(ledger["authorization_sha256"] == auth_sha, "Existing opening ledger belongs to a different authorization.")
        require(ledger["test_open_count"] == 1, "Existing opening ledger has an invalid open count.")
        require(ledger["maximum_test_open_count"] == 1, "Existing opening ledger changed the maximum open count.")
        require(ledger["opening_id"] == auth_sha, "Opening ID changed.")
        return ledger

    # This write is the irreversible 0 -> 1 transition.  It occurs before any
    # sealed Phase-3B test manifest is read or any Tier-C-v4 test field is generated.
    ledger = {
        "phase": "4G-R3G1 Tier C v4 single sealed-test opening ledger",
        "protocol_version": PROTOCOL_VERSION,
        "opening_id": auth_sha,
        "authorization_path": str(AUTH_PATH),
        "authorization_sha256": auth_sha,
        "test_open_count_before": 0,
        "test_open_count": 1,
        "maximum_test_open_count": 1,
        "single_opening_consumed": True,
        "opening_status": "in_progress",
        "resume_same_opening_allowed": True,
        "second_opening_authorized": False,
        "test_generation_started": False,
        "test_generation_completed": False,
        "test_metrics_computed": False,
    }
    write_json(LEDGER_PATH, ledger)
    return ledger


def update_ledger(**updates: Any) -> dict[str, Any]:
    ledger = load_json(LEDGER_PATH)
    ledger.update(updates)
    write_json(LEDGER_PATH, ledger)
    return ledger


def load_test_carrier_rows(authorization: dict[str, Any]) -> list[dict[str, str]]:
    rows = load_csv(CARRIER_SPLIT_PATH)
    test_rows = [row for row in rows if row["split"] == "test"]
    test_rows.sort(key=lambda row: int(row["split_position"]))

    require(len(test_rows) == EXPECTED_TEST_CARRIER_COUNT, "Expected 32 frozen Tier C v4 test carriers.")
    require([int(row["split_position"]) for row in test_rows] == list(range(EXPECTED_TEST_CARRIER_COUNT)), "Test split positions changed.")

    observed_ids = [int(row["carrier_id"]) for row in test_rows]
    require(observed_ids == [int(v) for v in authorization["test_carrier_ids"]], "Test carrier IDs changed from the one-time authorization.")
    require(all(row["test_source"] == "unopened_tier_c_v3_test" for row in test_rows), "Test carrier source changed.")
    return test_rows


def configure_field_generator() -> None:
    fieldgen.OUTPUT_DIR = FIELD_DIR
    fieldgen.PRIVILEGED_DIR = FIELD_PRIVILEGED_DIR
    fieldgen.DEVELOPMENT_CARRIER_COUNT = EXPECTED_TEST_CARRIER_COUNT
    fieldgen.TRAIN_CARRIER_COUNT = 0
    fieldgen.VALIDATION_CARRIER_COUNT = 0
    fieldgen.UNIQUE_PHASE_PAIR_COUNT = EXPECTED_TEST_CARRIER_COUNT * 2
    fieldgen.PAIRED_COMPARISON_COUNT = EXPECTED_TEST_CARRIER_COUNT * 2 * 2
    fieldgen.EXPECTED_FIELD_COUNT = EXPECTED_TEST_CARRIER_COUNT * fieldgen.STATE_COUNT


def generate_test_field_bank(authorization: dict[str, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    marker_path = stage_marker("field_bank")
    if marker_path.exists():
        marker = load_json(marker_path)
        require(marker["status"] == "completed", "Existing field-bank stage marker is not completed.")
        for path_string, expected_sha in marker["artifact_hashes"].items():
            verify_file_hash(Path(path_string), expected_sha)
        return marker

    FIELD_PRIVILEGED_DIR.mkdir(parents=True, exist_ok=True)

    configure_field_generator()
    # This validates the frozen numerical construction protocol, not the test outcome.
    fieldgen.validate_protocol()

    torch.manual_seed(fieldgen.SOLVER_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(fieldgen.SOLVER_SEED)
        torch.set_float32_matmul_precision("high")

    carrier_rows = load_test_carrier_rows(authorization)
    parameter_schema = fieldgen.load_parameter_schema()
    state_regimes, state_lookup = fieldgen.load_state_regimes()

    parameters, latent = fieldgen.generate_carrier_parameters(
        carrier_rows=carrier_rows,
        parameter_schema=parameter_schema,
    )
    native_parameter_path = FIELD_DIR / "development_carrier_parameters.csv"
    require(native_parameter_path.exists(), "Native parameter generator did not write its output.")
    test_parameter_path = FIELD_DIR / "test_carrier_parameters.csv"
    shutil.copy2(native_parameter_path, test_parameter_path)

    base_fields, preferred_angles = fieldgen.create_base_fields(
        carrier_rows=carrier_rows,
        parameters=parameters,
        latent=latent,
    )
    base_path = FIELD_PRIVILEGED_DIR / "test_base_fields.npy"
    np.save(base_path, base_fields)

    construction = fieldgen.construct_target_bank(
        carrier_rows=carrier_rows,
        parameters=parameters,
        base_fields=base_fields,
        preferred_angles=preferred_angles,
        state_lookup=state_lookup,
    )
    require(len(construction["failure_rows"]) == 0, "At least one sealed test carrier failed the frozen coarsening construction.")
    require(len(construction["selection_rows"]) == EXPECTED_TEST_CARRIER_COUNT * 2, "Test coarsening selection count changed.")

    selection_rows = construction["selection_rows"]
    length_ratios = np.asarray([float(row["constructor_coarse_to_fine_length_ratio"]) for row in selection_rows], dtype=np.float64)
    interface_ratios = np.asarray([float(row["constructor_coarse_to_fine_interface_ratio"]) for row in selection_rows], dtype=np.float64)
    require(np.all(length_ratios >= fieldgen.INTERNAL_TARGET_LENGTH_RATIO), "A test carrier failed the frozen internal length target.")
    require(np.all(interface_ratios <= fieldgen.INTERNAL_TARGET_INTERFACE_RATIO), "A test carrier failed the frozen internal interface target.")

    final_bank, physics_rows = fieldgen.run_relaxation(
        carrier_rows=carrier_rows,
        parameters=parameters,
        base_fields=base_fields,
        target_bank=construction["target_bank"],
        state_regimes=state_regimes,
        device=device,
        batch_size=batch_size,
    )
    require(final_bank.shape == (EXPECTED_TEST_CARRIER_COUNT, fieldgen.STATE_COUNT, fieldgen.FIELD_CHANNEL_COUNT, fieldgen.GRID_HEIGHT, fieldgen.GRID_WIDTH), "Test raw field-bank shape changed.")
    require(len(physics_rows) == EXPECTED_TEST_CARRIER_COUNT * fieldgen.STATE_COUNT, "Test physics diagnostic count changed.")
    require(np.isfinite(final_bank).all(), "Test raw field bank contains NaN or Inf.")

    normalization = load_json(NORMALIZATION_PATH)
    require(normalization["statistics_source"] == "clean training-carrier fields only", "Normalization source changed.")
    require(normalization["test_statistics_used"] is False, "Frozen normalization used test statistics.")
    mean = np.asarray(normalization["channel_mean"], dtype=np.float32)
    std = np.asarray(normalization["channel_standard_deviation"], dtype=np.float32)
    require(mean.shape == (fieldgen.FIELD_CHANNEL_COUNT,), "Normalization mean shape changed.")
    require(std.shape == (fieldgen.FIELD_CHANNEL_COUNT,), "Normalization std shape changed.")
    require(np.all(std > 0.0), "Invalid frozen normalization scale.")

    normalized_bank = ((final_bank - mean[None, None, :, None, None]) / std[None, None, :, None, None]).astype(np.float32)
    require(np.isfinite(normalized_bank).all(), "Normalized test field bank contains NaN or Inf.")

    carrier_ids = np.asarray([int(row["carrier_id"]) for row in carrier_rows], dtype=np.int64)
    carrier_splits = np.asarray(["test"] * EXPECTED_TEST_CARRIER_COUNT, dtype="<U5")

    ids_path = FIELD_PRIVILEGED_DIR / "test_carrier_ids.npy"
    splits_path = FIELD_PRIVILEGED_DIR / "test_carrier_splits.npy"
    raw_path = FIELD_PRIVILEGED_DIR / "test_carrier_state_fields_raw.npy"
    normalized_path = FIELD_PRIVILEGED_DIR / "test_carrier_state_fields_normalized.npy"
    physics_path = FIELD_PRIVILEGED_DIR / "test_physics_diagnostics.csv"

    np.save(ids_path, carrier_ids)
    np.save(splits_path, carrier_splits)
    np.save(raw_path, final_bank)
    np.save(normalized_path, normalized_bank)
    write_csv(physics_path, physics_rows)

    physics_summary = fieldgen.summarize_physics(physics_rows)
    require(physics_summary["all_values_finite"] is True, "Test physics contains non-finite values.")

    artifact_paths = [
        test_parameter_path,
        base_path,
        FIELD_PRIVILEGED_DIR / "coarsening_iteration_search.csv",
        FIELD_PRIVILEGED_DIR / "coarsening_pair_selection.csv",
        ids_path,
        splits_path,
        raw_path,
        normalized_path,
        physics_path,
    ]
    for path in artifact_paths:
        require(path.exists(), f"Expected test field artifact missing: {path}")

    artifact_hashes = {str(path): sha256_file(path) for path in artifact_paths}
    marker = {
        "phase": "4G-R3G1 Tier C v4 sealed test field-bank generation",
        "protocol_version": PROTOCOL_VERSION,
        "status": "completed",
        "test_open_count": 1,
        "test_carrier_count": EXPECTED_TEST_CARRIER_COUNT,
        "test_carrier_ids": carrier_ids.tolist(),
        "state_count": fieldgen.STATE_COUNT,
        "raw_field_shape": list(final_bank.shape),
        "normalization_source": "clean training-carrier fields only",
        "normalization_path": str(NORMALIZATION_PATH),
        "normalization_sha256": sha256_file(NORMALIZATION_PATH),
        "construction_failure_count": 0,
        "minimum_constructor_length_ratio": float(length_ratios.min()),
        "maximum_constructor_interface_ratio": float(interface_ratios.max()),
        "physics_summary": physics_summary,
        "test_statistics_used_for_normalization": False,
        "artifact_hashes": artifact_hashes,
    }
    write_json(marker_path, marker)
    return marker


def load_combined_field_data() -> dict[str, Any]:
    dev_ids = np.load(DEV_FIELD_DIR / "privileged/development_carrier_ids.npy").astype(np.int64)
    dev_splits = np.load(DEV_FIELD_DIR / "privileged/development_carrier_splits.npy").astype(str)
    dev_raw = np.load(DEV_FIELD_DIR / "privileged/development_carrier_state_fields_raw.npy", mmap_mode="r")
    dev_normalized = np.load(DEV_FIELD_DIR / "privileged/development_carrier_state_fields_normalized.npy", mmap_mode="r")

    test_ids = np.load(FIELD_PRIVILEGED_DIR / "test_carrier_ids.npy").astype(np.int64)
    test_raw = np.load(FIELD_PRIVILEGED_DIR / "test_carrier_state_fields_raw.npy", mmap_mode="r")
    test_normalized = np.load(FIELD_PRIVILEGED_DIR / "test_carrier_state_fields_normalized.npy", mmap_mode="r")

    require(dev_raw.shape[0] == 128 and dev_normalized.shape == dev_raw.shape, "Frozen development field bank shape changed.")
    require(test_raw.shape[0] == EXPECTED_TEST_CARRIER_COUNT and test_normalized.shape == test_raw.shape, "Sealed test field bank shape changed.")

    all_ids = np.concatenate([dev_ids, test_ids])
    require(len(set(all_ids.tolist())) == 160, "Combined Tier C v4 carrier IDs are not unique.")

    # Concatenating these modest field banks (~20 MiB total) keeps the original
    # point-bank indexing logic unchanged and simple.
    raw_bank = np.concatenate([np.asarray(dev_raw), np.asarray(test_raw)], axis=0).astype(np.float32, copy=False)
    normalized_bank = np.concatenate([np.asarray(dev_normalized), np.asarray(test_normalized)], axis=0).astype(np.float32, copy=False)

    carrier_lookup = {int(carrier_id): index for index, carrier_id in enumerate(all_ids.tolist())}
    split_ids = {
        "train": dev_ids[np.where(dev_splits == "train")[0]].astype(np.int64),
        "val": dev_ids[np.where(dev_splits == "val")[0]].astype(np.int64),
        "test": test_ids.astype(np.int64),
    }
    require(len(split_ids["train"]) == 96, "Training carrier count changed.")
    require(len(split_ids["val"]) == 32, "Validation carrier count changed.")
    require(len(split_ids["test"]) == 32, "Test carrier count changed.")

    return {
        "carrier_ids": all_ids,
        "carrier_lookup": carrier_lookup,
        "split_ids": split_ids,
        "raw_bank": raw_bank,
        "normalized_bank": normalized_bank,
    }


def configure_materializer(authorization: dict[str, Any]) -> None:
    materialize.OUTPUT_DIR = OUTPUT_DIR
    materialize.VISIBLE_DIR = VISIBLE_DIR
    materialize.PRIVILEGED_DIR = PRIVILEGED_DIR
    materialize.MANIFEST_DIR = MANIFEST_DIR
    materialize.NOISE_SEED = int(authorization["frozen_generation_seeds"]["fresh_noise_seed"])
    materialize.DEVELOPMENT_MANIFEST_SEED = int(authorization["frozen_generation_seeds"]["sealed_test_manifest_seed"])
    materialize.EXPECTED_SEQUENCE_COUNTS = {cell: EXPECTED_SEQUENCE_COUNT for cell in EXPECTED_TEST_CELLS}


def build_test_specifications() -> dict[str, dict[str, Any]]:
    specifications: dict[str, dict[str, Any]] = {}
    for cell_id in EXPECTED_TEST_CELLS:
        source = materialize.load_source_cell(cell_id)
        require(len(source["ordered_sequence_ids"]) == EXPECTED_SEQUENCE_COUNT, f"{cell_id} source sequence count changed.")
        specifications[cell_id] = {
            "source": source,
            "sequence_ids": source["ordered_sequence_ids"],
            "carrier_split": CELL_CARRIER_SPLIT[cell_id],
        }
    return specifications


def build_test_manifests(cell_id: str, specification: dict[str, Any], split_carrier_ids: dict[str, np.ndarray]) -> dict[str, Any]:
    manifests = materialize.build_fresh_manifests(
        cell_id=cell_id,
        specification=specification,
        split_carrier_ids=split_carrier_ids,
    )
    for row in manifests["sequence_manifest_rows"]:
        row["sealed_test_manifest_seed"] = materialize.DEVELOPMENT_MANIFEST_SEED
        row["tier_c_v4_sealed_test"] = True
    for row in manifests["privileged_trajectory_rows"]:
        row["tier_c_v4_test_carrier"] = (row["carrier_split"] == "test")
        row["tier_c_v4_sealed_test"] = True
    return manifests


def clean_partial_cell(cell_id: str) -> None:
    for path in (
        VISIBLE_DIR / f"{cell_id}_fields.npy",
        VISIBLE_DIR / f"{cell_id}_trajectory_index.csv",
        VISIBLE_DIR / f"{cell_id}_sequence_manifest.csv",
        PRIVILEGED_DIR / f"{cell_id}_trajectory_metadata.csv",
        PRIVILEGED_DIR / f"{cell_id}_point_bank_indices.npy",
    ):
        if path.exists():
            path.unlink()


def materialize_test_cell(
    cell_id: str,
    specification: dict[str, Any],
    field_data: dict[str, Any],
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    chunk_size: int,
) -> dict[str, Any]:
    marker_path = MANIFEST_DIR / f"{cell_id}_complete.json"
    field_path = VISIBLE_DIR / f"{cell_id}_fields.npy"

    if marker_path.exists():
        marker = load_json(marker_path)
        if marker.get("status") == "completed" and field_path.exists():
            verify_file_hash(field_path, marker["field_sha256"])
            return marker
        clean_partial_cell(cell_id)
        marker_path.unlink(missing_ok=True)

    manifests = build_test_manifests(cell_id, specification, field_data["split_ids"])
    point_bank_indices = materialize.convert_pairs_to_bank_indices(
        carrier_state_pairs=manifests["carrier_state_pairs"],
        carrier_lookup=field_data["carrier_lookup"],
    )

    trajectory_index_path = VISIBLE_DIR / f"{cell_id}_trajectory_index.csv"
    sequence_manifest_path = VISIBLE_DIR / f"{cell_id}_sequence_manifest.csv"
    trajectory_metadata_path = PRIVILEGED_DIR / f"{cell_id}_trajectory_metadata.csv"
    point_indices_path = PRIVILEGED_DIR / f"{cell_id}_point_bank_indices.npy"

    write_csv(trajectory_index_path, manifests["visible_trajectory_rows"])
    write_csv(sequence_manifest_path, manifests["sequence_manifest_rows"])
    write_csv(trajectory_metadata_path, manifests["privileged_trajectory_rows"])
    np.save(point_indices_path, point_bank_indices)

    point_count = int(manifests["point_count"])
    stored_shape = (
        len(NOISE_LEVELS),
        point_count,
        materialize.FIELD_CHANNEL_COUNT,
        materialize.GRID_HEIGHT,
        materialize.GRID_WIDTH,
    )
    target = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=stored_shape)

    raw_flat = field_data["raw_bank"].reshape(-1, materialize.FIELD_CHANNEL_COUNT, materialize.GRID_HEIGHT, materialize.GRID_WIDTH)
    normalized_flat = field_data["normalized_bank"].reshape(-1, materialize.FIELD_CHANNEL_COUNT, materialize.GRID_HEIGHT, materialize.GRID_WIDTH)
    mean = normalization_mean[None, :, None, None]
    std = normalization_std[None, :, None, None]
    generators = {
        noise: None if noise == 0.0 else np.random.default_rng(materialize.stable_noise_seed(cell_id, noise))
        for noise in NOISE_LEVELS
    }

    for start in range(0, point_count, chunk_size):
        end = min(point_count, start + chunk_size)
        selected_indices = point_bank_indices[start:end]
        raw_clean = np.asarray(raw_flat[selected_indices], dtype=np.float32)
        normalized_clean = np.asarray(normalized_flat[selected_indices], dtype=np.float32)

        for noise_index, noise in enumerate(NOISE_LEVELS):
            if noise == 0.0:
                values = normalized_clean
            else:
                epsilon = generators[noise].standard_normal(size=raw_clean.shape).astype(np.float32)
                noisy_raw = raw_clean + float(noise) * std * epsilon
                noisy_raw = materialize.postprocess_measurement_noise(noisy_raw)
                values = (noisy_raw - mean) / std
            require(np.isfinite(values).all(), f"{cell_id} contains non-finite values at noise {noise}.")
            target[noise_index, start:end] = values.astype(np.float16)

        if start == 0 or end == point_count or end % 4096 == 0:
            print(f"  {cell_id}: {end}/{point_count} points")

    target.flush()
    del target

    verification = np.load(field_path, mmap_mode="r")
    require(verification.shape == stored_shape, f"{cell_id} stored shape changed.")
    require(verification.dtype == np.float16, f"{cell_id} stored dtype changed.")
    require(np.isfinite(np.asarray(verification[:, :min(16, point_count)], dtype=np.float32)).all(), f"{cell_id} stored sample contains non-finite values.")
    del verification

    source_hashes = {
        name: sha256_file(Path(path_string))
        for name, path_string in specification["source"]["source_paths"].items()
    }
    carrier_split = specification["carrier_split"]
    unique_carriers = len({int(row["carrier_id"]) for row in manifests["privileged_trajectory_rows"]})

    artifact_paths = [trajectory_index_path, sequence_manifest_path, trajectory_metadata_path, point_indices_path, field_path]
    artifact_hashes = {str(path): sha256_file(path) for path in artifact_paths}

    marker = {
        "phase": "4G-R3G1 Tier C v4 sealed test cell materialization",
        "protocol_version": PROTOCOL_VERSION,
        "cell_id": cell_id,
        "status": "completed",
        "test_open_count": 1,
        "sequence_count": len(manifests["sequence_manifest_rows"]),
        "trajectory_count": len(manifests["visible_trajectory_rows"]),
        "point_count": point_count,
        "carrier_split": carrier_split,
        "unique_carrier_count": unique_carriers,
        "noise_levels": list(NOISE_LEVELS),
        "stored_shape": list(stored_shape),
        "stored_dtype": "float16",
        "field_path": str(field_path),
        "field_sha256": artifact_hashes[str(field_path)],
        "sealed_test_manifest_seed": materialize.DEVELOPMENT_MANIFEST_SEED,
        "measurement_noise_seed": materialize.NOISE_SEED,
        "test_manifest_seed_used": True,
        "test_carriers_used": carrier_split == "test",
        "sealed_test_cell_generated": True,
        "test_metrics_computed": False,
        "source_cell": cell_id,
        "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes,
    }
    require(marker["sequence_count"] == EXPECTED_SEQUENCE_COUNT, f"{cell_id} sequence count changed.")
    require(marker["trajectory_count"] == EXPECTED_TRAJECTORY_COUNT, f"{cell_id} trajectory count changed.")
    write_json(marker_path, marker)
    return marker


def generate_test_cells(authorization: dict[str, Any], chunk_size: int) -> list[dict[str, Any]]:
    configure_materializer(authorization)
    VISIBLE_DIR.mkdir(parents=True, exist_ok=True)
    PRIVILEGED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    np.save(VISIBLE_DIR / "noise_levels.npy", np.asarray(NOISE_LEVELS, dtype=np.float32))
    field_data = load_combined_field_data()

    normalization = load_json(NORMALIZATION_PATH)
    mean = np.asarray(normalization["channel_mean"], dtype=np.float32)
    std = np.asarray(normalization["channel_standard_deviation"], dtype=np.float32)
    require(normalization["statistics_source"] == "clean training-carrier fields only", "Normalization source changed.")
    require(normalization["test_statistics_used"] is False, "Test statistics entered normalization.")

    specifications = build_test_specifications()
    markers = []
    for index, cell_id in enumerate(EXPECTED_TEST_CELLS, start=1):
        print(f"[{index}/4] Materializing sealed {cell_id}")
        markers.append(
            materialize_test_cell(
                cell_id=cell_id,
                specification=specifications[cell_id],
                field_data=field_data,
                normalization_mean=mean,
                normalization_std=std,
                chunk_size=chunk_size,
            )
        )
    return markers


def build_artifact_manifest() -> list[dict[str, Any]]:
    excluded = {ARTIFACT_MANIFEST_PATH.resolve(), SUMMARY_PATH.resolve(), LEDGER_PATH.resolve()}
    rows: list[dict[str, Any]] = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        rows.append({
            "relative_path": str(path.relative_to(OUTPUT_DIR)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    require(args.batch_size > 0, "Batch size must be positive.")
    require(args.chunk_size > 0, "Chunk size must be positive.")

    # A completed opening is immutable and cannot be repeated.
    if SUMMARY_PATH.exists():
        summary = load_json(SUMMARY_PATH)
        require(summary["phase4gr3g1_status"] == "sealed_test_generation_frozen", "Existing G1 summary is not a completed freeze.")
        print("Phase 4G-R3G1 is already completed and frozen; refusing a second opening.")
        print(json.dumps(summary, indent=2))
        return

    authorization, _ = verify_pre_open_state()
    ledger = open_or_resume_ledger(authorization)

    # From this point onward the single opening has been consumed, even if the
    # process is interrupted.  A later invocation may only resume this same
    # opening under the same immutable authorization.
    update_ledger(test_generation_started=True)

    device = fieldgen.resolve_device(args.device)
    print(f"Single sealed opening ID: {ledger['opening_id']}")
    print(f"test_open_count: {ledger['test_open_count']}/1")
    print(f"Device: {device}")

    start_time = time.time()
    try:
        field_marker = generate_test_field_bank(
            authorization=authorization,
            device=device,
            batch_size=args.batch_size,
        )
        cell_markers = generate_test_cells(
            authorization=authorization,
            chunk_size=args.chunk_size,
        )

        manifest_rows = build_artifact_manifest()
        write_csv(ARTIFACT_MANIFEST_PATH, manifest_rows)
        artifact_manifest_sha = sha256_file(ARTIFACT_MANIFEST_PATH)

        summary = {
            "phase": "4G-R3G1 Tier C v4 single sealed-test generation and immutable artifact freeze",
            "protocol_version": PROTOCOL_VERSION,
            "opening_id": ledger["opening_id"],
            "authorization_sha256": ledger["authorization_sha256"],
            "test_open_count": 1,
            "maximum_test_open_count": 1,
            "single_opening_consumed": True,
            "second_opening_authorized": False,
            "sealed_test_cells": list(EXPECTED_TEST_CELLS),
            "test_carrier_count": EXPECTED_TEST_CARRIER_COUNT,
            "test_carrier_ids": authorization["test_carrier_ids"],
            "cell_carrier_split_binding": CELL_CARRIER_SPLIT,
            "sequence_count_per_cell": EXPECTED_SEQUENCE_COUNT,
            "trajectory_count_per_cell": EXPECTED_TRAJECTORY_COUNT,
            "noise_levels": list(NOISE_LEVELS),
            "frozen_generation_seeds": authorization["frozen_generation_seeds"],
            "normalization_source": "clean training-carrier fields only",
            "normalization_path": str(NORMALIZATION_PATH),
            "normalization_sha256": sha256_file(NORMALIZATION_PATH),
            "test_field_bank_generated": True,
            "test_manifests_generated": True,
            "test_fields_generated": True,
            "test_artifacts_hashed": True,
            "test_metrics_computed": False,
            "field_bank_marker": field_marker,
            "cell_markers": cell_markers,
            "artifact_count": len(manifest_rows),
            "artifact_manifest_path": str(ARTIFACT_MANIFEST_PATH),
            "artifact_manifest_sha256": artifact_manifest_sha,
            "elapsed_seconds": float(time.time() - start_time),
            "additional_tuning_authorized": False,
            "additional_final_fit_training_authorized": False,
            "checkpoint_reselection_authorized": False,
            "comparator_reselection_authorized": False,
            "test_regeneration_authorized": False,
            "tier_c_v5_authorized": False,
            "phase4gr3g1_status": "sealed_test_generation_frozen",
            "next_phase": "4G-R3G2 Tier C v4 frozen-checkpoint sealed-test evaluation",
        }
        write_json(SUMMARY_PATH, summary)
        update_ledger(
            opening_status="generation_completed",
            resume_same_opening_allowed=True,
            test_generation_completed=True,
            test_metrics_computed=False,
            g1_summary_path=str(SUMMARY_PATH),
            g1_summary_sha256=sha256_file(SUMMARY_PATH),
            artifact_manifest_sha256=artifact_manifest_sha,
        )

        print("Phase 4G-R3G1 sealed-test generation completed and frozen.")
        print(json.dumps(summary, indent=2))

    except Exception as exc:
        update_ledger(
            opening_status="in_progress_generation_interrupted_or_failed",
            test_generation_completed=False,
            last_error_type=type(exc).__name__,
            last_error_message=str(exc),
        )
        raise


if __name__ == "__main__":
    main()

