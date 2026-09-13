from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


PHASE1A = Path("outputs/phase1a_ground_truth")
PHASE1D = Path("outputs/phase1d_compositional_splits")
PHASE3A = Path("outputs/phase3a_tier_b_protocol")
PHASE3B = Path("outputs/phase3b_tier_b_data")
VISIBLE = PHASE3B / "visible"
PRIVILEGED = PHASE3B / "privileged"
OUTPUT = Path("outputs/phase3b_tier_b_audit")

STATE_COUNT = 8
OBS_DIM = 32
CARRIER_DIM = 6
NOISE_LEVELS = (0.0, 0.1, 0.25, 0.5, 1.0)

EXPECTED_CARRIERS = {"train": 96, "val": 32, "test": 32}
EXPECTED_TRAJECTORIES = {
    "train_joint": 11008,
    "val_composition": 1152,
    "val_carrier": 1152,
    "val_joint": 1152,
    "test_iid_pairing": 1152,
    "test_composition": 1152,
    "test_carrier": 1152,
    "test_joint": 1152,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def validate_sources():
    phase3a = load_json(PHASE3A / "phase3a_summary.json")
    generation = load_json(PHASE3B / "phase3b_generation_summary.json")

    if phase3a["status"] != "protocol_frozen":
        raise AssertionError("Phase 3A is not frozen.")
    if generation["generation_status"] != "completed":
        raise AssertionError("Phase 3B generation is incomplete.")
    if generation["test_iid_pairings_reused_from_training"] != 0:
        raise AssertionError("IID test pairing collision remains.")
    if generation["predictive_model_training_performed"] is not False:
        raise AssertionError("Predictive training was unexpectedly performed.")

    return phase3a, generation


def audit_carriers():
    rows = load_csv(PHASE3B / "carriers.csv")
    if len(rows) != 160:
        raise AssertionError(f"Expected 160 carriers, found {len(rows)}.")

    ids = [int(row["carrier_id"]) for row in rows]
    if len(set(ids)) != 160:
        raise AssertionError("Carrier IDs are not unique.")

    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED_CARRIERS:
        raise AssertionError(f"Carrier split mismatch: {dict(counts)}")

    vectors = np.asarray(
        [[float(row[f"u{i}"]) for i in range(CARRIER_DIM)] for row in rows],
        dtype=np.float64,
    )
    if not np.isfinite(vectors).all():
        raise FloatingPointError("Carrier vectors contain non-finite values.")

    coverage = []
    for split in ("train", "val", "test"):
        selected = vectors[[i for i, row in enumerate(rows) if row["split"] == split]]
        distances = np.sqrt(
            np.sum((selected[:, None, :] - selected[None, :, :]) ** 2, axis=-1)
        )
        upper = distances[np.triu_indices(len(selected), k=1)]

        bins = []
        for dimension in range(CARRIER_DIM):
            counts_dim, _ = np.histogram(
                selected[:, dimension],
                bins=8,
                range=(-1.0, 1.0),
            )
            bins.extend(counts_dim.tolist())

        coverage.append(
            {
                "split": split,
                "carrier_count": len(selected),
                "coordinate_minimum": float(selected.min()),
                "coordinate_maximum": float(selected.max()),
                "minimum_pairwise_distance": float(upper.min()),
                "median_pairwise_distance": float(np.median(upper)),
                "minimum_bin_occupancy": int(min(bins)),
                "maximum_bin_occupancy": int(max(bins)),
            }
        )

    write_csv(OUTPUT / "carrier_coverage.csv", coverage)
    return dict(counts)


def load_manifold():
    with np.load(PRIVILEGED / "carrier_state_observations.npz") as data:
        result = {
            "carrier_ids": np.asarray(data["carrier_ids"], dtype=np.int64),
            "carrier_splits": np.asarray(data["carrier_splits"]).astype(str),
            "carrier_vectors": np.asarray(data["carrier_vectors"], dtype=np.float64),
            "raw": np.asarray(data["raw_observations"], dtype=np.float64),
            "normalized": np.asarray(data["normalized_observations"], dtype=np.float64),
        }

    expected = (160, STATE_COUNT, OBS_DIM)
    if result["raw"].shape != expected or result["normalized"].shape != expected:
        raise AssertionError(
            f"Unexpected manifold shapes: raw={result['raw'].shape}, "
            f"normalized={result['normalized'].shape}"
        )
    return result


def audit_manifold(data):
    observations = data["normalized"]
    splits = data["carrier_splits"]

    if not np.isfinite(observations).all():
        raise FloatingPointError("Observation manifold contains non-finite values.")

    train = observations[splits == "train"]
    train_flat = train.reshape(-1, OBS_DIM)
    coordinate_std = train_flat.std(axis=0)

    state_variances = [
        float(np.mean(np.var(train[:, state, :], axis=0)))
        for state in range(STATE_COUNT)
    ]

    cross_state_distances = []
    for values in observations:
        distance = np.sqrt(
            np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=-1)
        )
        cross_state_distances.extend(
            distance[np.triu_indices(STATE_COUNT, k=1)].tolist()
        )

    result = {
        "all_values_finite": True,
        "minimum_train_coordinate_std": float(coordinate_std.min()),
        "maximum_train_coordinate_std": float(coordinate_std.max()),
        "minimum_within_state_variance": float(min(state_variances)),
        "maximum_within_state_variance": float(max(state_variances)),
        "minimum_same_carrier_cross_state_distance": float(
            min(cross_state_distances)
        ),
        "median_same_carrier_cross_state_distance": float(
            np.median(cross_state_distances)
        ),
        "all_coordinates_nonconstant": bool(np.all(coordinate_std > 1e-8)),
        "all_states_have_within_state_variation": bool(
            min(state_variances) > 1e-8
        ),
    }
    write_json(OUTPUT / "observation_manifold_audit.json", result)
    return result


def audit_dataset_files():
    rows = []

    for cell, expected_trajectories in EXPECTED_TRAJECTORIES.items():
        observation_path = VISIBLE / f"{cell}_observations.npz"
        trajectory_path = VISIBLE / f"{cell}_trajectory_index.csv"
        sequence_path = VISIBLE / f"{cell}_sequence_manifest.csv"
        clean_path = PRIVILEGED / f"{cell}_clean_vectors.npy"
        state_path = PRIVILEGED / f"{cell}_state_ids.npy"
        metadata_path = PRIVILEGED / f"{cell}_trajectory_metadata.csv"

        for path in (
            observation_path,
            trajectory_path,
            sequence_path,
            clean_path,
            state_path,
            metadata_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        trajectory_rows = load_csv(trajectory_path)
        metadata_rows = load_csv(metadata_path)

        if len(trajectory_rows) != expected_trajectories:
            raise AssertionError(
                f"{cell}: expected {expected_trajectories} trajectories, "
                f"found {len(trajectory_rows)}"
            )
        if len(metadata_rows) != expected_trajectories:
            raise AssertionError(f"{cell}: privileged metadata count mismatch.")

        point_count = sum(int(row["point_count"]) for row in trajectory_rows)

        with np.load(observation_path) as data:
            if set(data.files) != {"noise_fractions", "vectors"}:
                raise AssertionError(
                    f"{cell}: visible NPZ contains unexpected arrays {data.files}"
                )
            noise = np.asarray(data["noise_fractions"], dtype=np.float64)
            vectors = np.asarray(data["vectors"], dtype=np.float32)

        clean = np.load(clean_path)
        states = np.load(state_path)

        if not np.allclose(noise, NOISE_LEVELS):
            raise AssertionError(f"{cell}: noise levels changed.")
        if vectors.shape != (5, point_count, OBS_DIM):
            raise AssertionError(f"{cell}: vector shape is {vectors.shape}.")
        if clean.shape != (point_count, OBS_DIM):
            raise AssertionError(f"{cell}: clean shape is {clean.shape}.")
        if states.shape != (point_count,):
            raise AssertionError(f"{cell}: state shape is {states.shape}.")
        if not np.array_equal(vectors[0], clean):
            raise AssertionError(f"{cell}: zero-noise data differs from clean data.")
        if not np.isfinite(vectors).all() or not np.isfinite(clean).all():
            raise FloatingPointError(f"{cell}: non-finite data.")
        if not np.all((states >= 0) & (states < STATE_COUNT)):
            raise AssertionError(f"{cell}: invalid state IDs.")

        rows.append(
            {
                "cell_id": cell,
                "trajectory_count": len(trajectory_rows),
                "point_count": point_count,
                "noise_condition_count": 5,
                "observation_dimension": OBS_DIM,
                "visible_contains_clean_target_array": False,
                "zero_noise_matches_clean": True,
                "all_values_finite": True,
                "state_ids_valid": True,
            }
        )

    write_csv(OUTPUT / "dataset_file_audit.csv", rows)
    return rows


def privileged_knn(data):
    observations = data["normalized"]
    splits = data["carrier_splits"]

    train_values = observations[splits == "train"]
    train_x = train_values.reshape(-1, OBS_DIM)
    train_y = np.tile(np.arange(STATE_COUNT, dtype=np.int64), len(train_values))

    rows = []
    for split in ("train", "val", "test"):
        values = observations[splits == split]
        query_x = values.reshape(-1, OBS_DIM)
        query_y = np.tile(np.arange(STATE_COUNT, dtype=np.int64), len(values))

        distances = (
            np.sum(query_x**2, axis=1, keepdims=True)
            + np.sum(train_x**2, axis=1)[None, :]
            - 2.0 * query_x @ train_x.T
        )
        if split == "train":
            np.fill_diagonal(distances, np.inf)

        nearest = np.argpartition(distances, kth=4, axis=1)[:, :5]
        labels = train_y[nearest]

        predictions = np.asarray(
            [
                int(np.argmax(np.bincount(row, minlength=STATE_COUNT)))
                for row in labels
            ],
            dtype=np.int64,
        )
        accuracy = float(np.mean(predictions == query_y))

        rows.append(
            {
                "split": split,
                "carrier_count": len(values),
                "observation_count": len(query_x),
                "decoder": "5-nearest-neighbor",
                "training_carriers_only": True,
                "clean_state_accuracy": accuracy,
                "chance_accuracy": 1.0 / STATE_COUNT,
                "accuracy_above_chance": accuracy - 1.0 / STATE_COUNT,
            }
        )

    write_csv(OUTPUT / "privileged_state_decoder.csv", rows)
    return {row["split"]: row for row in rows}


def load_generators():
    raw = load_json(PHASE1A / "generators.json")
    if isinstance(raw, dict):
        return {
            str(name): tuple(
                int(value)
                for value in (
                    payload["mapping"]
                    if isinstance(payload, dict)
                    else payload
                )
            )
            for name, payload in raw.items()
        }
    return {
        row["name"]: tuple(int(value) for value in row["mapping"])
        for row in raw
    }


def parse_word(text, length):
    if length == 0:
        return tuple()
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed)
    except json.JSONDecodeError:
        pass
    delimiter = "->" if "->" in text else ","
    return tuple(part.strip() for part in text.split(delimiter) if part.strip())


def load_semigroup():
    output = []
    for row in load_csv(PHASE1A / "semigroup_elements.csv"):
        length = int(row["word_length"])
        word = parse_word(row["shortest_word"], length)
        if len(word) != length:
            raise AssertionError("Shortest word length mismatch.")
        output.append(
            {
                "element_id": str(row["element_id"]),
                "mapping": tuple(int(value) for value in json.loads(row["mapping"])),
                "word": word,
            }
        )
    if len(output) != 104:
        raise AssertionError(f"Expected 104 transformations, found {len(output)}.")
    return output


def fit_affine(inputs, targets):
    augmented = np.concatenate(
        [inputs, np.ones((len(inputs), 1), dtype=np.float64)],
        axis=1,
    )
    weights, _, _, _ = np.linalg.lstsq(augmented, targets, rcond=None)
    return weights[:-1], weights[-1]


def nearest_mapping(prediction, references):
    distances = np.sum(
        (prediction[:, None, :] - references[None, :, :]) ** 2,
        axis=-1,
    )
    return tuple(int(value) for value in np.argmin(distances, axis=1))


def affine_shortcut_audit(data):
    observations = data["normalized"]
    splits = data["carrier_splits"]
    generators = load_generators()
    semigroup = load_semigroup()

    transformation_split = {
        str(row["element_id"]): row["split"]
        for row in load_csv(PHASE1D / "transformation_split.csv")
    }
    test_indices = [
        index
        for index, row in enumerate(semigroup)
        if transformation_split[row["element_id"]] == "test"
    ]
    if len(test_indices) != 9:
        raise AssertionError("Expected nine test transformations.")

    train_ids = np.where(splits == "train")[0]
    inputs = observations[train_ids].reshape(-1, OBS_DIM)

    affine = {}
    for name, mapping in generators.items():
        targets = observations[train_ids][
            :, np.asarray(mapping, dtype=np.int64), :
        ].reshape(-1, OBS_DIM)
        affine[name] = fit_affine(inputs, targets)

    primitive_rows = []
    carrier_rows = []

    for split in ("train", "val", "test"):
        carrier_ids = np.where(splits == split)[0]

        for name, mapping in generators.items():
            matrix, bias = affine[name]
            prediction = observations[carrier_ids] @ matrix + bias
            target = observations[carrier_ids][
                :, np.asarray(mapping, dtype=np.int64), :
            ]

            decoded = [
                nearest_mapping(prediction[i], observations[carrier_id])
                for i, carrier_id in enumerate(carrier_ids)
            ]

            primitive_rows.append(
                {
                    "carrier_split": split,
                    "operation": name,
                    "carrier_count": len(carrier_ids),
                    "continuous_mse": float(np.mean((prediction - target) ** 2)),
                    "state_mapping_accuracy": float(
                        np.mean(
                            [
                                predicted == true
                                for result in decoded
                                for predicted, true in zip(result, mapping)
                            ]
                        )
                    ),
                    "exact_operation_rate": float(
                        np.mean([result == mapping for result in decoded])
                    ),
                }
            )

        for carrier_id in carrier_ids:
            primitive_exact = []
            for name, mapping in generators.items():
                matrix, bias = affine[name]
                decoded = nearest_mapping(
                    observations[carrier_id] @ matrix + bias,
                    observations[carrier_id],
                )
                primitive_exact.append(decoded == mapping)

            exact_all = []
            exact_test = []
            for index, transformation in enumerate(semigroup):
                current = observations[carrier_id].copy()
                for operation in transformation["word"]:
                    matrix, bias = affine[operation]
                    current = current @ matrix + bias

                decoded = nearest_mapping(
                    current,
                    observations[carrier_id],
                )
                exact = decoded == transformation["mapping"]
                exact_all.append(exact)
                if index in test_indices:
                    exact_test.append(exact)

            carrier_rows.append(
                {
                    "carrier_id": int(carrier_id),
                    "carrier_split": split,
                    "primitive_exact_rate": float(np.mean(primitive_exact)),
                    "all_primitives_exact": bool(all(primitive_exact)),
                    "all_exact_transformation_rate": float(np.mean(exact_all)),
                    "test_exact_transformation_rate": float(np.mean(exact_test)),
                }
            )

    write_csv(OUTPUT / "affine_primitive_audit.csv", primitive_rows)
    write_csv(OUTPUT / "affine_behavioral_by_carrier.csv", carrier_rows)

    split_results = {}
    for split in ("train", "val", "test"):
        primitive = [row for row in primitive_rows if row["carrier_split"] == split]
        carriers = [row for row in carrier_rows if row["carrier_split"] == split]

        split_results[split] = {
            "carrier_count": len(carriers),
            "primitive_continuous_mse": summarize(
                [row["continuous_mse"] for row in primitive]
            ),
            "primitive_mapping_accuracy": summarize(
                [row["state_mapping_accuracy"] for row in primitive]
            ),
            "primitive_exact_operation_rate": summarize(
                [row["exact_operation_rate"] for row in primitive]
            ),
            "all_exact_transformation_rate": summarize(
                [row["all_exact_transformation_rate"] for row in carriers]
            ),
            "test_exact_transformation_rate": summarize(
                [row["test_exact_transformation_rate"] for row in carriers]
            ),
            "fraction_with_all_primitives_exact": float(
                np.mean([row["all_primitives_exact"] for row in carriers])
            ),
            "fraction_with_all_104_transformations_exact": float(
                np.mean(
                    [
                        row["all_exact_transformation_rate"] == 1.0
                        for row in carriers
                    ]
                )
            ),
        }

    interpolation_fails = bool(
        min(
            split_results["val"]["primitive_continuous_mse"]["mean"],
            split_results["test"]["primitive_continuous_mse"]["mean"],
        )
        > 1e-10
    )
    primitive_fails = bool(
        split_results["val"]["fraction_with_all_primitives_exact"] < 1.0
        or split_results["test"]["fraction_with_all_primitives_exact"] < 1.0
    )
    composition_fails = bool(
        split_results["test"]["fraction_with_all_104_transformations_exact"]
        < 1.0
    )

    result = {
        "affine_interpolation_tolerance": 1e-10,
        "split_results": split_results,
        "heldout_continuous_interpolation_fails": interpolation_fails,
        "at_least_one_heldout_primitive_mapping_fails": primitive_fails,
        "all_104_transformations_not_exact_on_every_test_carrier":
            composition_fails,
        "affine_shortcut_removed": bool(
            interpolation_fails
            and primitive_fails
            and composition_fails
        ),
    }
    write_json(OUTPUT / "affine_shortcut_audit.json", result)
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    _, generation = validate_sources()
    carrier_counts = audit_carriers()
    manifold = load_manifold()
    manifold_result = audit_manifold(manifold)
    dataset_rows = audit_dataset_files()
    decoder = privileged_knn(manifold)
    affine = affine_shortcut_audit(manifold)

    manifold_valid = bool(
        manifold_result["all_coordinates_nonconstant"]
        and manifold_result["all_states_have_within_state_variation"]
        and decoder["val"]["clean_state_accuracy"] > 1.0 / STATE_COUNT
        and decoder["test"]["clean_state_accuracy"] > 1.0 / STATE_COUNT
    )
    files_valid = bool(
        len(dataset_rows) == 8
        and all(
            row["all_values_finite"]
            and row["zero_noise_matches_clean"]
            and not row["visible_contains_clean_target_array"]
            and row["state_ids_valid"]
            for row in dataset_rows
        )
    )
    accepted = bool(
        carrier_counts == EXPECTED_CARRIERS
        and manifold_valid
        and files_valid
        and affine["affine_shortcut_removed"]
    )

    summary = {
        "phase": "3B Tier B generation and pretraining audit",
        "carrier_count": 160,
        "carrier_split_counts": carrier_counts,
        "evaluation_cell_count": len(dataset_rows),
        "total_trajectory_count": generation["total_trajectory_count"],
        "total_point_count": generation["total_point_count"],
        "test_iid_pairings_reused_from_training": 0,
        "all_dataset_values_finite": all(
            row["all_values_finite"] for row in dataset_rows
        ),
        "visible_files_exclude_privileged_clean_arrays": all(
            not row["visible_contains_clean_target_array"]
            for row in dataset_rows
        ),
        "all_observation_coordinates_nonconstant":
            manifold_result["all_coordinates_nonconstant"],
        "all_states_have_within_state_variation":
            manifold_result["all_states_have_within_state_variation"],
        "privileged_clean_state_accuracy": {
            split: decoder[split]["clean_state_accuracy"]
            for split in ("train", "val", "test")
        },
        "chance_state_accuracy": 1.0 / STATE_COUNT,
        "affine_shortcut_removed": affine["affine_shortcut_removed"],
        "heldout_continuous_interpolation_fails":
            affine["heldout_continuous_interpolation_fails"],
        "at_least_one_heldout_primitive_mapping_fails":
            affine["at_least_one_heldout_primitive_mapping_fails"],
        "all_104_transformations_not_exact_on_every_test_carrier":
            affine[
                "all_104_transformations_not_exact_on_every_test_carrier"
            ],
        "predictive_model_training_performed": False,
        "diagnostic_reference_decoder_fitted": True,
        "checkpoint_selection_performed": False,
        "test_used_for_model_selection": False,
        "phase2_outputs_modified": False,
        "tier_b_data_accepted": accepted,
        "phase3b_status": "passed" if accepted else "failed",
    }

    write_json(OUTPUT / "phase3b_summary.json", summary)

    print("Phase 3B Tier B audit completed.")
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {OUTPUT}")


if __name__ == "__main__":
    main()
