from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import phase4fr3b_run_tier_c_v4_tuning as runtime


PROTOCOL_VERSION = "tier_c_v4"

PRIMARY_CELL = "test_joint"
PRIMARY_NOISE = 0.25
PRIMARY_NOISE_INDEX = 2

PRIMARY_MODEL = "OCM"
PRIMARY_COMPARATOR = "B4"

EXPECTED_SEQUENCE_COUNT = 144
EXPECTED_TRAJECTORY_COUNT = 1152
EXPECTED_TRAJECTORIES_PER_SEQUENCE = 8
EXPECTED_SEEDS = [11, 23, 37, 53, 71]

BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 73011

G1_ROOT = Path(
    "outputs/phase4gr3g1_tier_c_v4_sealed_test"
)

G1_SUMMARY = (
    G1_ROOT
    / "phase4gr3g1_sealed_test_generation_summary.json"
)

G1_ARTIFACT_MANIFEST = (
    G1_ROOT
    / "sealed_test_artifact_manifest.csv"
)

FINAL_REGISTRY = Path(
    "outputs/"
    "phase4gr3c_tier_c_v4_final_fit_freeze/"
    "frozen_final_fit_registry.csv"
)

COMPARATOR_PATH = Path(
    "outputs/"
    "phase4gr3d1_tier_c_v4_primary_comparator/"
    "primary_validation_comparator.json"
)

POLICY_PATH = Path(
    "outputs/"
    "phase4dr3_tier_c_v4_predictive_protocol/"
    "tier_c_v4_test_opening_policy.json"
)

G2A_REPORT = Path(
    "outputs/"
    "phase4gr3g2a_tier_c_v4_evaluation_interfaces/"
    "phase4gr3g2a_evaluation_interface_report.json"
)

OUTPUT_DIR = Path(
    "outputs/"
    "phase4gr3g2b_tier_c_v4_primary_confirmatory"
)

TRAJECTORY_RESULTS_PATH = (
    OUTPUT_DIR
    / "primary_trajectory_metrics.csv"
)

SEQUENCE_SEED_RESULTS_PATH = (
    OUTPUT_DIR
    / "primary_sequence_seed_metrics.csv"
)

PAIRED_RESULTS_PATH = (
    OUTPUT_DIR
    / "primary_paired_sequence_metrics.csv"
)

RESULT_PATH = (
    OUTPUT_DIR
    / "primary_confirmatory_result.json"
)

HASH_MANIFEST_PATH = (
    OUTPUT_DIR
    / "result_artifact_hashes.json"
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


def load_csv(
    path: Path,
) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    require(
        bool(rows),
        f"No rows supplied for {path}.",
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
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


def parse_int_list(
    value: str,
) -> list[int]:
    """
    Parse the frozen Tier-B/Tier-C operation-sequence
    serialization into the accepted runtime operation IDs.

    Accepted token forms are intentionally restricted to
    the six frozen primitive operations.
    """

    import ast
    import re

    primitive_map = {
        # Numeric runtime IDs.
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,

        # Frozen primitive names.
        "identity": 0,
        "cycle": 1,
        "pair_collapse": 2,
        "half_collapse": 3,
        "parity_collapse": 4,
        "reset": 5,

        # Common hyphen/space variants.
        "pair-collapse": 2,
        "half-collapse": 3,
        "parity-collapse": 4,

        # Frozen primitive transformation IDs.
        "t006": 0,
        "t020": 1,
        "t003": 2,
        "t001": 3,
        "t005": 4,
        "t000": 5,
    }

    raw = str(value).strip()

    require(
        bool(raw),
        "Empty operation sequence.",
    )

    parsed = None

    # First try JSON.
    try:
        candidate = json.loads(raw)

        if isinstance(
            candidate,
            (list, tuple),
        ):
            parsed = list(candidate)

    except Exception:
        pass

    # Then Python literal syntax, e.g.
    # ('cycle', 'reset') or ['cycle', 'reset'].
    if parsed is None:
        try:
            candidate = ast.literal_eval(raw)

            if isinstance(
                candidate,
                (list, tuple),
            ):
                parsed = list(candidate)

        except Exception:
            pass

    # Finally support the compact textual serialization
    # used by symbolic sequence manifests.
    if parsed is None:
        cleaned = raw

        # Remove common outer delimiters.
        cleaned = cleaned.strip(
            "[](){}"
        )

        # Normalize arrows and common delimiters.
        cleaned = cleaned.replace(
            "->",
            " ",
        )
        cleaned = cleaned.replace(
            "",
            " ",
        )

        parsed = [
            token
            for token in re.split(
                r"[\s,;|]+",
                cleaned,
            )
            if token
        ]

    result = []

    for item in parsed:
        # Integer values can pass directly.
        if isinstance(
            item,
            (int, np.integer),
        ):
            operation_id = int(item)

        else:
            token = str(item).strip()

            # Remove quotes left by compact serialization.
            token = token.strip(
                "'\""
            )

            normalized = (
                token
                .lower()
                .replace(" ", "_")
            )

            require(
                normalized
                in primitive_map,
                (
                    "Unknown frozen operation token: "
                    f"{token!r}"
                ),
            )

            operation_id = (
                primitive_map[
                    normalized
                ]
            )

        require(
            0 <= operation_id
            < int(
                runtime.OPERATION_COUNT
            ),
            (
                "Parsed operation ID is outside "
                f"the frozen primitive range: "
                f"{operation_id}"
            ),
        )

        result.append(
            operation_id
        )

    return result


class SealedTestCell:
    def __init__(
        self,
        cell_id: str,
    ) -> None:
        self.cell_id = cell_id

        visible = (
            G1_ROOT
            / "visible"
        )

        self.fields_path = (
            visible
            / f"{cell_id}_fields.npy"
        )

        index_path = (
            visible
            / f"{cell_id}_trajectory_index.csv"
        )

        manifest_path = (
            visible
            / f"{cell_id}_sequence_manifest.csv"
        )

        for path in (
            self.fields_path,
            index_path,
            manifest_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        self.fields = np.load(
            self.fields_path,
            mmap_mode="r",
        )

        require(
            self.fields.shape[0] == 5,
            "Expected five frozen noise levels.",
        )

        self.rows = []

        for row in load_csv(
            index_path
        ):
            self.rows.append(
                {
                    "trajectory_index":
                        int(
                            row[
                                "trajectory_index"
                            ]
                        ),

                    "cell_sequence_index":
                        int(
                            row[
                                "cell_sequence_index"
                            ]
                        ),

                    "source_sequence_id":
                        str(
                            row[
                                "source_sequence_id"
                            ]
                        ),

                    "point_start":
                        int(
                            row[
                                "point_start"
                            ]
                        ),

                    "point_count":
                        int(
                            row[
                                "point_count"
                            ]
                        ),

                    "sequence_length":
                        int(
                            row[
                                "sequence_length"
                            ]
                        ),
                }
            )

        require(
            len(self.rows)
            == EXPECTED_TRAJECTORY_COUNT,
            "Unexpected test trajectory count.",
        )

        manifest_rows = load_csv(
            manifest_path
        )

        require(
            len(manifest_rows)
            == EXPECTED_SEQUENCE_COUNT,
            "Unexpected test sequence count.",
        )

        # Resolve the operation-sequence column from the
        # frozen manifest rather than assuming a column name.
        #
        # A valid operation column must:
        #   1. parse as a JSON integer list,
        #   2. contain only frozen primitive IDs 0..5, and
        #   3. have exactly the frozen sequence length for
        #      every one of the 144 sequences.
        expected_lengths = {}

        for trajectory_row in self.rows:
            sequence_index = int(
                trajectory_row[
                    "cell_sequence_index"
                ]
            )

            sequence_length = int(
                trajectory_row[
                    "sequence_length"
                ]
            )

            if sequence_index in expected_lengths:
                require(
                    expected_lengths[
                        sequence_index
                    ] == sequence_length,
                    (
                        "Trajectories belonging to one "
                        "sequence disagree on sequence length."
                    ),
                )
            else:
                expected_lengths[
                    sequence_index
                ] = sequence_length

        manifest_columns = list(
            manifest_rows[0].keys()
        )

        preferred_operation_columns = [
            "operation_ids",
            "operation_id_sequence",
            "operation_sequence",
            "operations",
            "operator_ids",
            "primitive_operation_ids",
            "primitive_ids",
            "generator_ids",
            "generator_sequence",
            "word",
        ]

        def column_is_valid_operation_sequence(
            column_name,
        ):
            for manifest_row in manifest_rows:
                sequence_index = int(
                    manifest_row[
                        "cell_sequence_index"
                    ]
                )

                value = manifest_row.get(
                    column_name
                )

                if value is None:
                    return False

                try:
                    values = parse_int_list(
                        value
                    )
                except Exception:
                    return False

                if (
                    len(values)
                    != expected_lengths[
                        sequence_index
                    ]
                ):
                    return False

                if any(
                    operation_id < 0
                    or operation_id
                    >= int(
                        runtime.OPERATION_COUNT
                    )
                    for operation_id
                    in values
                ):
                    return False

            return True

        preferred_matches = [
            column
            for column
            in preferred_operation_columns
            if (
                column in manifest_columns
                and column_is_valid_operation_sequence(
                    column
                )
            )
        ]

        if len(
            preferred_matches
        ) == 1:
            operation_column = (
                preferred_matches[0]
            )

        elif len(
            preferred_matches
        ) > 1:
            raise AssertionError(
                "Multiple preferred operation columns "
                "match the frozen manifest: "
                + ", ".join(
                    preferred_matches
                )
            )

        else:
            inferred_matches = [
                column
                for column
                in manifest_columns
                if column_is_valid_operation_sequence(
                    column
                )
            ]

            require(
                len(
                    inferred_matches
                ) == 1,
                (
                    "Could not uniquely identify the "
                    "frozen operation-sequence column. "
                    f"Manifest columns={manifest_columns}; "
                    f"valid candidates={inferred_matches}"
                ),
            )

            operation_column = (
                inferred_matches[0]
            )

        print(
            "Resolved frozen operation column:",
            operation_column,
        )

        self.sequence_lookup = {}

        for row in manifest_rows:
            index = int(
                row[
                    "cell_sequence_index"
                ]
            )

            operation_ids = parse_int_list(
                row[
                    operation_column
                ]
            )

            require(
                len(
                    operation_ids
                )
                == expected_lengths[
                    index
                ],
                (
                    "Resolved operation sequence "
                    "length changed."
                ),
            )

            require(
                all(
                    0 <= operation_id
                    < int(
                        runtime.OPERATION_COUNT
                    )
                    for operation_id
                    in operation_ids
                ),
                (
                    "Resolved operation sequence "
                    "contains invalid primitive ID."
                ),
            )

            self.sequence_lookup[index] = {
                "source_sequence_id":
                    str(
                        row[
                            "source_sequence_id"
                        ]
                    ),

                "operation_ids":
                    operation_ids,
            }

        self.maximum_sequence_length = max(
            row[
                "sequence_length"
            ]
            for row in self.rows
        )

        counts = defaultdict(int)

        for row in self.rows:
            counts[
                row[
                    "cell_sequence_index"
                ]
            ] += 1

        require(
            len(counts)
            == EXPECTED_SEQUENCE_COUNT,
            "Sequence grouping count changed.",
        )

        require(
            all(
                value
                == EXPECTED_TRAJECTORIES_PER_SEQUENCE
                for value in counts.values()
            ),
            (
                "Expected exactly eight trajectories "
                "per sequence."
            ),
        )

    def make_batch(
        self,
        trajectory_indices: list[int],
        noise_index: int,
        device: torch.device,
    ) -> dict[str, Any]:
        selected_rows = [
            self.rows[index]
            for index in trajectory_indices
        ]

        lengths = np.asarray(
            [
                row[
                    "sequence_length"
                ]
                for row in selected_rows
            ],
            dtype=np.int64,
        )

        maximum_length = int(
            lengths.max()
        )

        require(
            maximum_length >= 1,
            (
                "Primary test unexpectedly contains "
                "zero-step-only batch."
            ),
        )

        batch_size = len(
            selected_rows
        )

        noisy = np.zeros(
            (
                batch_size,
                maximum_length + 1,
                4,
                32,
                32,
            ),
            dtype=np.float32,
        )

        clean = np.zeros_like(
            noisy
        )

        operation_ids = np.zeros(
            (
                batch_size,
                maximum_length,
            ),
            dtype=np.int64,
        )

        point_mask = np.zeros(
            (
                batch_size,
                maximum_length + 1,
            ),
            dtype=np.bool_,
        )

        step_mask = np.zeros(
            (
                batch_size,
                maximum_length,
            ),
            dtype=np.bool_,
        )

        sequence_indices = []
        source_sequence_ids = []
        actual_trajectory_indices = []

        for batch_index, row in enumerate(
            selected_rows
        ):
            point_start = row[
                "point_start"
            ]

            point_count = row[
                "point_count"
            ]

            sequence_length = row[
                "sequence_length"
            ]

            noisy_values = np.asarray(
                self.fields[
                    noise_index,
                    point_start:
                    point_start + point_count,
                ],
                dtype=np.float32,
            )

            clean_values = np.asarray(
                self.fields[
                    0,
                    point_start:
                    point_start + point_count,
                ],
                dtype=np.float32,
            )

            require(
                noisy_values.shape
                == (
                    point_count,
                    4,
                    32,
                    32,
                ),
                "Noisy field shape changed.",
            )

            require(
                clean_values.shape
                == noisy_values.shape,
                "Clean field shape changed.",
            )

            require(
                np.isfinite(
                    noisy_values
                ).all(),
                "Non-finite noisy test field.",
            )

            require(
                np.isfinite(
                    clean_values
                ).all(),
                "Non-finite clean test field.",
            )

            noisy[
                batch_index,
                :point_count,
            ] = noisy_values

            clean[
                batch_index,
                :point_count,
            ] = clean_values

            point_mask[
                batch_index,
                :point_count,
            ] = True

            manifest = (
                self.sequence_lookup[
                    row[
                        "cell_sequence_index"
                    ]
                ]
            )

            operations = manifest[
                "operation_ids"
            ]

            require(
                len(operations)
                == sequence_length,
                (
                    "Operation count and trajectory "
                    "length disagree."
                ),
            )

            operation_ids[
                batch_index,
                :sequence_length,
            ] = operations

            step_mask[
                batch_index,
                :sequence_length,
            ] = True

            sequence_indices.append(
                row[
                    "cell_sequence_index"
                ]
            )

            source_sequence_ids.append(
                row[
                    "source_sequence_id"
                ]
            )

            actual_trajectory_indices.append(
                row[
                    "trajectory_index"
                ]
            )

        return {
            "observations":
                torch.from_numpy(
                    noisy
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),

            "clean_observations":
                torch.from_numpy(
                    clean
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),

            "operation_ids":
                torch.from_numpy(
                    operation_ids
                ).to(
                    device=device,
                    dtype=torch.long,
                ),

            "point_mask":
                torch.from_numpy(
                    point_mask
                ).to(
                    device=device,
                    dtype=torch.bool,
                ),

            "step_mask":
                torch.from_numpy(
                    step_mask
                ).to(
                    device=device,
                    dtype=torch.bool,
                ),

            "sequence_lengths":
                torch.from_numpy(
                    lengths
                ).to(
                    device=device,
                    dtype=torch.long,
                ),

            "cell_sequence_indices":
                sequence_indices,

            "source_sequence_ids":
                source_sequence_ids,

            "trajectory_indices":
                actual_trajectory_indices,
        }


@torch.no_grad()
def predict_fields(
    model_id: str,
    model: torch.nn.Module,
    codec: torch.nn.Module,
    configuration: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    fields = batch[
        "observations"
    ]

    operation_ids = batch[
        "operation_ids"
    ]

    step_mask = batch[
        "step_mask"
    ]

    batch_size = int(
        fields.shape[0]
    )

    maximum_length = int(
        operation_ids.shape[1]
    )

    impl = runtime.accepted_impl

    if model_id == "OCM":
        temperature = float(
            configuration[
                "temperature"
            ]
        )

        latent_state_count = int(
            configuration[
                "latent_state_count"
            ]
        )

        encoded_fields = (
            impl.encode_all_fields(
                codec,
                fields,
            )
        )

        time_count = int(
            encoded_fields.shape[1]
        )

        flat_encoded = (
            encoded_fields.reshape(
                -1,
                runtime.OBSERVATION_DIMENSION,
            )
        )

        distributions_flat = (
            model.encode(
                flat_encoded,
                temperature=temperature,
            )
        )

        distributions = (
            distributions_flat.reshape(
                batch_size,
                time_count,
                latent_state_count,
            )
        )

        source_distributions = (
            distributions[
                :,
                :-1,
                :,
            ].reshape(
                batch_size
                * maximum_length,
                latent_state_count,
            )
        )

        flat_operations = (
            operation_ids.reshape(
                batch_size
                * maximum_length
            )
        )

        proposed_flat = (
            model.apply_operation(
                latent_distribution=(
                    source_distributions
                ),
                operation_ids=(
                    flat_operations
                ),
                temperature=temperature,
            )
        )

        predicted_step_latent = (
            model.decode(
                proposed_flat
            ).reshape(
                batch_size,
                maximum_length,
                runtime.OBSERVATION_DIMENSION,
            )
        )

        predicted_step_fields = (
            impl.decode_temporal_latent(
                codec,
                predicted_step_latent,
            )
        )

        final_distribution = (
            model.rollout(
                initial_distribution=(
                    distributions[
                        :,
                        0,
                        :,
                    ]
                ),
                operation_ids=(
                    operation_ids
                ),
                step_mask=step_mask,
                temperature=temperature,
            )
        )

        final_latent = (
            model.decode(
                final_distribution
            )
        )

        final_prediction = (
            codec.decode(
                final_latent
            )
        )

        return (
            predicted_step_fields,
            final_prediction,
        )

    encoded = (
        impl.encode_all_fields(
            codec,
            fields,
        )
    )

    source_latent = (
        encoded[
            :,
            :-1,
            :,
        ].reshape(
            batch_size
            * maximum_length,
            runtime.OBSERVATION_DIMENSION,
        )
    )

    flat_operations = (
        operation_ids.reshape(
            batch_size
            * maximum_length
        )
    )

    one_step_mask = torch.ones(
        (
            batch_size
            * maximum_length,
            1,
        ),
        dtype=torch.bool,
        device=fields.device,
    )

    predicted_step_latent = (
        model.rollout(
            initial_observation=(
                source_latent
            ),
            operation_ids=(
                flat_operations[:, None]
            ),
            step_mask=one_step_mask,
        )
    )

    predicted_step_fields = (
        codec.decode(
            predicted_step_latent
        ).reshape(
            batch_size,
            maximum_length,
            4,
            32,
            32,
        )
    )

    final_latent = model.rollout(
        initial_observation=(
            encoded[
                :,
                0,
                :,
            ]
        ),
        operation_ids=operation_ids,
        step_mask=step_mask,
    )

    final_prediction = (
        codec.decode(
            final_latent
        )
    )

    return (
        predicted_step_fields,
        final_prediction,
    )


def per_trajectory_metrics(
    predicted_step_fields: torch.Tensor,
    final_prediction: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, np.ndarray]:
    noisy = batch[
        "observations"
    ]

    clean = batch[
        "clean_observations"
    ]

    step_mask = batch[
        "step_mask"
    ]

    lengths = batch[
        "sequence_lengths"
    ]

    batch_size = int(
        noisy.shape[0]
    )

    device = noisy.device

    indices = torch.arange(
        batch_size,
        device=device,
    )

    noisy_final = noisy[
        indices,
        lengths,
    ]

    clean_final = clean[
        indices,
        lengths,
    ]

    noisy_rollout = (
        (
            final_prediction
            - noisy_final
        )
        .pow(2)
        .mean(
            dim=(1, 2, 3)
        )
    )

    clean_rollout = (
        (
            final_prediction
            - clean_final
        )
        .pow(2)
        .mean(
            dim=(1, 2, 3)
        )
    )

    noisy_step_values = (
        (
            predicted_step_fields
            - noisy[:, 1:, ...]
        )
        .pow(2)
        .mean(
            dim=(2, 3, 4)
        )
    )

    clean_step_values = (
        (
            predicted_step_fields
            - clean[:, 1:, ...]
        )
        .pow(2)
        .mean(
            dim=(2, 3, 4)
        )
    )

    mask_float = (
        step_mask.to(
            dtype=torch.float32
        )
    )

    denominator = (
        mask_float.sum(
            dim=1
        )
    )

    require(
        bool(
            torch.all(
                denominator > 0
            )
        ),
        (
            "Primary test contains "
            "zero-step trajectory."
        ),
    )

    noisy_one_step = (
        (
            noisy_step_values
            * mask_float
        ).sum(
            dim=1
        )
        / denominator
    )

    clean_one_step = (
        (
            clean_step_values
            * mask_float
        ).sum(
            dim=1
        )
        / denominator
    )

    return {
        "noisy_target_rollout_mse":
            noisy_rollout.detach(
            ).cpu().numpy(),

        "clean_target_rollout_mse":
            clean_rollout.detach(
            ).cpu().numpy(),

        "noisy_target_one_step_mse":
            noisy_one_step.detach(
            ).cpu().numpy(),

        "clean_target_one_step_mse":
            clean_one_step.detach(
            ).cpu().numpy(),
    }


def resolve_configuration_map(
) -> dict[str, dict[str, Any]]:
    records = (
        runtime.load_configurations()
    )

    result = {}

    for record in records:
        configuration_id = str(
            record[
                "configuration_id"
            ]
        )

        require(
            configuration_id
            not in result,
            (
                "Duplicate configuration ID: "
                f"{configuration_id}"
            ),
        )

        result[
            configuration_id
        ] = record[
            "configuration"
        ]

    return result


def determine_training_maximum_length(
) -> int:
    train_cell = runtime.TrajectoryCell(
        "train_joint",
        11008,
    )

    validation_cell = (
        runtime.TrajectoryCell(
            "val_joint",
            1152,
        )
    )

    maximum = max(
        int(
            train_cell
            .maximum_sequence_length
        ),
        int(
            validation_cell
            .maximum_sequence_length
        ),
    )

    del train_cell
    del validation_cell

    return maximum


def choose_primary_rows(
    registry_rows: list[
        dict[str, str]
    ],
) -> list[dict[str, str]]:
    selected = [
        row
        for row in registry_rows
        if (
            row[
                "model_id"
            ]
            in {
                PRIMARY_MODEL,
                PRIMARY_COMPARATOR,
            }
            and abs(
                float(
                    row[
                        "noise_fraction"
                    ]
                )
                - PRIMARY_NOISE
            ) < 1e-12
        )
    ]

    require(
        len(selected) == 10,
        (
            "Expected exactly 10 primary "
            "neural checkpoints."
        ),
    )

    for model_id in (
        PRIMARY_MODEL,
        PRIMARY_COMPARATOR,
    ):
        rows = [
            row
            for row in selected
            if row[
                "model_id"
            ] == model_id
        ]

        observed_seeds = sorted(
            int(
                row[
                    "effective_seed"
                ]
            )
            for row in rows
        )

        require(
            observed_seeds
            == EXPECTED_SEEDS,
            (
                f"{model_id} primary seed "
                f"set changed: {observed_seeds}"
            ),
        )

    selected.sort(
        key=lambda row: (
            row[
                "model_id"
            ],
            int(
                row[
                    "effective_seed"
                ]
            ),
        )
    )

    return selected


@torch.no_grad()
def evaluate_checkpoint(
    registry_row: dict[str, str],
    cell: SealedTestCell,
    configuration_map: dict[
        str,
        dict[str, Any],
    ],
    maximum_sequence_length: int,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    model_id = registry_row[
        "model_id"
    ]

    seed = int(
        registry_row[
            "effective_seed"
        ]
    )

    configuration_id = (
        registry_row[
            "configuration_id"
        ]
    )

    require(
        configuration_id
        in configuration_map,
        (
            "Configuration missing: "
            f"{configuration_id}"
        ),
    )

    configuration = (
        configuration_map[
            configuration_id
        ]
    )

    checkpoint_path = Path(
        registry_row[
            "best_checkpoint_path"
        ]
    )

    require(
        checkpoint_path.exists(),
        (
            "Checkpoint missing: "
            f"{checkpoint_path}"
        ),
    )

    require(
        sha256_file(
            checkpoint_path
        )
        == registry_row[
            "best_checkpoint_sha256"
        ],
        (
            "Frozen checkpoint hash "
            f"changed: {checkpoint_path}"
        ),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    codec, model, optimizer = (
        runtime.build_model_bundle(
            model_id=model_id,
            configuration=configuration,
            maximum_sequence_length=(
                maximum_sequence_length
            ),
            device=device,
        )
    )

    del optimizer

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

    codec.eval()
    model.eval()

    rows = []

    consistency_checks = 0

    for start in range(
        0,
        len(cell.rows),
        batch_size,
    ):
        selected = list(
            range(
                start,
                min(
                    start + batch_size,
                    len(cell.rows),
                ),
            )
        )

        batch = cell.make_batch(
            trajectory_indices=selected,
            noise_index=PRIMARY_NOISE_INDEX,
            device=device,
        )

        (
            predicted_steps,
            final_prediction,
        ) = predict_fields(
            model_id=model_id,
            model=model,
            codec=codec,
            configuration=configuration,
            batch=batch,
        )

        metrics = (
            per_trajectory_metrics(
                predicted_step_fields=(
                    predicted_steps
                ),
                final_prediction=(
                    final_prediction
                ),
                batch=batch,
            )
        )

        # Verify our per-trajectory implementation against
        # the already-frozen accepted scalar loss code.
        accepted_batch = {
            "observations":
                batch[
                    "observations"
                ],

            "operation_ids":
                batch[
                    "operation_ids"
                ],

            "point_mask":
                batch[
                    "point_mask"
                ],

            "step_mask":
                batch[
                    "step_mask"
                ],

            "sequence_lengths":
                batch[
                    "sequence_lengths"
                ],
        }

        accepted_losses = (
            runtime.compute_losses(
                model_id=model_id,
                model=model,
                codec=codec,
                batch=accepted_batch,
                configuration=configuration,
            )
        )

        observed_rollout = float(
            np.mean(
                metrics[
                    "noisy_target_rollout_mse"
                ]
            )
        )

        observed_one_step = float(
            np.mean(
                metrics[
                    "noisy_target_one_step_mse"
                ]
            )
        )

        accepted_rollout = float(
            accepted_losses[
                "rollout_field_loss"
            ].detach().cpu()
        )

        accepted_one_step = float(
            accepted_losses[
                "one_step_field_loss"
            ].detach().cpu()
        )

        require(
            abs(
                observed_rollout
                - accepted_rollout
            ) < 5e-5,
            (
                "Per-trajectory rollout metric "
                "does not reproduce frozen runtime: "
                f"{observed_rollout} vs "
                f"{accepted_rollout}"
            ),
        )

        require(
            abs(
                observed_one_step
                - accepted_one_step
            ) < 5e-5,
            (
                "Per-trajectory one-step metric "
                "does not reproduce frozen runtime: "
                f"{observed_one_step} vs "
                f"{accepted_one_step}"
            ),
        )

        consistency_checks += 1

        for local_index in range(
            len(selected)
        ):
            rows.append(
                {
                    "model_id":
                        model_id,

                    "configuration_id":
                        configuration_id,

                    "effective_seed":
                        seed,

                    "cell_id":
                        PRIMARY_CELL,

                    "noise_fraction":
                        PRIMARY_NOISE,

                    "trajectory_index":
                        batch[
                            "trajectory_indices"
                        ][local_index],

                    "cell_sequence_index":
                        batch[
                            "cell_sequence_indices"
                        ][local_index],

                    "source_sequence_id":
                        batch[
                            "source_sequence_ids"
                        ][local_index],

                    "noisy_target_rollout_mse":
                        float(
                            metrics[
                                "noisy_target_rollout_mse"
                            ][local_index]
                        ),

                    "clean_target_rollout_mse":
                        float(
                            metrics[
                                "clean_target_rollout_mse"
                            ][local_index]
                        ),

                    "noisy_target_one_step_mse":
                        float(
                            metrics[
                                "noisy_target_one_step_mse"
                            ][local_index]
                        ),

                    "clean_target_one_step_mse":
                        float(
                            metrics[
                                "clean_target_one_step_mse"
                            ][local_index]
                        ),
                }
            )

    require(
        len(rows)
        == EXPECTED_TRAJECTORY_COUNT,
        (
            "Checkpoint evaluation produced "
            "wrong trajectory count."
        ),
    )

    print(
        f"  {model_id} seed={seed}: "
        f"{len(rows)} trajectories, "
        f"{consistency_checks} runtime checks"
    )

    del model
    del codec
    del checkpoint

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


def aggregate_sequences(
    trajectory_rows: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    metric_names = [
        "noisy_target_rollout_mse",
        "clean_target_rollout_mse",
        "noisy_target_one_step_mse",
        "clean_target_one_step_mse",
    ]

    groups = defaultdict(list)

    for row in trajectory_rows:
        key = (
            row[
                "model_id"
            ],
            int(
                row[
                    "effective_seed"
                ]
            ),
            int(
                row[
                    "cell_sequence_index"
                ]
            ),
            str(
                row[
                    "source_sequence_id"
                ]
            ),
        )

        groups[key].append(row)

    expected_group_count = (
        2
        * 5
        * EXPECTED_SEQUENCE_COUNT
    )

    require(
        len(groups)
        == expected_group_count,
        (
            "Unexpected sequence/seed grouping "
            f"count: {len(groups)}"
        ),
    )

    result = []

    for (
        model_id,
        seed,
        sequence_index,
        source_sequence_id,
    ), rows in sorted(
        groups.items()
    ):
        require(
            len(rows)
            == EXPECTED_TRAJECTORIES_PER_SEQUENCE,
            (
                "Sequence does not contain "
                "eight trajectories."
            ),
        )

        record = {
            "model_id":
                model_id,

            "effective_seed":
                seed,

            "cell_id":
                PRIMARY_CELL,

            "noise_fraction":
                PRIMARY_NOISE,

            "cell_sequence_index":
                sequence_index,

            "source_sequence_id":
                source_sequence_id,

            "trajectory_count":
                len(rows),
        }

        for metric in metric_names:
            record[
                metric
            ] = float(
                np.mean(
                    [
                        float(
                            row[
                                metric
                            ]
                        )
                        for row in rows
                    ]
                )
            )

        result.append(
            record
        )

    return result


def paired_primary_analysis(
    sequence_seed_rows: list[
        dict[str, Any]
    ],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    groups = defaultdict(list)

    for row in sequence_seed_rows:
        groups[
            (
                row[
                    "model_id"
                ],
                int(
                    row[
                        "cell_sequence_index"
                    ]
                ),
                str(
                    row[
                        "source_sequence_id"
                    ]
                ),
            )
        ].append(row)

    paired_rows = []

    ocm_values = []
    b4_values = []

    for sequence_index in range(
        EXPECTED_SEQUENCE_COUNT
    ):
        candidates = [
            key
            for key in groups
            if (
                key[1]
                == sequence_index
            )
        ]

        source_ids = {
            key[2]
            for key in candidates
        }

        require(
            len(source_ids) == 1,
            (
                "Primary paired sequence "
                "source ID is ambiguous."
            ),
        )

        source_sequence_id = next(
            iter(source_ids)
        )

        per_model = {}

        for model_id in (
            PRIMARY_MODEL,
            PRIMARY_COMPARATOR,
        ):
            rows = groups[
                (
                    model_id,
                    sequence_index,
                    source_sequence_id,
                )
            ]

            require(
                len(rows) == 5,
                (
                    f"{model_id} sequence "
                    f"{sequence_index} does not "
                    "have five seeds."
                ),
            )

            seeds = sorted(
                int(
                    row[
                        "effective_seed"
                    ]
                )
                for row in rows
            )

            require(
                seeds == EXPECTED_SEEDS,
                (
                    "Primary neural seed set "
                    "changed."
                ),
            )

            per_model[
                model_id
            ] = float(
                np.mean(
                    [
                        float(
                            row[
                                "noisy_target_rollout_mse"
                            ]
                        )
                        for row in rows
                    ]
                )
            )

        ocm = per_model[
            PRIMARY_MODEL
        ]

        b4 = per_model[
            PRIMARY_COMPARATOR
        ]

        difference = (
            ocm - b4
        )

        ocm_values.append(
            ocm
        )

        b4_values.append(
            b4
        )

        paired_rows.append(
            {
                "cell_sequence_index":
                    sequence_index,

                "source_sequence_id":
                    source_sequence_id,

                "ocm_seed_averaged_noisy_target_rollout_mse":
                    ocm,

                "b4_seed_averaged_noisy_target_rollout_mse":
                    b4,

                "ocm_minus_b4":
                    difference,

                "ocm_better":
                    bool(
                        difference < 0.0
                    ),
            }
        )

    ocm_array = np.asarray(
        ocm_values,
        dtype=np.float64,
    )

    b4_array = np.asarray(
        b4_values,
        dtype=np.float64,
    )

    differences = (
        ocm_array
        - b4_array
    )

    observed_mean_difference = float(
        differences.mean()
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    bootstrap_means = np.empty(
        BOOTSTRAP_REPLICATES,
        dtype=np.float64,
    )

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):
        indices = rng.integers(
            0,
            EXPECTED_SEQUENCE_COUNT,
            size=(
                EXPECTED_SEQUENCE_COUNT
            ),
        )

        bootstrap_means[
            replicate
        ] = float(
            differences[
                indices
            ].mean()
        )

    ci_lower, ci_upper = (
        np.percentile(
            bootstrap_means,
            [2.5, 97.5],
        )
    )

    superiority_supported = bool(
        observed_mean_difference < 0.0
        and float(ci_upper) < 0.0
    )

    ocm_mean = float(
        ocm_array.mean()
    )

    b4_mean = float(
        b4_array.mean()
    )

    relative_improvement = float(
        (
            b4_mean
            - ocm_mean
        )
        / b4_mean
    )

    analysis = {
        "primary_model":
            PRIMARY_MODEL,

        "primary_comparator":
            PRIMARY_COMPARATOR,

        "cell":
            PRIMARY_CELL,

        "noise_fraction":
            PRIMARY_NOISE,

        "metric":
            "noisy_target_rollout_mse",

        "statistical_unit":
            "sequence",

        "sequence_count":
            EXPECTED_SEQUENCE_COUNT,

        "neural_seed_count_per_model":
            5,

        "neural_seed_aggregation":
            (
                "Average each sequence metric "
                "across five frozen seeds before "
                "paired comparison."
            ),

        "ocm_mean":
            ocm_mean,

        "b4_mean":
            b4_mean,

        "ocm_minus_b4_mean":
            observed_mean_difference,

        "relative_improvement_ocm_vs_b4":
            relative_improvement,

        "ocm_better_sequence_count":
            int(
                np.sum(
                    differences < 0.0
                )
            ),

        "ocm_better_sequence_fraction":
            float(
                np.mean(
                    differences < 0.0
                )
            ),

        "paired_difference_standard_deviation":
            float(
                differences.std(
                    ddof=1
                )
            ),

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "confidence_interval":
            "two-sided 95% percentile",

        "ci_95_lower":
            float(
                ci_lower
            ),

        "ci_95_upper":
            float(
                ci_upper
            ),

        "predictive_superiority_supported":
            superiority_supported,

        "decision_rule":
            (
                "Supported iff mean OCM-minus-B4 "
                "difference < 0 and upper endpoint "
                "of frozen 95% bootstrap CI < 0."
            ),
    }

    return (
        paired_rows,
        analysis,
    )


def main() -> None:
    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[1/8] Verifying frozen protocol state"
    )

    for path in (
        G1_SUMMARY,
        G1_ARTIFACT_MANIFEST,
        FINAL_REGISTRY,
        COMPARATOR_PATH,
        POLICY_PATH,
        G2A_REPORT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    g1 = load_json(
        G1_SUMMARY
    )

    require(
        g1[
            "phase4gr3g1_status"
        ] == "sealed_test_generation_frozen",
        "G1 is not frozen.",
    )

    require(
        g1[
            "test_open_count"
        ] == 1,
        "Frozen test opening count changed.",
    )

    require(
        g1[
            "test_regeneration_authorized"
        ] is False,
        "Test regeneration became authorized.",
    )

    g2a = load_json(
        G2A_REPORT
    )

    require(
        g2a[
            "phase4gr3g2a_status"
        ] == "evaluation_interfaces_frozen",
        "G2A interface freeze not passed.",
    )

    comparator = load_json(
        COMPARATOR_PATH
    )

    require(
        comparator[
            "selected_comparator_model_id"
        ] == PRIMARY_COMPARATOR,
        "Primary comparator changed.",
    )

    policy = load_json(
        POLICY_PATH
    )

    primary = policy[
        "primary_confirmatory_condition"
    ]

    require(
        primary[
            "primary_model"
        ] == PRIMARY_MODEL,
        "Primary model changed.",
    )

    require(
        primary[
            "cell"
        ] == PRIMARY_CELL,
        "Primary cell changed.",
    )

    require(
        float(
            primary[
                "noise_fraction"
            ]
        ) == PRIMARY_NOISE,
        "Primary noise fraction changed.",
    )

    require(
        int(
            primary[
                "bootstrap_replicates"
            ]
        )
        == BOOTSTRAP_REPLICATES,
        "Bootstrap replicate count changed.",
    )

    require(
        int(
            primary[
                "bootstrap_seed"
            ]
        )
        == BOOTSTRAP_SEED,
        "Bootstrap seed changed.",
    )

    print(
        "[2/8] Loading frozen primary checkpoints"
    )

    registry_rows = load_csv(
        FINAL_REGISTRY
    )

    primary_rows = (
        choose_primary_rows(
            registry_rows
        )
    )

    print(
        "Primary checkpoints:",
        len(
            primary_rows
        ),
    )

    print(
        "[3/8] Loading sealed test_joint"
    )

    cell = SealedTestCell(
        PRIMARY_CELL
    )

    print(
        "Sequences:",
        EXPECTED_SEQUENCE_COUNT,
        "Trajectories:",
        len(
            cell.rows
        ),
        "Test maximum sequence length:",
        cell.maximum_sequence_length,
    )

    print(
        "[4/8] Resolving frozen architecture length/configurations"
    )

    training_maximum_length = (
        determine_training_maximum_length()
    )

    require(
        training_maximum_length
        >= cell.maximum_sequence_length,
        (
            "Test sequence exceeds frozen "
            "training architecture length."
        ),
    )

    print(
        "Frozen model maximum sequence length:",
        training_maximum_length,
    )

    configuration_map = (
        resolve_configuration_map()
    )

    device = (
        runtime.resolve_device(
            "cuda"
        )
    )

    print(
        "Device:",
        device,
    )

    print(
        "[5/8] Running 10 frozen primary checkpoint evaluations"
    )

    all_trajectory_rows = []

    for index, row in enumerate(
        primary_rows,
        start=1,
    ):
        print(
            f"[{index}/10] "
            f"{row['model_id']} "
            f"seed={row['effective_seed']}"
        )

        checkpoint_rows = (
            evaluate_checkpoint(
                registry_row=row,
                cell=cell,
                configuration_map=(
                    configuration_map
                ),
                maximum_sequence_length=(
                    training_maximum_length
                ),
                device=device,
                batch_size=16,
            )
        )

        all_trajectory_rows.extend(
            checkpoint_rows
        )

    require(
        len(
            all_trajectory_rows
        )
        == (
            10
            * EXPECTED_TRAJECTORY_COUNT
        ),
        (
            "Unexpected total primary "
            "trajectory metric count."
        ),
    )

    write_csv(
        TRAJECTORY_RESULTS_PATH,
        all_trajectory_rows,
    )

    print(
        "[6/8] Aggregating trajectories within sequences"
    )

    sequence_seed_rows = (
        aggregate_sequences(
            all_trajectory_rows
        )
    )

    require(
        len(
            sequence_seed_rows
        )
        == (
            2
            * 5
            * EXPECTED_SEQUENCE_COUNT
        ),
        (
            "Unexpected sequence/seed "
            "metric count."
        ),
    )

    write_csv(
        SEQUENCE_SEED_RESULTS_PATH,
        sequence_seed_rows,
    )

    print(
        "[7/8] Running frozen paired sequence bootstrap"
    )

    (
        paired_rows,
        primary_analysis,
    ) = paired_primary_analysis(
        sequence_seed_rows
    )

    write_csv(
        PAIRED_RESULTS_PATH,
        paired_rows,
    )

    result = {
        "phase":
            (
                "4G-R3G2B Tier C v4 "
                "primary confirmatory test"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "test_open_count":
            1,

        "maximum_test_open_count":
            1,

        "test_regeneration_authorized":
            False,

        "additional_training_performed":
            False,

        "checkpoint_reselection_performed":
            False,

        "comparator_reselection_performed":
            False,

        "evaluated_checkpoint_count":
            len(
                primary_rows
            ),

        "evaluated_models":
            [
                PRIMARY_MODEL,
                PRIMARY_COMPARATOR,
            ],

        "evaluated_seeds":
            EXPECTED_SEEDS,

        "primary_analysis":
            primary_analysis,

        "trajectory_result_count":
            len(
                all_trajectory_rows
            ),

        "sequence_seed_result_count":
            len(
                sequence_seed_rows
            ),

        "paired_sequence_count":
            len(
                paired_rows
            ),

        "runtime_consistency_checks":
            "passed",

        "elapsed_seconds":
            float(
                time.time()
                - start_time
            ),

        "test_metrics_computed":
            True,

        "phase4gr3g2b_status":
            "primary_confirmatory_result_frozen",

        "next_phase":
            (
                "4G-R3G2C Tier C v4 full "
                "secondary predictive evaluation"
            ),
    }

    write_json(
        RESULT_PATH,
        result,
    )

    print(
        "[8/8] Freezing result hashes"
    )

    artifacts = [
        TRAJECTORY_RESULTS_PATH,
        SEQUENCE_SEED_RESULTS_PATH,
        PAIRED_RESULTS_PATH,
        RESULT_PATH,
    ]

    hashes = {
        str(path):
            sha256_file(
                path
            )
        for path in artifacts
    }

    write_json(
        HASH_MANIFEST_PATH,
        hashes,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PRIMARY CONFIRMATORY RESULT"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            primary_analysis,
            indent=2,
        )
    )

    print()
    print(
        "Phase 4G-R3G2B primary "
        "confirmatory test completed."
    )


if __name__ == "__main__":
    main()
