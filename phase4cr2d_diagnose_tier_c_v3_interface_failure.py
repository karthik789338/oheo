from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PHASE4AR2_DIR = Path(
    "outputs/phase4ar2_tier_c_v3_protocol"
)

PHASE4BR2_DIR = Path(
    "outputs/phase4br2_tier_c_v3_development_data"
)

PHASE4CR2_DIR = Path(
    "outputs/phase4cr2_tier_c_v3_predecoder_audit"
)

PRIVILEGED_DIR = (
    PHASE4BR2_DIR
    / "privileged"
)

VISIBLE_DIR = (
    PHASE4BR2_DIR
    / "visible"
)

MANIFEST_DIR = (
    PHASE4BR2_DIR
    / "cell_manifests"
)

OUTPUT_DIR = Path(
    "outputs/phase4cr2d_tier_c_v3_interface_diagnosis"
)


PROTOCOL_VERSION = "tier_c_v3"

INTERFACE_RATIO_THRESHOLD = 0.70
MEAN_LENGTH_RATIO_THRESHOLD = 2.50
WORST_LENGTH_RATIO_THRESHOLD = 1.80

EXPECTED_PAIR_COUNT = 512
EXPECTED_DEVELOPMENT_CARRIER_COUNT = 128

SEALED_TEST_CELLS = (
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
)

CARRIER_PARAMETER_NAMES = (
    "mean_phase_fraction",
    "base_length_scale_pixels",
    "interface_width",
    "phase_mobility",
    "orientation_correlation_length",
    "mean_defect_fraction",
    "defect_diffusivity",
    "bulk_driving_bias",
)


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


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


def sha256_file(path: Path):
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


def find_sealed_test_files():
    found = []

    for directory in (
        VISIBLE_DIR,
        PRIVILEGED_DIR,
        MANIFEST_DIR,
    ):
        if not directory.exists():
            continue

        for path in directory.iterdir():
            if any(
                cell_id in path.name
                for cell_id in SEALED_TEST_CELLS
            ):
                found.append(
                    str(path)
                )

    return found


def validate_sources():
    protocol = load_json(
        PHASE4AR2_DIR
        / "phase4ar2_summary.json"
    )

    field_summary = load_json(
        PHASE4BR2_DIR
        / "phase4br2_field_bank_summary.json"
    )

    dataset_summary = load_json(
        PHASE4BR2_DIR
        / "phase4br2_summary.json"
    )

    rejection_summary = load_json(
        PHASE4CR2_DIR
        / "phase4cr2_summary.json"
    )

    rejection_record = load_json(
        PHASE4CR2_DIR
        / "phase4cr2_rejection_record.json"
    )

    if (
        protocol["phase4ar2_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Tier C v3 protocol is not frozen."
        )

    if (
        protocol["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v3 protocol version changed."
        )

    if (
        field_summary[
            "field_bank_generation_status"
        ]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v3 field generation is incomplete."
        )

    if (
        dataset_summary["phase4br2_status"]
        != "completed"
    ):
        raise AssertionError(
            "Tier C v3 development dataset is incomplete."
        )

    if (
        rejection_summary["phase4cr2_status"]
        != "rejection_frozen"
    ):
        raise AssertionError(
            "Tier C v3 rejection is not frozen."
        )

    if (
        rejection_summary["tier_c_v3_rejected"]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 rejection changed."
        )

    if (
        rejection_record[
            "rejection_is_final"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 rejection is not final."
        )

    if (
        rejection_summary[
            "failed_morphology_checks"
        ]
        != [
            "mean_coarse_to_fine_interface_density_ratio"
        ]
    ):
        raise AssertionError(
            "Unexpected Tier C v3 failure reason."
        )

    if (
        rejection_summary[
            "state_decoder_fitting_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "State-decoder fitting was incorrectly authorized."
        )

    if (
        rejection_summary[
            "predictive_training_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Predictive training was incorrectly authorized."
        )

    if (
        rejection_summary[
            "sealed_test_set_remains_unopened"
        ]
        is not True
    ):
        raise AssertionError(
            "Tier C v3 test set is not sealed."
        )

    sealed_test_files = (
        find_sealed_test_files()
    )

    if sealed_test_files:
        raise AssertionError(
            "Sealed Tier C v3 test files exist: "
            + ", ".join(
                sealed_test_files
            )
        )

    parameter_rows = load_csv(
        PHASE4BR2_DIR
        / "development_carrier_parameters.csv"
    )

    if any(
        row["split"] == "test"
        for row in parameter_rows
    ):
        raise AssertionError(
            "Tier C v3 test-carrier parameters exist."
        )

    decoder_files = list(
        PHASE4CR2_DIR.glob(
            "*decoder*.pt"
        )
    )

    if decoder_files:
        raise AssertionError(
            "Unexpected Tier C v3 decoder checkpoints exist."
        )

    return {
        "protocol":
            protocol,

        "field_summary":
            field_summary,

        "dataset_summary":
            dataset_summary,

        "rejection_summary":
            rejection_summary,

        "rejection_record":
            rejection_record,
    }


def rank_values(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(values),
        dtype=np.float64,
    )

    start = 0

    while start < len(values):
        end = start + 1

        while (
            end < len(values)
            and values[
                order[end]
            ]
            == values[
                order[start]
            ]
        ):
            end += 1

        average_rank = (
            start + end - 1
        ) / 2.0

        ranks[
            order[start:end]
        ] = average_rank

        start = end

    return ranks


def correlation(
    first,
    second,
):
    first = np.asarray(
        first,
        dtype=np.float64,
    )

    second = np.asarray(
        second,
        dtype=np.float64,
    )

    if (
        first.std() < 1e-12
        or second.std() < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first,
            second,
        )[0, 1]
    )


def spearman_correlation(
    first,
    second,
):
    return correlation(
        rank_values(first),
        rank_values(second),
    )


def assign_quantile_labels(
    values,
    bin_count=4,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    edges = np.quantile(
        values,
        np.linspace(
            0.0,
            1.0,
            bin_count + 1,
        ),
    )

    interior_edges = edges[
        1:-1
    ]

    labels = np.searchsorted(
        interior_edges,
        values,
        side="right",
    )

    label_names = np.asarray(
        [
            f"Q{index + 1}"
            for index in range(
                bin_count
            )
        ],
        dtype=object,
    )

    return (
        label_names[labels],
        edges,
    )


def load_and_join_pairs():
    paired_rows = load_csv(
        PRIVILEGED_DIR
        / "paired_coarsening_diagnostics.csv"
    )

    morphology_rows = load_csv(
        PRIVILEGED_DIR
        / "development_morphology_diagnostics.csv"
    )

    parameter_rows = load_csv(
        PHASE4BR2_DIR
        / "development_carrier_parameters.csv"
    )

    if len(paired_rows) != EXPECTED_PAIR_COUNT:
        raise AssertionError(
            "Tier C v3 paired diagnostic count changed."
        )

    parameter_lookup = {
        int(row["carrier_id"]):
            row
        for row in parameter_rows
    }

    if (
        len(parameter_lookup)
        != EXPECTED_DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development carrier-parameter count changed."
        )

    morphology_lookup = {
        (
            int(row["carrier_id"]),
            int(row["state_id"]),
        ):
            row
        for row in morphology_rows
    }

    if len(morphology_lookup) != 1024:
        raise AssertionError(
            "Development morphology diagnostic count changed."
        )

    joined = []

    for row in paired_rows:
        carrier_id = int(
            row["carrier_id"]
        )

        fine_state_id = int(
            row["fine_state_id"]
        )

        coarse_state_id = int(
            row["coarse_state_id"]
        )

        parameter_row = parameter_lookup[
            carrier_id
        ]

        fine_morphology = morphology_lookup[
            (
                carrier_id,
                fine_state_id,
            )
        ]

        coarse_morphology = morphology_lookup[
            (
                carrier_id,
                coarse_state_id,
            )
        ]

        interface_ratio = float(
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
        )

        measured_length_ratio = float(
            row[
                "coarse_to_fine_length_ratio"
            ]
        )

        fine_target_length = float(
            fine_morphology[
                "target_characteristic_length"
            ]
        )

        coarse_target_length = float(
            coarse_morphology[
                "target_characteristic_length"
            ]
        )

        target_length_ratio = (
            coarse_target_length
            / fine_target_length
        )

        mean_phase_fraction = float(
            parameter_row[
                "mean_phase_fraction"
            ]
        )

        output = {
            "carrier_id":
                carrier_id,

            "carrier_split":
                row["carrier_split"],

            "anisotropy_level":
                int(
                    row[
                        "anisotropy_level"
                    ]
                ),

            "defect_recovery_level":
                int(
                    row[
                        "defect_recovery_level"
                    ]
                ),

            "fine_state_id":
                fine_state_id,

            "coarse_state_id":
                coarse_state_id,

            "fine_target_characteristic_length":
                fine_target_length,

            "coarse_target_characteristic_length":
                coarse_target_length,

            "target_coarse_to_fine_length_ratio":
                target_length_ratio,

            "fine_measured_characteristic_length":
                float(
                    row[
                        "fine_characteristic_length"
                    ]
                ),

            "coarse_measured_characteristic_length":
                float(
                    row[
                        "coarse_characteristic_length"
                    ]
                ),

            "measured_coarse_to_fine_length_ratio":
                measured_length_ratio,

            "fine_interface_density":
                float(
                    row[
                        "fine_interface_density"
                    ]
                ),

            "coarse_interface_density":
                float(
                    row[
                        "coarse_interface_density"
                    ]
                ),

            "coarse_to_fine_interface_density_ratio":
                interface_ratio,

            "interface_density_reduction_fraction":
                1.0 - interface_ratio,

            "interface_gate_passed":
                bool(
                    interface_ratio
                    <= INTERFACE_RATIO_THRESHOLD
                ),

            "interface_gate_excess":
                max(
                    0.0,
                    interface_ratio
                    - INTERFACE_RATIO_THRESHOLD,
                ),

            "coarse_has_more_interface_than_fine":
                bool(
                    interface_ratio > 1.0
                ),

            "mean_length_gate_passed_for_pair":
                bool(
                    measured_length_ratio
                    >= MEAN_LENGTH_RATIO_THRESHOLD
                ),

            "worst_length_gate_passed_for_pair":
                bool(
                    measured_length_ratio
                    >= WORST_LENGTH_RATIO_THRESHOLD
                ),

            "mean_phase_fraction":
                mean_phase_fraction,

            "phase_fraction_distance_from_half":
                abs(
                    mean_phase_fraction
                    - 0.5
                ),
        }

        for parameter_name in (
            CARRIER_PARAMETER_NAMES
        ):
            output[
                parameter_name
            ] = float(
                parameter_row[
                    parameter_name
                ]
            )

        joined.append(output)

    phase_fraction_values = np.asarray(
        [
            row["mean_phase_fraction"]
            for row in joined
        ],
        dtype=np.float64,
    )

    (
        phase_fraction_labels,
        phase_fraction_edges,
    ) = assign_quantile_labels(
        phase_fraction_values
    )

    base_length_values = np.asarray(
        [
            row[
                "base_length_scale_pixels"
            ]
            for row in joined
        ],
        dtype=np.float64,
    )

    (
        base_length_labels,
        base_length_edges,
    ) = assign_quantile_labels(
        base_length_values
    )

    coarse_target_values = np.asarray(
        [
            row[
                "coarse_target_characteristic_length"
            ]
            for row in joined
        ],
        dtype=np.float64,
    )

    (
        coarse_target_labels,
        coarse_target_edges,
    ) = assign_quantile_labels(
        coarse_target_values
    )

    for index, row in enumerate(joined):
        row[
            "phase_fraction_quartile"
        ] = str(
            phase_fraction_labels[
                index
            ]
        )

        row[
            "base_length_quartile"
        ] = str(
            base_length_labels[
                index
            ]
        )

        row[
            "coarse_target_length_quartile"
        ] = str(
            coarse_target_labels[
                index
            ]
        )

    quantile_definition = {
        "phase_fraction_edges":
            phase_fraction_edges.tolist(),

        "base_length_scale_edges":
            base_length_edges.tolist(),

        "coarse_target_length_edges":
            coarse_target_edges.tolist(),
    }

    write_json(
        OUTPUT_DIR
        / "quantile_definitions.json",
        quantile_definition,
    )

    write_csv(
        OUTPUT_DIR
        / "joined_paired_diagnostics.csv",
        joined,
    )

    return joined


def summarize_rows(
    rows,
    group_fields,
):
    grouped = defaultdict(list)

    for row in rows:
        key = tuple(
            row[field]
            for field in group_fields
        )

        grouped[key].append(row)

    output = []

    for key in sorted(grouped):
        group_rows = grouped[key]

        interface_ratios = np.asarray(
            [
                row[
                    "coarse_to_fine_interface_density_ratio"
                ]
                for row in group_rows
            ],
            dtype=np.float64,
        )

        length_ratios = np.asarray(
            [
                row[
                    "measured_coarse_to_fine_length_ratio"
                ]
                for row in group_rows
            ],
            dtype=np.float64,
        )

        fine_interfaces = np.asarray(
            [
                row[
                    "fine_interface_density"
                ]
                for row in group_rows
            ],
            dtype=np.float64,
        )

        coarse_interfaces = np.asarray(
            [
                row[
                    "coarse_interface_density"
                ]
                for row in group_rows
            ],
            dtype=np.float64,
        )

        result = {
            field:
                key[index]
            for index, field in enumerate(
                group_fields
            )
        }

        failure_count = int(
            np.sum(
                interface_ratios
                > INTERFACE_RATIO_THRESHOLD
            )
        )

        above_one_count = int(
            np.sum(
                interface_ratios > 1.0
            )
        )

        result.update(
            {
                "pair_count":
                    len(group_rows),

                "interface_gate_failure_count":
                    failure_count,

                "interface_gate_failure_fraction":
                    failure_count
                    / len(group_rows),

                "interface_ratio_mean":
                    float(
                        interface_ratios.mean()
                    ),

                "interface_ratio_median":
                    float(
                        np.median(
                            interface_ratios
                        )
                    ),

                "interface_ratio_standard_deviation":
                    float(
                        interface_ratios.std(
                            ddof=1
                        )
                    )
                    if len(interface_ratios) > 1
                    else 0.0,

                "interface_ratio_minimum":
                    float(
                        interface_ratios.min()
                    ),

                "interface_ratio_maximum":
                    float(
                        interface_ratios.max()
                    ),

                "pairs_with_interface_ratio_above_one":
                    above_one_count,

                "fraction_with_interface_ratio_above_one":
                    above_one_count
                    / len(group_rows),

                "measured_length_ratio_mean":
                    float(
                        length_ratios.mean()
                    ),

                "measured_length_ratio_minimum":
                    float(
                        length_ratios.min()
                    ),

                "fine_interface_density_mean":
                    float(
                        fine_interfaces.mean()
                    ),

                "coarse_interface_density_mean":
                    float(
                        coarse_interfaces.mean()
                    ),

                "mean_interface_reduction_fraction":
                    float(
                        1.0
                        - interface_ratios.mean()
                    ),
            }
        )

        output.append(result)

    return output


def calculate_factor_effects(rows):
    lookup = {
        (
            int(row["carrier_id"]),
            int(row["anisotropy_level"]),
            int(row["defect_recovery_level"]),
        ):
            row
        for row in rows
    }

    carrier_ids = sorted(
        {
            int(row["carrier_id"])
            for row in rows
        }
    )

    anisotropy_rows = []
    defect_rows = []

    for carrier_id in carrier_ids:
        for defect_level in (
            0,
            1,
        ):
            low = lookup[
                (
                    carrier_id,
                    0,
                    defect_level,
                )
            ]

            high = lookup[
                (
                    carrier_id,
                    1,
                    defect_level,
                )
            ]

            anisotropy_rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "carrier_split":
                        low[
                            "carrier_split"
                        ],

                    "defect_recovery_level":
                        defect_level,

                    "anisotropy_0_interface_ratio":
                        low[
                            "coarse_to_fine_interface_density_ratio"
                        ],

                    "anisotropy_1_interface_ratio":
                        high[
                            "coarse_to_fine_interface_density_ratio"
                        ],

                    "anisotropy_1_minus_0_interface_ratio":
                        (
                            high[
                                "coarse_to_fine_interface_density_ratio"
                            ]
                            - low[
                                "coarse_to_fine_interface_density_ratio"
                            ]
                        ),
                }
            )

        for anisotropy_level in (
            0,
            1,
        ):
            low = lookup[
                (
                    carrier_id,
                    anisotropy_level,
                    0,
                )
            ]

            high = lookup[
                (
                    carrier_id,
                    anisotropy_level,
                    1,
                )
            ]

            defect_rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "carrier_split":
                        low[
                            "carrier_split"
                        ],

                    "anisotropy_level":
                        anisotropy_level,

                    "defect_0_interface_ratio":
                        low[
                            "coarse_to_fine_interface_density_ratio"
                        ],

                    "defect_1_interface_ratio":
                        high[
                            "coarse_to_fine_interface_density_ratio"
                        ],

                    "defect_1_minus_0_interface_ratio":
                        (
                            high[
                                "coarse_to_fine_interface_density_ratio"
                            ]
                            - low[
                                "coarse_to_fine_interface_density_ratio"
                            ]
                        ),
                }
            )

    write_csv(
        OUTPUT_DIR
        / "paired_anisotropy_effects.csv",
        anisotropy_rows,
    )

    write_csv(
        OUTPUT_DIR
        / "paired_defect_recovery_effects.csv",
        defect_rows,
    )

    anisotropy_differences = np.asarray(
        [
            row[
                "anisotropy_1_minus_0_interface_ratio"
            ]
            for row in anisotropy_rows
        ],
        dtype=np.float64,
    )

    defect_differences = np.asarray(
        [
            row[
                "defect_1_minus_0_interface_ratio"
            ]
            for row in defect_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "anisotropy_effect": {
            "paired_comparison_count":
                len(anisotropy_rows),

            "mean_anisotropy_1_minus_0_interface_ratio":
                float(
                    anisotropy_differences.mean()
                ),

            "median_anisotropy_1_minus_0_interface_ratio":
                float(
                    np.median(
                        anisotropy_differences
                    )
                ),

            "fraction_anisotropy_1_higher":
                float(
                    np.mean(
                        anisotropy_differences
                        > 0.0
                    )
                ),
        },

        "defect_recovery_effect": {
            "paired_comparison_count":
                len(defect_rows),

            "mean_defect_1_minus_0_interface_ratio":
                float(
                    defect_differences.mean()
                ),

            "median_defect_1_minus_0_interface_ratio":
                float(
                    np.median(
                        defect_differences
                    )
                ),

            "fraction_defect_1_higher":
                float(
                    np.mean(
                        defect_differences
                        > 0.0
                    )
                ),
        },
    }

    write_json(
        OUTPUT_DIR
        / "paired_factor_effect_summary.json",
        summary,
    )

    return summary


def calculate_correlations(rows):
    outcome = np.asarray(
        [
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    variable_names = list(
        CARRIER_PARAMETER_NAMES
    ) + [
        "phase_fraction_distance_from_half",
        "fine_target_characteristic_length",
        "coarse_target_characteristic_length",
        "target_coarse_to_fine_length_ratio",
        "fine_measured_characteristic_length",
        "coarse_measured_characteristic_length",
        "measured_coarse_to_fine_length_ratio",
        "fine_interface_density",
        "coarse_interface_density",
    ]

    output = []

    for variable_name in variable_names:
        values = np.asarray(
            [
                row[variable_name]
                for row in rows
            ],
            dtype=np.float64,
        )

        pearson = correlation(
            values,
            outcome,
        )

        spearman = spearman_correlation(
            values,
            outcome,
        )

        output.append(
            {
                "variable":
                    variable_name,

                "pearson_correlation_with_interface_ratio":
                    pearson,

                "spearman_correlation_with_interface_ratio":
                    spearman,

                "absolute_spearman_correlation":
                    (
                        abs(spearman)
                        if math.isfinite(spearman)
                        else float("nan")
                    ),
            }
        )

    output.sort(
        key=lambda row: (
            -row[
                "absolute_spearman_correlation"
            ]
            if math.isfinite(
                row[
                    "absolute_spearman_correlation"
                ]
            )
            else float("inf")
        )
    )

    write_csv(
        OUTPUT_DIR
        / "interface_ratio_correlations.csv",
        output,
    )

    return output


def calculate_carrier_summary(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[
            int(row["carrier_id"])
        ].append(row)

    output = []

    for carrier_id in sorted(grouped):
        carrier_rows = grouped[
            carrier_id
        ]

        ratios = np.asarray(
            [
                row[
                    "coarse_to_fine_interface_density_ratio"
                ]
                for row in carrier_rows
            ],
            dtype=np.float64,
        )

        length_ratios = np.asarray(
            [
                row[
                    "measured_coarse_to_fine_length_ratio"
                ]
                for row in carrier_rows
            ],
            dtype=np.float64,
        )

        representative = carrier_rows[0]

        failure_count = int(
            np.sum(
                ratios
                > INTERFACE_RATIO_THRESHOLD
            )
        )

        output.append(
            {
                "carrier_id":
                    carrier_id,

                "carrier_split":
                    representative[
                        "carrier_split"
                    ],

                "mean_phase_fraction":
                    representative[
                        "mean_phase_fraction"
                    ],

                "base_length_scale_pixels":
                    representative[
                        "base_length_scale_pixels"
                    ],

                "interface_width":
                    representative[
                        "interface_width"
                    ],

                "pair_count":
                    len(carrier_rows),

                "interface_failure_count":
                    failure_count,

                "interface_failure_fraction":
                    failure_count
                    / len(carrier_rows),

                "all_four_pairs_fail":
                    bool(
                        failure_count
                        == len(carrier_rows)
                    ),

                "no_pairs_fail":
                    bool(
                        failure_count == 0
                    ),

                "mean_interface_ratio":
                    float(
                        ratios.mean()
                    ),

                "maximum_interface_ratio":
                    float(
                        ratios.max()
                    ),

                "minimum_interface_ratio":
                    float(
                        ratios.min()
                    ),

                "mean_measured_length_ratio":
                    float(
                        length_ratios.mean()
                    ),

                "minimum_measured_length_ratio":
                    float(
                        length_ratios.min()
                    ),
            }
        )

    output.sort(
        key=lambda row: (
            -row["mean_interface_ratio"],
            row["carrier_id"],
        )
    )

    write_csv(
        OUTPUT_DIR
        / "carrier_interface_summary.csv",
        output,
    )

    write_csv(
        OUTPUT_DIR
        / "worst_20_carriers.csv",
        output[:20],
    )

    return output


def calculate_threshold_cross_tab(rows):
    counts = Counter()

    for row in rows:
        length_pass = bool(
            row[
                "measured_coarse_to_fine_length_ratio"
            ]
            >= MEAN_LENGTH_RATIO_THRESHOLD
        )

        interface_pass = bool(
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
            <= INTERFACE_RATIO_THRESHOLD
        )

        counts[
            (
                length_pass,
                interface_pass,
            )
        ] += 1

    output = []

    for length_pass in (
        False,
        True,
    ):
        for interface_pass in (
            False,
            True,
        ):
            count = counts[
                (
                    length_pass,
                    interface_pass,
                )
            ]

            output.append(
                {
                    "mean_length_ratio_gate_passed":
                        length_pass,

                    "interface_ratio_gate_passed":
                        interface_pass,

                    "pair_count":
                        count,

                    "pair_fraction":
                        count / len(rows),
                }
            )

    write_csv(
        OUTPUT_DIR
        / "length_interface_gate_cross_tab.csv",
        output,
    )

    return output


def identify_failure_pattern(
    rows,
    combination_summary,
    carrier_summary,
    correlations,
    factor_effects,
):
    ratios = np.asarray(
        [
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    length_ratios = np.asarray(
        [
            row[
                "measured_coarse_to_fine_length_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    all_combination_means_fail = bool(
        all(
            row["interface_ratio_mean"]
            > INTERFACE_RATIO_THRESHOLD
            for row in combination_summary
        )
    )

    all_combination_failure_fractions = {
        (
            str(
                row["anisotropy_level"]
            )
            + "-"
            + str(
                row["defect_recovery_level"]
            )
        ):
            row[
                "interface_gate_failure_fraction"
            ]
        for row in combination_summary
    }

    all_four_fail_count = int(
        sum(
            row["all_four_pairs_fail"]
            for row in carrier_summary
        )
    )

    any_fail_count = int(
        sum(
            row[
                "interface_failure_count"
            ] > 0
            for row in carrier_summary
        )
    )

    no_fail_count = int(
        sum(
            row["no_pairs_fail"]
            for row in carrier_summary
        )
    )

    strongest_association = (
        correlations[0]
        if correlations
        else None
    )

    anisotropy_effect = factor_effects[
        "anisotropy_effect"
    ][
        "mean_anisotropy_1_minus_0_interface_ratio"
    ]

    defect_effect = factor_effects[
        "defect_recovery_effect"
    ][
        "mean_defect_1_minus_0_interface_ratio"
    ]

    if abs(anisotropy_effect) >= abs(
        defect_effect
    ):
        dominant_binary_factor = (
            "anisotropy_level"
        )

        dominant_binary_effect = (
            anisotropy_effect
        )

    else:
        dominant_binary_factor = (
            "defect_recovery_level"
        )

        dominant_binary_effect = (
            defect_effect
        )

    if all_combination_means_fail:
        failure_scope = (
            "systematic_across_anisotropy_and_"
            "defect_recovery_combinations"
        )

    else:
        failure_scope = (
            "concentrated_in_selected_factor_"
            "combinations"
        )

    length_interface_spearman = (
        spearman_correlation(
            length_ratios,
            ratios,
        )
    )

    result = {
        "pair_count":
            len(rows),

        "overall_interface_gate_failure_count":
            int(
                np.sum(
                    ratios
                    > INTERFACE_RATIO_THRESHOLD
                )
            ),

        "overall_interface_gate_failure_fraction":
            float(
                np.mean(
                    ratios
                    > INTERFACE_RATIO_THRESHOLD
                )
            ),

        "pairs_with_interface_ratio_above_one":
            int(
                np.sum(
                    ratios > 1.0
                )
            ),

        "fraction_with_interface_ratio_above_one":
            float(
                np.mean(
                    ratios > 1.0
                )
            ),

        "all_factor_combination_means_exceed_threshold":
            all_combination_means_fail,

        "factor_combination_failure_fractions":
            all_combination_failure_fractions,

        "failure_scope":
            failure_scope,

        "carrier_count":
            len(carrier_summary),

        "carriers_with_all_four_pairs_failing":
            all_four_fail_count,

        "fraction_carriers_with_all_four_pairs_failing":
            all_four_fail_count
            / len(carrier_summary),

        "carriers_with_at_least_one_pair_failing":
            any_fail_count,

        "fraction_carriers_with_at_least_one_pair_failing":
            any_fail_count
            / len(carrier_summary),

        "carriers_with_no_pair_failing":
            no_fail_count,

        "fraction_carriers_with_no_pair_failing":
            no_fail_count
            / len(carrier_summary),

        "dominant_binary_factor_by_paired_mean_effect":
            dominant_binary_factor,

        "dominant_binary_factor_mean_effect":
            dominant_binary_effect,

        "anisotropy_mean_paired_effect":
            anisotropy_effect,

        "defect_recovery_mean_paired_effect":
            defect_effect,

        "spearman_measured_length_ratio_vs_interface_ratio":
            length_interface_spearman,

        "strongest_continuous_association":
            (
                {
                    "variable":
                        strongest_association[
                            "variable"
                        ],

                    "spearman_correlation":
                        strongest_association[
                            "spearman_correlation_with_interface_ratio"
                        ],

                    "absolute_spearman_correlation":
                        strongest_association[
                            "absolute_spearman_correlation"
                        ],
                }
                if strongest_association
                else None
            ),

        "interpretation_limits": [
            (
                "This phase is descriptive and does not fit "
                "a predictive or causal model."
            ),
            (
                "Associations with carrier parameters do not "
                "establish causality."
            ),
            (
                "No acceptance threshold was changed or "
                "reinterpreted."
            ),
        ],
    }

    write_json(
        OUTPUT_DIR
        / "failure_pattern_diagnosis.json",
        result,
    )

    return result


def write_input_hashes():
    paths = {
        "phase4ar2_summary":
            PHASE4AR2_DIR
            / "phase4ar2_summary.json",

        "phase4br2_field_summary":
            PHASE4BR2_DIR
            / "phase4br2_field_bank_summary.json",

        "phase4br2_dataset_summary":
            PHASE4BR2_DIR
            / "phase4br2_summary.json",

        "phase4cr2_rejection_summary":
            PHASE4CR2_DIR
            / "phase4cr2_summary.json",

        "phase4cr2_rejection_record":
            PHASE4CR2_DIR
            / "phase4cr2_rejection_record.json",

        "paired_coarsening_diagnostics":
            PRIVILEGED_DIR
            / "paired_coarsening_diagnostics.csv",

        "development_morphology_diagnostics":
            PRIVILEGED_DIR
            / "development_morphology_diagnostics.csv",

        "development_carrier_parameters":
            PHASE4BR2_DIR
            / "development_carrier_parameters.csv",
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path":
                str(path),

            "sha256":
                sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = validate_sources()

    write_input_hashes()

    print(
        "[1/6] Joining frozen pair, morphology, "
        "and carrier diagnostics"
    )

    rows = load_and_join_pairs()

    print(
        "[2/6] Summarizing morphology subgroups"
    )

    overall_summary = summarize_rows(
        rows,
        group_fields=(
            "carrier_split",
        ),
    )

    anisotropy_summary = summarize_rows(
        rows,
        group_fields=(
            "anisotropy_level",
        ),
    )

    defect_summary = summarize_rows(
        rows,
        group_fields=(
            "defect_recovery_level",
        ),
    )

    combination_summary = summarize_rows(
        rows,
        group_fields=(
            "anisotropy_level",
            "defect_recovery_level",
        ),
    )

    phase_fraction_summary = summarize_rows(
        rows,
        group_fields=(
            "phase_fraction_quartile",
        ),
    )

    base_length_summary = summarize_rows(
        rows,
        group_fields=(
            "base_length_quartile",
        ),
    )

    coarse_target_summary = summarize_rows(
        rows,
        group_fields=(
            "coarse_target_length_quartile",
        ),
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_split.csv",
        overall_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_anisotropy.csv",
        anisotropy_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_defect_recovery.csv",
        defect_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_factor_combination.csv",
        combination_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_phase_fraction_quartile.csv",
        phase_fraction_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_base_length_quartile.csv",
        base_length_summary,
    )

    write_csv(
        OUTPUT_DIR
        / "interface_summary_by_coarse_target_length_quartile.csv",
        coarse_target_summary,
    )

    print(
        "[3/6] Computing paired anisotropy and "
        "defect-recovery effects"
    )

    factor_effects = (
        calculate_factor_effects(
            rows
        )
    )

    print(
        "[4/6] Computing carrier-level and "
        "continuous-variable diagnostics"
    )

    correlations = calculate_correlations(
        rows
    )

    carrier_summary = (
        calculate_carrier_summary(
            rows
        )
    )

    print(
        "[5/6] Comparing length and interface gates"
    )

    gate_cross_tab = (
        calculate_threshold_cross_tab(
            rows
        )
    )

    print(
        "[6/6] Freezing the descriptive diagnosis"
    )

    diagnosis = identify_failure_pattern(
        rows=rows,
        combination_summary=(
            combination_summary
        ),
        carrier_summary=carrier_summary,
        correlations=correlations,
        factor_effects=factor_effects,
    )

    ratios = np.asarray(
        [
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    worst_pairs = sorted(
        rows,
        key=lambda row: (
            -row[
                "coarse_to_fine_interface_density_ratio"
            ],
            row["carrier_id"],
            row["anisotropy_level"],
            row["defect_recovery_level"],
        ),
    )[:30]

    write_csv(
        OUTPUT_DIR
        / "worst_30_interface_pairs.csv",
        worst_pairs,
    )

    validation_rows = [
        row
        for row in rows
        if row["carrier_split"] == "val"
    ]

    training_rows = [
        row
        for row in rows
        if row["carrier_split"] == "train"
    ]

    summary = {
        "phase":
            (
                "4C-R2D Tier C v3 development-only "
                "interface-density failure diagnosis"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "source_rejection_frozen":
            True,

        "tier_c_v3_rejected":
            True,

        "pair_count":
            len(rows),

        "training_pair_count":
            len(training_rows),

        "validation_pair_count":
            len(validation_rows),

        "overall_interface_ratio_mean":
            float(
                ratios.mean()
            ),

        "overall_interface_ratio_median":
            float(
                np.median(ratios)
            ),

        "overall_interface_gate_failure_fraction":
            diagnosis[
                "overall_interface_gate_failure_fraction"
            ],

        "pairs_with_interface_ratio_above_one":
            diagnosis[
                "pairs_with_interface_ratio_above_one"
            ],

        "failure_scope":
            diagnosis[
                "failure_scope"
            ],

        "all_factor_combination_means_exceed_threshold":
            diagnosis[
                "all_factor_combination_means_exceed_threshold"
            ],

        "carriers_with_all_four_pairs_failing":
            diagnosis[
                "carriers_with_all_four_pairs_failing"
            ],

        "fraction_carriers_with_all_four_pairs_failing":
            diagnosis[
                "fraction_carriers_with_all_four_pairs_failing"
            ],

        "dominant_binary_factor_by_paired_mean_effect":
            diagnosis[
                "dominant_binary_factor_by_paired_mean_effect"
            ],

        "dominant_binary_factor_mean_effect":
            diagnosis[
                "dominant_binary_factor_mean_effect"
            ],

        "spearman_measured_length_ratio_vs_interface_ratio":
            diagnosis[
                "spearman_measured_length_ratio_vs_interface_ratio"
            ],

        "strongest_continuous_association":
            diagnosis[
                "strongest_continuous_association"
            ],

        "length_interface_gate_cross_tab":
            gate_cross_tab,

        "sealed_test_files_absent":
            True,

        "test_carrier_parameters_present":
            False,

        "test_fields_read":
            False,

        "test_metrics_computed":
            False,

        "state_decoder_fitted":
            False,

        "new_statistical_model_fitted":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "model_weights_created_or_modified":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_outputs_modified":
            False,

        "tier_c_v3_prior_outputs_modified":
            False,

        "tier_c_v4_protocol_frozen":
            False,

        "phase4cr2d_status":
            "passed",
    }

    write_json(
        OUTPUT_DIR
        / "phase4cr2d_summary.json",
        summary,
    )

    print()
    print(
        "Phase 4C-R2D Tier C v3 interface-density "
        "failure diagnosis completed."
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
