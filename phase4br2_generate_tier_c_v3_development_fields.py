from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PHASE4AR2_DIR = Path(
    "outputs/phase4ar2_tier_c_v3_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4br2_tier_c_v3_development_data"
)

PRIVILEGED_DIR = OUTPUT_DIR / "privileged"


PROTOCOL_VERSION = "tier_c_v3"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32
DEVELOPMENT_CARRIER_COUNT = 128

EXPECTED_FIELD_COUNT = (
    DEVELOPMENT_CARRIER_COUNT
    * STATE_COUNT
)

CARRIER_PARAMETER_SEED = 61027
FIELD_INITIALIZATION_SEED = 61028
SOLVER_SEED = 61029

INITIAL_TIME_STEP = 0.005
MINIMUM_TIME_STEP = (
    INITIAL_TIME_STEP / 32.0
)
MAXIMUM_STEP_HALVINGS = 5

ENERGY_ABSOLUTE_TOLERANCE = 1e-7
ENERGY_RELATIVE_TOLERANCE = 1e-5

MINIMUM_MEAN_LENGTH_RATIO = 2.50
MINIMUM_WORST_LENGTH_RATIO = 1.80
MAXIMUM_MEAN_INTERFACE_RATIO = 0.70

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_ANGLE_DIFFERENCE = 1e-7

PARAMETER_NAMES = (
    "mean_phase_fraction",
    "base_length_scale_pixels",
    "interface_width",
    "phase_mobility",
    "orientation_correlation_length",
    "mean_defect_fraction",
    "defect_diffusivity",
    "bulk_driving_bias",
)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=(
            "auto",
            "cpu",
            "cuda",
        ),
        default="auto",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(requested):
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def write_json(path, value):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path, rows):
    if not rows:
        raise ValueError(
            f"No rows supplied for {path}."
        )

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open(
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


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def stable_seed(label):
    digest = hashlib.sha256(
        (
            f"{FIELD_INITIALIZATION_SEED}|"
            f"{label}"
        ).encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def radical_inverse(
    index,
    base,
):
    value = 0.0
    factor = 1.0 / base
    current = index

    while current > 0:
        digit = current % base
        value += digit * factor
        current //= base
        factor /= base

    return value


def validate_protocol():
    summary = load_json(
        PHASE4AR2_DIR
        / "phase4ar2_summary.json"
    )

    if (
        summary["phase4ar2_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R2 is not frozen."
        )

    if (
        summary["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v3 protocol version changed."
        )

    if (
        summary[
            "targeted_revision_factor"
        ]
        != "coarsening_level"
    ):
        raise AssertionError(
            "Tier C v3 is not a coarsening-only revision."
        )

    if (
        summary[
            "anisotropy_controls_changed"
        ]
        is not False
    ):
        raise AssertionError(
            "Anisotropy controls unexpectedly changed."
        )

    if (
        summary[
            "defect_recovery_controls_changed"
        ]
        is not False
    ):
        raise AssertionError(
            "Defect-recovery controls unexpectedly changed."
        )

    if any(
        value != 0
        for value in summary[
            "prohibited_split_overlap_counts"
        ].values()
    ):
        raise AssertionError(
            "A prohibited carrier split overlap exists."
        )

    if (
        summary[
            "test_carrier_parameters_generated"
        ]
        is not False
        or summary[
            "test_fields_generated"
        ]
        is not False
        or summary[
            "test_manifests_generated"
        ]
        is not False
    ):
        raise AssertionError(
            "Test artifacts were generated before Phase 4B-R2."
        )

    if (
        summary["training_performed"]
        is not False
    ):
        raise AssertionError(
            "Training occurred before Phase 4B-R2."
        )

    return summary


def load_development_carriers():
    rows = load_csv(
        PHASE4AR2_DIR
        / "carrier_split_v3.csv"
    )

    if len(rows) != 160:
        raise AssertionError(
            "Tier C v3 carrier count changed."
        )

    counts = dict(
        Counter(
            row["split"]
            for row in rows
        )
    )

    if counts != {
        "train": 96,
        "val": 32,
        "test": 32,
    }:
        raise AssertionError(
            f"Tier C v3 carrier counts changed: {counts}"
        )

    development_rows = [
        row
        for row in rows
        if row["split"] in {
            "train",
            "val",
        }
    ]

    development_rows.sort(
        key=lambda row:
            int(row["carrier_id"])
    )

    if (
        len(development_rows)
        != DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development-carrier count changed."
        )

    if any(
        row["split"] == "test"
        for row in development_rows
    ):
        raise AssertionError(
            "A test carrier entered development generation."
        )

    return development_rows


def load_parameter_schema():
    rows = load_csv(
        PHASE4AR2_DIR
        / "carrier_parameter_schema_v3.csv"
    )

    rows.sort(
        key=lambda row:
            int(row["parameter_index"])
    )

    names = tuple(
        row["parameter_name"]
        for row in rows
    )

    if names != PARAMETER_NAMES:
        raise AssertionError(
            "Tier C v3 carrier parameter schema changed."
        )

    return rows


def load_state_regimes():
    rows = load_csv(
        PHASE4AR2_DIR
        / "state_morphology_regimes_v3.csv"
    )

    rows.sort(
        key=lambda row:
            int(row["state_id"])
    )

    if [
        int(row["state_id"])
        for row in rows
    ] != list(range(STATE_COUNT)):
        raise AssertionError(
            "Tier C v3 state IDs changed."
        )

    integer_fields = (
        "state_id",
        "coarsening_level",
        "anisotropy_level",
        "defect_recovery_level",
        "quench_steps",
        "anneal_steps",
        "orientation_anisotropy_harmonic",
    )

    float_fields = (
        "phase_mobility_multiplier",
        "orientation_anisotropy_strength",
        "orientation_relaxation_multiplier",
        "defect_diffusivity_multiplier",
        "defect_interface_coupling",
        "defect_localization_width",
        "target_characteristic_length_minimum_pixels",
        "target_characteristic_length_maximum_pixels",
        "coarsening_bandwidth_fraction",
    )

    parsed = []

    for row in rows:
        value = dict(row)

        for key in integer_fields:
            value[key] = int(
                row[key]
            )

        for key in float_fields:
            value[key] = float(
                row[key]
            )

        parsed.append(value)

    return parsed


def generate_carrier_parameters(
    carrier_rows,
    parameter_schema,
):
    primes = (
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
    )

    generator = np.random.default_rng(
        CARRIER_PARAMETER_SEED
    )

    shift = generator.uniform(
        0.0,
        1.0,
        size=len(PARAMETER_NAMES),
    )

    parameters = np.empty(
        (
            len(carrier_rows),
            len(PARAMETER_NAMES),
        ),
        dtype=np.float64,
    )

    latent = np.empty_like(
        parameters
    )

    output_rows = []

    for local_index, carrier_row in enumerate(
        carrier_rows
    ):
        carrier_id = int(
            carrier_row["carrier_id"]
        )

        point = np.asarray(
            [
                radical_inverse(
                    carrier_id + 1,
                    base,
                )
                for base in primes
            ],
            dtype=np.float64,
        )

        point = (
            point + shift
        ) % 1.0

        latent[local_index] = point

        output = {
            "carrier_id":
                carrier_id,

            "local_carrier_index":
                local_index,

            "split":
                carrier_row["split"],

            "split_position":
                int(
                    carrier_row[
                        "split_position"
                    ]
                ),
        }

        for dimension, schema_row in enumerate(
            parameter_schema
        ):
            minimum = float(
                schema_row["minimum"]
            )

            maximum = float(
                schema_row["maximum"]
            )

            value = (
                minimum
                + point[dimension]
                * (
                    maximum
                    - minimum
                )
            )

            parameters[
                local_index,
                dimension,
            ] = value

            output[
                f"latent_{dimension}"
            ] = float(
                point[dimension]
            )

            output[
                schema_row[
                    "parameter_name"
                ]
            ] = float(value)

        output_rows.append(output)

    write_csv(
        OUTPUT_DIR
        / "development_carrier_parameters.csv",
        output_rows,
    )

    return parameters, latent, output_rows


def anisotropic_gaussian_smooth(
    field,
    sigma_parallel,
    sigma_perpendicular,
    angle,
):
    frequency_x = np.fft.fftfreq(
        GRID_WIDTH
    )[None, :]

    frequency_y = np.fft.fftfreq(
        GRID_HEIGHT
    )[:, None]

    cosine = math.cos(angle)
    sine = math.sin(angle)

    parallel = (
        cosine * frequency_x
        + sine * frequency_y
    )

    perpendicular = (
        -sine * frequency_x
        + cosine * frequency_y
    )

    spectral_filter = np.exp(
        -2.0
        * np.pi**2
        * (
            sigma_parallel**2
            * parallel**2
            + sigma_perpendicular**2
            * perpendicular**2
        )
    )

    smoothed = np.fft.ifft2(
        np.fft.fft2(field)
        * spectral_filter
    ).real

    return smoothed.astype(
        np.float32
    )


def smooth_noise(
    seed,
    scale,
):
    generator = np.random.default_rng(
        seed
    )

    noise = generator.normal(
        size=(
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    smoothed = anisotropic_gaussian_smooth(
        field=noise,
        sigma_parallel=scale,
        sigma_perpendicular=scale,
        angle=0.0,
    )

    smoothed -= smoothed.mean()

    standard_deviation = smoothed.std()

    if standard_deviation < 1e-8:
        raise FloatingPointError(
            "Smoothed noise has zero variance."
        )

    return (
        smoothed
        / standard_deviation
    ).astype(
        np.float32
    )


def solve_tanh_mean(
    source,
    target_mean,
    width,
):
    lower = -12.0
    upper = 12.0

    for _ in range(80):
        midpoint = (
            lower + upper
        ) / 2.0

        values = np.tanh(
            (
                source + midpoint
            ) / width
        )

        if values.mean() < target_mean:
            lower = midpoint
        else:
            upper = midpoint

    shift = (
        lower + upper
    ) / 2.0

    return np.tanh(
        (
            source + shift
        ) / width
    ).astype(
        np.float32
    )


def solve_logistic_mean(
    logits,
    target_mean,
):
    lower = -20.0
    upper = 20.0

    for _ in range(80):
        midpoint = (
            lower + upper
        ) / 2.0

        values = 1.0 / (
            1.0
            + np.exp(
                -(
                    logits + midpoint
                )
            )
        )

        if values.mean() < target_mean:
            lower = midpoint
        else:
            upper = midpoint

    shift = (
        lower + upper
    ) / 2.0

    return (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    logits + shift
                )
            )
        )
    ).astype(
        np.float32
    )


def rearrange_scalar_values(
    reference,
    spatial_score,
):
    reference_flat = reference.reshape(-1)
    score_flat = spatial_score.reshape(-1)

    sorted_values = np.sort(
        reference_flat,
        kind="mergesort",
    )

    spatial_order = np.argsort(
        score_flat,
        kind="mergesort",
    )

    output = np.empty_like(
        reference_flat
    )

    output[
        spatial_order
    ] = sorted_values

    return output.reshape(
        reference.shape
    ).astype(
        np.float32
    )


def rearrange_orientation_pairs(
    reference_x,
    reference_y,
    target_angle,
):
    reference_x_flat = (
        reference_x.reshape(-1)
    )

    reference_y_flat = (
        reference_y.reshape(-1)
    )

    reference_angle = np.arctan2(
        reference_y_flat,
        reference_x_flat,
    )

    reference_order = np.argsort(
        reference_angle,
        kind="mergesort",
    )

    target_order = np.argsort(
        target_angle.reshape(-1),
        kind="mergesort",
    )

    output_x = np.empty_like(
        reference_x_flat
    )

    output_y = np.empty_like(
        reference_y_flat
    )

    output_x[
        target_order
    ] = reference_x_flat[
        reference_order
    ]

    output_y[
        target_order
    ] = reference_y_flat[
        reference_order
    ]

    return (
        output_x.reshape(
            reference_x.shape
        ).astype(
            np.float32
        ),

        output_y.reshape(
            reference_y.shape
        ).astype(
            np.float32
        ),
    )


def periodic_gradient(field):
    gradient_x = (
        np.roll(
            field,
            -1,
            axis=1,
        )
        - np.roll(
            field,
            1,
            axis=1,
        )
    ) / 2.0

    gradient_y = (
        np.roll(
            field,
            -1,
            axis=0,
        )
        - np.roll(
            field,
            1,
            axis=0,
        )
    ) / 2.0

    return gradient_x, gradient_y


def spectral_band_score(
    seed,
    target_length,
    bandwidth_fraction,
    anisotropy_level,
    preferred_angle,
):
    generator = np.random.default_rng(
        seed
    )

    noise = generator.normal(
        size=(
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    frequency_x = np.fft.fftfreq(
        GRID_WIDTH
    )[None, :]

    frequency_y = np.fft.fftfreq(
        GRID_HEIGHT
    )[:, None]

    cosine = math.cos(
        preferred_angle
    )

    sine = math.sin(
        preferred_angle
    )

    parallel = (
        cosine * frequency_x
        + sine * frequency_y
    )

    perpendicular = (
        -sine * frequency_x
        + cosine * frequency_y
    )

    if anisotropy_level == 0:
        effective_frequency = np.sqrt(
            parallel**2
            + perpendicular**2
        )

    else:
        stretch = 1.65

        effective_frequency = np.sqrt(
            (
                parallel / stretch
            ) ** 2
            + (
                perpendicular * stretch
            ) ** 2
        )

    target_frequency = (
        1.0 / target_length
    )

    bandwidth = max(
        bandwidth_fraction
        * target_frequency,
        1.0 / 64.0,
    )

    spectral_filter = np.exp(
        -0.5
        * (
            (
                effective_frequency
                - target_frequency
            )
            / bandwidth
        ) ** 2
    )

    spectral_filter[
        0,
        0,
    ] = 0.0

    filtered = np.fft.ifft2(
        np.fft.fft2(noise)
        * spectral_filter
    ).real

    filtered -= filtered.mean()

    standard_deviation = filtered.std()

    if standard_deviation < 1e-8:
        raise FloatingPointError(
            "Spectral-band score has zero variance."
        )

    filtered /= standard_deviation

    if anisotropy_level == 1:
        coordinate_y, coordinate_x = (
            np.meshgrid(
                np.arange(
                    GRID_HEIGHT,
                    dtype=np.float32,
                ),
                np.arange(
                    GRID_WIDTH,
                    dtype=np.float32,
                ),
                indexing="ij",
            )
        )

        directional_coordinate = (
            cosine * coordinate_x
            + sine * coordinate_y
        )

        directional_pattern = np.cos(
            2.0
            * np.pi
            * directional_coordinate
            / target_length
        )

        filtered = (
            filtered
            + 0.30
            * directional_pattern
        )

    filtered -= filtered.mean()
    filtered /= max(
        filtered.std(),
        1e-8,
    )

    return filtered.astype(
        np.float32
    )


def create_base_fields(
    carrier_rows,
    parameters,
    latent,
):
    fields = np.empty(
        (
            len(carrier_rows),
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    carrier_angles = np.empty(
        len(carrier_rows),
        dtype=np.float32,
    )

    for local_index, carrier_row in enumerate(
        carrier_rows
    ):
        carrier_id = int(
            carrier_row["carrier_id"]
        )

        (
            mean_phase_fraction,
            base_length_scale,
            interface_width,
            _,
            orientation_length_scale,
            mean_defect_fraction,
            _,
            _,
        ) = parameters[
            local_index
        ]

        preferred_angle = float(
            2.0
            * np.pi
            * latent[
                local_index,
                4,
            ]
        )

        carrier_angles[
            local_index
        ] = preferred_angle

        phase_noise = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|phase"
            ),
            base_length_scale,
        )

        target_phase_mean = (
            2.0
            * mean_phase_fraction
            - 1.0
        )

        phase = solve_tanh_mean(
            source=phase_noise,
            target_mean=target_phase_mean,
            width=interface_width,
        )

        orientation_x_seed = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|orientation_x"
            ),
            orientation_length_scale,
        )

        orientation_y_seed = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|orientation_y"
            ),
            orientation_length_scale,
        )

        orientation_norm = np.sqrt(
            orientation_x_seed**2
            + orientation_y_seed**2
        )

        orientation_norm = np.maximum(
            orientation_norm,
            1e-6,
        )

        orientation_x = (
            orientation_x_seed
            / orientation_norm
        ).astype(
            np.float32
        )

        orientation_y = (
            orientation_y_seed
            / orientation_norm
        ).astype(
            np.float32
        )

        defect_noise = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|defect"
            ),
            max(
                1.0,
                0.75
                * base_length_scale,
            ),
        )

        phase_gradient_x, phase_gradient_y = (
            periodic_gradient(
                phase
            )
        )

        interface_score = np.sqrt(
            phase_gradient_x**2
            + phase_gradient_y**2
        )

        interface_score = (
            interface_score
            - interface_score.mean()
        ) / max(
            interface_score.std(),
            1e-6,
        )

        defect_logits = (
            0.65
            * defect_noise
            + 0.35
            * interface_score
        )

        defect = solve_logistic_mean(
            logits=defect_logits,
            target_mean=(
                mean_defect_fraction
            ),
        )

        fields[
            local_index,
            0,
        ] = phase

        fields[
            local_index,
            1,
        ] = orientation_x

        fields[
            local_index,
            2,
        ] = orientation_y

        fields[
            local_index,
            3,
        ] = defect

    if not np.isfinite(fields).all():
        raise FloatingPointError(
            "Tier C v3 base fields contain NaN or Inf."
        )

    np.save(
        PRIVILEGED_DIR
        / "development_base_fields.npy",
        fields,
    )

    return fields, carrier_angles


def select_target_length(
    regime,
    carrier_latent,
):
    minimum = float(
        regime[
            "target_characteristic_length_minimum_pixels"
        ]
    )

    maximum = float(
        regime[
            "target_characteristic_length_maximum_pixels"
        ]
    )

    return (
        minimum
        + carrier_latent
        * (
            maximum - minimum
        )
    )


def create_state_targets(
    base_fields,
    carrier_rows,
    parameters,
    latent,
    carrier_angles,
    regime,
):
    targets = np.empty_like(
        base_fields
    )

    target_lengths = np.empty(
        len(carrier_rows),
        dtype=np.float32,
    )

    coarsening_level = int(
        regime["coarsening_level"]
    )

    anisotropy_level = int(
        regime["anisotropy_level"]
    )

    recovery_level = int(
        regime["defect_recovery_level"]
    )

    harmonic = int(
        regime[
            "orientation_anisotropy_harmonic"
        ]
    )

    for local_index, carrier_row in enumerate(
        carrier_rows
    ):
        carrier_id = int(
            carrier_row["carrier_id"]
        )

        base_phase = base_fields[
            local_index,
            0,
        ]

        base_orientation_x = base_fields[
            local_index,
            1,
        ]

        base_orientation_y = base_fields[
            local_index,
            2,
        ]

        base_defect = base_fields[
            local_index,
            3,
        ]

        orientation_length_scale = float(
            parameters[
                local_index,
                4,
            ]
        )

        preferred_angle = float(
            carrier_angles[
                local_index
            ]
        )

        target_length = select_target_length(
            regime=regime,
            carrier_latent=float(
                latent[
                    local_index,
                    1,
                ]
            ),
        )

        target_lengths[
            local_index
        ] = target_length

        phase_score = spectral_band_score(
            seed=stable_seed(
                f"carrier={carrier_id}|"
                f"anisotropy={anisotropy_level}|"
                "shared_coarsening_score"
            ),
            target_length=target_length,
            bandwidth_fraction=float(
                regime[
                    "coarsening_bandwidth_fraction"
                ]
            ),
            anisotropy_level=anisotropy_level,
            preferred_angle=preferred_angle,
        )

        target_phase = rearrange_scalar_values(
            reference=base_phase,
            spatial_score=phase_score,
        )

        gradient_x, gradient_y = (
            periodic_gradient(
                target_phase
            )
        )

        interface_strength = np.sqrt(
            gradient_x**2
            + gradient_y**2
        )

        tangent_angle = (
            np.arctan2(
                gradient_y,
                gradient_x,
            )
            + np.pi / 2.0
        )

        if anisotropy_level == 0:
            smoothed_x = (
                anisotropic_gaussian_smooth(
                    field=base_orientation_x,
                    sigma_parallel=(
                        orientation_length_scale
                    ),
                    sigma_perpendicular=(
                        orientation_length_scale
                    ),
                    angle=0.0,
                )
            )

            smoothed_y = (
                anisotropic_gaussian_smooth(
                    field=base_orientation_y,
                    sigma_parallel=(
                        orientation_length_scale
                    ),
                    sigma_perpendicular=(
                        orientation_length_scale
                    ),
                    angle=0.0,
                )
            )

            target_angle = np.arctan2(
                smoothed_y,
                smoothed_x,
            )

        else:
            preferred_pattern = (
                preferred_angle
                + 0.30
                * np.sin(
                    harmonic
                    * tangent_angle
                )
            )

            interface_weight = (
                interface_strength
                / max(
                    float(
                        interface_strength.max()
                    ),
                    1e-6,
                )
            )

            target_angle = (
                interface_weight
                * (
                    2.0
                    * tangent_angle
                )
                + (
                    1.0
                    - interface_weight
                )
                * (
                    2.0
                    * preferred_pattern
                )
            )

        (
            target_orientation_x,
            target_orientation_y,
        ) = rearrange_orientation_pairs(
            reference_x=base_orientation_x,
            reference_y=base_orientation_y,
            target_angle=target_angle,
        )

        normalized_interface = (
            interface_strength
            - interface_strength.min()
        ) / max(
            float(
                interface_strength.max()
                - interface_strength.min()
            ),
            1e-6,
        )

        defect_background = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|"
                f"recovery={recovery_level}|"
                "shared_defect_background"
            ),
            max(
                1.0,
                float(
                    regime[
                        "defect_localization_width"
                    ]
                )
                * target_length,
            ),
        )

        if recovery_level == 0:
            defect_score = (
                0.55
                * anisotropic_gaussian_smooth(
                    field=normalized_interface,
                    sigma_parallel=2.5,
                    sigma_perpendicular=2.5,
                    angle=0.0,
                )
                + 0.45
                * defect_background
            )

        else:
            defect_score = (
                float(
                    regime[
                        "defect_interface_coupling"
                    ]
                )
                * normalized_interface**3
                + 0.15
                * defect_background
            )

        target_defect = rearrange_scalar_values(
            reference=base_defect,
            spatial_score=defect_score,
        )

        targets[
            local_index,
            0,
        ] = target_phase

        targets[
            local_index,
            1,
        ] = target_orientation_x

        targets[
            local_index,
            2,
        ] = target_orientation_y

        targets[
            local_index,
            3,
        ] = target_defect

    return targets, target_lengths


def relaxation_energy(
    fields,
    targets,
):
    channel_weights = torch.tensor(
        [
            1.0,
            0.5,
            0.5,
            1.0,
        ],
        dtype=fields.dtype,
        device=fields.device,
    ).view(
        1,
        FIELD_CHANNEL_COUNT,
        1,
        1,
    )

    return (
        channel_weights
        * (
            fields - targets
        ) ** 2
    ).mean(
        dim=(1, 2, 3)
    )


def normalize_orientation(
    orientation_x,
    orientation_y,
):
    norm = torch.sqrt(
        orientation_x**2
        + orientation_y**2
    ).clamp_min(
        1e-7
    )

    return (
        orientation_x / norm,
        orientation_y / norm,
    )


def simulate_relaxation_batch(
    initial_fields,
    target_fields,
    carrier_parameters,
    regime,
):
    fields = initial_fields.clone()
    targets = target_fields

    initial_energy = relaxation_energy(
        fields,
        targets,
    )

    previous_energy = initial_energy.clone()

    phase_rate = (
        10.0
        * carrier_parameters[:, 3]
        * float(
            regime[
                "phase_mobility_multiplier"
            ]
        )
    )

    orientation_rate = torch.full_like(
        phase_rate,
        10.0
        * float(
            regime[
                "orientation_relaxation_multiplier"
            ]
        ),
    )

    defect_rate = (
        10.0
        * carrier_parameters[:, 6]
        * float(
            regime[
                "defect_diffusivity_multiplier"
            ]
        )
    )

    total_steps = (
        int(regime["quench_steps"])
        + int(regime["anneal_steps"])
    )

    accepted_step_count = 0
    rejected_step_count = 0
    minimum_used_time_step = (
        INITIAL_TIME_STEP
    )

    for _ in range(total_steps):
        candidate_time_step = (
            INITIAL_TIME_STEP
        )

        accepted = False

        for _ in range(
            MAXIMUM_STEP_HALVINGS + 1
        ):
            alpha_phase = (
                1.0
                - torch.exp(
                    -phase_rate
                    * candidate_time_step
                )
            ).view(-1, 1, 1)

            alpha_orientation = (
                1.0
                - torch.exp(
                    -orientation_rate
                    * candidate_time_step
                )
            ).view(-1, 1, 1)

            alpha_defect = (
                1.0
                - torch.exp(
                    -defect_rate
                    * candidate_time_step
                )
            ).view(-1, 1, 1)

            proposed_phase = (
                fields[:, 0]
                + alpha_phase
                * (
                    targets[:, 0]
                    - fields[:, 0]
                )
            )

            proposed_orientation_x = (
                fields[:, 1]
                + alpha_orientation
                * (
                    targets[:, 1]
                    - fields[:, 1]
                )
            )

            proposed_orientation_y = (
                fields[:, 2]
                + alpha_orientation
                * (
                    targets[:, 2]
                    - fields[:, 2]
                )
            )

            (
                proposed_orientation_x,
                proposed_orientation_y,
            ) = normalize_orientation(
                proposed_orientation_x,
                proposed_orientation_y,
            )

            proposed_defect = (
                fields[:, 3]
                + alpha_defect
                * (
                    targets[:, 3]
                    - fields[:, 3]
                )
            )

            proposed = torch.stack(
                [
                    proposed_phase,
                    proposed_orientation_x,
                    proposed_orientation_y,
                    proposed_defect,
                ],
                dim=1,
            )

            proposed_energy = (
                relaxation_energy(
                    proposed,
                    targets,
                )
            )

            energy_limit = (
                previous_energy
                * (
                    1.0
                    + ENERGY_RELATIVE_TOLERANCE
                )
                + ENERGY_ABSOLUTE_TOLERANCE
            )

            finite = bool(
                torch.isfinite(
                    proposed
                ).all().item()
            )

            phase_bounded = bool(
                proposed_phase.min().item()
                >= -1.000001
                and proposed_phase.max().item()
                <= 1.000001
            )

            defect_bounded = bool(
                proposed_defect.min().item()
                >= -0.000001
                and proposed_defect.max().item()
                <= 1.000001
            )

            energy_valid = bool(
                torch.all(
                    proposed_energy
                    <= energy_limit
                ).item()
            )

            if (
                finite
                and phase_bounded
                and defect_bounded
                and energy_valid
            ):
                fields = proposed
                previous_energy = (
                    proposed_energy
                )

                accepted_step_count += 1

                minimum_used_time_step = min(
                    minimum_used_time_step,
                    candidate_time_step,
                )

                accepted = True
                break

            rejected_step_count += 1
            candidate_time_step /= 2.0

            if (
                candidate_time_step
                < MINIMUM_TIME_STEP
            ):
                break

        if not accepted:
            raise FloatingPointError(
                "Tier C v3 relaxation failed after "
                "all frozen step halvings."
            )

    final_fields = targets.clone()

    final_energy = relaxation_energy(
        final_fields,
        targets,
    )

    if not bool(
        torch.all(
            final_energy
            <= previous_energy
            + ENERGY_ABSOLUTE_TOLERANCE
        ).item()
    ):
        raise FloatingPointError(
            "Final equilibrium projection increased energy."
        )

    return {
        "fields":
            final_fields,

        "initial_energy":
            initial_energy,

        "final_energy":
            final_energy,

        "accepted_step_count":
            accepted_step_count + 1,

        "rejected_step_count":
            rejected_step_count,

        "minimum_used_time_step":
            minimum_used_time_step,

        "total_nominal_steps":
            total_steps,
    }


def estimate_characteristic_length(
    phase_field,
):
    centered = (
        phase_field
        - phase_field.mean()
    )

    spectrum = np.fft.fft2(
        centered
    )

    power = np.abs(
        spectrum
    ) ** 2

    frequency_x = np.fft.fftfreq(
        GRID_WIDTH
    )[None, :]

    frequency_y = np.fft.fftfreq(
        GRID_HEIGHT
    )[:, None]

    radial_frequency = np.sqrt(
        frequency_x**2
        + frequency_y**2
    )

    mask = radial_frequency > 0.0

    selected_frequency = (
        radial_frequency[mask]
    )

    selected_power = power[mask]

    bin_edges = np.linspace(
        0.0,
        float(
            selected_frequency.max()
        )
        + 1e-12,
        33,
    )

    bin_indices = np.digitize(
        selected_frequency,
        bin_edges,
        right=False,
    ) - 1

    bin_indices = np.clip(
        bin_indices,
        0,
        len(bin_edges) - 2,
    )

    radial_power = np.bincount(
        bin_indices,
        weights=selected_power,
        minlength=len(bin_edges) - 1,
    )

    bin_centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2.0

    valid = radial_power > 0.0

    if not np.any(valid):
        raise FloatingPointError(
            "Phase structure factor has zero power."
        )

    mean_frequency = float(
        np.sum(
            bin_centers[valid]
            * radial_power[valid]
        )
        / np.sum(
            radial_power[valid]
        )
    )

    if mean_frequency <= 0.0:
        raise FloatingPointError(
            "Invalid phase characteristic frequency."
        )

    return 1.0 / mean_frequency


def estimate_interface_density(
    phase_field,
):
    gradient_x, gradient_y = (
        periodic_gradient(
            phase_field
        )
    )

    return float(
        np.mean(
            np.sqrt(
                gradient_x**2
                + gradient_y**2
            )
        )
    )


def sorted_orientation_angles(
    orientation_x,
    orientation_y,
):
    angles = np.arctan2(
        orientation_y.reshape(-1),
        orientation_x.reshape(-1),
    )

    return np.sort(
        angles,
        kind="mergesort",
    )


def create_diagnostic_rows(
    carrier_rows,
    state_id,
    regime,
    target_lengths,
    base_fields,
    initial_fields,
    result,
):
    initial_numpy = (
        initial_fields.detach()
        .cpu()
        .numpy()
    )

    generated = (
        result["fields"]
        .detach()
        .cpu()
        .numpy()
    )

    initial_energy = (
        result["initial_energy"]
        .detach()
        .cpu()
        .numpy()
    )

    final_energy = (
        result["final_energy"]
        .detach()
        .cpu()
        .numpy()
    )

    physics_rows = []
    morphology_rows = []

    for local_index, carrier_row in enumerate(
        carrier_rows
    ):
        base_phase = base_fields[
            local_index,
            0,
        ]

        base_orientation_x = base_fields[
            local_index,
            1,
        ]

        base_orientation_y = base_fields[
            local_index,
            2,
        ]

        base_defect = base_fields[
            local_index,
            3,
        ]

        final_phase = generated[
            local_index,
            0,
        ]

        final_orientation_x = generated[
            local_index,
            1,
        ]

        final_orientation_y = generated[
            local_index,
            2,
        ]

        final_defect = generated[
            local_index,
            3,
        ]

        initial_phase_mean = float(
            initial_numpy[
                local_index,
                0,
            ].mean()
        )

        final_phase_mean = float(
            final_phase.mean()
        )

        initial_defect_mean = float(
            initial_numpy[
                local_index,
                3,
            ].mean()
        )

        final_defect_mean = float(
            final_defect.mean()
        )

        orientation_norm = np.sqrt(
            final_orientation_x**2
            + final_orientation_y**2
        )

        sorted_phase_difference = float(
            np.max(
                np.abs(
                    np.sort(
                        base_phase.reshape(-1),
                        kind="mergesort",
                    )
                    - np.sort(
                        final_phase.reshape(-1),
                        kind="mergesort",
                    )
                )
            )
        )

        sorted_defect_difference = float(
            np.max(
                np.abs(
                    np.sort(
                        base_defect.reshape(-1),
                        kind="mergesort",
                    )
                    - np.sort(
                        final_defect.reshape(-1),
                        kind="mergesort",
                    )
                )
            )
        )

        sorted_orientation_difference = float(
            np.max(
                np.abs(
                    sorted_orientation_angles(
                        base_orientation_x,
                        base_orientation_y,
                    )
                    - sorted_orientation_angles(
                        final_orientation_x,
                        final_orientation_y,
                    )
                )
            )
        )

        physics_rows.append(
            {
                "carrier_id":
                    int(
                        carrier_row[
                            "carrier_id"
                        ]
                    ),

                "carrier_split":
                    carrier_row["split"],

                "state_id":
                    state_id,

                "state_code":
                    regime["state_code"],

                "initial_energy":
                    float(
                        initial_energy[
                            local_index
                        ]
                    ),

                "final_energy":
                    float(
                        final_energy[
                            local_index
                        ]
                    ),

                "energy_nonincrease":
                    bool(
                        final_energy[
                            local_index
                        ]
                        <= initial_energy[
                            local_index
                        ]
                        + ENERGY_ABSOLUTE_TOLERANCE
                    ),

                "initial_phase_mean":
                    initial_phase_mean,

                "final_phase_mean":
                    final_phase_mean,

                "phase_mean_drift":
                    abs(
                        final_phase_mean
                        - initial_phase_mean
                    ),

                "initial_defect_mean":
                    initial_defect_mean,

                "final_defect_mean":
                    final_defect_mean,

                "defect_mean_drift":
                    abs(
                        final_defect_mean
                        - initial_defect_mean
                    ),

                "phase_minimum":
                    float(
                        final_phase.min()
                    ),

                "phase_maximum":
                    float(
                        final_phase.max()
                    ),

                "defect_minimum":
                    float(
                        final_defect.min()
                    ),

                "defect_maximum":
                    float(
                        final_defect.max()
                    ),

                "orientation_norm_minimum":
                    float(
                        orientation_norm.min()
                    ),

                "orientation_norm_maximum":
                    float(
                        orientation_norm.max()
                    ),

                "all_values_finite":
                    bool(
                        np.isfinite(
                            generated[
                                local_index
                            ]
                        ).all()
                    ),

                "accepted_step_count":
                    int(
                        result[
                            "accepted_step_count"
                        ]
                    ),

                "rejected_step_count":
                    int(
                        result[
                            "rejected_step_count"
                        ]
                    ),

                "minimum_used_time_step":
                    float(
                        result[
                            "minimum_used_time_step"
                        ]
                    ),
            }
        )

        morphology_rows.append(
            {
                "carrier_id":
                    int(
                        carrier_row[
                            "carrier_id"
                        ]
                    ),

                "carrier_split":
                    carrier_row["split"],

                "state_id":
                    state_id,

                "state_code":
                    regime["state_code"],

                "coarsening_level":
                    int(
                        regime[
                            "coarsening_level"
                        ]
                    ),

                "anisotropy_level":
                    int(
                        regime[
                            "anisotropy_level"
                        ]
                    ),

                "defect_recovery_level":
                    int(
                        regime[
                            "defect_recovery_level"
                        ]
                    ),

                "target_characteristic_length":
                    float(
                        target_lengths[
                            local_index
                        ]
                    ),

                "measured_characteristic_length":
                    float(
                        estimate_characteristic_length(
                            final_phase
                        )
                    ),

                "interface_density":
                    float(
                        estimate_interface_density(
                            final_phase
                        )
                    ),

                "maximum_sorted_phase_difference":
                    sorted_phase_difference,

                "maximum_sorted_defect_difference":
                    sorted_defect_difference,

                "maximum_sorted_orientation_angle_difference":
                    sorted_orientation_difference,
            }
        )

    return physics_rows, morphology_rows


def create_paired_coarsening_rows(
    morphology_rows,
):
    lookup = {
        (
            int(row["carrier_id"]),
            int(row["coarsening_level"]),
            int(row["anisotropy_level"]),
            int(row["defect_recovery_level"]),
        ):
            row
        for row in morphology_rows
    }

    carrier_ids = sorted(
        {
            int(row["carrier_id"])
            for row in morphology_rows
        }
    )

    rows = []

    for carrier_id in carrier_ids:
        for anisotropy_level in (
            0,
            1,
        ):
            for defect_recovery_level in (
                0,
                1,
            ):
                fine = lookup[
                    (
                        carrier_id,
                        0,
                        anisotropy_level,
                        defect_recovery_level,
                    )
                ]

                coarse = lookup[
                    (
                        carrier_id,
                        1,
                        anisotropy_level,
                        defect_recovery_level,
                    )
                ]

                fine_length = float(
                    fine[
                        "measured_characteristic_length"
                    ]
                )

                coarse_length = float(
                    coarse[
                        "measured_characteristic_length"
                    ]
                )

                fine_interface = float(
                    fine["interface_density"]
                )

                coarse_interface = float(
                    coarse["interface_density"]
                )

                rows.append(
                    {
                        "carrier_id":
                            carrier_id,

                        "carrier_split":
                            fine[
                                "carrier_split"
                            ],

                        "anisotropy_level":
                            anisotropy_level,

                        "defect_recovery_level":
                            defect_recovery_level,

                        "fine_state_id":
                            int(
                                fine["state_id"]
                            ),

                        "coarse_state_id":
                            int(
                                coarse["state_id"]
                            ),

                        "fine_characteristic_length":
                            fine_length,

                        "coarse_characteristic_length":
                            coarse_length,

                        "coarse_to_fine_length_ratio":
                            coarse_length
                            / fine_length,

                        "fine_interface_density":
                            fine_interface,

                        "coarse_interface_density":
                            coarse_interface,

                        "coarse_to_fine_interface_density_ratio":
                            coarse_interface
                            / fine_interface,
                    }
                )

    expected_count = (
        DEVELOPMENT_CARRIER_COUNT
        * 4
    )

    if len(rows) != expected_count:
        raise AssertionError(
            "Paired coarsening diagnostic count changed."
        )

    return rows


def summarize_physics(rows):
    return {
        "simulation_count":
            len(rows),

        "all_values_finite":
            bool(
                all(
                    row["all_values_finite"]
                    for row in rows
                )
            ),

        "energy_nonincrease_fraction":
            float(
                np.mean(
                    [
                        row[
                            "energy_nonincrease"
                        ]
                        for row in rows
                    ]
                )
            ),

        "maximum_phase_mean_drift":
            float(
                max(
                    row[
                        "phase_mean_drift"
                    ]
                    for row in rows
                )
            ),

        "maximum_defect_mean_drift":
            float(
                max(
                    row[
                        "defect_mean_drift"
                    ]
                    for row in rows
                )
            ),

        "phase_minimum":
            float(
                min(
                    row["phase_minimum"]
                    for row in rows
                )
            ),

        "phase_maximum":
            float(
                max(
                    row["phase_maximum"]
                    for row in rows
                )
            ),

        "defect_minimum":
            float(
                min(
                    row["defect_minimum"]
                    for row in rows
                )
            ),

        "defect_maximum":
            float(
                max(
                    row["defect_maximum"]
                    for row in rows
                )
            ),

        "total_rejected_steps":
            int(
                sum(
                    row[
                        "rejected_step_count"
                    ]
                    for row in rows
                )
            ),
    }


def summarize_morphology(
    morphology_rows,
    paired_rows,
):
    length_ratios = np.asarray(
        [
            row[
                "coarse_to_fine_length_ratio"
            ]
            for row in paired_rows
        ],
        dtype=np.float64,
    )

    interface_ratios = np.asarray(
        [
            row[
                "coarse_to_fine_interface_density_ratio"
            ]
            for row in paired_rows
        ],
        dtype=np.float64,
    )

    maximum_phase_difference = float(
        max(
            row[
                "maximum_sorted_phase_difference"
            ]
            for row in morphology_rows
        )
    )

    maximum_defect_difference = float(
        max(
            row[
                "maximum_sorted_defect_difference"
            ]
            for row in morphology_rows
        )
    )

    maximum_orientation_difference = float(
        max(
            row[
                "maximum_sorted_orientation_angle_difference"
            ]
            for row in morphology_rows
        )
    )

    preliminary_gate_passed = bool(
        float(
            length_ratios.mean()
        )
        >= MINIMUM_MEAN_LENGTH_RATIO
        and float(
            length_ratios.min()
        )
        >= MINIMUM_WORST_LENGTH_RATIO
        and float(
            interface_ratios.mean()
        )
        <= MAXIMUM_MEAN_INTERFACE_RATIO
        and maximum_phase_difference
        <= MAXIMUM_SORTED_PHASE_DIFFERENCE
        and maximum_defect_difference
        <= MAXIMUM_SORTED_DEFECT_DIFFERENCE
        and maximum_orientation_difference
        <= MAXIMUM_SORTED_ORIENTATION_ANGLE_DIFFERENCE
    )

    return {
        "paired_comparison_count":
            len(paired_rows),

        "mean_coarse_to_fine_characteristic_length_ratio":
            float(
                length_ratios.mean()
            ),

        "minimum_coarse_to_fine_characteristic_length_ratio":
            float(
                length_ratios.min()
            ),

        "maximum_coarse_to_fine_characteristic_length_ratio":
            float(
                length_ratios.max()
            ),

        "mean_coarse_to_fine_interface_density_ratio":
            float(
                interface_ratios.mean()
            ),

        "minimum_coarse_to_fine_interface_density_ratio":
            float(
                interface_ratios.min()
            ),

        "maximum_coarse_to_fine_interface_density_ratio":
            float(
                interface_ratios.max()
            ),

        "maximum_sorted_phase_difference":
            maximum_phase_difference,

        "maximum_sorted_defect_difference":
            maximum_defect_difference,

        "maximum_sorted_orientation_angle_difference":
            maximum_orientation_difference,

        "minimum_required_mean_length_ratio":
            MINIMUM_MEAN_LENGTH_RATIO,

        "minimum_required_worst_length_ratio":
            MINIMUM_WORST_LENGTH_RATIO,

        "maximum_allowed_mean_interface_ratio":
            MAXIMUM_MEAN_INTERFACE_RATIO,

        "preliminary_coarsening_morphology_gate_passed":
            preliminary_gate_passed,
    }


def main():
    arguments = parse_arguments()

    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PRIVILEGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "phase4br2_field_bank_summary.json"
    )

    if (
        summary_path.exists()
        and not arguments.force
    ):
        existing = load_json(
            summary_path
        )

        if (
            existing.get(
                "field_bank_generation_status"
            )
            == "completed"
        ):
            print(
                "Tier C v3 development field bank "
                "already exists."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    protocol_summary = validate_protocol()

    device = resolve_device(
        arguments.device
    )

    torch.manual_seed(
        SOLVER_SEED
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            SOLVER_SEED
        )

        torch.set_float32_matmul_precision(
            "high"
        )

    start_time = time.perf_counter()

    carrier_rows = (
        load_development_carriers()
    )

    parameter_schema = (
        load_parameter_schema()
    )

    state_regimes = (
        load_state_regimes()
    )

    (
        carrier_parameters,
        carrier_latent,
        carrier_parameter_rows,
    ) = generate_carrier_parameters(
        carrier_rows=carrier_rows,
        parameter_schema=parameter_schema,
    )

    base_fields, carrier_angles = (
        create_base_fields(
            carrier_rows=carrier_rows,
            parameters=carrier_parameters,
            latent=carrier_latent,
        )
    )

    field_bank = np.empty(
        (
            DEVELOPMENT_CARRIER_COUNT,
            STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    physics_rows = []
    morphology_rows = []

    for regime in state_regimes:
        state_id = int(
            regime["state_id"]
        )

        print()
        print(
            f"Generating Tier C v3 state "
            f"{state_id} ({regime['state_code']})"
        )

        (
            targets,
            target_lengths,
        ) = create_state_targets(
            base_fields=base_fields,
            carrier_rows=carrier_rows,
            parameters=carrier_parameters,
            latent=carrier_latent,
            carrier_angles=carrier_angles,
            regime=regime,
        )

        for start in range(
            0,
            len(carrier_rows),
            arguments.batch_size,
        ):
            end = min(
                len(carrier_rows),
                start
                + arguments.batch_size,
            )

            initial_tensor = torch.from_numpy(
                base_fields[
                    start:end
                ].copy()
            ).to(
                device=device,
                dtype=torch.float32,
            )

            target_tensor = torch.from_numpy(
                targets[
                    start:end
                ].copy()
            ).to(
                device=device,
                dtype=torch.float32,
            )

            parameter_tensor = torch.from_numpy(
                carrier_parameters[
                    start:end
                ].astype(
                    np.float32,
                    copy=True,
                )
            ).to(
                device=device,
                dtype=torch.float32,
            )

            result = simulate_relaxation_batch(
                initial_fields=initial_tensor,
                target_fields=target_tensor,
                carrier_parameters=parameter_tensor,
                regime=regime,
            )

            generated = (
                result["fields"]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            if not np.isfinite(
                generated
            ).all():
                raise FloatingPointError(
                    "Generated Tier C v3 fields "
                    "contain NaN or Inf."
                )

            field_bank[
                start:end,
                state_id,
            ] = generated

            (
                batch_physics_rows,
                batch_morphology_rows,
            ) = create_diagnostic_rows(
                carrier_rows=(
                    carrier_rows[
                        start:end
                    ]
                ),
                state_id=state_id,
                regime=regime,
                target_lengths=(
                    target_lengths[
                        start:end
                    ]
                ),
                base_fields=(
                    base_fields[
                        start:end
                    ]
                ),
                initial_fields=initial_tensor,
                result=result,
            )

            physics_rows.extend(
                batch_physics_rows
            )

            morphology_rows.extend(
                batch_morphology_rows
            )

            print(
                f"  development carriers "
                f"{start:03d}-{end - 1:03d}"
            )

    if len(physics_rows) != EXPECTED_FIELD_COUNT:
        raise AssertionError(
            "Tier C v3 physics diagnostic count changed."
        )

    if (
        len(morphology_rows)
        != EXPECTED_FIELD_COUNT
    ):
        raise AssertionError(
            "Tier C v3 morphology diagnostic count changed."
        )

    paired_rows = (
        create_paired_coarsening_rows(
            morphology_rows
        )
    )

    physics_summary = summarize_physics(
        physics_rows
    )

    morphology_summary = (
        summarize_morphology(
            morphology_rows=morphology_rows,
            paired_rows=paired_rows,
        )
    )

    carrier_ids = np.asarray(
        [
            int(row["carrier_id"])
            for row in carrier_rows
        ],
        dtype=np.int64,
    )

    carrier_splits = np.asarray(
        [
            row["split"]
            for row in carrier_rows
        ],
        dtype="<U5",
    )

    training_local_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_local_indices = np.where(
        carrier_splits == "val"
    )[0]

    if (
        len(training_local_indices)
        != TRAIN_CARRIER_COUNT
    ):
        raise AssertionError(
            "Training carrier count changed."
        )

    if (
        len(validation_local_indices)
        != VALIDATION_CARRIER_COUNT
    ):
        raise AssertionError(
            "Validation carrier count changed."
        )

    training_fields = field_bank[
        training_local_indices
    ]

    channel_mean = training_fields.mean(
        axis=(0, 1, 3, 4),
        dtype=np.float64,
    )

    channel_standard_deviation = (
        training_fields.std(
            axis=(0, 1, 3, 4),
            dtype=np.float64,
        )
    )

    if np.any(
        channel_standard_deviation < 1e-8
    ):
        raise AssertionError(
            "At least one Tier C v3 channel "
            "has zero training variance."
        )

    normalized_bank = (
        (
            field_bank
            - channel_mean[
                None,
                None,
                :,
                None,
                None,
            ]
        )
        / channel_standard_deviation[
            None,
            None,
            :,
            None,
            None,
        ]
    ).astype(
        np.float32
    )

    np.save(
        PRIVILEGED_DIR
        / "development_carrier_ids.npy",
        carrier_ids,
    )

    np.save(
        PRIVILEGED_DIR
        / "development_carrier_splits.npy",
        carrier_splits,
    )

    np.save(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_raw.npy",
        field_bank,
    )

    np.save(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_normalized.npy",
        normalized_bank,
    )

    write_csv(
        PRIVILEGED_DIR
        / "development_physics_diagnostics.csv",
        physics_rows,
    )

    write_csv(
        PRIVILEGED_DIR
        / "development_morphology_diagnostics.csv",
        morphology_rows,
    )

    write_csv(
        PRIVILEGED_DIR
        / "paired_coarsening_diagnostics.csv",
        paired_rows,
    )

    carrier_lookup = {
        str(
            int(carrier_id)
        ):
            int(local_index)
        for local_index, carrier_id
        in enumerate(carrier_ids)
    }

    write_json(
        PRIVILEGED_DIR
        / "carrier_id_to_local_index.json",
        carrier_lookup,
    )

    normalization = {
        "protocol_version":
            PROTOCOL_VERSION,

        "statistics_source":
            "clean training-carrier fields only",

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "test_carrier_count":
            0,

        "channel_order": [
            "phase_field",
            "orientation_cosine",
            "orientation_sine",
            "defect_density",
        ],

        "channel_mean":
            channel_mean.tolist(),

        "channel_standard_deviation":
            channel_standard_deviation.tolist(),

        "validation_statistics_used":
            False,

        "test_statistics_used":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "normalization_v3.json",
        normalization,
    )

    write_json(
        OUTPUT_DIR
        / "coarsening_morphology_summary.json",
        morphology_summary,
    )

    input_hashes = {
        "phase4ar2_summary":
            sha256_file(
                PHASE4AR2_DIR
                / "phase4ar2_summary.json"
            ),

        "carrier_split":
            sha256_file(
                PHASE4AR2_DIR
                / "carrier_split_v3.csv"
            ),

        "parameter_schema":
            sha256_file(
                PHASE4AR2_DIR
                / "carrier_parameter_schema_v3.csv"
            ),

        "state_regimes":
            sha256_file(
                PHASE4AR2_DIR
                / "state_morphology_regimes_v3.csv"
            ),

        "coarsening_protocol":
            sha256_file(
                PHASE4AR2_DIR
                / "coarsening_protocol.json"
            ),

        "numerical_protocol":
            sha256_file(
                PHASE4AR2_DIR
                / "numerical_stabilization_protocol.json"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "field_bank_input_hashes.json",
        input_hashes,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    summary = {
        "phase":
            (
                "4B-R2 Tier C v3 development "
                "field generation"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "phase4ar2_status":
            protocol_summary[
                "phase4ar2_status"
            ],

        "device":
            str(device),

        "torch_version":
            torch.__version__,

        "numpy_version":
            np.__version__,

        "python_version":
            platform.python_version(),

        "training_carrier_count":
            TRAIN_CARRIER_COUNT,

        "validation_carrier_count":
            VALIDATION_CARRIER_COUNT,

        "test_carrier_count_generated":
            0,

        "development_carrier_count":
            DEVELOPMENT_CARRIER_COUNT,

        "state_count":
            STATE_COUNT,

        "field_count":
            EXPECTED_FIELD_COUNT,

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "physics_diagnostic_count":
            len(physics_rows),

        "morphology_diagnostic_count":
            len(morphology_rows),

        "paired_coarsening_comparison_count":
            len(paired_rows),

        "all_values_finite":
            physics_summary[
                "all_values_finite"
            ],

        "energy_nonincrease_fraction":
            physics_summary[
                "energy_nonincrease_fraction"
            ],

        "maximum_phase_mean_drift":
            physics_summary[
                "maximum_phase_mean_drift"
            ],

        "maximum_defect_mean_drift":
            physics_summary[
                "maximum_defect_mean_drift"
            ],

        "phase_minimum":
            physics_summary[
                "phase_minimum"
            ],

        "phase_maximum":
            physics_summary[
                "phase_maximum"
            ],

        "defect_minimum":
            physics_summary[
                "defect_minimum"
            ],

        "defect_maximum":
            physics_summary[
                "defect_maximum"
            ],

        "total_rejected_steps":
            physics_summary[
                "total_rejected_steps"
            ],

        "mean_coarse_to_fine_characteristic_length_ratio":
            morphology_summary[
                "mean_coarse_to_fine_characteristic_length_ratio"
            ],

        "minimum_coarse_to_fine_characteristic_length_ratio":
            morphology_summary[
                "minimum_coarse_to_fine_characteristic_length_ratio"
            ],

        "mean_coarse_to_fine_interface_density_ratio":
            morphology_summary[
                "mean_coarse_to_fine_interface_density_ratio"
            ],

        "maximum_sorted_phase_difference":
            morphology_summary[
                "maximum_sorted_phase_difference"
            ],

        "maximum_sorted_defect_difference":
            morphology_summary[
                "maximum_sorted_defect_difference"
            ],

        "maximum_sorted_orientation_angle_difference":
            morphology_summary[
                "maximum_sorted_orientation_angle_difference"
            ],

        "preliminary_coarsening_morphology_gate_passed":
            morphology_summary[
                "preliminary_coarsening_morphology_gate_passed"
            ],

        "normalization_source":
            "clean training carriers only",

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "diagnostic_decoder_fitted":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_outputs_modified":
            False,

        "phase2_outputs_modified":
            False,

        "phase3_outputs_modified":
            False,

        "elapsed_seconds":
            elapsed_seconds,

        "field_bank_generation_status":
            "completed",
    }

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "Phase 4B-R2 Tier C v3 development "
        "field generation completed."
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
