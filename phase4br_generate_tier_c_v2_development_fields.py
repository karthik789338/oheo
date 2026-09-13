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


PHASE4AR_DIR = Path(
    "outputs/phase4ar_tier_c_v2_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4br_tier_c_v2_development_data"
)

PRIVILEGED_DIR = OUTPUT_DIR / "privileged"


PROTOCOL_VERSION = "tier_c_v2"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

EXPECTED_DEVELOPMENT_CARRIER_COUNT = 128
EXPECTED_FIELD_COUNT = (
    EXPECTED_DEVELOPMENT_CARRIER_COUNT
    * STATE_COUNT
)

CARRIER_PARAMETER_SEED = 51027
FIELD_INITIALIZATION_SEED = 51028
SOLVER_SEED = 51029

INITIAL_TIME_STEP = 0.005
MINIMUM_TIME_STEP = (
    INITIAL_TIME_STEP / 32.0
)
MAXIMUM_STEP_HALVINGS = 5

ENERGY_ABSOLUTE_TOLERANCE = 1e-7
ENERGY_RELATIVE_TOLERANCE = 1e-5

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
        PHASE4AR_DIR
        / "phase4ar_summary.json"
    )

    if (
        summary["phase4ar_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R is not frozen."
        )

    if (
        summary["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v2 protocol version changed."
        )

    if (
        summary[
            "v1_v2_test_carrier_id_overlap"
        ]
        != 0
    ):
        raise AssertionError(
            "Tier C v2 test carriers overlap v1."
        )

    if (
        summary[
            "test_carrier_parameters_generated"
        ]
        is not False
    ):
        raise AssertionError(
            "Test parameters were already generated."
        )

    if (
        summary["test_fields_generated"]
        is not False
    ):
        raise AssertionError(
            "Test fields were already generated."
        )

    if (
        summary["training_performed"]
        is not False
    ):
        raise AssertionError(
            "Training occurred before Phase 4B-R."
        )

    return summary


def load_development_carriers():
    rows = load_csv(
        PHASE4AR_DIR
        / "carrier_split_v2.csv"
    )

    if len(rows) != 160:
        raise AssertionError(
            "Tier C v2 carrier count changed."
        )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != {
        "train": 96,
        "val": 32,
        "test": 32,
    }:
        raise AssertionError(
            f"Carrier counts changed: {dict(counts)}"
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
        != EXPECTED_DEVELOPMENT_CARRIER_COUNT
    ):
        raise AssertionError(
            "Development-carrier count changed."
        )

    return development_rows


def load_parameter_schema():
    rows = load_csv(
        PHASE4AR_DIR
        / "carrier_parameter_schema_v2.csv"
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
            "Tier C v2 parameter schema changed."
        )

    return rows


def load_state_regimes():
    rows = load_csv(
        PHASE4AR_DIR
        / "state_morphology_regimes_v2.csv"
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
            "Tier C v2 state IDs changed."
        )

    parsed = []

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
    )

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
    schema,
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
            schema
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


def anisotropic_smooth(
    field,
    sigma_parallel,
    sigma_perpendicular,
    angle,
):
    kx = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            GRID_WIDTH
        )
    )

    ky = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            GRID_HEIGHT
        )
    )

    kx_grid = kx[None, :]
    ky_grid = ky[:, None]

    cosine = math.cos(angle)
    sine = math.sin(angle)

    parallel = (
        cosine * kx_grid
        + sine * ky_grid
    )

    perpendicular = (
        -sine * kx_grid
        + cosine * ky_grid
    )

    filter_value = np.exp(
        -0.5
        * (
            sigma_parallel**2
            * parallel**2
            + sigma_perpendicular**2
            * perpendicular**2
        )
    )

    result = np.fft.ifft2(
        np.fft.fft2(field)
        * filter_value
    ).real

    return result.astype(
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

    smoothed = anisotropic_smooth(
        field=noise,
        sigma_parallel=scale,
        sigma_perpendicular=scale,
        angle=0.0,
    )

    smoothed -= smoothed.mean()

    standard_deviation = smoothed.std()

    if standard_deviation < 1e-8:
        raise FloatingPointError(
            "Smoothed field has zero variance."
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
                    logits
                    + midpoint
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
                    logits
                    + shift
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
        ).astype(np.float32),

        output_y.reshape(
            reference_y.shape
        ).astype(np.float32),
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

    coordinate_y, coordinate_x = np.meshgrid(
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

    carrier_angles = np.empty(
        len(carrier_rows),
        dtype=np.float32,
    )

    for local_index, row in enumerate(
        carrier_rows
    ):
        carrier_id = int(
            row["carrier_id"]
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
            "Base fields contain NaN or Inf."
        )

    np.save(
        PRIVILEGED_DIR
        / "development_base_fields.npy",
        fields,
    )

    return fields, carrier_angles


def create_state_targets(
    base_fields,
    carrier_rows,
    parameters,
    carrier_angles,
    regime,
):
    targets = np.empty_like(
        base_fields
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

    coordinate_y, coordinate_x = np.meshgrid(
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

    for local_index, row in enumerate(
        carrier_rows
    ):
        carrier_id = int(
            row["carrier_id"]
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

        base_length_scale = float(
            parameters[
                local_index,
                1,
            ]
        )

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

        if coarsening_level == 0:
            morphology_scale = (
                0.80
                * base_length_scale
            )
        else:
            morphology_scale = (
                2.20
                * base_length_scale
            )

        if anisotropy_level == 0:
            parallel_scale = (
                morphology_scale
            )

            perpendicular_scale = (
                morphology_scale
            )

        else:
            parallel_scale = (
                2.75
                * morphology_scale
            )

            perpendicular_scale = max(
                0.45
                * morphology_scale,
                0.8,
            )

        phase_score = anisotropic_smooth(
            field=base_phase,
            sigma_parallel=parallel_scale,
            sigma_perpendicular=(
                perpendicular_scale
            ),
            angle=preferred_angle,
        )

        if anisotropy_level == 1:
            directional_coordinate = (
                math.cos(
                    preferred_angle
                )
                * coordinate_x
                + math.sin(
                    preferred_angle
                )
                * coordinate_y
            )

            phase_score = (
                phase_score
                + 0.20
                * np.cos(
                    2.0
                    * np.pi
                    * directional_coordinate
                    / max(
                        6.0,
                        morphology_scale
                        * 2.0,
                    )
                )
            )

        target_phase = (
            rearrange_scalar_values(
                reference=base_phase,
                spatial_score=phase_score,
            )
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
            smoothed_x = anisotropic_smooth(
                field=base_orientation_x,
                sigma_parallel=(
                    orientation_length_scale
                ),
                sigma_perpendicular=(
                    orientation_length_scale
                ),
                angle=0.0,
            )

            smoothed_y = anisotropic_smooth(
                field=base_orientation_y,
                sigma_parallel=(
                    orientation_length_scale
                ),
                sigma_perpendicular=(
                    orientation_length_scale
                ),
                angle=0.0,
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
                f"state={regime['state_id']}|"
                "defect_target"
            ),
            max(
                1.0,
                float(
                    regime[
                        "defect_localization_width"
                    ]
                )
                * base_length_scale,
            ),
        )

        if recovery_level == 0:
            defect_score = (
                0.55
                * anisotropic_smooth(
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

        target_defect = (
            rearrange_scalar_values(
                reference=base_defect,
                spatial_score=defect_score,
            )
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

    return targets


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

    previous_energy = (
        relaxation_energy(
            fields,
            targets,
        )
    )

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

            finite = torch.isfinite(
                proposed
            ).all()

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
                bool(finite.item())
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
                "Tier C v2 relaxation step failed "
                "after all frozen step halvings."
            )

    final_candidate = targets.clone()

    final_energy = relaxation_energy(
        final_candidate,
        targets,
    )

    final_energy_valid = bool(
        torch.all(
            final_energy
            <= previous_energy
            + ENERGY_ABSOLUTE_TOLERANCE
        ).item()
    )

    if not final_energy_valid:
        raise FloatingPointError(
            "Final equilibrium relaxation step "
            "increased energy."
        )

    fields = final_candidate

    return {
        "fields":
            fields,

        "initial_energy":
            relaxation_energy(
                initial_fields,
                targets,
            ),

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


def create_diagnostic_rows(
    carrier_rows,
    state_id,
    initial_fields,
    result,
):
    initial_numpy = (
        initial_fields.detach()
        .cpu()
        .numpy()
    )

    fields_numpy = (
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

    rows = []

    for local_index, carrier_row in enumerate(
        carrier_rows
    ):
        initial_phase_mean = float(
            initial_numpy[
                local_index,
                0,
            ].mean()
        )

        final_phase_mean = float(
            fields_numpy[
                local_index,
                0,
            ].mean()
        )

        initial_defect_mean = float(
            initial_numpy[
                local_index,
                3,
            ].mean()
        )

        final_defect_mean = float(
            fields_numpy[
                local_index,
                3,
            ].mean()
        )

        orientation_norm = np.sqrt(
            fields_numpy[
                local_index,
                1,
            ] ** 2
            + fields_numpy[
                local_index,
                2,
            ] ** 2
        )

        rows.append(
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
                        fields_numpy[
                            local_index,
                            0,
                        ].min()
                    ),

                "phase_maximum":
                    float(
                        fields_numpy[
                            local_index,
                            0,
                        ].max()
                    ),

                "defect_minimum":
                    float(
                        fields_numpy[
                            local_index,
                            3,
                        ].min()
                    ),

                "defect_maximum":
                    float(
                        fields_numpy[
                            local_index,
                            3,
                        ].max()
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
                            fields_numpy[
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

    return rows


def summarize_diagnostics(rows):
    energy_fraction = float(
        np.mean(
            [
                row["energy_nonincrease"]
                for row in rows
            ]
        )
    )

    result = {
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
            energy_fraction,

        "maximum_phase_mean_drift":
            float(
                max(
                    row["phase_mean_drift"]
                    for row in rows
                )
            ),

        "maximum_defect_mean_drift":
            float(
                max(
                    row["defect_mean_drift"]
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
                    row["rejected_step_count"]
                    for row in rows
                )
            ),
    }

    return result


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
        / "phase4br_field_bank_summary.json"
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
                "Tier C v2 development field bank "
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
        schema=parameter_schema,
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
            EXPECTED_DEVELOPMENT_CARRIER_COUNT,
            STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    diagnostic_rows = []

    for regime in state_regimes:
        state_id = int(
            regime["state_id"]
        )

        print()
        print(
            f"Generating Tier C v2 state {state_id} "
            f"({regime['state_code']})"
        )

        targets = create_state_targets(
            base_fields=base_fields,
            carrier_rows=carrier_rows,
            parameters=carrier_parameters,
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
                    "Generated Tier C v2 fields "
                    "contain NaN or Inf."
                )

            field_bank[
                start:end,
                state_id,
            ] = generated

            diagnostic_rows.extend(
                create_diagnostic_rows(
                    carrier_rows=(
                        carrier_rows[
                            start:end
                        ]
                    ),
                    state_id=state_id,
                    initial_fields=(
                        initial_tensor
                    ),
                    result=result,
                )
            )

            print(
                f"  development carriers "
                f"{start:03d}-{end - 1:03d}"
            )

    if (
        len(diagnostic_rows)
        != EXPECTED_FIELD_COUNT
    ):
        raise AssertionError(
            "Tier C v2 diagnostic count changed."
        )

    if not np.isfinite(
        field_bank
    ).all():
        raise FloatingPointError(
            "Tier C v2 field bank is not finite."
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
        ]
    )

    training_local_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_local_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(training_local_indices) != 96:
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(validation_local_indices) != 32:
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
        channel_standard_deviation
        < 1e-8
    ):
        raise AssertionError(
            "At least one Tier C v2 channel "
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
        diagnostic_rows,
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
            96,

        "validation_carrier_count":
            32,

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
        / "normalization_v2.json",
        normalization,
    )

    diagnostics = summarize_diagnostics(
        diagnostic_rows
    )

    input_hashes = {
        "phase4ar_summary":
            sha256_file(
                PHASE4AR_DIR
                / "phase4ar_summary.json"
            ),

        "carrier_split":
            sha256_file(
                PHASE4AR_DIR
                / "carrier_split_v2.csv"
            ),

        "parameter_schema":
            sha256_file(
                PHASE4AR_DIR
                / "carrier_parameter_schema_v2.csv"
            ),

        "state_regimes":
            sha256_file(
                PHASE4AR_DIR
                / "state_morphology_regimes_v2.csv"
            ),

        "numerical_protocol":
            sha256_file(
                PHASE4AR_DIR
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
            "4B-R Tier C v2 development field generation",

        "protocol_version":
            PROTOCOL_VERSION,

        "phase4ar_status":
            protocol_summary[
                "phase4ar_status"
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
            96,

        "validation_carrier_count":
            32,

        "test_carrier_count_generated":
            0,

        "development_carrier_count":
            EXPECTED_DEVELOPMENT_CARRIER_COUNT,

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
            len(diagnostic_rows),

        "all_values_finite":
            diagnostics[
                "all_values_finite"
            ],

        "energy_nonincrease_fraction":
            diagnostics[
                "energy_nonincrease_fraction"
            ],

        "maximum_phase_mean_drift":
            diagnostics[
                "maximum_phase_mean_drift"
            ],

        "maximum_defect_mean_drift":
            diagnostics[
                "maximum_defect_mean_drift"
            ],

        "phase_minimum":
            diagnostics[
                "phase_minimum"
            ],

        "phase_maximum":
            diagnostics[
                "phase_maximum"
            ],

        "defect_minimum":
            diagnostics[
                "defect_minimum"
            ],

        "defect_maximum":
            diagnostics[
                "defect_maximum"
            ],

        "total_rejected_steps":
            diagnostics[
                "total_rejected_steps"
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

        "training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "tier_c_v1_outputs_modified":
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
        "Phase 4B-R Tier C v2 development "
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
