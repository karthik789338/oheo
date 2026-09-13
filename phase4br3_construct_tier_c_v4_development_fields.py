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


PHASE4AR3_DIR = Path(
    "outputs/phase4ar3_tier_c_v4_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4br3_tier_c_v4_development_fields"
)

PRIVILEGED_DIR = OUTPUT_DIR / "privileged"


PROTOCOL_VERSION = "tier_c_v4"

STATE_COUNT = 8
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

TRAIN_CARRIER_COUNT = 96
VALIDATION_CARRIER_COUNT = 32
DEVELOPMENT_CARRIER_COUNT = 128

UNIQUE_PHASE_PAIR_COUNT = (
    DEVELOPMENT_CARRIER_COUNT * 2
)

PAIRED_COMPARISON_COUNT = (
    DEVELOPMENT_CARRIER_COUNT * 2 * 2
)

EXPECTED_FIELD_COUNT = (
    DEVELOPMENT_CARRIER_COUNT * STATE_COUNT
)

CARRIER_PARAMETER_SEED = 71027
FIELD_INITIALIZATION_SEED = 71028
SOLVER_SEED = 71029

FINE_DIFFUSION_SIGMA = 0.75
COARSE_DIFFUSION_SIGMA = 1.25

ANISOTROPIC_PARALLEL_MULTIPLIER = 1.50
ANISOTROPIC_PERPENDICULAR_MULTIPLIER = 0.80

MINIMUM_COARSE_ITERATIONS = 2
MAXIMUM_COARSE_ITERATIONS = 32

INTERNAL_TARGET_INTERFACE_RATIO = 0.62
INTERNAL_TARGET_LENGTH_RATIO = 2.60

EXTERNAL_MAXIMUM_MEAN_INTERFACE_RATIO = 0.70
EXTERNAL_MAXIMUM_PAIR_INTERFACE_RATIO = 0.85
EXTERNAL_MINIMUM_PAIR_PASS_FRACTION = 0.95

EXTERNAL_MINIMUM_MEAN_LENGTH_RATIO = 2.50
EXTERNAL_MINIMUM_WORST_LENGTH_RATIO = 1.80

MAXIMUM_SORTED_PHASE_DIFFERENCE = 1e-7
MAXIMUM_SORTED_DEFECT_DIFFERENCE = 1e-7
MAXIMUM_SORTED_ORIENTATION_PAIR_DIFFERENCE = 1e-7

INITIAL_TIME_STEP = 0.005
MINIMUM_TIME_STEP = INITIAL_TIME_STEP / 32.0
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
        choices=("auto", "cpu", "cuda"),
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
            lambda: handle.read(1024 * 1024),
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


def radical_inverse(index, base):
    value = 0.0
    factor = 1.0 / base
    current = index

    while current > 0:
        digit = current % base
        value += digit * factor
        current //= base
        factor /= base

    return value


def parse_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def validate_protocol():
    summary = load_json(
        PHASE4AR3_DIR
        / "phase4ar3_summary.json"
    )

    protocol = load_json(
        PHASE4AR3_DIR
        / "tier_c_v4_protocol.json"
    )

    terminal_policy = load_json(
        PHASE4AR3_DIR
        / "terminal_revision_policy.json"
    )

    if (
        summary["phase4ar3_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A-R3 is not frozen."
        )

    if (
        summary["protocol_version"]
        != PROTOCOL_VERSION
    ):
        raise AssertionError(
            "Tier C v4 protocol version changed."
        )

    if (
        summary["tier_c_v4_is_final_revision"]
        is not True
    ):
        raise AssertionError(
            "Tier C v4 is not frozen as the final revision."
        )

    if (
        summary["tier_c_v5_authorized"]
        is not False
    ):
        raise AssertionError(
            "Tier C v5 was unexpectedly authorized."
        )

    if (
        summary[
            "coarsening_construction"
        ]
        != (
            "volume-preserving diffusion-threshold "
            "with exact rank transport"
        )
    ):
        raise AssertionError(
            "Tier C v4 coarsening construction changed."
        )

    if (
        float(
            summary[
                "internal_target_interface_ratio"
            ]
        )
        != INTERNAL_TARGET_INTERFACE_RATIO
    ):
        raise AssertionError(
            "Internal interface target changed."
        )

    if (
        float(
            summary[
                "internal_target_length_ratio"
            ]
        )
        != INTERNAL_TARGET_LENGTH_RATIO
    ):
        raise AssertionError(
            "Internal length target changed."
        )

    if (
        summary[
            "validation_test_overlap_count"
        ]
        != 0
    ):
        raise AssertionError(
            "Tier C v4 validation and test IDs overlap."
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
        or summary[
            "test_metrics_computed"
        ]
        is not False
    ):
        raise AssertionError(
            "Tier C v4 test artifacts already exist."
        )

    if (
        summary["diagnostic_decoder_fitted"]
        is not False
        or summary[
            "predictive_training_performed"
        ]
        is not False
        or summary[
            "checkpoint_selection_performed"
        ]
        is not False
    ):
        raise AssertionError(
            "Training occurred before Phase 4B-R3."
        )

    if (
        terminal_policy[
            "tier_c_v5_authorized"
        ]
        is not False
    ):
        raise AssertionError(
            "Terminal revision policy changed."
        )

    return {
        "summary": summary,
        "protocol": protocol,
        "terminal_policy": terminal_policy,
    }


def load_development_carriers():
    rows = load_csv(
        PHASE4AR3_DIR
        / "carrier_split_v4.csv"
    )

    if len(rows) != 160:
        raise AssertionError(
            "Tier C v4 carrier count changed."
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
            f"Tier C v4 carrier counts changed: {counts}"
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
            "A Tier C v4 test carrier entered development."
        )

    return development_rows


def load_parameter_schema():
    rows = load_csv(
        PHASE4AR3_DIR
        / "carrier_parameter_schema_v4.csv"
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
            "Tier C v4 parameter schema changed."
        )

    return rows


def load_state_regimes():
    rows = load_csv(
        PHASE4AR3_DIR
        / "state_morphology_regimes_v4.csv"
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
            "Tier C v4 state IDs changed."
        )

    integer_fields = (
        "state_id",
        "coarsening_level",
        "anisotropy_level",
        "defect_recovery_level",
        "quench_steps",
        "anneal_steps",
        "orientation_anisotropy_harmonic",
        "minimum_coarse_iterations",
        "maximum_coarse_iterations",
    )

    float_fields = (
        "phase_mobility_multiplier",
        "orientation_anisotropy_strength",
        "orientation_relaxation_multiplier",
        "defect_diffusivity_multiplier",
        "defect_interface_coupling",
        "defect_localization_width",
        "internal_target_interface_ratio",
        "internal_target_length_ratio",
    )

    parsed = []

    for row in rows:
        value = dict(row)

        for key in integer_fields:
            value[key] = int(row[key])

        for key in float_fields:
            value[key] = float(row[key])

        parsed.append(value)

    lookup = {}

    for row in parsed:
        key = (
            row["coarsening_level"],
            row["anisotropy_level"],
            row["defect_recovery_level"],
        )

        if key in lookup:
            raise AssertionError(
                f"Duplicate state-factor key: {key}"
            )

        lookup[key] = row

    expected_keys = {
        (coarsening, anisotropy, defect)
        for coarsening in (0, 1)
        for anisotropy in (0, 1)
        for defect in (0, 1)
    }

    if set(lookup) != expected_keys:
        raise AssertionError(
            "Tier C v4 state-factor registry is incomplete."
        )

    return parsed, lookup


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

    latent = np.empty_like(parameters)

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
                    maximum - minimum
                )
            )

            parameters[
                local_index,
                dimension,
            ] = value

            output[
                f"latent_{dimension}"
            ] = float(point[dimension])

            output[
                schema_row["parameter_name"]
            ] = float(value)

        output_rows.append(output)

    write_csv(
        OUTPUT_DIR
        / "development_carrier_parameters.csv",
        output_rows,
    )

    return parameters, latent


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

    return smoothed.astype(np.float32)


def normalize_score(field):
    field = np.asarray(
        field,
        dtype=np.float32,
    )

    field = field - field.mean()

    standard_deviation = float(
        field.std()
    )

    if standard_deviation < 1e-8:
        raise FloatingPointError(
            "Spatial score has zero variance."
        )

    return (
        field / standard_deviation
    ).astype(np.float32)


def smooth_noise(seed, scale):
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

    return normalize_score(smoothed)


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
    ).astype(np.float32)


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
    ).astype(np.float32)


def rank_transport_scalar_values(
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
    ).astype(np.float32)


def rank_transport_orientation_pairs(
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
        np.roll(field, -1, axis=1)
        - np.roll(field, 1, axis=1)
    ) / 2.0

    gradient_y = (
        np.roll(field, -1, axis=0)
        - np.roll(field, 1, axis=0)
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

    preferred_angles = np.empty(
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
        ) = parameters[local_index]

        preferred_angle = float(
            2.0
            * np.pi
            * latent[
                local_index,
                4,
            ]
        )

        preferred_angles[
            local_index
        ] = preferred_angle

        phase_noise = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|base_phase"
            ),
            base_length_scale,
        )

        phase = solve_tanh_mean(
            source=phase_noise,
            target_mean=(
                2.0
                * mean_phase_fraction
                - 1.0
            ),
            width=interface_width,
        )

        orientation_x_seed = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|base_orientation_x"
            ),
            orientation_length_scale,
        )

        orientation_y_seed = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|base_orientation_y"
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
        ).astype(np.float32)

        orientation_y = (
            orientation_y_seed
            / orientation_norm
        ).astype(np.float32)

        defect_noise = smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|base_defect"
            ),
            max(
                1.0,
                0.75
                * base_length_scale,
            ),
        )

        gradient_x, gradient_y = (
            periodic_gradient(phase)
        )

        interface_score = np.sqrt(
            gradient_x**2
            + gradient_y**2
        )

        interface_score = normalize_score(
            interface_score
        )

        defect_logits = (
            0.65
            * defect_noise
            + 0.35
            * interface_score
        )

        defect = solve_logistic_mean(
            logits=defect_logits,
            target_mean=mean_defect_fraction,
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
            "Tier C v4 base fields contain NaN or Inf."
        )

    return fields, preferred_angles


# ------------------------------------------------------------------
# Constructor metrics used only to choose the deterministic iteration.
# Final acceptance metrics are recomputed separately from saved fields.
# ------------------------------------------------------------------

def constructor_characteristic_length(
    phase_field,
):
    centered = (
        phase_field
        - phase_field.mean()
    )

    spectrum = np.fft.fft2(
        centered
    )

    power = np.abs(spectrum) ** 2

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

    total_power = float(
        power[mask].sum()
    )

    if total_power <= 0.0:
        raise FloatingPointError(
            "Constructor structure factor has zero power."
        )

    mean_frequency = float(
        (
            radial_frequency[mask]
            * power[mask]
        ).sum()
        / total_power
    )

    if mean_frequency <= 0.0:
        raise FloatingPointError(
            "Constructor characteristic frequency is invalid."
        )

    return 1.0 / mean_frequency


def constructor_interface_density(
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


def create_fine_score(
    carrier_id,
    anisotropy_level,
    preferred_angle,
):
    generator = np.random.default_rng(
        stable_seed(
            f"carrier={carrier_id}|"
            f"anisotropy={anisotropy_level}|"
            "shared_fine_score"
        )
    )

    raw_score = generator.normal(
        size=(
            GRID_HEIGHT,
            GRID_WIDTH,
        )
    )

    if anisotropy_level == 0:
        sigma_parallel = (
            FINE_DIFFUSION_SIGMA
        )

        sigma_perpendicular = (
            FINE_DIFFUSION_SIGMA
        )

    else:
        sigma_parallel = (
            FINE_DIFFUSION_SIGMA
            * ANISOTROPIC_PARALLEL_MULTIPLIER
        )

        sigma_perpendicular = (
            FINE_DIFFUSION_SIGMA
            * ANISOTROPIC_PERPENDICULAR_MULTIPLIER
        )

    fine_score = anisotropic_gaussian_smooth(
        field=raw_score,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        angle=preferred_angle,
    )

    return normalize_score(fine_score)


def diffuse_coarse_score(
    score,
    anisotropy_level,
    preferred_angle,
):
    if anisotropy_level == 0:
        sigma_parallel = (
            COARSE_DIFFUSION_SIGMA
        )

        sigma_perpendicular = (
            COARSE_DIFFUSION_SIGMA
        )

    else:
        sigma_parallel = (
            COARSE_DIFFUSION_SIGMA
            * ANISOTROPIC_PARALLEL_MULTIPLIER
        )

        sigma_perpendicular = (
            COARSE_DIFFUSION_SIGMA
            * ANISOTROPIC_PERPENDICULAR_MULTIPLIER
        )

    diffused = anisotropic_gaussian_smooth(
        field=score,
        sigma_parallel=sigma_parallel,
        sigma_perpendicular=sigma_perpendicular,
        angle=preferred_angle,
    )

    return normalize_score(diffused)


def construct_phase_pair(
    carrier_id,
    anisotropy_level,
    base_phase,
    preferred_angle,
):
    fine_score = create_fine_score(
        carrier_id=carrier_id,
        anisotropy_level=anisotropy_level,
        preferred_angle=preferred_angle,
    )

    fine_phase = rank_transport_scalar_values(
        reference=base_phase,
        spatial_score=fine_score,
    )

    fine_length = (
        constructor_characteristic_length(
            fine_phase
        )
    )

    fine_interface = (
        constructor_interface_density(
            fine_phase
        )
    )

    if fine_interface <= 0.0:
        raise FloatingPointError(
            "Fine interface density is invalid."
        )

    current_score = fine_score.copy()

    search_rows = []
    selected = None
    best_candidate = None

    for iteration in range(
        1,
        MAXIMUM_COARSE_ITERATIONS + 1,
    ):
        current_score = diffuse_coarse_score(
            score=current_score,
            anisotropy_level=anisotropy_level,
            preferred_angle=preferred_angle,
        )

        coarse_phase = (
            rank_transport_scalar_values(
                reference=base_phase,
                spatial_score=current_score,
            )
        )

        coarse_length = (
            constructor_characteristic_length(
                coarse_phase
            )
        )

        coarse_interface = (
            constructor_interface_density(
                coarse_phase
            )
        )

        length_ratio = (
            coarse_length / fine_length
        )

        interface_ratio = (
            coarse_interface
            / fine_interface
        )

        internal_length_passed = bool(
            length_ratio
            >= INTERNAL_TARGET_LENGTH_RATIO
        )

        internal_interface_passed = bool(
            interface_ratio
            <= INTERNAL_TARGET_INTERFACE_RATIO
        )

        eligible_iteration = bool(
            iteration
            >= MINIMUM_COARSE_ITERATIONS
        )

        both_targets_passed = bool(
            eligible_iteration
            and internal_length_passed
            and internal_interface_passed
        )

        search_rows.append(
            {
                "carrier_id":
                    carrier_id,

                "anisotropy_level":
                    anisotropy_level,

                "iteration":
                    iteration,

                "eligible_iteration":
                    eligible_iteration,

                "fine_characteristic_length":
                    fine_length,

                "coarse_characteristic_length":
                    coarse_length,

                "coarse_to_fine_length_ratio":
                    length_ratio,

                "fine_interface_density":
                    fine_interface,

                "coarse_interface_density":
                    coarse_interface,

                "coarse_to_fine_interface_ratio":
                    interface_ratio,

                "internal_length_target_passed":
                    internal_length_passed,

                "internal_interface_target_passed":
                    internal_interface_passed,

                "both_internal_targets_passed":
                    both_targets_passed,
            }
        )

        score_value = (
            max(
                0.0,
                length_ratio
                - INTERNAL_TARGET_LENGTH_RATIO,
            )
            + max(
                0.0,
                INTERNAL_TARGET_INTERFACE_RATIO
                - interface_ratio,
            )
        )

        if (
            best_candidate is None
            or score_value
            > best_candidate["score_value"]
        ):
            best_candidate = {
                "iteration":
                    iteration,

                "phase":
                    coarse_phase.copy(),

                "coarse_length":
                    coarse_length,

                "coarse_interface":
                    coarse_interface,

                "length_ratio":
                    length_ratio,

                "interface_ratio":
                    interface_ratio,

                "score_value":
                    score_value,
            }

        if both_targets_passed:
            selected = {
                "iteration":
                    iteration,

                "fine_phase":
                    fine_phase,

                "coarse_phase":
                    coarse_phase,

                "fine_length":
                    fine_length,

                "coarse_length":
                    coarse_length,

                "fine_interface":
                    fine_interface,

                "coarse_interface":
                    coarse_interface,

                "length_ratio":
                    length_ratio,

                "interface_ratio":
                    interface_ratio,
            }

            break

    return {
        "selected":
            selected,

        "best_candidate":
            best_candidate,

        "search_rows":
            search_rows,
    }


def create_orientation_target(
    base_orientation_x,
    base_orientation_y,
    phase_field,
    anisotropy_level,
    preferred_angle,
    regime,
    orientation_length_scale,
):
    gradient_x, gradient_y = (
        periodic_gradient(
            phase_field
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
        harmonic = int(
            regime[
                "orientation_anisotropy_harmonic"
            ]
        )

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

    return rank_transport_orientation_pairs(
        reference_x=base_orientation_x,
        reference_y=base_orientation_y,
        target_angle=target_angle,
    )


def create_defect_target(
    carrier_id,
    base_defect,
    phase_field,
    defect_recovery_level,
    regime,
):
    gradient_x, gradient_y = (
        periodic_gradient(
            phase_field
        )
    )

    interface_strength = np.sqrt(
        gradient_x**2
        + gradient_y**2
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

    phase_length = (
        constructor_characteristic_length(
            phase_field
        )
    )

    defect_background = smooth_noise(
        stable_seed(
            f"carrier={carrier_id}|"
            f"defect_recovery={defect_recovery_level}|"
            "shared_defect_background"
        ),
        max(
            1.0,
            float(
                regime[
                    "defect_localization_width"
                ]
            )
            * phase_length,
        ),
    )

    if defect_recovery_level == 0:
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

    return rank_transport_scalar_values(
        reference=base_defect,
        spatial_score=defect_score,
    )


def construct_target_bank(
    carrier_rows,
    parameters,
    base_fields,
    preferred_angles,
    state_lookup,
):
    target_bank = np.empty(
        (
            DEVELOPMENT_CARRIER_COUNT,
            STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    search_rows = []
    selection_rows = []
    failure_rows = []

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

        preferred_angle = float(
            preferred_angles[
                local_index
            ]
        )

        orientation_length_scale = float(
            parameters[
                local_index,
                4,
            ]
        )

        for anisotropy_level in (
            0,
            1,
        ):
            result = construct_phase_pair(
                carrier_id=carrier_id,
                anisotropy_level=anisotropy_level,
                base_phase=base_phase,
                preferred_angle=preferred_angle,
            )

            search_rows.extend(
                result["search_rows"]
            )

            selected = result["selected"]

            if selected is None:
                best = result[
                    "best_candidate"
                ]

                failure_rows.append(
                    {
                        "carrier_id":
                            carrier_id,

                        "carrier_split":
                            carrier_row["split"],

                        "anisotropy_level":
                            anisotropy_level,

                        "maximum_iteration":
                            MAXIMUM_COARSE_ITERATIONS,

                        "best_iteration":
                            best["iteration"],

                        "best_length_ratio":
                            best["length_ratio"],

                        "best_interface_ratio":
                            best["interface_ratio"],

                        "internal_length_target":
                            INTERNAL_TARGET_LENGTH_RATIO,

                        "internal_interface_target":
                            INTERNAL_TARGET_INTERFACE_RATIO,
                    }
                )

                continue

            selection_rows.append(
                {
                    "carrier_id":
                        carrier_id,

                    "carrier_split":
                        carrier_row["split"],

                    "anisotropy_level":
                        anisotropy_level,

                    "selected_iteration":
                        selected["iteration"],

                    "constructor_fine_characteristic_length":
                        selected["fine_length"],

                    "constructor_coarse_characteristic_length":
                        selected["coarse_length"],

                    "constructor_coarse_to_fine_length_ratio":
                        selected["length_ratio"],

                    "constructor_fine_interface_density":
                        selected["fine_interface"],

                    "constructor_coarse_interface_density":
                        selected["coarse_interface"],

                    "constructor_coarse_to_fine_interface_ratio":
                        selected["interface_ratio"],

                    "internal_length_target_passed":
                        bool(
                            selected["length_ratio"]
                            >= INTERNAL_TARGET_LENGTH_RATIO
                        ),

                    "internal_interface_target_passed":
                        bool(
                            selected["interface_ratio"]
                            <= INTERNAL_TARGET_INTERFACE_RATIO
                        ),

                    "both_internal_targets_passed":
                        True,
                }
            )

            phase_by_coarsening = {
                0: selected["fine_phase"],
                1: selected["coarse_phase"],
            }

            for coarsening_level in (
                0,
                1,
            ):
                phase_field = (
                    phase_by_coarsening[
                        coarsening_level
                    ]
                )

                for defect_recovery_level in (
                    0,
                    1,
                ):
                    regime = state_lookup[
                        (
                            coarsening_level,
                            anisotropy_level,
                            defect_recovery_level,
                        )
                    ]

                    (
                        orientation_x,
                        orientation_y,
                    ) = create_orientation_target(
                        base_orientation_x=(
                            base_orientation_x
                        ),
                        base_orientation_y=(
                            base_orientation_y
                        ),
                        phase_field=phase_field,
                        anisotropy_level=(
                            anisotropy_level
                        ),
                        preferred_angle=(
                            preferred_angle
                        ),
                        regime=regime,
                        orientation_length_scale=(
                            orientation_length_scale
                        ),
                    )

                    defect_field = (
                        create_defect_target(
                            carrier_id=carrier_id,
                            base_defect=base_defect,
                            phase_field=phase_field,
                            defect_recovery_level=(
                                defect_recovery_level
                            ),
                            regime=regime,
                        )
                    )

                    state_id = int(
                        regime["state_id"]
                    )

                    target_bank[
                        local_index,
                        state_id,
                        0,
                    ] = phase_field

                    target_bank[
                        local_index,
                        state_id,
                        1,
                    ] = orientation_x

                    target_bank[
                        local_index,
                        state_id,
                        2,
                    ] = orientation_y

                    target_bank[
                        local_index,
                        state_id,
                        3,
                    ] = defect_field

    write_csv(
        PRIVILEGED_DIR
        / "coarsening_iteration_search.csv",
        search_rows,
    )

    if selection_rows:
        write_csv(
            PRIVILEGED_DIR
            / "coarsening_pair_selection.csv",
            selection_rows,
        )

    if failure_rows:
        write_csv(
            PRIVILEGED_DIR
            / "coarsening_construction_failures.csv",
            failure_rows,
        )

    return {
        "target_bank":
            target_bank,

        "search_rows":
            search_rows,

        "selection_rows":
            selection_rows,

        "failure_rows":
            failure_rows,
    }


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


def normalize_orientation_torch(
    orientation_x,
    orientation_y,
):
    norm = torch.sqrt(
        orientation_x**2
        + orientation_y**2
    ).clamp_min(1e-7)

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

    initial_energy = relaxation_energy(
        fields,
        target_fields,
    )

    previous_energy = (
        initial_energy.clone()
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

    accepted_steps = 0
    rejected_steps = 0
    minimum_used_time_step = (
        INITIAL_TIME_STEP
    )

    for _ in range(total_steps):
        time_step = INITIAL_TIME_STEP
        accepted = False

        for _ in range(
            MAXIMUM_STEP_HALVINGS + 1
        ):
            phase_alpha = (
                1.0
                - torch.exp(
                    -phase_rate * time_step
                )
            ).view(-1, 1, 1)

            orientation_alpha = (
                1.0
                - torch.exp(
                    -orientation_rate
                    * time_step
                )
            ).view(-1, 1, 1)

            defect_alpha = (
                1.0
                - torch.exp(
                    -defect_rate * time_step
                )
            ).view(-1, 1, 1)

            phase = (
                fields[:, 0]
                + phase_alpha
                * (
                    target_fields[:, 0]
                    - fields[:, 0]
                )
            )

            orientation_x = (
                fields[:, 1]
                + orientation_alpha
                * (
                    target_fields[:, 1]
                    - fields[:, 1]
                )
            )

            orientation_y = (
                fields[:, 2]
                + orientation_alpha
                * (
                    target_fields[:, 2]
                    - fields[:, 2]
                )
            )

            (
                orientation_x,
                orientation_y,
            ) = normalize_orientation_torch(
                orientation_x,
                orientation_y,
            )

            defect = (
                fields[:, 3]
                + defect_alpha
                * (
                    target_fields[:, 3]
                    - fields[:, 3]
                )
            )

            proposed = torch.stack(
                [
                    phase,
                    orientation_x,
                    orientation_y,
                    defect,
                ],
                dim=1,
            )

            proposed_energy = (
                relaxation_energy(
                    proposed,
                    target_fields,
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
                phase.min().item()
                >= -1.000001
                and phase.max().item()
                <= 1.000001
            )

            defect_bounded = bool(
                defect.min().item()
                >= -0.000001
                and defect.max().item()
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

                accepted_steps += 1

                minimum_used_time_step = min(
                    minimum_used_time_step,
                    time_step,
                )

                accepted = True
                break

            rejected_steps += 1
            time_step /= 2.0

            if time_step < MINIMUM_TIME_STEP:
                break

        if not accepted:
            raise FloatingPointError(
                "Tier C v4 relaxation failed after "
                "all frozen step halvings."
            )

    final_fields = target_fields.clone()

    final_energy = relaxation_energy(
        final_fields,
        target_fields,
    )

    if not bool(
        torch.all(
            final_energy
            <= previous_energy
            + ENERGY_ABSOLUTE_TOLERANCE
        ).item()
    ):
        raise FloatingPointError(
            "Final target projection increased energy."
        )

    return {
        "fields":
            final_fields,

        "initial_energy":
            initial_energy,

        "final_energy":
            final_energy,

        "accepted_step_count":
            accepted_steps + 1,

        "rejected_step_count":
            rejected_steps,

        "minimum_used_time_step":
            minimum_used_time_step,
    }


def create_physics_rows(
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

    final_numpy = (
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
            final_numpy[
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
            final_numpy[
                local_index,
                3,
            ].mean()
        )

        orientation_norm = np.sqrt(
            final_numpy[
                local_index,
                1,
            ] ** 2
            + final_numpy[
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

                "phase_mean_drift":
                    abs(
                        final_phase_mean
                        - initial_phase_mean
                    ),

                "defect_mean_drift":
                    abs(
                        final_defect_mean
                        - initial_defect_mean
                    ),

                "phase_minimum":
                    float(
                        final_numpy[
                            local_index,
                            0,
                        ].min()
                    ),

                "phase_maximum":
                    float(
                        final_numpy[
                            local_index,
                            0,
                        ].max()
                    ),

                "defect_minimum":
                    float(
                        final_numpy[
                            local_index,
                            3,
                        ].min()
                    ),

                "defect_maximum":
                    float(
                        final_numpy[
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
                            final_numpy[
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


def run_relaxation(
    carrier_rows,
    parameters,
    base_fields,
    target_bank,
    state_regimes,
    device,
    batch_size,
):
    final_bank = np.empty_like(
        target_bank
    )

    physics_rows = []

    for regime in state_regimes:
        state_id = int(
            regime["state_id"]
        )

        print(
            f"Relaxing state {state_id} "
            f"({regime['state_code']})"
        )

        for start in range(
            0,
            DEVELOPMENT_CARRIER_COUNT,
            batch_size,
        ):
            end = min(
                DEVELOPMENT_CARRIER_COUNT,
                start + batch_size,
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
                target_bank[
                    start:end,
                    state_id,
                ].copy()
            ).to(
                device=device,
                dtype=torch.float32,
            )

            parameter_tensor = torch.from_numpy(
                parameters[
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
                .astype(np.float32)
            )

            final_bank[
                start:end,
                state_id,
            ] = generated

            physics_rows.extend(
                create_physics_rows(
                    carrier_rows=(
                        carrier_rows[
                            start:end
                        ]
                    ),
                    state_id=state_id,
                    initial_fields=initial_tensor,
                    result=result,
                )
            )

    return final_bank, physics_rows


# ------------------------------------------------------------------
# Independent verification code.
# These functions do not call the constructor metric implementations.
# ------------------------------------------------------------------

def independent_characteristic_length(
    phase_field,
):
    centered = np.asarray(
        phase_field,
        dtype=np.float64,
    )

    centered = (
        centered - centered.mean()
    )

    power = np.abs(
        np.fft.fft2(centered)
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
            "Independent structure factor has zero power."
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
            "Independent characteristic frequency is invalid."
        )

    return 1.0 / mean_frequency


def independent_interface_density(
    phase_field,
):
    phase = np.asarray(
        phase_field,
        dtype=np.float64,
    )

    gradient_x = (
        np.roll(
            phase,
            -1,
            axis=1,
        )
        - np.roll(
            phase,
            1,
            axis=1,
        )
    ) / 2.0

    gradient_y = (
        np.roll(
            phase,
            -1,
            axis=0,
        )
        - np.roll(
            phase,
            1,
            axis=0,
        )
    ) / 2.0

    return float(
        np.mean(
            np.sqrt(
                gradient_x**2
                + gradient_y**2
            )
        )
    )


def sorted_scalar_difference(
    first,
    second,
):
    first_sorted = np.sort(
        np.asarray(
            first,
            dtype=np.float32,
        ).reshape(-1),
        kind="mergesort",
    )

    second_sorted = np.sort(
        np.asarray(
            second,
            dtype=np.float32,
        ).reshape(-1),
        kind="mergesort",
    )

    return float(
        np.max(
            np.abs(
                first_sorted
                - second_sorted
            )
        )
    )


def sorted_orientation_pair_difference(
    base_x,
    base_y,
    observed_x,
    observed_y,
):
    base_pairs = np.stack(
        [
            base_x.reshape(-1),
            base_y.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    observed_pairs = np.stack(
        [
            observed_x.reshape(-1),
            observed_y.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    base_order = np.lexsort(
        (
            base_pairs[:, 1],
            base_pairs[:, 0],
        )
    )

    observed_order = np.lexsort(
        (
            observed_pairs[:, 1],
            observed_pairs[:, 0],
        )
    )

    return float(
        np.max(
            np.abs(
                base_pairs[base_order]
                - observed_pairs[
                    observed_order
                ]
            )
        )
    )


def independent_verify_saved_fields(
    carrier_rows,
    base_fields,
    state_lookup,
    selection_lookup,
):
    saved_bank = np.load(
        PRIVILEGED_DIR
        / "development_carrier_state_fields_raw.npy",
        mmap_mode="r",
    )

    expected_shape = (
        DEVELOPMENT_CARRIER_COUNT,
        STATE_COUNT,
        FIELD_CHANNEL_COUNT,
        GRID_HEIGHT,
        GRID_WIDTH,
    )

    if saved_bank.shape != expected_shape:
        raise AssertionError(
            f"Saved Tier C v4 field shape changed: "
            f"{saved_bank.shape}"
        )

    rows = []

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

        for anisotropy_level in (
            0,
            1,
        ):
            selection = selection_lookup[
                (
                    carrier_id,
                    anisotropy_level,
                )
            ]

            for defect_recovery_level in (
                0,
                1,
            ):
                fine_state = state_lookup[
                    (
                        0,
                        anisotropy_level,
                        defect_recovery_level,
                    )
                ]

                coarse_state = state_lookup[
                    (
                        1,
                        anisotropy_level,
                        defect_recovery_level,
                    )
                ]

                fine_field = np.asarray(
                    saved_bank[
                        local_index,
                        int(
                            fine_state[
                                "state_id"
                            ]
                        ),
                    ],
                    dtype=np.float32,
                )

                coarse_field = np.asarray(
                    saved_bank[
                        local_index,
                        int(
                            coarse_state[
                                "state_id"
                            ]
                        ),
                    ],
                    dtype=np.float32,
                )

                fine_length = (
                    independent_characteristic_length(
                        fine_field[0]
                    )
                )

                coarse_length = (
                    independent_characteristic_length(
                        coarse_field[0]
                    )
                )

                fine_interface = (
                    independent_interface_density(
                        fine_field[0]
                    )
                )

                coarse_interface = (
                    independent_interface_density(
                        coarse_field[0]
                    )
                )

                length_ratio = (
                    coarse_length
                    / fine_length
                )

                interface_ratio = (
                    coarse_interface
                    / fine_interface
                )

                phase_difference = max(
                    sorted_scalar_difference(
                        base_phase,
                        fine_field[0],
                    ),
                    sorted_scalar_difference(
                        base_phase,
                        coarse_field[0],
                    ),
                )

                defect_difference = max(
                    sorted_scalar_difference(
                        base_defect,
                        fine_field[3],
                    ),
                    sorted_scalar_difference(
                        base_defect,
                        coarse_field[3],
                    ),
                )

                orientation_difference = max(
                    sorted_orientation_pair_difference(
                        base_x=base_orientation_x,
                        base_y=base_orientation_y,
                        observed_x=fine_field[1],
                        observed_y=fine_field[2],
                    ),
                    sorted_orientation_pair_difference(
                        base_x=base_orientation_x,
                        base_y=base_orientation_y,
                        observed_x=coarse_field[1],
                        observed_y=coarse_field[2],
                    ),
                )

                rows.append(
                    {
                        "carrier_id":
                            carrier_id,

                        "carrier_split":
                            carrier_row[
                                "split"
                            ],

                        "anisotropy_level":
                            anisotropy_level,

                        "defect_recovery_level":
                            defect_recovery_level,

                        "fine_state_id":
                            int(
                                fine_state[
                                    "state_id"
                                ]
                            ),

                        "coarse_state_id":
                            int(
                                coarse_state[
                                    "state_id"
                                ]
                            ),

                        "selected_constructor_iteration":
                            int(
                                selection[
                                    "selected_iteration"
                                ]
                            ),

                        "constructor_length_ratio":
                            float(
                                selection[
                                    "constructor_coarse_to_fine_length_ratio"
                                ]
                            ),

                        "constructor_interface_ratio":
                            float(
                                selection[
                                    "constructor_coarse_to_fine_interface_ratio"
                                ]
                            ),

                        "independent_fine_characteristic_length":
                            fine_length,

                        "independent_coarse_characteristic_length":
                            coarse_length,

                        "independent_coarse_to_fine_length_ratio":
                            length_ratio,

                        "independent_fine_interface_density":
                            fine_interface,

                        "independent_coarse_interface_density":
                            coarse_interface,

                        "independent_coarse_to_fine_interface_ratio":
                            interface_ratio,

                        "independent_interface_mean_threshold_passed":
                            bool(
                                interface_ratio
                                <= EXTERNAL_MAXIMUM_MEAN_INTERFACE_RATIO
                            ),

                        "independent_interface_pair_maximum_passed":
                            bool(
                                interface_ratio
                                <= EXTERNAL_MAXIMUM_PAIR_INTERFACE_RATIO
                            ),

                        "maximum_sorted_phase_difference":
                            phase_difference,

                        "maximum_sorted_defect_difference":
                            defect_difference,

                        "maximum_sorted_orientation_pair_difference":
                            orientation_difference,
                    }
                )

    if len(rows) != PAIRED_COMPARISON_COUNT:
        raise AssertionError(
            "Independent pair-verification count changed."
        )

    write_csv(
        PRIVILEGED_DIR
        / "independent_pair_verification.csv",
        rows,
    )

    return rows


def summarize_physics(rows):
    return {
        "diagnostic_count":
            len(rows),

        "all_values_finite":
            bool(
                all(
                    row[
                        "all_values_finite"
                    ]
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


def summarize_independent_verification(rows):
    length_ratios = np.asarray(
        [
            row[
                "independent_coarse_to_fine_length_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    interface_ratios = np.asarray(
        [
            row[
                "independent_coarse_to_fine_interface_ratio"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    phase_differences = np.asarray(
        [
            row[
                "maximum_sorted_phase_difference"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    defect_differences = np.asarray(
        [
            row[
                "maximum_sorted_defect_difference"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    orientation_differences = np.asarray(
        [
            row[
                "maximum_sorted_orientation_pair_difference"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    pair_pass_fraction = float(
        np.mean(
            interface_ratios
            <= EXTERNAL_MAXIMUM_MEAN_INTERFACE_RATIO
        )
    )

    checks = {
        "mean_characteristic_length_ratio":
            bool(
                length_ratios.mean()
                >= EXTERNAL_MINIMUM_MEAN_LENGTH_RATIO
            ),

        "worst_characteristic_length_ratio":
            bool(
                length_ratios.min()
                >= EXTERNAL_MINIMUM_WORST_LENGTH_RATIO
            ),

        "mean_interface_density_ratio":
            bool(
                interface_ratios.mean()
                <= EXTERNAL_MAXIMUM_MEAN_INTERFACE_RATIO
            ),

        "maximum_pair_interface_density_ratio":
            bool(
                interface_ratios.max()
                <= EXTERNAL_MAXIMUM_PAIR_INTERFACE_RATIO
            ),

        "pair_interface_pass_fraction":
            bool(
                pair_pass_fraction
                >= EXTERNAL_MINIMUM_PAIR_PASS_FRACTION
            ),

        "phase_multiset_preserved":
            bool(
                phase_differences.max()
                <= MAXIMUM_SORTED_PHASE_DIFFERENCE
            ),

        "defect_multiset_preserved":
            bool(
                defect_differences.max()
                <= MAXIMUM_SORTED_DEFECT_DIFFERENCE
            ),

        "orientation_pair_multiset_preserved":
            bool(
                orientation_differences.max()
                <= MAXIMUM_SORTED_ORIENTATION_PAIR_DIFFERENCE
            ),
    }

    return {
        "pair_count":
            len(rows),

        "mean_characteristic_length_ratio":
            float(
                length_ratios.mean()
            ),

        "minimum_characteristic_length_ratio":
            float(
                length_ratios.min()
            ),

        "maximum_characteristic_length_ratio":
            float(
                length_ratios.max()
            ),

        "mean_interface_density_ratio":
            float(
                interface_ratios.mean()
            ),

        "minimum_interface_density_ratio":
            float(
                interface_ratios.min()
            ),

        "maximum_interface_density_ratio":
            float(
                interface_ratios.max()
            ),

        "fraction_pairs_at_or_below_mean_interface_threshold":
            pair_pass_fraction,

        "maximum_sorted_phase_difference":
            float(
                phase_differences.max()
            ),

        "maximum_sorted_defect_difference":
            float(
                defect_differences.max()
            ),

        "maximum_sorted_orientation_pair_difference":
            float(
                orientation_differences.max()
            ),

        "checks":
            checks,

        "failed_checks": [
            name
            for name, passed
            in checks.items()
            if not passed
        ],

        "independent_morphology_gate_passed":
            bool(
                all(checks.values())
            ),
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

    ranks[order] = np.arange(
        len(values),
        dtype=np.float64,
    )

    return ranks


def spearman_correlation(
    first,
    second,
):
    first_ranks = rank_values(first)
    second_ranks = rank_values(second)

    if (
        first_ranks.std() < 1e-12
        or second_ranks.std() < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first_ranks,
            second_ranks,
        )[0, 1]
    )


def write_input_hashes():
    paths = {
        "phase4ar3_summary":
            PHASE4AR3_DIR
            / "phase4ar3_summary.json",

        "tier_c_v4_protocol":
            PHASE4AR3_DIR
            / "tier_c_v4_protocol.json",

        "carrier_split_v4":
            PHASE4AR3_DIR
            / "carrier_split_v4.csv",

        "carrier_parameter_schema_v4":
            PHASE4AR3_DIR
            / "carrier_parameter_schema_v4.csv",

        "state_morphology_regimes_v4":
            PHASE4AR3_DIR
            / "state_morphology_regimes_v4.csv",

        "coarsening_protocol_v4":
            PHASE4AR3_DIR
            / "coarsening_protocol_v4.json",

        "acceptance_gates_v4":
            PHASE4AR3_DIR
            / "acceptance_gates_v4.json",

        "terminal_revision_policy":
            PHASE4AR3_DIR
            / "terminal_revision_policy.json",
    }

    hashes = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        hashes[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    write_json(
        OUTPUT_DIR
        / "input_hashes.json",
        hashes,
    )


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
        / "phase4br3_field_summary.json"
    )

    if (
        summary_path.exists()
        and not arguments.force
    ):
        existing = load_json(
            summary_path
        )

        if existing.get(
            "phase4br3_status"
        ) in {
            "completed_and_verified",
            "failed_pre_materialization",
        }:
            print(
                "Phase 4B-R3.1 already has a frozen result."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    protocol = validate_protocol()

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

    (
        state_regimes,
        state_lookup,
    ) = load_state_regimes()

    (
        parameters,
        latent,
    ) = generate_carrier_parameters(
        carrier_rows=carrier_rows,
        parameter_schema=parameter_schema,
    )

    (
        base_fields,
        preferred_angles,
    ) = create_base_fields(
        carrier_rows=carrier_rows,
        parameters=parameters,
        latent=latent,
    )

    np.save(
        PRIVILEGED_DIR
        / "development_base_fields.npy",
        base_fields,
    )

    print(
        "[1/4] Constructing deterministic "
        "fine/coarse phase pairs"
    )

    construction = construct_target_bank(
        carrier_rows=carrier_rows,
        parameters=parameters,
        base_fields=base_fields,
        preferred_angles=preferred_angles,
        state_lookup=state_lookup,
    )

    construction_failure_count = len(
        construction["failure_rows"]
    )

    if construction_failure_count > 0:
        failure_summary = {
            "phase":
                (
                    "4B-R3 Tier C v4 development "
                    "pair construction"
                ),

            "protocol_version":
                PROTOCOL_VERSION,

            "unique_phase_pair_count_expected":
                UNIQUE_PHASE_PAIR_COUNT,

            "unique_phase_pair_count_completed":
                len(
                    construction[
                        "selection_rows"
                    ]
                ),

            "construction_failure_count":
                construction_failure_count,

            "all_pairs_found_valid_iteration":
                False,

            "internal_target_interface_ratio":
                INTERNAL_TARGET_INTERFACE_RATIO,

            "internal_target_length_ratio":
                INTERNAL_TARGET_LENGTH_RATIO,

            "trajectory_materialization_authorized":
                False,

            "diagnostic_decoder_fitting_authorized":
                False,

            "predictive_training_authorized":
                False,

            "test_generation_authorized":
                False,

            "tier_c_v5_authorized":
                False,

            "phase4br3_status":
                "failed_pre_materialization",
        }

        write_json(
            OUTPUT_DIR
            / "construction_failure_summary.json",
            failure_summary,
        )

        write_json(
            summary_path,
            failure_summary,
        )

        print(
            json.dumps(
                failure_summary,
                indent=2,
            )
        )

        raise SystemExit(
            "Tier C v4 construction failed for one or "
            "more development pairs. Under the terminal "
            "revision policy, do not materialize trajectories "
            "and do not create Tier C v5."
        )

    if (
        len(
            construction[
                "selection_rows"
            ]
        )
        != UNIQUE_PHASE_PAIR_COUNT
    ):
        raise AssertionError(
            "Completed phase-pair count changed."
        )

    selection_lookup = {
        (
            int(row["carrier_id"]),
            int(row["anisotropy_level"]),
        ):
            row
        for row in construction[
            "selection_rows"
        ]
    }

    if (
        len(selection_lookup)
        != UNIQUE_PHASE_PAIR_COUNT
    ):
        raise AssertionError(
            "Construction selection lookup is incomplete."
        )

    selected_iterations = np.asarray(
        [
            int(
                row[
                    "selected_iteration"
                ]
            )
            for row in construction[
                "selection_rows"
            ]
        ],
        dtype=np.int64,
    )

    constructor_length_ratios = np.asarray(
        [
            float(
                row[
                    "constructor_coarse_to_fine_length_ratio"
                ]
            )
            for row in construction[
                "selection_rows"
            ]
        ],
        dtype=np.float64,
    )

    constructor_interface_ratios = np.asarray(
        [
            float(
                row[
                    "constructor_coarse_to_fine_interface_ratio"
                ]
            )
            for row in construction[
                "selection_rows"
            ]
        ],
        dtype=np.float64,
    )

    all_internal_targets_passed = bool(
        np.all(
            constructor_length_ratios
            >= INTERNAL_TARGET_LENGTH_RATIO
        )
        and np.all(
            constructor_interface_ratios
            <= INTERNAL_TARGET_INTERFACE_RATIO
        )
    )

    if not all_internal_targets_passed:
        raise AssertionError(
            "A selected Tier C v4 pair does not satisfy "
            "the frozen internal targets."
        )

    print(
        "[2/4] Running unchanged numerical relaxation"
    )

    (
        final_bank,
        physics_rows,
    ) = run_relaxation(
        carrier_rows=carrier_rows,
        parameters=parameters,
        base_fields=base_fields,
        target_bank=construction[
            "target_bank"
        ],
        state_regimes=state_regimes,
        device=device,
        batch_size=arguments.batch_size,
    )

    if (
        len(physics_rows)
        != EXPECTED_FIELD_COUNT
    ):
        raise AssertionError(
            "Physics diagnostic count changed."
        )

    if not np.isfinite(
        final_bank
    ).all():
        raise FloatingPointError(
            "Tier C v4 final field bank contains NaN or Inf."
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

    training_indices = np.where(
        carrier_splits == "train"
    )[0]

    validation_indices = np.where(
        carrier_splits == "val"
    )[0]

    if len(training_indices) != 96:
        raise AssertionError(
            "Training carrier count changed."
        )

    if len(validation_indices) != 32:
        raise AssertionError(
            "Validation carrier count changed."
        )

    training_fields = final_bank[
        training_indices
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
            "At least one Tier C v4 channel "
            "has zero training variance."
        )

    normalized_bank = (
        (
            final_bank
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
    ).astype(np.float32)

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
        final_bank,
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

    write_json(
        OUTPUT_DIR
        / "normalization_v4.json",
        {
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
        },
    )

    print(
        "[3/4] Independently verifying saved "
        "development fields"
    )

    verification_rows = (
        independent_verify_saved_fields(
            carrier_rows=carrier_rows,
            base_fields=base_fields,
            state_lookup=state_lookup,
            selection_lookup=selection_lookup,
        )
    )

    physics_summary = summarize_physics(
        physics_rows
    )

    verification_summary = (
        summarize_independent_verification(
            verification_rows
        )
    )

    parameter_lookup = {
        int(row["carrier_id"]):
            float(
                row[
                    "base_length_scale_pixels"
                ]
            )
        for row in load_csv(
            OUTPUT_DIR
            / "development_carrier_parameters.csv"
        )
    }

    base_lengths = np.asarray(
        [
            parameter_lookup[
                int(row["carrier_id"])
            ]
            for row in verification_rows
        ],
        dtype=np.float64,
    )

    independent_interface_ratios = np.asarray(
        [
            row[
                "independent_coarse_to_fine_interface_ratio"
            ]
            for row in verification_rows
        ],
        dtype=np.float64,
    )

    base_length_interface_spearman = (
        spearman_correlation(
            base_lengths,
            independent_interface_ratios,
        )
    )

    physics_gate_passed = bool(
        physics_summary[
            "all_values_finite"
        ]
        and physics_summary[
            "energy_nonincrease_fraction"
        ] >= 0.99
        and physics_summary[
            "maximum_phase_mean_drift"
        ] <= 1e-5
        and physics_summary[
            "maximum_defect_mean_drift"
        ] <= 1e-5
        and physics_summary[
            "phase_minimum"
        ] >= -1.000001
        and physics_summary[
            "phase_maximum"
        ] <= 1.000001
        and physics_summary[
            "defect_minimum"
        ] >= -0.000001
        and physics_summary[
            "defect_maximum"
        ] <= 1.000001
    )

    ready_for_trajectory_materialization = bool(
        all_internal_targets_passed
        and physics_gate_passed
        and verification_summary[
            "independent_morphology_gate_passed"
        ]
    )

    write_json(
        OUTPUT_DIR
        / "independent_morphology_summary.json",
        verification_summary,
    )

    write_input_hashes()

    print(
        "[4/4] Freezing the Phase 4B-R3.1 result"
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    summary = {
        "phase":
            (
                "4B-R3 Tier C v4 development "
                "field construction and verification"
            ),

        "protocol_version":
            PROTOCOL_VERSION,

        "phase4ar3_status":
            protocol[
                "summary"
            ][
                "phase4ar3_status"
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

        "unique_phase_pair_count":
            UNIQUE_PHASE_PAIR_COUNT,

        "paired_coarsening_comparison_count":
            PAIRED_COMPARISON_COUNT,

        "all_pairs_found_valid_iteration":
            True,

        "construction_failure_count":
            0,

        "minimum_selected_iteration":
            int(
                selected_iterations.min()
            ),

        "maximum_selected_iteration":
            int(
                selected_iterations.max()
            ),

        "mean_selected_iteration":
            float(
                selected_iterations.mean()
            ),

        "constructor_minimum_length_ratio":
            float(
                constructor_length_ratios.min()
            ),

        "constructor_maximum_interface_ratio":
            float(
                constructor_interface_ratios.max()
            ),

        "all_internal_targets_passed":
            all_internal_targets_passed,

        "physics_gate_passed":
            physics_gate_passed,

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

        "independent_mean_length_ratio":
            verification_summary[
                "mean_characteristic_length_ratio"
            ],

        "independent_minimum_length_ratio":
            verification_summary[
                "minimum_characteristic_length_ratio"
            ],

        "independent_mean_interface_ratio":
            verification_summary[
                "mean_interface_density_ratio"
            ],

        "independent_maximum_interface_ratio":
            verification_summary[
                "maximum_interface_density_ratio"
            ],

        "independent_interface_pair_pass_fraction":
            verification_summary[
                "fraction_pairs_at_or_below_mean_interface_threshold"
            ],

        "maximum_sorted_phase_difference":
            verification_summary[
                "maximum_sorted_phase_difference"
            ],

        "maximum_sorted_defect_difference":
            verification_summary[
                "maximum_sorted_defect_difference"
            ],

        "maximum_sorted_orientation_pair_difference":
            verification_summary[
                "maximum_sorted_orientation_pair_difference"
            ],

        "independent_morphology_gate_passed":
            verification_summary[
                "independent_morphology_gate_passed"
            ],

        "independent_morphology_failed_checks":
            verification_summary[
                "failed_checks"
            ],

        "base_length_interface_ratio_spearman":
            base_length_interface_spearman,

        "trajectory_materialization_authorized":
            ready_for_trajectory_materialization,

        "diagnostic_decoder_fitted":
            False,

        "diagnostic_decoder_fitting_authorized":
            False,

        "predictive_training_performed":
            False,

        "checkpoint_selection_performed":
            False,

        "test_carrier_parameters_generated":
            False,

        "test_fields_generated":
            False,

        "test_manifests_generated":
            False,

        "test_metrics_computed":
            False,

        "tier_c_v1_outputs_modified":
            False,

        "tier_c_v2_outputs_modified":
            False,

        "tier_c_v3_outputs_modified":
            False,

        "tier_c_v5_authorized":
            False,

        "elapsed_seconds":
            elapsed_seconds,

        "phase4br3_status":
            (
                "completed_and_verified"
                if ready_for_trajectory_materialization
                else "failed_pre_materialization"
            ),
    }

    write_json(
        summary_path,
        summary,
    )

    print()
    print(
        "Phase 4B-R3.1 completed."
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

    if not ready_for_trajectory_materialization:
        raise SystemExit(
            "Tier C v4 failed before trajectory "
            "materialization. Under the terminal revision "
            "policy, do not fit decoders, do not train "
            "predictive models, and do not create Tier C v5."
        )


if __name__ == "__main__":
    main()
