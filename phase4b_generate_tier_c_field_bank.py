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


PHASE4A_DIR = Path(
    "outputs/phase4a_tier_c_physical_protocol"
)

OUTPUT_DIR = Path(
    "outputs/phase4b_tier_c_data"
)

PRIVILEGED_DIR = OUTPUT_DIR / "privileged"


STATE_COUNT = 8
CARRIER_COUNT = 160
FIELD_CHANNEL_COUNT = 4
GRID_HEIGHT = 32
GRID_WIDTH = 32

CARRIER_DIMENSION = 8

CARRIER_PARAMETER_SEED = 41027
FIELD_INITIALIZATION_SEED = 41028
SOLVER_SEED = 41029

TIME_STEP = 0.01

EXPECTED_CARRIER_SPLITS = {
    "train": 96,
    "val": 32,
    "test": 32,
}

PARAMETER_NAMES = (
    "mean_phase_fraction",
    "initial_length_scale_pixels",
    "interface_width",
    "phase_mobility",
    "orientation_anisotropy",
    "initial_defect_fraction",
    "defect_mobility",
    "process_driving_bias",
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


def resolve_device(requested: str):
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


def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value):
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            value,
            handle,
            indent=2,
        )


def write_csv(path: Path, rows):
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
        newline="",
        encoding="utf-8",
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
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def stable_seed(label: str):
    digest = hashlib.sha256(
        (
            f"{FIELD_INITIALIZATION_SEED}|{label}"
        ).encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def validate_phase4a():
    summary = load_json(
        PHASE4A_DIR
        / "phase4a_summary.json"
    )

    if (
        summary["phase4a_status"]
        != "protocol_frozen"
    ):
        raise AssertionError(
            "Phase 4A is not frozen."
        )

    if summary["sanity_checks"] != "passed":
        raise AssertionError(
            "Phase 4A sanity checks failed."
        )

    if summary["phase3_results_frozen"] is not True:
        raise AssertionError(
            "Phase 3 results are not frozen."
        )

    if summary["physics_simulation_performed"] is not False:
        raise AssertionError(
            "Phase 4A unexpectedly reports simulation."
        )

    if summary["fields_generated"] is not False:
        raise AssertionError(
            "Phase 4A unexpectedly reports generated fields."
        )

    return summary


def radical_inverse(
    index: int,
    base: int,
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


def load_carrier_manifest():
    rows = load_csv(
        PHASE4A_DIR
        / "carrier_reuse_manifest.csv"
    )

    if len(rows) != CARRIER_COUNT:
        raise AssertionError(
            "Carrier count changed."
        )

    rows.sort(
        key=lambda row:
            int(row["carrier_id"])
    )

    carrier_ids = [
        int(row["carrier_id"])
        for row in rows
    ]

    if carrier_ids != list(
        range(CARRIER_COUNT)
    ):
        raise AssertionError(
            "Carrier IDs are not contiguous."
        )

    counts = Counter(
        row["split"]
        for row in rows
    )

    if dict(counts) != EXPECTED_CARRIER_SPLITS:
        raise AssertionError(
            f"Carrier splits changed: {dict(counts)}"
        )

    return rows


def load_parameter_schema():
    rows = load_csv(
        PHASE4A_DIR
        / "physical_carrier_parameter_schema.csv"
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
            "Physical carrier parameter schema changed."
        )

    return rows


def generate_carrier_parameters(
    carrier_manifest,
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
        size=CARRIER_DIMENSION,
    )

    latent = np.empty(
        (
            CARRIER_COUNT,
            CARRIER_DIMENSION,
        ),
        dtype=np.float64,
    )

    parameters = np.empty_like(
        latent
    )

    for carrier_id in range(
        CARRIER_COUNT
    ):
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

        latent[carrier_id] = point

        for dimension, schema in enumerate(
            parameter_schema
        ):
            minimum = float(
                schema["minimum"]
            )

            maximum = float(
                schema["maximum"]
            )

            parameters[
                carrier_id,
                dimension,
            ] = (
                minimum
                + point[dimension]
                * (
                    maximum
                    - minimum
                )
            )

    output_rows = []

    for row in carrier_manifest:
        carrier_id = int(
            row["carrier_id"]
        )

        output = {
            "carrier_id": carrier_id,
            "split": row["split"],
            "split_position":
                int(row["split_position"]),
        }

        for dimension, name in enumerate(
            PARAMETER_NAMES
        ):
            output[
                f"latent_{dimension}"
            ] = float(
                latent[
                    carrier_id,
                    dimension,
                ]
            )

            output[name] = float(
                parameters[
                    carrier_id,
                    dimension,
                ]
            )

        output_rows.append(output)

    write_csv(
        OUTPUT_DIR
        / "physical_carrier_parameters.csv",
        output_rows,
    )

    return parameters, output_rows


def spectral_smooth_noise(
    seed: int,
    length_scale: float,
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

    kx = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            GRID_WIDTH,
            d=1.0,
        )
    )

    ky = (
        2.0
        * np.pi
        * np.fft.fftfreq(
            GRID_HEIGHT,
            d=1.0,
        )
    )

    squared_wave_number = (
        ky[:, None] ** 2
        + kx[None, :] ** 2
    )

    spectral_filter = np.exp(
        -0.5
        * squared_wave_number
        * float(length_scale) ** 2
    )

    smoothed = np.fft.ifft2(
        np.fft.fft2(noise)
        * spectral_filter
    ).real

    smoothed -= smoothed.mean()

    standard_deviation = smoothed.std()

    if standard_deviation < 1e-10:
        raise FloatingPointError(
            "Smoothed random field has zero variance."
        )

    smoothed /= standard_deviation

    return smoothed.astype(
        np.float32
    )


def match_tanh_mean(
    base_field,
    target_mean,
    interface_width,
):
    lower = -8.0
    upper = 8.0

    for _ in range(64):
        midpoint = (
            lower + upper
        ) / 2.0

        field = np.tanh(
            (
                base_field
                + midpoint
            )
            / interface_width
        )

        if field.mean() < target_mean:
            lower = midpoint
        else:
            upper = midpoint

    shift = (
        lower + upper
    ) / 2.0

    field = np.tanh(
        (
            base_field
            + shift
        )
        / interface_width
    )

    return field.astype(
        np.float32
    )


def generate_base_fields(
    carrier_parameters,
):
    fields = np.empty(
        (
            CARRIER_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    summary_rows = []

    for carrier_id in range(
        CARRIER_COUNT
    ):
        (
            mean_phase_fraction,
            initial_length_scale,
            interface_width,
            _,
            _,
            initial_defect_fraction,
            _,
            _,
        ) = carrier_parameters[
            carrier_id
        ]

        phase_noise = spectral_smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|phase"
            ),
            initial_length_scale,
        )

        target_phase_mean = (
            2.0
            * mean_phase_fraction
            - 1.0
        )

        phase = match_tanh_mean(
            base_field=phase_noise,
            target_mean=target_phase_mean,
            interface_width=interface_width,
        )

        orientation_x = spectral_smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|orientation_x"
            ),
            max(
                1.0,
                0.8
                * initial_length_scale,
            ),
        )

        orientation_y = spectral_smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|orientation_y"
            ),
            max(
                1.0,
                0.8
                * initial_length_scale,
            ),
        )

        orientation_norm = np.sqrt(
            orientation_x**2
            + orientation_y**2
        )

        orientation_norm = np.maximum(
            orientation_norm,
            1e-6,
        )

        orientation_x = (
            orientation_x
            / orientation_norm
        )

        orientation_y = (
            orientation_y
            / orientation_norm
        )

        defect_noise = spectral_smooth_noise(
            stable_seed(
                f"carrier={carrier_id}|defect"
            ),
            max(
                1.0,
                0.5
                * initial_length_scale,
            ),
        )

        interface_indicator = (
            0.5
            * np.maximum(
                0.0,
                1.0 - phase**2,
            )
        )

        defect = (
            initial_defect_fraction
            + 0.45
            * interface_indicator
            + 0.035
            * defect_noise
        )

        defect = np.clip(
            defect,
            0.0,
            1.0,
        )

        fields[
            carrier_id,
            0,
        ] = phase

        fields[
            carrier_id,
            1,
        ] = orientation_x

        fields[
            carrier_id,
            2,
        ] = orientation_y

        fields[
            carrier_id,
            3,
        ] = defect

        summary_rows.append(
            {
                "carrier_id":
                    carrier_id,

                "phase_mean":
                    float(
                        phase.mean()
                    ),

                "target_phase_mean":
                    float(
                        target_phase_mean
                    ),

                "phase_mean_error":
                    float(
                        abs(
                            phase.mean()
                            - target_phase_mean
                        )
                    ),

                "phase_minimum":
                    float(
                        phase.min()
                    ),

                "phase_maximum":
                    float(
                        phase.max()
                    ),

                "orientation_norm_mean":
                    float(
                        np.sqrt(
                            orientation_x**2
                            + orientation_y**2
                        ).mean()
                    ),

                "defect_mean":
                    float(
                        defect.mean()
                    ),

                "defect_minimum":
                    float(
                        defect.min()
                    ),

                "defect_maximum":
                    float(
                        defect.max()
                    ),
            }
        )

    if not np.isfinite(
        fields
    ).all():
        raise FloatingPointError(
            "Base fields contain NaN or Inf."
        )

    np.save(
        PRIVILEGED_DIR
        / "base_initial_fields.npy",
        fields,
    )

    write_csv(
        OUTPUT_DIR
        / "base_field_summary.csv",
        summary_rows,
    )

    return fields


def load_state_regimes():
    rows = load_csv(
        PHASE4A_DIR
        / "state_physics_regimes.csv"
    )

    rows.sort(
        key=lambda row:
            int(row["state_id"])
    )

    if [
        int(row["state_id"])
        for row in rows
    ] != list(
        range(STATE_COUNT)
    ):
        raise AssertionError(
            "State regime IDs changed."
        )

    numeric_fields = (
        "phase_mobility_multiplier",
        "interfacial_energy_multiplier",
        "bulk_driving_multiplier",
        "anisotropy_multiplier",
        "defect_coupling_multiplier",
        "orientation_relaxation_multiplier",
    )

    parsed = []

    for row in rows:
        result = {
            "state_id":
                int(row["state_id"]),

            "regime_label":
                row["regime_label"],

            "relaxation_steps":
                int(
                    row["relaxation_steps"]
                ),
        }

        for field in numeric_fields:
            result[field] = float(
                row[field]
            )

        parsed.append(result)

    return parsed


def build_wave_numbers(
    device,
):
    kx = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            GRID_WIDTH,
            d=1.0,
            device=device,
        )
    )

    ky = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            GRID_HEIGHT,
            d=1.0,
            device=device,
        )
    )

    squared = (
        ky[:, None] ** 2
        + kx[None, :] ** 2
    )

    return {
        "kx": kx,
        "ky": ky,
        "k2": squared,
        "k4": squared**2,
    }


def gradient_squared(
    field,
    wave_numbers,
):
    spectrum = torch.fft.fft2(
        field
    )

    gradient_x = torch.fft.ifft2(
        1j
        * wave_numbers["kx"][
            None,
            None,
            :,
        ]
        * spectrum
    ).real

    gradient_y = torch.fft.ifft2(
        1j
        * wave_numbers["ky"][
            None,
            :,
            None,
        ]
        * spectrum
    ).real

    return (
        gradient_x**2
        + gradient_y**2
    )


def total_energy(
    phase,
    orientation_x,
    orientation_y,
    defect,
    coefficients,
    wave_numbers,
):
    phase_target_interface = (
        0.5
        * (
            1.0
            - phase**2
        )
    )

    phase_energy = (
        0.25
        * coefficients["bulk"][
            :,
            None,
            None,
        ]
        * (
            phase**2
            - 1.0
        ) ** 2
        + 0.5
        * coefficients["kappa_phase"][
            :,
            None,
            None,
        ]
        * gradient_squared(
            phase,
            wave_numbers,
        )
        + (
            coefficients["bias"][
                :,
                None,
                None,
            ]
            / 3.0
        )
        * phase**3
    )

    orientation_norm_squared = (
        orientation_x**2
        + orientation_y**2
    )

    orientation_energy = (
        0.25
        * coefficients["orientation_bulk"][
            :,
            None,
            None,
        ]
        * (
            orientation_norm_squared
            - 1.0
        ) ** 2
        + 0.5
        * coefficients["kappa_orientation"][
            :,
            None,
            None,
        ]
        * (
            gradient_squared(
                orientation_x,
                wave_numbers,
            )
            + gradient_squared(
                orientation_y,
                wave_numbers,
            )
        )
        + 0.25
        * coefficients["anisotropy"][
            :,
            None,
            None,
        ]
        * (
            orientation_x**4
            + orientation_y**4
        )
    )

    defect_energy = (
        0.5
        * coefficients["defect_coupling"][
            :,
            None,
            None,
        ]
        * (
            defect
            - phase_target_interface
        ) ** 2
        + 0.5
        * coefficients["kappa_defect"][
            :,
            None,
            None,
        ]
        * gradient_squared(
            defect,
            wave_numbers,
        )
    )

    return (
        phase_energy
        + orientation_energy
        + defect_energy
    ).mean(
        dim=(-2, -1)
    )


def build_coefficients(
    carrier_parameters,
    regime,
    device,
):
    parameters = torch.as_tensor(
        carrier_parameters,
        dtype=torch.float32,
        device=device,
    )

    interface_width = parameters[:, 2]

    coefficients = {
        "phase_mobility":
            parameters[:, 3]
            * regime[
                "phase_mobility_multiplier"
            ],

        "bulk":
            0.9
            * regime[
                "bulk_driving_multiplier"
            ]
            * (
                0.9
                + 0.2
                * parameters[:, 0]
            ),

        "kappa_phase":
            0.075
            * interface_width**2
            * regime[
                "interfacial_energy_multiplier"
            ],

        "bias":
            parameters[:, 7]
            * regime[
                "bulk_driving_multiplier"
            ],

        "orientation_mobility":
            torch.full_like(
                parameters[:, 0],
                0.9
                * regime[
                    "orientation_relaxation_multiplier"
                ],
            ),

        "orientation_bulk":
            torch.full_like(
                parameters[:, 0],
                1.0,
            ),

        "kappa_orientation":
            0.06
            + 0.035
            * interface_width,

        "anisotropy":
            parameters[:, 4]
            * regime[
                "anisotropy_multiplier"
            ],

        "defect_mobility":
            parameters[:, 6],

        "defect_coupling":
            torch.full_like(
                parameters[:, 0],
                0.55
                * regime[
                    "defect_coupling_multiplier"
                ],
            ),

        "kappa_defect":
            0.045
            + 0.025
            * interface_width,
    }

    return coefficients


def simulate_batch(
    initial_fields,
    coefficients,
    relaxation_steps,
    wave_numbers,
):
    phase = initial_fields[:, 0].clone()
    orientation_x = initial_fields[:, 1].clone()
    orientation_y = initial_fields[:, 2].clone()
    defect = initial_fields[:, 3].clone()

    initial_phase_mean = phase.mean(
        dim=(-2, -1)
    )

    initial_energy = total_energy(
        phase=phase,
        orientation_x=orientation_x,
        orientation_y=orientation_y,
        defect=defect,
        coefficients=coefficients,
        wave_numbers=wave_numbers,
    )

    k2 = wave_numbers["k2"][
        None,
        :,
        :,
    ]

    k4 = wave_numbers["k4"][
        None,
        :,
        :,
    ]

    for _ in range(
        relaxation_steps
    ):
        target_interface = (
            0.5
            * (
                1.0
                - phase**2
            )
        )

        nonlinear_phase = (
            coefficients["bulk"][
                :,
                None,
                None,
            ]
            * (
                phase**3
                - phase
            )
            + coefficients["bias"][
                :,
                None,
                None,
            ]
            * phase**2
            + coefficients[
                "defect_coupling"
            ][
                :,
                None,
                None,
            ]
            * phase
            * (
                defect
                - target_interface
            )
        )

        phase_spectrum = torch.fft.fft2(
            phase
        )

        phase = torch.fft.ifft2(
            (
                phase_spectrum
                - TIME_STEP
                * coefficients[
                    "phase_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * k2
                * torch.fft.fft2(
                    nonlinear_phase
                )
            )
            / (
                1.0
                + TIME_STEP
                * coefficients[
                    "phase_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * coefficients[
                    "kappa_phase"
                ][
                    :,
                    None,
                    None,
                ]
                * k4
            )
        ).real

        orientation_norm_squared = (
            orientation_x**2
            + orientation_y**2
        )

        nonlinear_orientation_x = (
            coefficients[
                "orientation_bulk"
            ][
                :,
                None,
                None,
            ]
            * (
                orientation_norm_squared
                - 1.0
            )
            * orientation_x
            + coefficients["anisotropy"][
                :,
                None,
                None,
            ]
            * orientation_x**3
        )

        nonlinear_orientation_y = (
            coefficients[
                "orientation_bulk"
            ][
                :,
                None,
                None,
            ]
            * (
                orientation_norm_squared
                - 1.0
            )
            * orientation_y
            + coefficients["anisotropy"][
                :,
                None,
                None,
            ]
            * orientation_y**3
        )

        orientation_denominator = (
            1.0
            + TIME_STEP
            * coefficients[
                "orientation_mobility"
            ][
                :,
                None,
                None,
            ]
            * coefficients[
                "kappa_orientation"
            ][
                :,
                None,
                None,
            ]
            * k2
        )

        orientation_x = torch.fft.ifft2(
            (
                torch.fft.fft2(
                    orientation_x
                )
                - TIME_STEP
                * coefficients[
                    "orientation_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * torch.fft.fft2(
                    nonlinear_orientation_x
                )
            )
            / orientation_denominator
        ).real

        orientation_y = torch.fft.ifft2(
            (
                torch.fft.fft2(
                    orientation_y
                )
                - TIME_STEP
                * coefficients[
                    "orientation_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * torch.fft.fft2(
                    nonlinear_orientation_y
                )
            )
            / orientation_denominator
        ).real

        target_interface = (
            0.5
            * (
                1.0
                - phase**2
            )
        )

        nonlinear_defect = (
            coefficients[
                "defect_coupling"
            ][
                :,
                None,
                None,
            ]
            * (
                defect
                - target_interface
            )
        )

        defect = torch.fft.ifft2(
            (
                torch.fft.fft2(
                    defect
                )
                - TIME_STEP
                * coefficients[
                    "defect_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * torch.fft.fft2(
                    nonlinear_defect
                )
            )
            / (
                1.0
                + TIME_STEP
                * coefficients[
                    "defect_mobility"
                ][
                    :,
                    None,
                    None,
                ]
                * coefficients[
                    "kappa_defect"
                ][
                    :,
                    None,
                    None,
                ]
                * k2
            )
        ).real

    final_energy = total_energy(
        phase=phase,
        orientation_x=orientation_x,
        orientation_y=orientation_y,
        defect=defect,
        coefficients=coefficients,
        wave_numbers=wave_numbers,
    )

    final_phase_mean = phase.mean(
        dim=(-2, -1)
    )

    preprocessed = torch.stack(
        [
            phase,
            orientation_x,
            orientation_y,
            defect,
        ],
        dim=1,
    )

    orientation_norm = torch.sqrt(
        orientation_x**2
        + orientation_y**2
    ).clamp_min(
        1e-6
    )

    observation = torch.stack(
        [
            phase.clamp(
                -1.0,
                1.0,
            ),

            orientation_x
            / orientation_norm,

            orientation_y
            / orientation_norm,

            defect.clamp(
                0.0,
                1.0,
            ),
        ],
        dim=1,
    )

    return {
        "observation":
            observation,

        "preprocessed":
            preprocessed,

        "initial_energy":
            initial_energy,

        "final_energy":
            final_energy,

        "initial_phase_mean":
            initial_phase_mean,

        "final_phase_mean":
            final_phase_mean,
    }


def create_diagnostic_rows(
    carrier_ids,
    state_id,
    relaxation_steps,
    simulation,
):
    preprocessed = (
        simulation[
            "preprocessed"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    observations = (
        simulation[
            "observation"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    initial_energy = (
        simulation[
            "initial_energy"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    final_energy = (
        simulation[
            "final_energy"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    initial_phase_mean = (
        simulation[
            "initial_phase_mean"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    final_phase_mean = (
        simulation[
            "final_phase_mean"
        ]
        .detach()
        .cpu()
        .numpy()
    )

    rows = []

    for local_index, carrier_id in enumerate(
        carrier_ids
    ):
        initial_value = float(
            initial_energy[
                local_index
            ]
        )

        final_value = float(
            final_energy[
                local_index
            ]
        )

        tolerance = max(
            1e-6,
            1e-4
            * abs(
                initial_value
            ),
        )

        q_norm = np.sqrt(
            observations[
                local_index,
                1,
            ] ** 2
            + observations[
                local_index,
                2,
            ] ** 2
        )

        rows.append(
            {
                "carrier_id":
                    int(carrier_id),

                "state_id":
                    int(state_id),

                "relaxation_steps":
                    int(
                        relaxation_steps
                    ),

                "time_step":
                    TIME_STEP,

                "initial_energy":
                    initial_value,

                "final_energy":
                    final_value,

                "energy_change":
                    final_value
                    - initial_value,

                "energy_nonincrease":
                    bool(
                        final_value
                        <= initial_value
                        + tolerance
                    ),

                "initial_phase_mean":
                    float(
                        initial_phase_mean[
                            local_index
                        ]
                    ),

                "final_phase_mean":
                    float(
                        final_phase_mean[
                            local_index
                        ]
                    ),

                "phase_mean_drift":
                    float(
                        abs(
                            final_phase_mean[
                                local_index
                            ]
                            - initial_phase_mean[
                                local_index
                            ]
                        )
                    ),

                "preclip_phase_minimum":
                    float(
                        preprocessed[
                            local_index,
                            0,
                        ].min()
                    ),

                "preclip_phase_maximum":
                    float(
                        preprocessed[
                            local_index,
                            0,
                        ].max()
                    ),

                "preclip_defect_minimum":
                    float(
                        preprocessed[
                            local_index,
                            3,
                        ].min()
                    ),

                "preclip_defect_maximum":
                    float(
                        preprocessed[
                            local_index,
                            3,
                        ].max()
                    ),

                "postprocess_orientation_norm_mean":
                    float(
                        q_norm.mean()
                    ),

                "all_values_finite":
                    bool(
                        np.isfinite(
                            observations[
                                local_index
                            ]
                        ).all()
                    ),
            }
        )

    return rows


def aggregate_diagnostics(
    rows,
):
    energy_fraction = float(
        np.mean(
            [
                row["energy_nonincrease"]
                for row in rows
            ]
        )
    )

    maximum_phase_drift = float(
        max(
            row["phase_mean_drift"]
            for row in rows
        )
    )

    all_finite = bool(
        all(
            row["all_values_finite"]
            for row in rows
        )
    )

    state_rows = []

    for state_id in range(
        STATE_COUNT
    ):
        selected = [
            row
            for row in rows
            if row["state_id"] == state_id
        ]

        state_rows.append(
            {
                "state_id":
                    state_id,

                "simulation_count":
                    len(selected),

                "energy_nonincrease_fraction":
                    float(
                        np.mean(
                            [
                                row[
                                    "energy_nonincrease"
                                ]
                                for row in selected
                            ]
                        )
                    ),

                "maximum_phase_mean_drift":
                    float(
                        max(
                            row[
                                "phase_mean_drift"
                            ]
                            for row in selected
                        )
                    ),

                "mean_energy_change":
                    float(
                        np.mean(
                            [
                                row[
                                    "energy_change"
                                ]
                                for row in selected
                            ]
                        )
                    ),
            }
        )

    write_csv(
        OUTPUT_DIR
        / "physics_diagnostics_by_state.csv",
        state_rows,
    )

    return {
        "simulation_count":
            len(rows),

        "energy_nonincrease_fraction":
            energy_fraction,

        "maximum_phase_mean_drift":
            maximum_phase_drift,

        "all_values_finite":
            all_finite,
    }


def main():
    arguments = parse_arguments()

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
        / "phase4b_field_bank_summary.json"
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
                "Phase 4B field bank already exists. "
                "Use --force to regenerate it."
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            return

    phase4a = validate_phase4a()

    device = resolve_device(
        arguments.device
    )

    if arguments.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive."
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

    torch.backends.cudnn.benchmark = False

    start_time = time.perf_counter()

    carrier_manifest = (
        load_carrier_manifest()
    )

    parameter_schema = (
        load_parameter_schema()
    )

    (
        carrier_parameters,
        carrier_parameter_rows,
    ) = generate_carrier_parameters(
        carrier_manifest=carrier_manifest,
        parameter_schema=parameter_schema,
    )

    base_fields = generate_base_fields(
        carrier_parameters
    )

    state_regimes = (
        load_state_regimes()
    )

    wave_numbers = build_wave_numbers(
        device
    )

    field_bank = np.empty(
        (
            CARRIER_COUNT,
            STATE_COUNT,
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ),
        dtype=np.float32,
    )

    diagnostic_rows = []

    for regime in state_regimes:
        state_id = regime[
            "state_id"
        ]

        print()
        print(
            f"Generating state {state_id} "
            f"with {regime['relaxation_steps']} steps"
        )

        for start in range(
            0,
            CARRIER_COUNT,
            arguments.batch_size,
        ):
            end = min(
                CARRIER_COUNT,
                start
                + arguments.batch_size,
            )

            carrier_ids = np.arange(
                start,
                end,
                dtype=np.int64,
            )

            initial = torch.from_numpy(
                base_fields[
                    start:end
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            coefficients = (
                build_coefficients(
                    carrier_parameters=(
                        carrier_parameters[
                            start:end
                        ]
                    ),
                    regime=regime,
                    device=device,
                )
            )

            simulation = simulate_batch(
                initial_fields=initial,
                coefficients=coefficients,
                relaxation_steps=(
                    regime[
                        "relaxation_steps"
                    ]
                ),
                wave_numbers=wave_numbers,
            )

            observations = (
                simulation[
                    "observation"
                ]
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

            if not np.isfinite(
                observations
            ).all():
                raise FloatingPointError(
                    "Generated fields contain NaN or Inf."
                )

            field_bank[
                start:end,
                state_id,
            ] = observations

            diagnostic_rows.extend(
                create_diagnostic_rows(
                    carrier_ids=carrier_ids,
                    state_id=state_id,
                    relaxation_steps=(
                        regime[
                            "relaxation_steps"
                        ]
                    ),
                    simulation=simulation,
                )
            )

            print(
                f"  carriers {start:03d}-{end - 1:03d}"
            )

    if len(diagnostic_rows) != (
        CARRIER_COUNT
        * STATE_COUNT
    ):
        raise AssertionError(
            "Physics diagnostic count changed."
        )

    if not np.isfinite(
        field_bank
    ).all():
        raise FloatingPointError(
            "Field bank contains non-finite values."
        )

    train_carrier_ids = np.asarray(
        [
            int(row["carrier_id"])
            for row in carrier_manifest
            if row["split"] == "train"
        ],
        dtype=np.int64,
    )

    training_fields = field_bank[
        train_carrier_ids
    ]

    channel_mean = training_fields.mean(
        axis=(
            0,
            1,
            3,
            4,
        ),
        dtype=np.float64,
    )

    channel_std = training_fields.std(
        axis=(
            0,
            1,
            3,
            4,
        ),
        dtype=np.float64,
    )

    if np.any(
        channel_std < 1e-8
    ):
        raise AssertionError(
            "At least one field channel has "
            "effectively zero variance."
        )

    normalized_bank = (
        field_bank
        - channel_mean[
            None,
            None,
            :,
            None,
            None,
        ]
    ) / channel_std[
        None,
        None,
        :,
        None,
        None,
    ]

    normalized_bank = (
        normalized_bank.astype(
            np.float32
        )
    )

    np.save(
        PRIVILEGED_DIR
        / "carrier_state_fields_raw.npy",
        field_bank,
    )

    np.save(
        PRIVILEGED_DIR
        / "carrier_state_fields_normalized.npy",
        normalized_bank,
    )

    write_csv(
        PRIVILEGED_DIR
        / "physics_diagnostics.csv",
        diagnostic_rows,
    )

    normalization = {
        "statistics_source":
            "clean training-carrier fields only",

        "training_carrier_count":
            len(
                train_carrier_ids
            ),

        "channel_order": [
            "phase_field",
            "orientation_cosine",
            "orientation_sine",
            "defect_density",
        ],

        "channel_mean":
            channel_mean.tolist(),

        "channel_standard_deviation":
            channel_std.tolist(),

        "validation_statistics_used":
            False,

        "test_statistics_used":
            False,
    }

    write_json(
        OUTPUT_DIR
        / "normalization.json",
        normalization,
    )

    diagnostic_summary = (
        aggregate_diagnostics(
            diagnostic_rows
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    input_hashes = {
        "phase4a_summary":
            sha256_file(
                PHASE4A_DIR
                / "phase4a_summary.json"
            ),

        "physics_protocol":
            sha256_file(
                PHASE4A_DIR
                / "physics_protocol.json"
            ),

        "carrier_manifest":
            sha256_file(
                PHASE4A_DIR
                / "carrier_reuse_manifest.csv"
            ),

        "state_regimes":
            sha256_file(
                PHASE4A_DIR
                / "state_physics_regimes.csv"
            ),
    }

    write_json(
        OUTPUT_DIR
        / "field_bank_input_hashes.json",
        input_hashes,
    )

    summary = {
        "phase":
            "4B Tier C carrier-state field generation",

        "phase4a_status":
            phase4a[
                "phase4a_status"
            ],

        "device":
            str(device),

        "torch_version":
            torch.__version__,

        "numpy_version":
            np.__version__,

        "python_version":
            platform.python_version(),

        "carrier_count":
            CARRIER_COUNT,

        "state_count":
            STATE_COUNT,

        "field_count":
            (
                CARRIER_COUNT
                * STATE_COUNT
            ),

        "field_shape": [
            FIELD_CHANNEL_COUNT,
            GRID_HEIGHT,
            GRID_WIDTH,
        ],

        "carrier_parameter_dimension":
            CARRIER_DIMENSION,

        "time_step":
            TIME_STEP,

        "batch_size":
            arguments.batch_size,

        "physics_diagnostic_count":
            len(
                diagnostic_rows
            ),

        "energy_nonincrease_fraction":
            diagnostic_summary[
                "energy_nonincrease_fraction"
            ],

        "maximum_phase_mean_drift":
            diagnostic_summary[
                "maximum_phase_mean_drift"
            ],

        "all_values_finite":
            diagnostic_summary[
                "all_values_finite"
            ],

        "normalization_source":
            "clean training carriers only",

        "clean_field_bank_stored_once":
            True,

        "trajectory_fields_materialized":
            False,

        "test_metrics_computed":
            False,

        "training_performed":
            False,

        "checkpoint_selection_performed":
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
        "Phase 4B carrier-state field bank "
        "generation completed."
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
