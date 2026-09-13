from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ======================================================================
# Paths
# ======================================================================

OUT = Path(
    "paper_artifacts/iclr_results"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR = OUT / "tables"
FIGURE_DIR = OUT / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


G2E_SUMMARY = Path(
    "outputs/"
    "phase4gr3g2e_tier_c_v4_final_freeze/"
    "phase4gr3g2e_final_summary.json"
)

FIELD_SUMMARY = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_fields/"
    "phase4br3_field_summary.json"
)

MORPHOLOGY_SUMMARY = Path(
    "outputs/"
    "phase4br3_tier_c_v4_development_fields/"
    "independent_morphology_summary.json"
)

STATE_DECODER_SUMMARY = Path(
    "outputs/"
    "phase4cr3_tier_c_v4_audit/"
    "state_decoder_summary.json"
)

GLOBAL_STATS_SUMMARY = Path(
    "outputs/"
    "phase4cr3_tier_c_v4_audit/"
    "global_statistics_summary.json"
)

AFFINE_SUMMARY = Path(
    "outputs/"
    "phase4cr3_tier_c_v4_audit/"
    "affine_shortcut_summary.json"
)

STATE_PATH_SUMMARY = Path(
    "outputs/"
    "phase4gr3g2dr1_tier_c_v4_exact_phase3g_structural/"
    "state_path_condition_summary.csv"
)


# ======================================================================
# Frozen predictive results
# ======================================================================
#
# These values are the frozen G2C noisy-target rollout means.
# No result is recomputed here.
#

PRIMARY_JOINT_NOISE_025 = {
    "B0": 1.216326,
    "B1": 1.022196,
    "B2": 1.030627,
    "B3": 0.994846,
    "B4": 1.000112,
    "B5": 1.034443,
    "OCM": 1.039329,
}


JOINT_NOISY_ROLLOUT = {
    "noise": [
        0.00,
        0.10,
        0.25,
        0.50,
        1.00,
    ],

    "B0": [
        1.149519,
        1.160897,
        1.216326,
        1.397044,
        1.973852,
    ],

    "B3": [
        0.973354,
        0.975048,
        0.994846,
        1.053326,
        1.230496,
    ],

    "B4": [
        0.982108,
        0.987772,
        1.000112,
        1.057369,
        1.228444,
    ],

    "OCM": [
        1.016008,
        1.019531,
        1.039329,
        1.095925,
        1.273872,
    ],
}


# Frozen descriptive winners across the 20
# cell x noise noisy-target rollout conditions.
PREDICTIVE_WIN_COUNTS = {
    "B1": 4,
    "B2": 0,
    "B3": 10,
    "B4": 6,
    "B5": 0,
    "OCM": 0,
}


STATE_CHANCE = 1.0 / 8.0


# ======================================================================
# Helpers
# ======================================================================

def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def load_csv(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    require(
        len(rows) > 0,
        f"No rows for {path}.",
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
        writer.writerows(
            rows
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def latex_escape(
    value,
) -> str:
    text = str(
        value
    )

    replacements = {
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
    }

    for source, target in (
        replacements.items()
    ):
        text = text.replace(
            source,
            target,
        )

    return text


def write_latex_tabular(
    path: Path,
    headers: list[str],
    rows: list[list[str]],
    alignment: str,
) -> None:
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(
            latex_escape(
                value
            )
            for value in headers
        )
        + r" \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(
                latex_escape(
                    value
                )
                for value in row
            )
            + r" \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    path.write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )


def configure_plot():
    """
    Compact ICLR-friendly plotting defaults.

    Important:
    - no figure title
    - no embedded caption
    """

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    fig,
    stem: str,
):
    png = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    pdf = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(
        fig
    )

    return png, pdf


# ======================================================================
# Load and verify frozen sources
# ======================================================================

print(
    "[1/8] Loading frozen results"
)

for path in (
    G2E_SUMMARY,
    FIELD_SUMMARY,
    MORPHOLOGY_SUMMARY,
    STATE_DECODER_SUMMARY,
    GLOBAL_STATS_SUMMARY,
    AFFINE_SUMMARY,
    STATE_PATH_SUMMARY,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )


g2e = load_json(
    G2E_SUMMARY
)

field = load_json(
    FIELD_SUMMARY
)

morphology = load_json(
    MORPHOLOGY_SUMMARY
)

decoder = load_json(
    STATE_DECODER_SUMMARY
)

global_stats = load_json(
    GLOBAL_STATS_SUMMARY
)

affine = load_json(
    AFFINE_SUMMARY
)

state_path_rows = load_csv(
    STATE_PATH_SUMMARY
)


require(
    g2e[
        "experimental_campaign_status"
    ]
    == "CLOSED",
    (
        "Experimental campaign is not "
        "frozen closed."
    ),
)

require(
    g2e[
        "primary_confirmatory_result"
    ][
        "predictive_superiority_supported"
    ]
    is False,
    (
        "Primary frozen result changed."
    ),
)

require(
    decoder[
        "state_separability_gate_passed"
    ]
    is True,
    "State-separability gate changed.",
)

require(
    global_stats[
        "global_statistics_gate_passed"
    ]
    is True,
    "Global shortcut gate changed.",
)

require(
    affine[
        "affine_shortcut_gate_passed"
    ]
    is True,
    "Affine shortcut gate changed.",
)


# ======================================================================
# Reframed research questions
# ======================================================================

print(
    "[2/8] Writing reframed supported RQs"
)

rq_text = r"""% Reframed research questions used in the main paper.
% Unsupported earlier research questions are intentionally omitted
% from the main narrative and remain only in the frozen reproducibility
% record.

\textbf{RQ1: Benchmark Validity and State Identifiability.}
Can a nontrivial, shortcut-resistant physical evolution benchmark
preserve reliably identifiable latent physical states across controlled
morphology variations?

\textbf{RQ2: Predictive Learnability under Distribution Shift.}
Can learned models capture nontrivial predictive dynamics beyond
persistence under composition, carrier, and joint distribution shifts?

\textbf{RQ3: Identifiability versus Process-Memory Emergence.}
Can highly identifiable physical states coexist with weak emergence
of the corresponding symbolic process representation under predictive
training?
"""

(
    OUT
    / "reframed_research_questions.tex"
).write_text(
    rq_text,
    encoding="utf-8",
)


rq_markdown = """# Reframed Research Questions

**RQ1  Benchmark Validity and State Identifiability.**  
Can a nontrivial, shortcut-resistant physical evolution benchmark preserve reliably identifiable latent physical states across controlled morphology variations?

**RQ2  Predictive Learnability under Distribution Shift.**  
Can learned models capture nontrivial predictive dynamics beyond persistence under composition, carrier, and joint distribution shifts?

**RQ3  Identifiability versus Process-Memory Emergence.**  
Can highly identifiable physical states coexist with weak emergence of the corresponding symbolic process representation under predictive training?
"""

(
    OUT
    / "reframed_research_questions.md"
).write_text(
    rq_markdown,
    encoding="utf-8",
)


# ======================================================================
# Table 1
# Benchmark validity and state identifiability
# ======================================================================

print(
    "[3/8] Generating Table 1"
)

spearman = float(
    field[
        "base_length_interface_ratio_spearman"
    ]
)

mean_length_ratio = float(
    morphology[
        "mean_characteristic_length_ratio"
    ]
)

minimum_length_ratio = float(
    morphology[
        "minimum_characteristic_length_ratio"
    ]
)

mean_interface_ratio = float(
    morphology[
        "mean_interface_density_ratio"
    ]
)

state_mean = float(
    decoder[
        "mean_validation_state_accuracy"
    ]
)

state_worst = float(
    decoder[
        "worst_seed_validation_state_accuracy"
    ]
)

coarsening_mean = float(
    decoder[
        "mean_validation_coarsening_accuracy"
    ]
)

global_validation = float(
    global_stats[
        "validation_accuracy"
    ]
)

global_threshold = float(
    global_stats[
        "maximum_allowed_validation_accuracy"
    ]
)

affine_latent_mse = float(
    affine[
        "validation_continuous_latent_mse"
    ]
)

affine_min_mse = float(
    affine[
        "minimum_required_validation_continuous_mse"
    ]
)

affine_primitive_exact = float(
    affine[
        "validation_exact_primitive_recovery_rate"
    ]
)

affine_all104 = float(
    affine[
        "fraction_validation_carriers_all_104_exact"
    ]
)


table1 = [
    {
        "Property":
            "Characteristic-length separation",
        "Metric":
            "Mean coarse/fine characteristic-length ratio",
        "Result":
            f"{mean_length_ratio:.3f}",
        "Reference":
            "Frozen morphology gate passed",
    },
    {
        "Property":
            "Characteristic-length separation",
        "Metric":
            "Minimum coarse/fine characteristic-length ratio",
        "Result":
            f"{minimum_length_ratio:.3f}",
        "Reference":
            "Frozen morphology gate passed",
    },
    {
        "Property":
            "Interface morphology",
        "Metric":
            "Mean coarse/fine interface-density ratio",
        "Result":
            f"{mean_interface_ratio:.3f}",
        "Reference":
            "Frozen morphology gate passed",
    },
    {
        "Property":
            "Carrier-scale decoupling",
        "Metric":
            "Base-length/interface-ratio Spearman rho",
        "Result":
            f"{spearman:.4f}",
        "Reference":
            "Near-zero final Tier-C-v4 association",
    },
    {
        "Property":
            "Eight-state identifiability",
        "Metric":
            "Mean validation state accuracy",
        "Result":
            f"{100.0 * state_mean:.2f}%",
        "Reference":
            "Required >= 80%",
    },
    {
        "Property":
            "Eight-state identifiability",
        "Metric":
            "Worst-seed validation state accuracy",
        "Result":
            f"{100.0 * state_worst:.2f}%",
        "Reference":
            "Required >= 75%",
    },
    {
        "Property":
            "Coarsening-factor identifiability",
        "Metric":
            "Mean validation accuracy",
        "Result":
            f"{100.0 * coarsening_mean:.2f}%",
        "Reference":
            "Required >= 90%",
    },
    {
        "Property":
            "Global-statistics shortcut control",
        "Metric":
            "28-feature validation accuracy",
        "Result":
            f"{100.0 * global_validation:.2f}%",
        "Reference":
            f"Required < {100.0 * global_threshold:.0f}%",
    },
    {
        "Property":
            "Affine shortcut control",
        "Metric":
            "Validation continuous latent MSE",
        "Result":
            f"{affine_latent_mse:.4f}",
        "Reference":
            f"Required > {affine_min_mse:.0e}",
    },
    {
        "Property":
            "Affine shortcut control",
        "Metric":
            "Exact primitive recovery",
        "Result":
            f"{100.0 * affine_primitive_exact:.2f}%",
        "Reference":
            "Required < 100%",
    },
    {
        "Property":
            "Affine shortcut control",
        "Metric":
            "Validation carriers exact on all 104 transformations",
        "Result":
            f"{100.0 * affine_all104:.2f}%",
        "Reference":
            "Required < 100%",
    },
]


write_csv(
    TABLE_DIR
    / "table_01_benchmark_validity.csv",
    table1,
)

write_latex_tabular(
    TABLE_DIR
    / "table_01_benchmark_validity.tex",
    [
        "Property",
        "Metric",
        "Result",
        "Reference",
    ],
    [
        [
            row["Property"],
            row["Metric"],
            row["Result"],
            row["Reference"],
        ]
        for row in table1
    ],
    "llrl",
)


# ======================================================================
# Table 2
# Primary joint-OOD predictive condition
# ======================================================================

print(
    "[4/8] Generating Table 2"
)

roles = {
    "B0":
        "Persistence baseline",

    "B1":
        "Learned baseline",

    "B2":
        "Learned baseline",

    "B3":
        "Learned baseline; lowest test MSE",

    "B4":
        "Frozen confirmatory comparator",

    "B5":
        "Learned baseline",

    "OCM":
        "Primary model",
}


table2 = []

for model in (
    "B0",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
):
    table2.append(
        {
            "Model":
                model,

            "Role":
                roles[
                    model
                ],

            "NoisyTargetRolloutMSE":
                f"{PRIMARY_JOINT_NOISE_025[model]:.6f}",
        }
    )


write_csv(
    TABLE_DIR
    / "table_02_primary_joint_ood_prediction.csv",
    table2,
)

write_latex_tabular(
    TABLE_DIR
    / "table_02_primary_joint_ood_prediction.tex",
    [
        "Model",
        "Role",
        "Noisy-target rollout MSE",
    ],
    [
        [
            row["Model"],
            row["Role"],
            row["NoisyTargetRolloutMSE"],
        ]
        for row in table2
    ],
    "llr",
)


primary = (
    g2e[
        "primary_confirmatory_result"
    ]
)

primary_test_rows = [
    {
        "Comparison":
            "OCM - B4",
        "Result":
            f"{primary['ocm_minus_b4']:+.6f}",
    },
    {
        "Comparison":
            "95% paired-bootstrap CI",
        "Result":
            (
                "["
                f"{primary['bootstrap_ci_lower']:+.6f}, "
                f"{primary['bootstrap_ci_upper']:+.6f}"
                "]"
            ),
    },
    {
        "Comparison":
            "OCM-better sequences",
        "Result":
            (
                f"{primary['ocm_better_sequence_count']}/"
                f"{primary['sequence_count']} "
                f"({100.0 * primary['ocm_better_sequence_fraction']:.1f}%)"
            ),
    },
]

write_csv(
    TABLE_DIR
    / "table_02_primary_confirmatory_test.csv",
    primary_test_rows,
)


# ======================================================================
# Table 3
# Identifiabilityrepresentation gap
# ======================================================================

print(
    "[5/8] Generating Table 3"
)

structural = (
    g2e[
        "secondary_structural_diagnostics"
    ]
)

alignment_mean = float(
    structural[
        "training_carrier_state_alignment_accuracy"
    ][
        "grand_mean"
    ]
)

state_path_mean = float(
    structural[
        "sealed_state_path_accuracy"
    ][
        "grand_mean"
    ]
)

exact_transform = float(
    structural[
        "exact_transformation_rate"
    ][
        "grand_mean"
    ]
)

heldout_exact = float(
    structural[
        "heldout_exact_transformation_rate"
    ][
        "grand_mean"
    ]
)

information_accuracy = float(
    structural[
        "information_class_accuracy"
    ][
        "grand_mean"
    ]
)

blackwell_ba = float(
    structural[
        "blackwell_relation_balanced_accuracy"
    ][
        "grand_mean"
    ]
)


table3 = [
    {
        "Level":
            "Physical observation space",
        "Metric":
            "State-decoder validation accuracy",
        "Result":
            f"{100.0 * state_mean:.2f}%",
    },
    {
        "Level":
            "Physical observation space",
        "Metric":
            "Worst-seed state accuracy",
        "Result":
            f"{100.0 * state_worst:.2f}%",
    },
    {
        "Level":
            "Reference",
        "Metric":
            "Eight-state chance accuracy",
        "Result":
            f"{100.0 * STATE_CHANCE:.2f}%",
    },
    {
        "Level":
            "OCM representation",
        "Metric":
            "Training-carrier state alignment",
        "Result":
            f"{100.0 * alignment_mean:.2f}%",
    },
    {
        "Level":
            "OCM representation",
        "Metric":
            "Sealed state-path accuracy",
        "Result":
            f"{100.0 * state_path_mean:.2f}%",
    },
    {
        "Level":
            "Process algebra",
        "Metric":
            "Exact transformation recovery",
        "Result":
            (
                f"{100.0 * exact_transform:.2f}% "
                "(1/104)"
            ),
    },
    {
        "Level":
            "Process algebra",
        "Metric":
            "Held-out exact transformation recovery",
        "Result":
            (
                f"{100.0 * heldout_exact:.2f}% "
                "(0/9)"
            ),
    },
    {
        "Level":
            "Information structure",
        "Metric":
            "Information-class accuracy",
        "Result":
            f"{100.0 * information_accuracy:.2f}%",
    },
    {
        "Level":
            "Information structure",
        "Metric":
            "Blackwell relation balanced accuracy",
        "Result":
            f"{100.0 * blackwell_ba:.2f}%",
    },
]


write_csv(
    TABLE_DIR
    / "table_03_identifiability_representation_gap.csv",
    table3,
)

write_latex_tabular(
    TABLE_DIR
    / "table_03_identifiability_representation_gap.tex",
    [
        "Level",
        "Metric",
        "Result",
    ],
    [
        [
            row["Level"],
            row["Metric"],
            row["Result"],
        ]
        for row in table3
    ],
    "llr",
)


# ======================================================================
# Table 4
# Predictive winner frequency
# ======================================================================

print(
    "[6/8] Generating Table 4"
)

require(
    sum(
        PREDICTIVE_WIN_COUNTS.values()
    ) == 20,
    (
        "Predictive winner counts "
        "must sum to 20."
    ),
)


table4 = []

for model in (
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "OCM",
):
    count = (
        PREDICTIVE_WIN_COUNTS[
            model
        ]
    )

    table4.append(
        {
            "Model":
                model,

            "BestConditions":
                count,

            "TotalConditions":
                20,

            "Fraction":
                f"{100.0 * count / 20.0:.1f}%",
        }
    )


write_csv(
    TABLE_DIR
    / "table_04_predictive_winner_frequency.csv",
    table4,
)

write_latex_tabular(
    TABLE_DIR
    / "table_04_predictive_winner_frequency.tex",
    [
        "Model",
        "Best conditions",
        "Total conditions",
        "Fraction",
    ],
    [
        [
            row["Model"],
            row["BestConditions"],
            row["TotalConditions"],
            row["Fraction"],
        ]
        for row in table4
    ],
    "lrrr",
)


# ======================================================================
# Figures
# ======================================================================

print(
    "[7/8] Generating publication figures"
)

configure_plot()


# ----------------------------------------------------------------------
# Figure 1
# Identifiability vs learned representation
# ----------------------------------------------------------------------

labels = [
    "Physical\nDecoder",
    "OCM State\nAlignment",
    "OCM State\nPath",
]

values = [
    state_mean,
    alignment_mean,
    state_path_mean,
]

fig, ax = plt.subplots(
    figsize=(
        3.35,
        2.55,
    )
)

x = np.arange(
    len(
        labels
    )
)

ax.bar(
    x,
    values,
    width=0.62,
)

ax.axhline(
    STATE_CHANCE,
    linestyle="--",
    linewidth=1.1,
    label="8-state chance",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels
)

ax.set_xlabel(
    "Representation / Diagnostic"
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_ylim(
    0.0,
    1.05,
)

ax.legend(
    frameon=False,
    loc="upper right",
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

fig.tight_layout()

save_figure(
    fig,
    "fig_01_identifiability_representation_gap",
)


# ----------------------------------------------------------------------
# Figure 2
# Joint-OOD prediction
# ----------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        3.35,
        2.55,
    )
)

noise = np.asarray(
    JOINT_NOISY_ROLLOUT[
        "noise"
    ],
    dtype=float,
)

for model in (
    "B0",
    "B3",
    "B4",
    "OCM",
):
    ax.plot(
        noise,
        JOINT_NOISY_ROLLOUT[
            model
        ],
        marker="o",
        label=model,
    )

ax.set_xlabel(
    "Noise Fraction"
)

ax.set_ylabel(
    "Noisy-Target Rollout MSE"
)

ax.set_xticks(
    noise
)

ax.legend(
    frameon=False,
    ncol=2,
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

fig.tight_layout()

save_figure(
    fig,
    "fig_02_joint_ood_prediction",
)


# ----------------------------------------------------------------------
# Figure 3
# State-path recovery
# ----------------------------------------------------------------------

expected_cells = [
    "test_iid_pairing",
    "test_composition",
    "test_carrier",
    "test_joint",
]

cell_labels = {
    "test_iid_pairing":
        "IID Pairing",

    "test_composition":
        "Composition",

    "test_carrier":
        "Carrier",

    "test_joint":
        "Joint",
}

noise_levels = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
]

state_lookup = {}

for row in state_path_rows:
    key = (
        str(
            row[
                "cell_id"
            ]
        ),
        float(
            row[
                "noise_fraction"
            ]
        ),
    )

    state_lookup[
        key
    ] = float(
        row[
            "state_path_accuracy_mean"
        ]
    )


require(
    len(
        state_lookup
    ) == 20,
    (
        "Expected exactly 20 "
        "state-path conditions."
    ),
)


fig, ax = plt.subplots(
    figsize=(
        3.35,
        2.55,
    )
)

for cell in expected_cells:
    values = [
        state_lookup[
            (
                cell,
                noise,
            )
        ]
        for noise in noise_levels
    ]

    ax.plot(
        noise_levels,
        values,
        marker="o",
        label=cell_labels[
            cell
        ],
    )

ax.axhline(
    STATE_CHANCE,
    linestyle="--",
    linewidth=1.1,
    label="8-state chance",
)

ax.set_xlabel(
    "Noise Fraction"
)

ax.set_ylabel(
    "State-Path Accuracy"
)

ax.set_xticks(
    noise_levels
)

ax.set_ylim(
    0.08,
    0.19,
)

ax.legend(
    frameon=False,
    ncol=2,
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

fig.tight_layout()

save_figure(
    fig,
    "fig_03_state_path_recovery",
)


# ======================================================================
# Manifest
# ======================================================================

print(
    "[8/8] Writing artifact manifest"
)

source_paths = [
    G2E_SUMMARY,
    FIELD_SUMMARY,
    MORPHOLOGY_SUMMARY,
    STATE_DECODER_SUMMARY,
    GLOBAL_STATS_SUMMARY,
    AFFINE_SUMMARY,
    STATE_PATH_SUMMARY,
]

manifest = {
    "artifact_set":
        "ICLR results tables and figures",

    "experimental_campaign_status":
        g2e[
            "experimental_campaign_status"
        ],

    "main_research_questions": [
        "RQ1 Benchmark Validity and State Identifiability",
        "RQ2 Predictive Learnability under Distribution Shift",
        "RQ3 Identifiability versus Process-Memory Emergence",
    ],

    "main_scientific_message":
        (
            "Highly identifiable physical states do not "
            "necessarily induce the corresponding symbolic "
            "process representation under predictive learning."
        ),

    "figure_policy": {
        "overall_title":
            False,

        "embedded_caption":
            False,

        "axis_titles":
            True,

        "png_dpi":
            600,

        "pdf_vector_output":
            True,
    },

    "tables": [
        "table_01_benchmark_validity",
        "table_02_primary_joint_ood_prediction",
        "table_02_primary_confirmatory_test",
        "table_03_identifiability_representation_gap",
        "table_04_predictive_winner_frequency",
    ],

    "figures": [
        "fig_01_identifiability_representation_gap",
        "fig_02_joint_ood_prediction",
        "fig_03_state_path_recovery",
    ],

    "source_hashes": {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path in source_paths
    },

    "no_training_performed":
        True,

    "no_tuning_performed":
        True,

    "no_test_reopening":
        True,

    "paper_artifact_generation_only":
        True,
}


with (
    OUT
    / "artifact_manifest.json"
).open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        manifest,
        handle,
        indent=2,
    )


print()
print(
    "=" * 78
)
print(
    "ICLR RESULTS ARTIFACT PACKAGE COMPLETE"
)
print(
    "=" * 78
)

print()
print(
    "Output directory:"
)
print(
    OUT
)

print()
print(
    "Tables:"
)

for path in sorted(
    TABLE_DIR.iterdir()
):
    print(
        " ",
        path
    )

print()
print(
    "Figures:"
)

for path in sorted(
    FIGURE_DIR.iterdir()
):
    print(
        " ",
        path
    )

print()
print(
    "Main RQs:"
)
print(
    "  RQ1: Benchmark Validity and State Identifiability"
)
print(
    "  RQ2: Predictive Learnability under Distribution Shift"
)
print(
    "  RQ3: Identifiability versus Process-Memory Emergence"
)

print()
print(
    "Main message:"
)
print(
    "  Physical-state identifiability does not imply "
    "process-memory emergence."
)
