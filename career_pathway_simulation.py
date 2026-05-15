#!/usr/bin/env python3
"""
career_pathway_simulation.py

Monte Carlo + simple ML model for comparing two rare-success pathways:

1. Elite academic/professional study route:
   - examples: doctor, elite engineer, elite scientific/technical professional.
2. High-level professional football/soccer route:
   - sustained professional contract and top-tier/high-level career.

This model is a transparent stage-gate simulation. It is not a prophecy for one child.
It estimates population-level probabilities under explicit assumptions.

Darija note:
    Had lmodel kayqaren joj toro9 b statistic: qraya elite vs football pro.

Run:
    python career_pathway_simulation.py --n 300000 --seed 42 --out outputs

Optional:
    python career_pathway_simulation.py --n 1000000 --seed 42 --out outputs
    python career_pathway_simulation.py --skip-ml
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = [
    "academic_ability",
    "athletic_ability",
    "discipline",
    "family_resources",
    "environment_quality",
    "health_resilience",
    "psychological_resilience",
    "luck",
    "goal_commitment",
    "relative_age_advantage",
    "physical_profile",
    "tactical_learning",
    "annual_severe_injury_probability",
    "severe_injuries",
    "career_ending_injury",
    "academic_life_disruption",
]


def expit(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    # Kanclipiw x bach probabilities ma ytl3och l 0/1 b tariqa extreme bzaf.
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def one_in_n(p: float) -> str:
    """Format probability as '1 in N'."""
    if p <= 0 or not np.isfinite(p):
        return "not observed"
    n = 1.0 / p
    if n < 10:
        return f"1 in {n:.1f}"
    if n < 100:
        return f"1 in {n:.0f}"
    return f"1 in {n:,.0f}"


def pct(p: float, digits: int = 3) -> str:
    return f"{100*p:.{digits}f}%"


def normal_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Approximate binomial confidence interval."""
    if n <= 0:
        return (float("nan"), float("nan"))
    se = math.sqrt(max(p * (1 - p), 0) / n)
    return (max(0.0, p - z * se), min(1.0, p + z * se))


def make_correlation_matrix() -> Tuple[List[str], np.ndarray]:
    """Build a positive-definite-ish correlation matrix for latent traits."""
    names = [
        "academic_ability",
        "athletic_ability",
        "discipline",
        "family_resources",
        "environment_quality",
        "health_resilience",
        "psychological_resilience",
        "luck",
    ]
    idx = {n: i for i, n in enumerate(names)}
    corr = np.eye(len(names))

    def setc(a: str, b: str, val: float) -> None:
        corr[idx[a], idx[b]] = val
        corr[idx[b], idx[a]] = val

    # Had correlations rah modelling assumptions: machi facts thabtin.
    # Lfikra: resources/environment kaytla9aw m3a qraia, health m3a sport, etc.
    setc("academic_ability", "family_resources", 0.25)
    setc("academic_ability", "discipline", 0.20)
    setc("academic_ability", "environment_quality", 0.20)
    setc("family_resources", "environment_quality", 0.45)
    setc("family_resources", "discipline", 0.15)
    setc("athletic_ability", "health_resilience", 0.25)
    setc("athletic_ability", "discipline", 0.10)
    setc("athletic_ability", "environment_quality", 0.10)
    setc("discipline", "psychological_resilience", 0.25)
    setc("health_resilience", "psychological_resilience", 0.15)
    setc("academic_ability", "luck", 0.05)
    setc("athletic_ability", "luck", 0.05)

    eig_min = np.linalg.eigvalsh(corr).min()
    if eig_min <= 1e-8:
        raise ValueError(f"Correlation matrix is not positive definite; min eigenvalue={eig_min}")
    return names, corr


def generate_population(n: int, seed: int) -> pd.DataFrame:
    """
    Generate a synthetic population of children who set a serious goal early.
    All latent variables are standardized; 0 = average in this goal-committed cohort.
    """
    rng = np.random.default_rng(seed)
    names, corr = make_correlation_matrix()
    raw = rng.multivariate_normal(np.zeros(len(names)), corr, size=n)
    df = pd.DataFrame(raw, columns=names)

    # Hna kanbniw population synthetic: kol row = wa7ed child f cohort goal-committed.
    # Both pathways assume a goal set early. This creates high motivation on average,
    # but still variable commitment across children and years.
    df["goal_commitment"] = np.clip(rng.normal(loc=1.0, scale=0.35, size=n), -0.25, 2.25)

    # Soccer-specific relative-age and physical/tactical factors.
    # F football, selection machi ghir talent: physique, maturity, learning, timing kaydakhlo.
    # relative_age_advantage captures the selection advantage from being older/more mature
    # within an age cohort. It is not the same as long-run talent.
    df["relative_age_advantage"] = rng.normal(loc=0.0, scale=1.0, size=n)
    df["physical_profile"] = (
        0.75 * df["athletic_ability"]
        + 0.25 * df["health_resilience"]
        + rng.normal(0, 0.20, size=n)
    )
    df["tactical_learning"] = (
        0.35 * df["academic_ability"]
        + 0.45 * df["discipline"]
        + 0.20 * df["psychological_resilience"]
        + rng.normal(0, 0.20, size=n)
    )

    # Injury process. This is deliberately simplified: it turns multi-year training
    # exposure into severe injuries and rare career-ending injuries.
    # Injuries hna mkhtasra: enough bach n7esbo risk, machi medical simulator.
    annual_p = expit(
        -2.20
        - 0.55 * df["health_resilience"]
        + 0.25 * df["athletic_ability"]      # high performers train/play more
        + 0.15 * df["goal_commitment"]
        + rng.normal(0, 0.25, size=n)
    )
    df["annual_severe_injury_probability"] = annual_p
    # Development window from roughly age 12 to 20.
    df["severe_injuries"] = rng.poisson(lam=np.clip(annual_p * 7.0, 0, 10), size=n)
    career_end_p = np.clip(
        0.015
        + 0.045 * df["severe_injuries"]
        + 0.030 * np.maximum(-df["health_resilience"], 0),
        0.0,
        0.60,
    )
    df["career_ending_injury"] = rng.binomial(1, career_end_p, size=n)

    # Academic life disruption: illness, family instability, financial shock, migration,
    # caring obligations, etc. This is not "ability"; it is external disruption.
    disruption_p = expit(
        -2.50
        - 0.35 * df["family_resources"]
        - 0.30 * df["psychological_resilience"]
        + rng.normal(0, 0.25, size=n)
    )
    df["academic_life_disruption"] = rng.binomial(1, disruption_p, size=n)

    return df


def simulate_paths(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Simulate stage-gate paths.

    The intercepts are calibrated so that, under the default population,
    teen soccer-selected players have roughly single-digit probability of high-level
    football, close to published academy follow-up anchors. Edit these intercepts
    if you want a different country, league pyramid, or definition of success.
    """
    rng = np.random.default_rng(seed + 1009)
    out = df.copy()

    # Academic / elite study pathway.
    # Hadi stage-gate: ila tfawt stage, kaymchi l next; ila la, pathway kaywqaf.
    # Outcome definition:
    # elite_study = reaches and completes a selective medical/engineering/scientific
    # route leading to elite professional status.
    out["p_acad_foundation"] = expit(
        0.80
        + 1.00 * out["academic_ability"]
        + 0.60 * out["discipline"]
        + 0.60 * out["family_resources"]
        + 0.45 * out["environment_quality"]
        + 0.40 * out["goal_commitment"]
        + 0.20 * out["health_resilience"]
        + 0.15 * out["luck"]
        - 0.60 * out["academic_life_disruption"]
    )
    out["p_acad_strong_secondary"] = expit(
        -0.10
        + 1.35 * out["academic_ability"]
        + 0.90 * out["discipline"]
        + 0.55 * out["family_resources"]
        + 0.45 * out["environment_quality"]
        + 0.35 * out["goal_commitment"]
        + 0.15 * out["health_resilience"]
        + 0.20 * out["luck"]
        - 0.70 * out["academic_life_disruption"]
    )
    out["p_acad_elite_entry"] = expit(
        -3.50
        + 1.50 * out["academic_ability"]
        + 1.10 * out["discipline"]
        + 0.65 * out["family_resources"]
        + 0.55 * out["environment_quality"]
        + 0.30 * out["goal_commitment"]
        + 0.20 * out["luck"]
        - 0.60 * out["academic_life_disruption"]
    )
    out["p_acad_elite_completion"] = expit(
        -0.20
        + 0.85 * out["academic_ability"]
        + 1.15 * out["discipline"]
        + 0.45 * out["family_resources"]
        + 0.35 * out["environment_quality"]
        + 0.25 * out["goal_commitment"]
        + 0.35 * out["health_resilience"]
        + 0.20 * out["psychological_resilience"]
        + 0.25 * out["luck"]
        - 0.80 * out["academic_life_disruption"]
    )

    out["acad_foundation"] = rng.binomial(1, out["p_acad_foundation"])
    out["acad_strong_secondary"] = out["acad_foundation"] * rng.binomial(
        1, out["p_acad_strong_secondary"]
    )
    out["acad_elite_entry"] = out["acad_strong_secondary"] * rng.binomial(
        1, out["p_acad_elite_entry"]
    )
    out["elite_study"] = out["acad_elite_entry"] * rng.binomial(
        1, out["p_acad_elite_completion"]
    )

    # Football/soccer pathway.
    # Nafs l logic dyal gates, walakin probabilities hna rare bzaf.
    # Outcome definitions:
    # any_pro_soccer = earns a professional contract / pro senior pathway.
    # high_level_soccer = reaches a high-level/top-tier professional standard.
    out["p_soc_regional_selection"] = expit(
        -7.50
        + 1.90 * out["athletic_ability"]
        + 0.80 * out["physical_profile"]
        + 0.65 * out["discipline"]
        + 0.45 * out["family_resources"]
        + 0.65 * out["environment_quality"]
        + 0.55 * out["goal_commitment"]
        + 0.30 * out["health_resilience"]
        + 0.20 * out["relative_age_advantage"]
        + 0.25 * out["luck"]
        - 0.30 * out["career_ending_injury"]
    )
    out["p_soc_academy_retention"] = expit(
        -7.50
        + 2.10 * out["athletic_ability"]
        + 0.95 * out["physical_profile"]
        + 0.90 * out["discipline"]
        + 0.35 * out["family_resources"]
        + 0.75 * out["environment_quality"]
        + 0.45 * out["goal_commitment"]
        + 0.35 * out["health_resilience"]
        + 0.25 * out["relative_age_advantage"]
        + 0.20 * out["luck"]
        - 0.35 * out["severe_injuries"]
        - 2.00 * out["career_ending_injury"]
    )
    out["p_soc_first_contract"] = expit(
        -10.50
        + 2.30 * out["athletic_ability"]
        + 1.10 * out["physical_profile"]
        + 1.00 * out["discipline"]
        + 0.25 * out["family_resources"]
        + 0.70 * out["environment_quality"]
        + 0.35 * out["goal_commitment"]
        + 0.30 * out["tactical_learning"]
        + 0.25 * out["luck"]
        - 0.65 * out["severe_injuries"]
        - 2.00 * out["career_ending_injury"]
    )
    out["p_soc_high_level"] = expit(
        -12.00
        + 2.40 * out["athletic_ability"]
        + 1.20 * out["physical_profile"]
        + 1.10 * out["discipline"]
        + 0.20 * out["family_resources"]
        + 0.70 * out["environment_quality"]
        + 0.30 * out["goal_commitment"]
        + 0.35 * out["tactical_learning"]
        + 0.30 * out["luck"]
        - 0.85 * out["severe_injuries"]
        - 2.50 * out["career_ending_injury"]
    )

    out["soc_regional_selection"] = rng.binomial(1, out["p_soc_regional_selection"])
    out["soc_academy_retention"] = out["soc_regional_selection"] * rng.binomial(
        1, out["p_soc_academy_retention"]
    )
    out["any_pro_soccer"] = out["soc_academy_retention"] * rng.binomial(
        1, out["p_soc_first_contract"]
    )
    out["high_level_soccer"] = out["any_pro_soccer"] * rng.binomial(
        1, out["p_soc_high_level"]
    )

    return out


def summarize_stage_rates(df: pd.DataFrame) -> pd.DataFrame:
    # Kanjm3o funnel dyal kol pathway bach results ybano wad7in, stage by stage.
    stages = [
        ("Academic: foundation", "acad_foundation"),
        ("Academic: strong secondary", "acad_strong_secondary"),
        ("Academic: elite entry", "acad_elite_entry"),
        ("Academic: elite study success", "elite_study"),
        ("Soccer: regional/academy selection", "soc_regional_selection"),
        ("Soccer: academy retention", "soc_academy_retention"),
        ("Soccer: any pro contract", "any_pro_soccer"),
        ("Soccer: high-level pro", "high_level_soccer"),
    ]
    rows = []
    n = len(df)
    for label, col in stages:
        p = float(df[col].mean())
        lo, hi = normal_ci(p, n)
        rows.append(
            {
                "stage": label,
                "count": int(df[col].sum()),
                "n": n,
                "probability": p,
                "ci_low": lo,
                "ci_high": hi,
                "percent": pct(p),
                "one_in": one_in_n(p),
            }
        )
    return pd.DataFrame(rows)


def summarize_profiles(df: pd.DataFrame) -> pd.DataFrame:
    # Profiles kayjawbo 3la "chno ila": high academic, high athletic, injuries, resources...
    q = df.quantile([0.10, 0.25, 0.75, 0.90], numeric_only=True)
    profiles = {
        "All goal-committed children": np.ones(len(df), dtype=bool),
        "Low current academic ability (bottom 10%)": df["academic_ability"] <= q.loc[0.10, "academic_ability"],
        "High current academic ability (top 10%)": df["academic_ability"] >= q.loc[0.90, "academic_ability"],
        "Low athletic ability (bottom 10%)": df["athletic_ability"] <= q.loc[0.10, "athletic_ability"],
        "High athletic ability (top 10%)": df["athletic_ability"] >= q.loc[0.90, "athletic_ability"],
        "Low family resources (bottom 25%)": df["family_resources"] <= q.loc[0.25, "family_resources"],
        "High family resources (top 25%)": df["family_resources"] >= q.loc[0.75, "family_resources"],
        "High discipline (top 10%)": df["discipline"] >= q.loc[0.90, "discipline"],
        "Low academic + high athletic": (
            (df["academic_ability"] <= q.loc[0.25, "academic_ability"]) &
            (df["athletic_ability"] >= q.loc[0.75, "athletic_ability"])
        ),
        "High academic + low athletic": (
            (df["academic_ability"] >= q.loc[0.75, "academic_ability"]) &
            (df["athletic_ability"] <= q.loc[0.25, "athletic_ability"])
        ),
        "High athletic + no severe injury": (
            (df["athletic_ability"] >= q.loc[0.90, "athletic_ability"]) &
            (df["severe_injuries"] == 0)
        ),
        "High athletic + at least one severe injury": (
            (df["athletic_ability"] >= q.loc[0.90, "athletic_ability"]) &
            (df["severe_injuries"] >= 1)
        ),
        "At least one severe football injury": df["severe_injuries"] >= 1,
        "No severe football injury": df["severe_injuries"] == 0,
        "Soccer-selected by early teens": df["soc_regional_selection"] == 1,
        "Strong academic secondary profile": df["acad_strong_secondary"] == 1,
    }

    rows = []
    for name, mask in profiles.items():
        sub = df.loc[mask]
        n = len(sub)
        if n == 0:
            continue
        p_study = float(sub["elite_study"].mean())
        p_anypro = float(sub["any_pro_soccer"].mean())
        p_highsoc = float(sub["high_level_soccer"].mean())
        ratio = p_study / p_highsoc if p_highsoc > 0 else float("inf")
        rows.append(
            {
                "profile": name,
                "n": n,
                "elite_study_probability": p_study,
                "any_pro_soccer_probability": p_anypro,
                "high_level_soccer_probability": p_highsoc,
                "study_to_high_soccer_ratio": ratio,
                "elite_study_percent": pct(p_study),
                "any_pro_soccer_percent": pct(p_anypro),
                "high_level_soccer_percent": pct(p_highsoc),
                "elite_study_one_in": one_in_n(p_study),
                "high_level_soccer_one_in": one_in_n(p_highsoc),
            }
        )
    return pd.DataFrame(rows)


def plot_stage_funnel(stage_rates: pd.DataFrame, outdir: Path) -> None:
    # Two separate plots because the scales are very different.
    # Qraia probabilities kbar bzaf mn soccer, donc plot wa7ed ghadi ykhebi football.
    acad = stage_rates[stage_rates["stage"].str.startswith("Academic")].copy()
    soc = stage_rates[stage_rates["stage"].str.startswith("Soccer")].copy()

    for subset, filename, title in [
        (acad, "academic_stage_funnel.png", "Academic pathway stage funnel"),
        (soc, "soccer_stage_funnel.png", "Soccer pathway stage funnel"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [s.replace("Academic: ", "").replace("Soccer: ", "") for s in subset["stage"]]
        ax.bar(labels, subset["probability"] * 100)
        ax.set_ylabel("Share of original goal-committed cohort (%)")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=180)
        plt.close(fig)


def plot_deciles(df: pd.DataFrame, outdir: Path) -> None:
    tmp = df.copy()
    tmp["academic_decile"] = pd.qcut(tmp["academic_ability"], 10, labels=False, duplicates="drop") + 1
    tmp["athletic_decile"] = pd.qcut(tmp["athletic_ability"], 10, labels=False, duplicates="drop") + 1

    acad = tmp.groupby("academic_decile")["elite_study"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(acad["academic_decile"], acad["elite_study"] * 100, marker="o")
    ax.set_xlabel("Academic ability decile in the simulated cohort")
    ax.set_ylabel("Elite study success (%)")
    ax.set_title("Elite study probability by academic decile")
    fig.tight_layout()
    fig.savefig(outdir / "elite_study_by_academic_decile.png", dpi=180)
    plt.close(fig)

    soc = tmp.groupby("athletic_decile")["high_level_soccer"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(soc["athletic_decile"], soc["high_level_soccer"] * 100, marker="o")
    ax.set_xlabel("Athletic ability decile in the simulated cohort")
    ax.set_ylabel("High-level soccer success (%)")
    ax.set_title("High-level soccer probability by athletic decile")
    fig.tight_layout()
    fig.savefig(outdir / "high_level_soccer_by_athletic_decile.png", dpi=180)
    plt.close(fig)


def run_ml(df: pd.DataFrame, outdir: Path, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Fit simple random forests to the simulated outcomes and export feature importances.
    This is not the main model; it is an interpretability check showing which variables
    predict success under the simulation.
    """
    # ML hna ghir explanation layer: kaychre7 simulation, ma kaybdelch proba
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        warnings.warn(
            "scikit-learn is not installed. Skipping ML. Install with: pip install scikit-learn"
        )
        return pd.DataFrame(
            [{"outcome": "ML skipped", "feature": str(exc), "importance": np.nan, "auc": np.nan}]
        )

    rng = np.random.default_rng(seed + 202)
    if len(df) > sample_size:
        idx = rng.choice(df.index.to_numpy(), size=sample_size, replace=False)
        data = df.loc[idx].copy()
    else:
        data = df.copy()

    X = data[FEATURES].astype(float)
    outcomes = ["elite_study", "any_pro_soccer", "high_level_soccer"]
    rows = []

    for outcome in outcomes:
        y = data[outcome].astype(int)
        positives = int(y.sum())
        if positives < 20:
            warnings.warn(
                f"Only {positives} positives for {outcome}. Increase --n or interpret ML cautiously."
            )

        stratify = y if positives >= 2 and positives < len(y) - 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=stratify
        )

        clf = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, pred)
        except ValueError:
            auc = float("nan")

        importances = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
        for feature, importance in importances.items():
            rows.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "importance": float(importance),
                    "auc": float(auc),
                    "positives_in_ml_sample": positives,
                    "ml_sample_n": len(data),
                }
            )

        # Plot top 12 importances.
        top = importances.head(12).sort_values()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top.index, top.values)
        ax.set_xlabel("Random forest feature importance")
        ax.set_title(f"Predictors of {outcome} (AUC={auc:.3f})")
        fig.tight_layout()
        fig.savefig(outdir / f"feature_importance_{outcome}.png", dpi=180)
        plt.close(fig)

    return pd.DataFrame(rows)


def latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for table text."""
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def write_results_snippet(
    outdir: Path,
    stage_rates: pd.DataFrame,
    profiles: pd.DataFrame,
    ml_importances: pd.DataFrame,
    metadata: Dict[str, object],
) -> None:
    """Write a LaTeX snippet that the report can input."""
    all_row = profiles[profiles["profile"] == "All goal-committed children"].iloc[0]
    selected_row = profiles[profiles["profile"] == "Soccer-selected by early teens"].iloc[0]
    academic_row = profiles[profiles["profile"] == "Strong academic secondary profile"].iloc[0]

    p_study = all_row["elite_study_probability"]
    p_soc = all_row["high_level_soccer_probability"]
    p_any = all_row["any_pro_soccer_probability"]
    ratio = all_row["study_to_high_soccer_ratio"]

    lines = []
    lines.append("% Auto-generated by career_pathway_simulation.py")
    lines.append("\\section{Simulation results from the current run}")
    lines.append(f"Run size: $N={metadata['n']:,}$ simulated goal-committed children; seed: {metadata['seed']}.")
    lines.append("")
    lines.append("\\subsection{Main result}")
    lines.append(
        "Under the current calibration, the estimated probability of the elite study pathway is "
        f"\\textbf{{{latex_escape(pct(p_study))}}} ({one_in_n(p_study)}). "
        "The estimated probability of any professional soccer contract is "
        f"\\textbf{{{latex_escape(pct(p_any))}}} ({one_in_n(p_any)}), and the probability of high-level professional soccer is "
        f"\\textbf{{{latex_escape(pct(p_soc))}}} ({one_in_n(p_soc)}). "
        f"The elite study route is therefore about \\textbf{{{ratio:.1f} times}} as likely as the high-level soccer route "
        "for the default goal-committed child cohort."
    )
    lines.append("")
    lines.append(
        "Conditioning changes the picture: among simulated children who are soccer-selected by early teens, "
        f"high-level soccer is {latex_escape(pct(selected_row['high_level_soccer_probability']))} "
        f"({one_in_n(selected_row['high_level_soccer_probability'])}); "
        "among simulated children with a strong academic secondary profile, elite study success is "
        f"{latex_escape(pct(academic_row['elite_study_probability']))} ({one_in_n(academic_row['elite_study_probability'])})."
    )
    lines.append("")

    lines.append("\\subsection{Stage-gate funnel}")
    lines.append("\\begin{center}\\small")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Stage & Count & Probability & One-in \\\\")
    lines.append("\\midrule")
    for _, r in stage_rates.iterrows():
        lines.append(
            f"{latex_escape(str(r['stage']))} & {int(r['count']):,} & {latex_escape(str(r['percent']))} & {latex_escape(str(r['one_in']))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")

    lines.append("\\subsection{Circumstance sensitivity}")
    show_profiles = profiles[
        profiles["profile"].isin(
            [
                "All goal-committed children",
                "Low current academic ability (bottom 10%)",
                "High current academic ability (top 10%)",
                "Low athletic ability (bottom 10%)",
                "High athletic ability (top 10%)",
                "Low family resources (bottom 25%)",
                "High family resources (top 25%)",
                "Low academic + high athletic",
                "High academic + low athletic",
                "High athletic + no severe injury",
                "High athletic + at least one severe injury",
                "At least one severe football injury",
                "No severe football injury",
                "Soccer-selected by early teens",
                "Strong academic secondary profile",
            ]
        )
    ]
    lines.append("\\begin{center}\\scriptsize")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Profile & N & Elite study & Any pro soccer & High-level soccer \\\\")
    lines.append("\\midrule")
    for _, r in show_profiles.iterrows():
        lines.append(
            f"{latex_escape(str(r['profile']))} & {int(r['n']):,} & "
            f"{latex_escape(str(r['elite_study_percent']))} & "
            f"{latex_escape(str(r['any_pro_soccer_percent']))} & "
            f"{latex_escape(str(r['high_level_soccer_percent']))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{center}")

    if not ml_importances.empty and "outcome" in ml_importances.columns:
        lines.append("\\subsection{Machine-learning interpretability check}")
        for outcome in ["elite_study", "any_pro_soccer", "high_level_soccer"]:
            sub = ml_importances[ml_importances["outcome"] == outcome].head(6)
            if sub.empty:
                continue
            auc = sub["auc"].iloc[0]
            top_features = ", ".join(
                [f"{latex_escape(str(f))}" for f in sub["feature"].tolist()]
            )
            lines.append(
                f"For \\texttt{{{latex_escape(outcome)}}}, the random-forest AUC was approximately {auc:.3f}. "
                f"The strongest predictors were: {top_features}."
            )

    lines.append("\\subsection{Generated figures}")
    lines.append("The Python run also generated these plot/data files in the output folder:")
    lines.append("\\begin{itemize}")
    for filename in [
        "academic_stage_funnel.png",
        "soccer_stage_funnel.png",
        "elite_study_by_academic_decile.png",
        "high_level_soccer_by_athletic_decile.png",
        "ml_feature_importances.csv",
    ]:
        if (outdir / filename).exists():
            lines.append(f"\\item \\texttt{{{latex_escape(filename)}}}")
    lines.append("\\end{itemize}")

    (outdir / "results_snippet.tex").write_text("\n".join(lines), encoding="utf-8")


def write_text_summary(outdir: Path, stage_rates: pd.DataFrame, profiles: pd.DataFrame, metadata: Dict[str, object]) -> None:
    all_row = profiles[profiles["profile"] == "All goal-committed children"].iloc[0]
    text = []
    text.append("CAREER PATHWAY MODEL SUMMARY")
    text.append("=" * 34)
    text.append(f"N = {metadata['n']:,}")
    text.append(f"seed = {metadata['seed']}")
    text.append("")
    text.append("Main default-cohort estimates:")
    text.append(f"- Elite study pathway: {all_row['elite_study_percent']} ({all_row['elite_study_one_in']})")
    text.append(f"- Any pro soccer contract: {all_row['any_pro_soccer_percent']} ({one_in_n(all_row['any_pro_soccer_probability'])})")
    text.append(f"- High-level pro soccer: {all_row['high_level_soccer_percent']} ({all_row['high_level_soccer_one_in']})")
    text.append(f"- Study / high-level soccer ratio: {all_row['study_to_high_soccer_ratio']:.1f}x")
    text.append("")
    text.append("Stage rates:")
    text.append(stage_rates.to_string(index=False))
    text.append("")
    text.append("Profile rates:")
    text.append(profiles.to_string(index=False))
    (outdir / "model_summary.txt").write_text("\n".join(text), encoding="utf-8")


def save_outputs(df: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> None:
    # Had function katsift kolchi l outputs folder: CSVs, plots, summary, LaTeX snippet.
    outdir.mkdir(parents=True, exist_ok=True)

    stage_rates = summarize_stage_rates(df)
    profiles = summarize_profiles(df)

    stage_rates.to_csv(outdir / "stage_rates.csv", index=False)
    profiles.to_csv(outdir / "profile_rates.csv", index=False)

    plot_stage_funnel(stage_rates, outdir)
    plot_deciles(df, outdir)

    if args.skip_ml:
        ml = pd.DataFrame([{"outcome": "ML skipped by user", "feature": "", "importance": np.nan, "auc": np.nan}])
    else:
        ml = run_ml(df, outdir=outdir, sample_size=args.ml_sample, seed=args.seed)
    ml.to_csv(outdir / "ml_feature_importances.csv", index=False)

    # Save a sample only; the full simulated dataset can be very large.
    sample_n = min(args.save_sample, len(df))
    df.sample(n=sample_n, random_state=args.seed).to_csv(outdir / "simulated_individuals_sample.csv", index=False)

    metadata = {
        "n": len(df),
        "seed": args.seed,
        "ml_sample": args.ml_sample,
        "skip_ml": args.skip_ml,
        "model_version": "1.0",
        "notes": "Synthetic Monte Carlo stage-gate model; parameters documented in LaTeX report.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_results_snippet(outdir, stage_rates, profiles, ml, metadata)
    write_text_summary(outdir, stage_rates, profiles, metadata)

    print((outdir / "model_summary.txt").read_text(encoding="utf-8").split("Stage rates:")[0])
    print(f"Outputs written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Career pathway Monte Carlo model: elite study vs high-level soccer.")
    parser.add_argument("--n", type=int, default=300_000, help="Number of simulated children.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output folder.")
    parser.add_argument("--ml-sample", type=int, default=60_000, help="Max sample size for ML feature importance.")
    parser.add_argument("--save-sample", type=int, default=10_000, help="Number of simulated rows to save as CSV sample.")
    parser.add_argument("--skip-ml", action="store_true", help="Skip scikit-learn random-forest interpretability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 50_000:
        warnings.warn(
            "N is small for rare soccer events. Use at least 300000 for stable high-level soccer estimates."
        )

    # Pipeline sahl: generate population -> simulate paths -> write outputs.
    pop = generate_population(args.n, args.seed)
    sim = simulate_paths(pop, args.seed)
    save_outputs(sim, args.out, args)


if __name__ == "__main__":
    main()
