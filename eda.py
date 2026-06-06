# eda.py – Exploratory Data Analysis for AQI Predictor (Karachi)
"""
Exploratory Data Analysis (EDA) script
============================================================
Purpose:
  * Load the engineered feature set (data/features.parquet).
  * Summarize column data types, missing values, and basic statistics.
  * Visualise:
    - Correlation matrix (heatmap)
    - Distribution of each numeric feature (histograms / KDE)
    - Pair‑wise relationships for the top correlated features.
    - Target variable (`target_aqi_24h`, `target_aqi_48h`, `target_aqi_72h`) distribution.
  * Save all plots to a `plots/` folder for quick inspection.

The script is intentionally self‑contained – you can simply run:
    python eda.py
and it will create the visual assets without affecting the rest of the pipeline.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --------------------------
# Configuration
# --------------------------
DATA_PATH = os.path.join("data", "features.parquet")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Use a dark, modern aesthetic that works well on both light and dark terminals.
plt.style.use("dark_background")
sns.set_context("talk", font_scale=1.1)
sns.set_style("darkgrid")

# --------------------------
# Helper Functions
# --------------------------
def save_plot(fig, name: str) -> None:
    """Save a Matplotlib figure to the plots folder as PNG.
    Args:
        fig: Matplotlib figure instance.
        name: Filename without extension.
    """
    out_path = PLOTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ED] Saved plot → {out_path}")

def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising data type, missing count, % missing, and basic stats.
    """
    overview = []
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isna().sum()
        miss_pct = missing / len(df) * 100
        if np.issubdtype(dtype, np.number):
            stats = df[col].describe()
            mean = stats.get("mean", np.nan)
            std = stats.get("std", np.nan)
            min_ = stats.get("min", np.nan)
            max_ = stats.get("max", np.nan)
        else:
            mean = std = min_ = max_ = np.nan
        overview.append({
            "column": col,
            "dtype": str(dtype),
            "missing": missing,
            "%missing": round(miss_pct, 2),
            "mean": round(mean, 3) if pd.notna(mean) else None,
            "std": round(std, 3) if pd.notna(std) else None,
            "min": round(min_, 3) if pd.notna(min_) else None,
            "max": round(max_, 3) if pd.notna(max_) else None,
        })
    return pd.DataFrame(overview).set_index("column")

def plot_correlation(df: pd.DataFrame, target: str = None) -> None:
    """Plot a correlation heatmap for numeric columns.
    If a target column is supplied the heatmap will be sorted so the target appears
    at the top‑left corner for easier visual inspection.
    """
    corr = df.select_dtypes(include=[np.number]).corr()
    if target and target in corr.columns:
        # reorder rows/cols to bring the target close to the centre of the matrix
        ordered = corr.abs()[target].sort_values(ascending=False).index
        corr = corr.loc[ordered, ordered]
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="viridis",
        linewidths=0.5,
        linecolor="gray",
        annot=False,
        fmt=".2f",
        cbar_kws={"shrink": .8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16, weight="bold")
    save_plot(fig, "correlation_heatmap")

def plot_histograms(df: pd.DataFrame, cols_per_fig: int = 6) -> None:
    """Plot histograms (with KDE) for all numeric columns, grouped into batches.
    Each figure contains up to `cols_per_fig` sub‑plots arranged in a grid.
    """
    numeric = df.select_dtypes(include=[np.number])
    cols = list(numeric.columns)
    for i in range(0, len(cols), cols_per_fig):
        batch = cols[i : i + cols_per_fig]
        n = len(batch)
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
        axes = axes.flatten()
        for ax, col in zip(axes, batch):
            sns.histplot(df[col].dropna(), kde=True, stat="density", ax=ax, color="#ff6f61")
            ax.set_title(col, fontsize=12)
        # hide any unused sub‑plots
        for j in range(len(batch), len(axes)):
            axes[j].axis("off")
        save_plot(fig, f"histograms_batch_{i // cols_per_fig + 1}")

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Visualise the distribution of each target variable (24h/48h/72h).
    """
    targets = [c for c in df.columns if c.startswith("target_aqi_")]
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, targets):
        sns.kdeplot(df[col].dropna(), fill=True, ax=ax, color="#4a90e2")
        ax.set_title(col)
        ax.set_xlabel("AQI")
    save_plot(fig, "target_distributions")

def main() -> None:
    print("[ED] Loading feature parquet …")
    df = pd.read_parquet(DATA_PATH)
    print(f"[ED] Loaded {len(df)} rows × {len(df.columns)} columns")

    # ---------------------------------------------------------------------
    # 1️⃣ Column overview – data types, missing values, basic stats
    # ---------------------------------------------------------------------
    overview = column_overview(df)
    overview_path = PLOTS_DIR / "column_overview.json"
    overview.to_json(overview_path, orient="index", indent=2)
    print(f"[ED] Column overview saved → {overview_path}")

    # ---------------------------------------------------------------------
    # 2️⃣ Correlation matrix (focus on the most relevant target)
    # ---------------------------------------------------------------------
    plot_correlation(df, target="target_aqi_24h")

    # ---------------------------------------------------------------------
    # 3️⃣ Distribution of numeric features – histograms/KDE
    # ---------------------------------------------------------------------
    plot_histograms(df)

    # ---------------------------------------------------------------------
    # 4️⃣ Target variable analysis
    # ---------------------------------------------------------------------
    plot_target_distribution(df)

    # ---------------------------------------------------------------------
    # 5️⃣ Quick pair‑wise scatter for the top‑3 correlated features with target
    # ---------------------------------------------------------------------
    numeric = df.select_dtypes(include=[np.number])
    corr_target = numeric.corr()["target_aqi_24h"].abs().sort_values(ascending=False)
    top_features = corr_target.iloc[1:4].index.tolist()  # skip the target itself
    if top_features:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.pairplot(df[["target_aqi_24h"] + top_features], diag_kind="kde", plot_kws={"alpha": 0.6})
        save_plot(fig, "pairplot_top_features")

    print("[ED] EDA complete – explore the generated PNG files in the 'plots' folder.")

if __name__ == "__main__":
    main()
