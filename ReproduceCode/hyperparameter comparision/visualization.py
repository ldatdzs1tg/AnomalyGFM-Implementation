import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

DATASET_STYLE = {
    "reddit_svd":        {"label": "Reddit",        "marker": "^"},
    "Amazon_upu_svd":    {"label": "Amazon",        "marker": "v"},
    "Disney_svd":        {"label": "Disney",        "marker": "D"},
    "amazon_svd":    {"label": "Amazon-all",    "marker": "o"},
    "yelp_svd":   {"label": "YelpChi-all",   "marker": ">"},
    "questions_svd":      {"label": "Question",      "marker": "s"},
    "tolokers_svd":      {"label": "Tolokers",      "marker": "*"},
    "elliptic_svd":      {"label": "Elliptic",      "marker": "<"},
    "t_finance_svd":     {"label": "T-Finance",     "marker": "p"},
}


def plot_T(csv_path):
    df = pd.read_csv(csv_path)

    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True)

    ax_auc, ax_ap = axes

    for ds in datasets:
        if ds not in DATASET_STYLE:
            continue

        sub = df[df["dataset"] == ds].sort_values("T")

        style = DATASET_STYLE[ds]

        ax_auc.plot(
            sub["T"],
            sub["auc"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"]
        )

        ax_ap.plot(
            sub["T"],
            sub["ap"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"]
        )

    # auc subplot
    ax_auc.set_title("(a) AUROC")
    ax_auc.set_ylabel("AUROC")
    ax_auc.set_xlabel("T")
    ax_auc.set_ylim(0.5, 1.0)

    # ap subplot
    ax_ap.set_title("(b) AUPRC")
    ax_ap.set_ylabel("AUPRC")
    ax_ap.set_xlabel("T")
    ax_ap.set_ylim(0.0, 1.0)

    # legend
    handles, labels = ax_auc.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

def plot_alpha(csv_path):
    df = pd.read_csv(csv_path)

    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True)

    ax_auc, ax_ap = axes

    for ds in datasets:
        if ds not in DATASET_STYLE:
            continue

        sub = df[df["dataset"] == ds].sort_values("alpha")

        style = DATASET_STYLE[ds]

        ax_auc.plot(
            sub["alpha"],
            sub["auc"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"]
        )

        ax_ap.plot(
            sub["alpha"],
            sub["ap"],
            marker=style["marker"],
            linewidth=2,
            markersize=6,
            label=style["label"]
        )

    # auc subplot
    ax_auc.set_title("(a) AUROC")
    ax_auc.set_ylabel("AUROC")
    ax_auc.set_xlabel("Alpha")
    ax_auc.set_ylim(0.5, 1.0)

    # ap subplot
    ax_ap.set_title("(b) AUPRC")
    ax_ap.set_ylabel("AUPRC")
    ax_ap.set_xlabel("Alpha")
    ax_ap.set_ylim(0.0, 1.0)

    # legend
    handles, labels = ax_auc.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

plot_T(str(SCRIPT_DIR / "T_results.csv"))
plot_alpha(str(SCRIPT_DIR / "ALPHA_results.csv"))