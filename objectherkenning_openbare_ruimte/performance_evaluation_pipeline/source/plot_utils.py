import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (
    ObjectClass,
)


def _extract_plot_df(
    results_df: pd.DataFrame, split: str, target_class: ObjectClass
) -> pd.DataFrame:
    plot_df = results_df[
        (results_df["Size"] == "all")
        & (results_df["Split"] == split)
        & (results_df["Object Class"] == target_class.name)
    ].set_index("Conf")

    return plot_df


def save_pr_curve(
    results_df: pd.DataFrame,
    dataset: str,
    split: str,
    target_class: ObjectClass,
    model_name: str,
    result_type: str,
    output_dir: str = "",
    filename: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class=target_class
    )

    ax = plot_df[["Precision", "Recall"]].plot(
        kind="line",
        title=f"{result_type.upper()}\nDataset: {dataset}_{split}, Model: {model_name}, Object: {target_class.name}",
        xlabel="Confidence threshold",
        xticks=np.arange(0.1, 1, 0.1),
        ylim=[0.39, 1.01],
    )
    fig = ax.get_figure()

    if not show_plot:
        plt.close()

    if not filename:
        filename = f"{result_type}_{split}_{target_class.name}_pr-curve.png"
    fig.savefig(os.path.join(output_dir, filename))


def save_fscore_curve(
    results_df: pd.DataFrame,
    dataset: str,
    split: str,
    target_class: ObjectClass,
    model_name: str,
    result_type: str,
    output_dir: str = "",
    filename: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class=target_class
    )
    ax = plot_df[["F1", "F0.5", "F2"]].plot(
        kind="line",
        title=f"{result_type.upper()}\nDataset: {dataset}_{split}, Model: {model_name}, Object: {target_class.name}",
        xlabel="Confidence threshold",
        xticks=np.arange(0.1, 1, 0.1),
        ylim=[0.39, 1.01],
    )
    fig = ax.get_figure()

    if not show_plot:
        plt.close()

    if not filename:
        filename = f"{result_type}_{split}_{target_class.name}_f-score.png"
    fig.savefig(os.path.join(output_dir, filename))
