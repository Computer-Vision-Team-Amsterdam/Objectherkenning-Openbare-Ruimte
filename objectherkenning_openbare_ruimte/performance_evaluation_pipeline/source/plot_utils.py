import os
from typing import Optional, Tuple

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


def _save_pr_f_curve(
    result_df: pd.DataFrame, title: str, output_path: str, show_plot: bool = False
):
    ax = result_df.plot(
        kind="line",
        title=title,
        xlabel="Confidence threshold",
        xticks=list(np.arange(0.1, 1, 0.1)),
        xlim=[0, 1],
        ylim=[0.39, 1.01],
    ).legend(loc="lower left")
    fig = ax.get_figure()

    if not show_plot:
        plt.close()

    fig.savefig(output_path)


def _generate_plot_title_and_filename(
    result_type: str,
    plot_type: str,
    dataset: str,
    split: str,
    model_name: str,
    target_class: str,
    filename: Optional[str] = None,
) -> Tuple[str, str]:
    title = f"{result_type.upper()}\nDataset: {dataset}_{split}, Model: {model_name}, Object: {target_class}"
    if not filename:
        filename = f"{'-'.join(result_type.split())}_{split}_{target_class.replace('_', '-')}_{plot_type}.png"
    return title, filename


def save_pr_curve(
    results_df: pd.DataFrame,
    split: str,
    target_class: ObjectClass,
    model_name: str,
    result_type: str,
    dataset: str = "",
    output_dir: str = "",
    filename: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """
    Plot and save the precision and recall curve for a particular split and target_class.
    """
    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class=target_class
    )[["Precision", "Recall"]]

    (title, filename) = _generate_plot_title_and_filename(
        result_type=result_type,
        plot_type="pr-curve",
        dataset=dataset,
        split=split,
        model_name=model_name,
        target_class=target_class.name,
        filename=filename,
    )

    _save_pr_f_curve(
        result_df=plot_df,
        title=title,
        output_path=os.path.join(output_dir, filename),
        show_plot=show_plot,
    )


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
    """
    Plot and save the F-score curve for a particular split and target_class.
    """
    plot_df = _extract_plot_df(
        results_df=results_df, split=split, target_class=target_class
    )[["F1", "F0.5", "F2"]]

    (title, filename) = _generate_plot_title_and_filename(
        result_type=result_type,
        plot_type="f-score",
        dataset=dataset,
        split=split,
        model_name=model_name,
        target_class=target_class.name,
        filename=filename,
    )

    _save_pr_f_curve(
        result_df=plot_df,
        title=title,
        output_path=os.path.join(output_dir, filename),
        show_plot=show_plot,
    )
