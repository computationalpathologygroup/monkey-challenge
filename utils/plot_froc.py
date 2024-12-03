"""
FROC curve plotting function that can be used with metrics.json file that is downloaded from Grand-challenge platform.
Adapted form https://github.com/computationalpathologygroup/evaluation-tools/blob/main/src/cpg_evaluation_tools/plotting/froc.py
"""
import os.path

import numpy as np
import seaborn as sns
from sklearn.utils._plotting import _BinaryClassifierCurveDisplayMixin
import argparse
import json
import matplotlib.pyplot as plt


class FrocCurveDisplay(_BinaryClassifierCurveDisplayMixin):
    """FROC curve display."""

    def __init__(
        self,
        *,
        fps,
        tpr,
        estimator_name=None,
        froc_score=None,
        pos_label=None,
    ):
        """
        Initialize FrocCurveDisplay.

        Parameters
        ----------
        fps : ndarray
            False positive rate.

        tpr : ndarray
            True positive rate.

        estimator_name : str, default=None
            Name of estimator. If None, the estimator name is not shown.

        froc_score : float, default=None
            FROC score. If None, the froc score is not shown.

        pos_label : int, float, bool or str, default=None
            The label of the positive class. When `pos_label=None`, if `y_true`
            is in {-1, 1} or {0, 1}, `pos_label` is set to 1, otherwise an
            error will be raised.

        Attributes
        ----------
        line_ : matplotlib Artist
            ROC Curve.

        ax_ : matplotlib Axes
            Axes with ROC Curve.

        figure_ : matplotlib Figure
            Figure containing the curve.

        """
        self.estimator_name = estimator_name
        self.fps = fps
        self.tpr = tpr
        self.froc_score = froc_score
        self.pos_label = pos_label

    def plot(
        self,
        ax=None,
        *,
        name=None,
        xticks=None,
        legend_loc="best",
        **kwargs,
    ):
        """
        Plot visualization.

        Extra keyword arguments will be passed to matplotlib's ``plot``.

        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.
        xticks : np.ndarray | list, default=None
            List of xticks to be displayed. If `None`, xticks are computed
            automatically. May fail if FFPI is too large.
        legend_loc : str, default='best'
            Location of legend. See matplotlib legend documentation for valid
            locations.

        **kwargs : dict
            Keyword arguments to be passed to matplotlib's `plot`.

        Returns
        -------
        display : :class:`~cpg_evaluation_tools.plotting.FrocCurveDisplay`
            Object that stores computed values.

        """
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)

        line_kwargs = {}
        if self.froc_score is not None and name is not None:
            line_kwargs["label"] = f"{name} (FROC score = {self.froc_score:0.2f})"
        elif self.froc_score is not None:
            line_kwargs["label"] = f"FROC score = {self.froc_score:0.2f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        (self.line_,) = self.ax_.plot(self.fps, self.tpr, **line_kwargs)
        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )

        xlabel = "False Positive per mm2" + info_pos_label
        ylabel = "True Positives Rate (Sensitivity)" + info_pos_label
        if xticks is None:
            xtick_step = 1
            # xticks = list(range(0, int(self.fps.max() + 1)))
            xticks = list(range(0, 301))
            other_possible_xticks_steps = iter(
                [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            )
            while len(xticks) > 7:
                xtick_step = next(other_possible_xticks_steps)
                xticks = list(
                    range(0, max(xticks)+ xtick_step, xtick_step)
                    # range(0, int(self.fps.max() + xtick_step), xtick_step)
                )
        self.ax_.set(
            xlabel=xlabel,
            xticks=xticks,
            xlim=(-0.01 * max(xticks), max(xticks) + 0.01 * max(xticks)),
            ylabel=ylabel,
            ylim=(-0.01, 1.01),
            aspect=max(xticks),
            # aspect=self.fps.max(),
        )

        if "label" in line_kwargs:
            self.ax_.legend(loc=legend_loc)

        return self


def plot_froc(
    fps: np.ndarray,
    total_sensitivity: np.ndarray,
    froc_score: float | None = None,
    name: str | None = None,
    pos_label: str | None = None,
    froc_thresholds: list[float] | None = None,
    **kwargs,
) -> FrocCurveDisplay:
    """
    Plot FROC curve from predictions.

    Parameters
    ----------
    fps : ndarray
        False positives per mm.
    total_sensitivity : ndarray
        True positive rate.
    froc_score : float, default=None
        FROC score.
    name : str, default=None
        Name of estimator.
    pos_label : int, float, bool or str, default=None
        The label of the positive class. Only for display.
    **kwargs : dict
        Keyword arguments to be passed to `FrocCurveDisplay.plot`.

    """
    marker = kwargs.pop("marker", "")
    disp = FrocCurveDisplay(
        fps=fps,
        tpr=total_sensitivity,
        estimator_name=name,
        froc_score=froc_score,
        pos_label=pos_label,
    )
    disp = disp.plot(marker=marker, **kwargs)
    sns.despine(ax=disp.ax_, trim=True)
    # Add vertical lines at eval_thresholds
    if froc_thresholds is not None:
        for threshold in froc_thresholds:
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.7)

    return disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot FROC curve")
    parser.add_argument("input_path", type=str, help="Path to the file or folder with metrics json files"
                                                     "from Grand-challenge.")
    parser.add_argument("output_path", type=str, help="Path to the output folder to save the FROC curve plot.")
    parser.add_argument("--per-file", type=bool, default=False, help="If set, the per slide FROC curves will also be plotted.")
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    plot_per_file = args.per_file
    eval_thresholds = (10, 20, 50, 100, 200, 300)
    # input_path = 'example_files/metrics_gc.json'
    # output_path = 'example_files/'
    # plot_per_file = False

    with open(input_path, 'r') as f:
        metric_dict = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cell_type, metrics in metric_dict['aggregates'].items():
        # Plot the FROC curve for each cell type on the same axis
        disp = plot_froc(
            fps=np.array(metrics['fp_per_mm2_aggr']),
            total_sensitivity=np.array(metrics['sensitivity_aggr']),
            name=cell_type,
            froc_thresholds=eval_thresholds,
            ax=ax
        )

    # Customize the shared plot
    ax.set_title('Aggregated FROC curves')
    ax.legend(loc='lower right')
    plt.tight_layout()

    # Show and save the figure
    plt.show()
    plt.savefig(os.path.join(output_path, 'froc_curves_aggregated.png'))

    if plot_per_file:
        for file_id, file_metrics in metric_dict['per_slide'].items():
            # plot the froc curve
            fig, ax = plt.subplots(figsize=(10, 8))
            for cell_type, metrics in metric_dict['aggregates'].items():
                # Plot the FROC curve for each cell type on the same axis
                disp = plot_froc(
                    fps=np.array(metrics['fp_per_mm2_aggr']),
                    total_sensitivity=np.array(metrics['sensitivity_aggr']),
                    name=cell_type,
                    froc_thresholds=eval_thresholds,
                    ax=ax
                )

            # Customize the shared plot
            ax.set_title(f'FROC Curves for image {file_id}')
            ax.legend(loc='lower right')
            plt.tight_layout()

            # Show and save the figure
            plt.show()
            plt.savefig(os.path.join(output_path, f'froc_curves_{file_id}.png'))