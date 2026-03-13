import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from scipy.stats import norm  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore

from metrics import score_decomposition


def prepare_corp_rel_diagram(
    y_true, y_prob, bands=None, m=1000, asymptotic=None, confidence=0.90
):
    """
    Prepare data for the CORP reliability diagram.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.
    bands : str, optional
        Possible values are 'consistency' or 'confidence'. If 'constistency', then consitency bands are computed, which assume that the forecast is calibrated, so the bands are positioned around the diagonal. If 'confidence', then confidence bands are computed which cluster around the CORP estimate and follow the frequentist conficence interval. If None, no bands are computed.
    m : int, optional
        Number of points used in resampling for the bands. Default is 100.
    asymptotic : bool, optional
        If True, use asymptotic theory to compute the bands. If False, use resampling. If None, it is determined based on the size of the data.
    confidence : float, optional
        Confidence level for the confidence bands. Default is 0.90.

    Returns
    -------
    dict
        A dictionary containing:
        - 'y_prob': Sorted predicted probabilities.
        - 'y_cal': Corresponding calibrated probabilities.
        - 'y_true': Corresponding true labels.
        - 'consistency': Optional bands if specified.
        - 'confidence': Optional bands if specified.
    source:
    [1] T. Dimitriadis, T. Gneiting, and A. I. Jordan, “Stable reliability diagrams for probabilistic classifiers,” Proceedings of the National Academy of Sciences, vol. 118, no. 8, p. e2016191118, Feb. 2021, doi: 10.1073/pnas.2016191118.
    """
    if not isinstance(y_true, np.ndarray) or y_true.ndim != 1:
        raise ValueError("y_true must be a 1D numpy array")
    if not isinstance(y_prob, np.ndarray) or y_prob.ndim != 1:
        raise ValueError("y_prob must be a 1D numpy array")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("y_prob must be in the range [0, 1]")
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_prob)):
        raise ValueError("y_true and y_prob must not contain NaN values")
    y_prob_sort_idx = np.argsort(y_prob)
    y_prob = y_prob[y_prob_sort_idx]
    y_true = y_true[y_prob_sort_idx]
    iso_reg = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    y_cal = iso_reg.fit_transform(y_prob, y_true)

    if bands is not None:
        if bands == "consistency":
            n = len(y_true)
            k = np.unique(y_true).size
            if (len(y_true) <= 1000) or (n <= 5000 and n <= 50 * k) and not asymptotic:
                asymptotic = False
                # we use resampling to compute the consistency bands
                samples = np.random.choice(y_prob, size=(m, n), replace=True)
                draws = np.random.binomial(n=1, p=samples)
            # we rely on asymptotic theory to compute the consistency bands
            elif n >= 8 * k**2:
                asymptotic = True
                # we use the discrete asymptotic distribution
                lower_bound = np.zeros((len(y_prob),))
                upper_bound = np.zeros((len(y_prob),))
                for z_k in np.unique(y_prob):
                    idx = np.where(y_prob == z_k)[0]
                    bound = np.sqrt(
                        (y_prob[idx] * (1 - y_prob[idx]) / len(idx))
                    ) * norm.ppf((1 - confidence) / 2)
                    lower_bound[idx] = y_prob[idx] - bound
                    upper_bound[idx] = y_prob[idx] + bound

            else:
                asymptotic = True
                # For implementation see Apendix S3 of [1]. (Dimitriadis et al., 2021)
                raise NotImplementedError(
                    "Consistency bands using asymptotic theory are not implemented yet."
                )

        elif bands == "confidence":
            if asymptotic:
                # we use the asymptotic distribution to compute the confidence bands
                raise NotImplementedError(
                    "Confidence bands using asymptotic theory are not implemented yet."
                )
            else:
                # we use resampling to compute the confidence bands
                # y_cal_clean = np.clip(y_cal, 1e-12, 1 - 1e-12)
                # clean_mask = np.isfinite(y_cal_clean)
                # y_cal_clean = y_cal_clean[clean_mask]  # remove NaN or inf
                samples = np.random.choice(
                    y_cal, size=(m, len(y_true)), replace=True
                )
                samples = np.clip(samples, 1e-12, 1 - 1e-12)

                draws = np.random.binomial(n=1, p=samples)

        else:
            raise ValueError(
                f"bands must be either 'consistency' or 'confidence', got {bands}"
            )

        if not asymptotic:
            y_cal_samples = np.zeros((m, len(y_true)))
            for i in range(m):
                y_cal_samples[i] = iso_reg.fit_transform(samples[i], draws[i])
            lower_bound = np.zeros((len(y_prob),))
            upper_bound = np.zeros((len(y_prob),))

            for z_k in np.unique(y_prob if bands == "consistency" else y_cal):
                if bands == "consistency":
                    idx = np.where(y_prob == z_k)[0]
                else:
                    idx = np.where(y_cal == z_k)[0]
                samples_with_z_k = []
                for i in range(m):
                    idx_i = np.where(samples[i] == z_k)[0]
                    if len(idx_i) > 0:
                        samples_with_z_k.append(y_cal_samples[i, idx_i[0]])
                if len(samples_with_z_k) > 0:
                    lower_bound[idx] = np.percentile(
                        samples_with_z_k, (1.0 - confidence) / 2 * 100
                    )
                    upper_bound[idx] = np.percentile(
                        samples_with_z_k, (1.0 - (1.0 - confidence) / 2) * 100
                    )
                else:
                    lower_bound[idx] = 0
                    upper_bound[idx] = 1

    return {
        "y_prob": y_prob,
        "y_cal": y_cal,
        "y_true": y_true,
        "consistency": (
            {"lower": np.clip(lower_bound, 0, 1), "upper": np.clip(upper_bound, 0, 1)}
            if bands == "consistency"
            else None
        ),
        "confidence": (
            {"lower": np.clip(lower_bound, 0, 1), "upper": np.clip(upper_bound, 0, 1)}
            if bands == "confidence"
            else None
        ),
    }


def corp_rel_diagram(
    y_true,
    y_prob,
    ax=None,
    title="",
    bands="confidence",
    confidence=0.90,
    plot_prob_hist=True,
    plot_cal_hist=True,
    score=True,
    color="red",
    label=None
):
    """
    Plot/compute the CORP reliability diagram, with optional histograms of predicted probabilities and calibrated probabilities.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes will be created.
    title : str, optional
        Title of the plot.
    bands : str, optional
        Possible values are 'consistency' or 'confidence'. If 'consistency', then consistency bands are computed, which assume that the forecast is calibrated, so the bands are positioned around the diagonal. If 'confidence', then confidence bands are computed which cluster around the CORP estimate and follow the frequentist confidence interval. If None, no bands are computed.
    confidence : float, optional
        Confidence level for the confidence bands. Default is 0.90.
    plot_prob_hist : bool, optional
        If True, plot histogram of predicted probabilities on the x-axis.
    plot_cal_hist : bool, optional
        If True, plot histogram of calibrated probabilities (y_cal) on the y-axis.
    score : bool, optional
        If True, compute the score decomposition for the reliability diagram.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the reliability diagram.

    source:
    [1] T. Dimitriadis, T. Gneiting, and A. I. Jordan, “Stable reliability diagrams for probabilistic classifiers,” Proceedings of the National Academy of Sciences, vol. 118, no. 8, p. e2016191118, Feb. 2021, doi: 10.1073/pnas.2016191118.
    """

    diagram = prepare_corp_rel_diagram(
        y_true, y_prob, bands=bands, confidence=confidence
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = None

    # Optionally add axes for histograms
    if plot_prob_hist or plot_cal_hist:
        divider = make_axes_locatable(ax)
        ax_histx = ax_histy = None
        if plot_prob_hist:
            ax_histx = divider.append_axes("top", 1.0, pad=0.1, sharex=ax)
            plt.setp(ax_histx.get_xticklabels(), visible=False)
            # Remove all spines except bottom (x-axis)
            for spine in ["left", "right", "top"]:
                ax_histx.spines[spine].set_visible(False)
        if plot_cal_hist:
            ax_histy = divider.append_axes("right", 1.0, pad=0.1, sharey=ax)
            plt.setp(ax_histy.get_yticklabels(), visible=False)
            # Remove all spines except left (y-axis)
            for spine in ["right", "top", "bottom"]:
                ax_histy.spines[spine].set_visible(False)
        bins = len(y_prob) // 10 if len(y_prob) > 10 else 5

    # Main reliability diagram
    ax.plot(diagram["y_prob"], diagram["y_cal"], marker="o", linestyle="-", color=color, label=label)
    if diagram["consistency"] is not None:
        ax.fill_between(
            diagram["y_prob"],
            diagram["consistency"]["lower"],
            diagram["consistency"]["upper"],
            color="blue",
            alpha=0.2,
            label="Consistency bands",
        )
    elif diagram["confidence"] is not None:
        ax.fill_between(
            diagram["y_prob"],
            diagram["confidence"]["lower"],
            diagram["confidence"]["upper"],
            color="blue",
            alpha=0.2,
            label="Confidence bands",
        )
    ax.set_xlabel("Forecast value", fontweight="bold")
    ax.set_ylabel("CEP", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    # ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Plot histogram of predicted probabilities on x-axis
    if plot_prob_hist:
        ax_histx.hist(
            diagram["y_prob"],
            bins=bins,
            range=(0, 1),
            color="gray",
            alpha=0.5,
            edgecolor="black",
        )
        ax_histx.set_yticks([])
        ax_histx.set_xlim(0, 1)
        ax_histx.grid(False)
        ax.axvline(
            np.mean(diagram["y_prob"]),
            color="gray",
            linestyle="-",
            alpha=0.5,
            linewidth=3,
        )

    # Plot histogram of calibrated probabilities on y-axis
    if plot_cal_hist:
        ax_histy.hist(
            diagram["y_cal"],
            bins=bins,
            range=(0, 1),
            orientation="horizontal",
            color="red",
            alpha=0.5,
            edgecolor="black",
        )
        ax_histy.set_xticks([])
        ax_histy.set_ylim(0, 1)
        ax_histy.grid(False)
        ax.axhline(
            np.mean(diagram["y_cal"]),
            color="red",
            linestyle="-",
            alpha=0.5,
            linewidth=3,
        )

    if score:
        scores = score_decomposition(
            diagram["y_true"], diagram["y_prob"], diagram["y_cal"], score="brier"
        )
        # Show MCB, DSC, UNC with DSC in green
        ax.text(
            0.05,
            0.95,
            f"MCB: {scores['mcb']:.4f}\n",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="none", edgecolor="none"),
            color="black",
        )
        ax.text(
            0.05,
            0.95 - 0.04,
            f"DSC: {scores['dsc']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="none", edgecolor="none"),
            color="green",
        )
        ax.text(
            0.05,
            0.95 - 0.08,
            f"UNC: {scores['unc']:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="none", edgecolor="none"),
            color="black",
        )
    if plot_prob_hist:
        ax_histx.set_title(title)
    else:
        ax.set_title(title)
    if fig is not None:
        plt.show()
    return ax
