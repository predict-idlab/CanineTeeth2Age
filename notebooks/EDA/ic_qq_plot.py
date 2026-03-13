from lifelines import *
import numpy as np
from scipy import stats as stats


def get_distribution_name_of_lifelines_model(model):
    return model._class_name.replace("Fitter", "").replace("AFT", "").lower()


def create_scipy_stats_model_from_lifelines_model(model):
    from lifelines.fitters import KnownModelParametricUnivariateFitter

    is_univariate_model = isinstance(model, KnownModelParametricUnivariateFitter)
    dist = get_distribution_name_of_lifelines_model(model)

    if not (is_univariate_model):
        raise TypeError(
            "Cannot use qq-plot with this model. See notes here: https://lifelines.readthedocs.io/en/latest/Examples.html?highlight=qq_plot#selecting-a-parametric-model-using-qq-plots"
        )
    if dist == "weibull":
        scipy_dist = "weibull_min"
        sparams = (model.rho_, 0, model.lambda_)
    elif dist == "lognormal":
        scipy_dist = "lognorm"
        sparams = (model.sigma_, 0, np.exp(model.mu_))
    elif dist == "loglogistic":
        scipy_dist = "fisk"
        sparams = (model.beta_, 0, model.alpha_)
    elif dist == "exponential":
        scipy_dist = "expon"
        sparams = (0, model.lambda_)
    else:
        raise NotImplementedError("Distribution not implemented in SciPy")
    return getattr(stats, scipy_dist)(*sparams)


def qq_plot(model, ax=None, scatter_color="k", **plot_kwargs):
    """
    Produces a quantile-quantile plot of the empirical CDF against
    the fitted parametric CDF. Large deviances away from the line y=x
    can invalidate a model (though we expect some natural deviance in the tails).

    Parameters
    -----------
    model: obj
        A fitted lifelines univariate parametric model, like ``WeibullFitter``
    plot_kwargs:
        kwargs for the plot.

    Returns
    --------
    ax:
        The axes which was used.

    Examples
    ---------
    .. code:: python

        from lifelines import *
        from lifelines.plotting import qq_plot
        from lifelines.datasets import load_rossi
        df = load_rossi()
        wf = WeibullFitter().fit(df['week'], df['arrest'])
        qq_plot(wf)

    Notes
    ------
    The interval censoring case uses the mean between the upper and lower bounds.

    """
    from lifelines.utils import qth_survival_times
    from lifelines import KaplanMeierFitter
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    dist = get_distribution_name_of_lifelines_model(model)
    dist_object = create_scipy_stats_model_from_lifelines_model(model)

    COL_EMP = "empirical quantiles"
    COL_THEO = "fitted %s quantiles" % dist

    kmf = KaplanMeierFitter()
    kmf.fit_interval_censoring(model.lower_bound, model.upper_bound, label=COL_EMP)
    sf, cdf = (
        kmf.survival_function_.mean(1),
        kmf.cumulative_density_[COL_EMP + "_lower"],
    )

    q = np.unique(cdf.values)

    quantiles = qth_survival_times(1 - q, sf)
    quantiles[COL_THEO] = dist_object.ppf(q)
    quantiles = quantiles.replace([-np.inf, 0, np.inf], np.nan).dropna()

    max_, min_ = quantiles[0].max(), quantiles[0].min()

    quantiles.plot.scatter(
        COL_THEO, 0, c="none", edgecolor=scatter_color, lw=0.5, ax=ax
    )
    ax.plot([min_, max_], [min_, max_], c="k", ls=":", lw=1.0)
    ax.set_ylabel(COL_EMP)
    ax.set_ylim(min_, max_)
    ax.set_xlim(min_, max_)

    return ax
