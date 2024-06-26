################### HEADER ###################
# Author: SKLEARN Library                    #
# Date: 2024-04-04                           #
# ############################################
# The following function was provided by     #
# the sklearn library on helper functions    #
# for studying the Gaussian Process          #
# Regressor from their library.              #
##############################################


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style='darkgrid')

def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        sns.lineplot(
            x=x,
            y=single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",ax=ax,
        )
    sns.lineplot(x=x, y=y_mean, color="black", label="Mean",ax=ax)
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev."
    )
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('y')