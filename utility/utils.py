#!/usr/bin/env python
'''
Author Met
07/11/17 - 11:00
'''
from __future__ import division, print_function
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
try:
    from skopt.plots import LogLocator, partial_dependence
    from skopt.plots import _format_scatter_plot_axes
except:
    print('Import module missing: pip install skopt')
import logging
log = logging.getLogger(__name__)

def print_time_pass(title_to_display, seconds):
    ''' Display time in format (hh:mm:ss) '''

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    log.info(title_to_display + "%d:%02d:%02d" % (h, m, s))


def plot_evaluations(result, bins=20, path=None):
    """Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `bins` [int, bins=20]:
        Number of bins to use for histograms on the diagonal.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    order = range(samples.shape[0])
    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(2 * space.n_dims, 2 * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                ax[i, i].hist(samples[:, j], bins=bins,
                              range=space.dimensions[j].bounds)

            # lower triangle
            elif i > j:
                ax[i, j].scatter(samples[:, j], samples[:, i], c=order,
                                 s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)
    if path:
        fig.savefig(path, format="pdf")
    return _format_scatter_plot_axes(ax, space, "Number of samples")


def plot_objective(result, levels=10, n_points=40, n_samples=250, size=2,
                   zscale='linear', path=None):
    """Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`

    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.

    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=2]
        Height (in inches) of each facet.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space
    samples = np.asarray(result.x_iters)
    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(size * space.n_dims, size * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi = partial_dependence(space, result.models[-1], i,
                                            j=None,
                                            sample_points=rvs_transformed,
                                            n_points=n_points)

                ax[i, i].plot(xi, yi)
                ax[i, i].axvline(result.x[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(space, result.models[-1],
                                                i, j,
                                                rvs_transformed, n_points)
                ax[i, j].contourf(xi, yi, zi, levels,
                                  locator=locator, cmap='viridis_r')
                ax[i, j].scatter(samples[:, j], samples[:, i],
                                 c='k', s=10, lw=0.)
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    if path:
        fig.savefig(path, format="pdf")

    return _format_scatter_plot_axes(ax, space, "Partial dependence")


def plot_confusion_matrix(cm, fname):
    labels = ['0', '1']
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.savefig(fname + '.png')


def plot_correlation_test(corr_testing, cca, save_flag, pathname, my_title):
    # Plot correlation between U and V in testing
    data_to_plot_0 = corr_testing[:, 0]
    data_to_plot_1 = corr_testing[:, 1]
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    width = 0.35  # the width of the bars
    ax.bar(np.arange(len(data_to_plot_0)), data_to_plot_0, width, color="steelblue")

    ax.set_ylabel("Pearson correlation")
    ax.set_xticks(np.arange(cca.numCC))
    ax.set_xticklabels(["Comp %d" % i for i in range(cca.numCC)], rotation=45)
    ax.set_title((my_title + "correlation U,V. With p-values."))
    ax.set_ylim([0, 1])

    plt.plot((0, 10), (data_to_plot_0.mean(), data_to_plot_0.mean()), 'r--', lw=3)
    ax.text(8, 0.9, 'Mean: '+str(data_to_plot_0.mean())[:5], style='italic',
        bbox={'facecolor':'red', 'alpha':0.7, 'pad':2})

    # Now make some labels
    rects = ax.patches
    labels = ["%s" % str('%.4f'%data_to_plot_1[i]) for i in xrange(len(data_to_plot_1))]

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 0.05, label, ha='center', va='bottom')

    #plt.show()
    if (save_flag):
        fig.savefig(pathname, format='pdf')
