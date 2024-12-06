# Import global packages
import os
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.ticker as tlt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import datetime

def nice_nrow_ncol(nfig, increasing=True, n=20, maxratio=3):
    """Proposes ideal number of columns and rows for the total number of figures.
    | 1 | 2 |...|ncol|
    | 2 |   |   |    |
    |...|   |   |    |
    |nrow|  |   |    |

    Total number of cells should be higher than the number of figures.

    Args:
        nfig: Total number of figures to be covered
        increasing: True --> return the smaller value first
                    False--> return the higher value first
        n: number of trials
    Returns:
        nrow: [int]
        ncol: [int]
    """
    sq = np.sqrt(nfig)
    sqlow, squpp = int(np.floor(sq)), int(np.ceil(sq))
    upper = np.empty([], dtype=int)
    lower = np.empty([], dtype=int)
    remainder = np.array(999999, dtype=int)
    x = squpp
    y = squpp
    i = 0
    while (i < n) and (y >= 1) and (x/y <= maxratio):
        while (y*x >= nfig):
            y -= 1
        lower = np.append(lower, y+1)
        upper = np.append(upper, x)
        remainder = np.append(remainder, (y+1)*x-nfig)
        x += 1
        i += 1

    # find the first pair [x, y] minimizing the remainder
    i = remainder.argmin()
    if increasing:
        nrow, ncol = lower[i], upper[i]
    else:
        nrow, ncol = upper[i], lower[i]

    return nrow, ncol

def plot_several_bar_plots(figures, file, nrows = 1, ncols=1, width=0.8, size=(15, 15)):
    """Plot a dictionary of barplots. Midpoints and heights are expected.

    Args:
        figures: <title, figure> dictionary containing midpoints and heights
        file: where to save the figure [str]
        ncols: number of columns of subplots wanted in the display [integer]
        nrows: number of rows of subplots wanted in the figure [integer]
        width: width of the plotted bars [float]
    """
    if (ncols == 1) and (nrows == 1):
        title = 'Topic 0'
        mids, heights = figures[title]
        plt.bar(mids, height=heights, width=width)
        plt.title(title)
    else:
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        ylim = 0
        for ind,title in enumerate(figures):
            mids, heights = figures[title]
            ylim = np.maximum(ylim, np.max(heights))
        for ind,title in enumerate(figures):
            mids, heights = figures[title]
            axeslist.ravel()[ind].bar(mids, height=heights, width=width)
            axeslist.ravel()[ind].set_ylim(0, ylim)
            axeslist.ravel()[ind].set_title(title)
    plt.gcf().set_size_inches(size)
    #plt.figure(figsize=(15, 15))
    plt.tight_layout()
    plt.savefig(file)
    plt.close()

def hist_values(values, grid):
    """Compute the counts of values in bins given by the grid.

    Args:
        values: vector of values to create a histogram
        grid: grid of bounds for the bins

    Returns:
        mids: midpoints of the bins
        counts: number of topics-specific locations in the bins
    """
    lower = grid[:-1]
    upper = grid[1:]
    count = np.full(lower.shape, 0.0)
    for i, x in enumerate(lower):
        count[i] = tfm.reduce_sum(
            tf.dtypes.cast(values > lower[i], tf.int32) * tf.dtypes.cast(values <= upper[i], tf.int32))
    return ((lower + upper) / 2, count)

def hist_log_theta_means(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(15,15)):
    """ Histograms of log theta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    grid = np.linspace(grid_from, grid_to, nbins + 1)
    Eqmeanlog = model.get_Eqmean(model.theta_varfam, log=True)
    figures = {'Topic '+str(k+1): hist_values(Eqmeanlog[:, k], grid) for k in range(model.num_topics)}
    nrow, ncol = nice_nrow_ncol(model.num_topics)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def hist_max_theta_over_k(model, path, nbins=20, size=(15,15)):
    """ Histograms of log theta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        nbins: number of bins
    """
    Eqmean = model.get_Eqmean(model.theta_varfam, log=True)
    # Scale wrt the sum of all thetas for the document OR sum to scale it back
    Eqmeansum = tfm.reduce_mean(Eqmean, axis=1)
    Eqmeanscaled = Eqmean / Eqmeansum[:, tf.newaxis]
    # the maximal value of theta
    Eqmean_max_over_k = tfm.reduce_max(Eqmeanscaled, axis=1)
    # which topic was maximal
    Eqmean_which_k_max = tf.argmax(Eqmeanscaled, axis=1)
    fig, axs = plt.subplots(1, 2, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(Eqmean_max_over_k, bins=nbins)
    axs[0].set_title('Maximal scaled theta')
    axs[1].hist(Eqmean_which_k_max, bins=model.num_topics)
    axs[1].set_title('Topic of maximal theta')
    plt.gcf().set_size_inches(size)
    plt.savefig(path)
    plt.close()

def hist_log_beta_means(model, path, grid_from=-2.5, grid_to=2.5, nbins=20, size=(15,15)):
    """ Histograms of log beta means for each topic separately.

    Args:
        model: TBIP class object
        path: directory + file_name.png
        grid_from: the lowest value of the grid
        grid_to: the highest value of the grid
        nbins: number of bins

    """
    grid = np.linspace(grid_from, grid_to, nbins + 1)
    Eqmeanlog = model.get_Eqmean(model.beta_varfam, log=True)
    figures = {'Topic '+str(k+1): hist_values(Eqmeanlog[k, :], grid) for k in range(model.num_topics)}
    nrow, ncol = nice_nrow_ncol(model.num_topics)
    plot_several_bar_plots(figures, path, nrow, ncol, grid[1] - grid[0], size)

def barplot_ordered_labels_top(x, label, path, size=(15,15), do_label=True):
    xsort = -np.sort(-x)
    xord = np.argsort(-x)
    labup = 0.02*np.max(x)
    plt.figure(figsize=size)
    #plt.axis('off')
    plt.xticks([])
    plt.bar(range(x.shape[0]), height=xsort, align='edge')
    if do_label:
        for i in range(x.shape[0]):
            plt.text(i, xsort[i]+labup, label+str(xord[i]))
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)
    plt.close()

def get_Ekvt(model, logscale=True):
    if logscale:
        Ekvt = model.ar_kv_varfam.location
    else:
        # covariance = scale_tril @ scale_tril.T      is of shape: [num_topics, num_words, num_times, num_times]
        ar_kv_cov, ar_kv_var = model.get_ar_cov(model.ar_kv_varfam)
        ar_kv = model.ar_kv_varfam.location + 0.5 * ar_kv_var
        Ekvt = tfm.exp(ar_kv)
    return Ekvt


def plot_topics_in_time(model, path, vocabulary, breaks, k, Ekvt, hkt, hline, nwords=0, size=(15,8)):
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]
    middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(model.num_times)]

    ymin = tfm.reduce_min(hkt)
    ymax = tfm.reduce_max(hkt)
    rng = ymax - ymin
    plt.figure(figsize=size)
    plt.step(dbreaks, hkt, where='pre')
    plt.hlines(hline, xmin=dbreaks[0], xmax=dbreaks[-1], colors='grey', linestyles='--')
    var = tfm.reduce_mean(
        tf.gather(model.ar_kv_prec_varfam.rate, k) / (tf.gather(model.ar_kv_prec_varfam.shape, k) - 1.0))
    delta = tfm.reduce_mean(tf.gather(model.ar_kv_delta_varfam.distribution.mean(), k))
    ar_kv_mean = tfm.reduce_mean(tf.gather(model.ar_kv_mean_varfam.location, k, axis=0))
    plt.title('Topic ' + str(k+1) +
              ', variance = ' + str(tfm.round(1000 * var).numpy() / 1000) +
              ', delta = ' + str(tfm.round(100 * delta).numpy() / 100) +
              ', mean = ' + str(tfm.round(100 * ar_kv_mean).numpy() / 100))
    # Top words for this topic in each time period
    if nwords > 0:
        plt.ylim(ymin - (0.05 + 0.1*nwords) * rng, ymax + 0.01 * rng)
        ywmin = ymin - (0.1 + 0.1*nwords) * rng
        ywmax = ymin - 0.1 * rng
        ywords = tf.range(ywmax, ywmin, delta=(ywmin - ywmax) / nwords) # decreasing
        Evt = tf.gather(Ekvt, k, axis=0)
        # Find the top words
        words = np.array([])
        for t in range(model.num_times):
            Ev_val, Ev_ind = tfm.top_k(tf.gather(Evt, t, axis=-1), k=nwords)
            words = np.append(words, Ev_ind)
            if t % 2:
                addt = 0.0
            else:
                addt = -0.5 * (ywmin - ywmax) / nwords
            for iy in range(nwords):
                plt.text(middbreaks[t], ywords[iy]+addt, vocabulary[Ev_ind[iy]], horizontalalignment='center')
    else:
        plt.ylim(ymin - 0.01 * rng, ymax + 0.01 * rng)
        words = np.array([])
        # No need to find top words
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return np.unique(words).astype(np.int32)

def plot_words_in_time(model, path, title, vocabulary, breaks, k, words, ar_v, size=(15,8), how="ar"):
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]

    if how == "ar":
        ar_v = tf.gather(tf.gather(model.ar_kv_varfam.location, k, axis=0), words, axis=0)
    hvt = tf.concat([tf.gather(ar_v, 0, axis=1)[:, tf.newaxis], ar_v], axis=1)

    legend = vocabulary[words]
    # vars = tf.gather(model.ar_v_prec_varfam.rate, words) / (tf.gather(model.ar_v_prec_varfam.shape, words) - 1.0)
    # deltas = tf.gather(model.ar_v_delta_varfam.distribution.mean(), words)
    # legend = [vocabulary[words[v]] + ', var=' + str(tfm.round(vars[v]*1000).numpy()/1000) + ', delta=' + str(tfm.round(deltas[v]*100).numpy()/100) for v in range(len(words))]

    ymin = tfm.reduce_min(hvt)
    ymax = tfm.reduce_max(hvt)
    rng = ymax - ymin
    plt.figure(figsize=size)
    for v in range(len(words)):
        plt.step(dbreaks, tf.gather(hvt, v, axis=0), where='pre')
    #if how != "ar":
    #    plt.yscale("log")
    plt.legend(legend, loc='lower center', ncols=5)
    plt.title(title)
    plt.ylim(ymin - 0.1 * rng, ymax + 0.01 * rng)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_word_in_time(path, hkv_t, word, breaks, k, size=(15,8)):
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]
    ymin = tfm.reduce_min(hkv_t)
    ymax = tfm.reduce_max(hkv_t)
    rng = ymax - ymin
    plt.figure(figsize=size)
    plt.step(dbreaks, tf.concat([tf.cast(tf.zeros((1)), "float32"), hkv_t], axis=0), where='pre')
    plt.title('Evolution of ' + word + ' for topic ' + str(k+1))
    plt.ylim(ymin - 0.1 * rng, ymax + 0.01 * rng)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def spineplot(x, xlabels, ylabels, title, path, sort='unsorted', size=(10, 10)):
    xsum = tfm.reduce_sum(x, axis=0)
    xavg = tfm.reduce_mean(x, axis=1)

    weights = {}
    if sort == 'unsorted':
        iis = range(xavg.shape[0])
    elif sort == 'by_avg_rate':
        val, iis = tfm.top_k(xavg, xavg.shape[0])
    else:
        raise ValueError('Unrecognized choice of sorting for spineplot.')

    for i in iis:
        weights[ylabels[i]] = tf.gather(x, i, axis=0) / xsum
    width = 1.0

    fig, ax = plt.subplots(figsize=size)
    bottom = np.zeros(x.shape[1])
    for boolean, weight in weights.items():
        p = ax.bar(xlabels, weight, width, label=boolean, bottom=bottom)
        bottom += weight
    plt.xlim(left=-0.6, right=len(xlabels) + 1.1)
    plt.ylim(0.0, 1.005)
    plt.yticks([])
    plt.title(title)
    ax.legend(loc="upper right", reverse=True)
    # plt.show()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_avg_rates(model):
    ### Averaging over rates
    avg_tk = tf.zeros([model.num_times, model.num_topics])
    ar_kv_cov, ar_kv_var = model.get_ar_cov(model.ar_kv_varfam)
    ar_kv = tfm.exp(model.ar_kv_varfam.location + 0.5 * ar_kv_var)
    for t in range(model.num_times):
        t_indices = tf.where(model.all_time_indices == t)[:, 0]
        theta = model.get_gamma_distribution_Eqmean_subset(model.theta_varfam, t_indices, log=False)
        ar_kv_t = tf.gather(ar_kv, t, axis=-1)
        avg = tfm.reduce_mean(tfm.reduce_mean(theta, axis=0)[:, tf.newaxis] * ar_kv_t, axis=1)  # [K]
        avg_tk = tf.tensor_scatter_nd_update(avg_tk, tf.constant([[t]]), avg[tf.newaxis, :])
    avg_kt = tf.transpose(avg_tk)
    return avg_kt

def get_beta_kvt(model, what="log_beta"):
    if what == "beta":
        beta_kvt = get_Ekvt(model, logscale=False)
    elif what == "log_beta":
        beta_kvt = get_Ekvt(model, logscale=True)
    elif what == "fe":
        beta_kvt = get_fe_kvt3(model, w=0.5)
    else:
        raise ValueError('Unrecognized choice for computing cosine similarities')
    return beta_kvt
def get_cosine_similarity_times(model, what="log_beta"):
    beta_kvt = get_beta_kvt(model, what)
    cosine_similarity = tf.ones([model.num_times, model.num_topics])
    for t in tf.range(1, model.num_times):
        a = tf.gather(beta_kvt, t, axis=-1)
        b = tf.gather(beta_kvt, t-1, axis=-1)
        cos_k = tfm.reduce_sum(a * b, axis=-1) / tf.norm(a, axis=-1) / tf.norm(b, axis=-1)
        cosine_similarity = tf.tensor_scatter_nd_update(cosine_similarity, [[t]], cos_k[tf.newaxis, :])

    return cosine_similarity


def get_cosine_similarity_topics(model, what="log_beta"):
    beta_kvt = get_beta_kvt(model, what)
    cosine_similarity = tf.zeros([model.num_topics, model.num_topics, model.num_times])
    for k1 in tf.range(model.num_topics):
        a = tf.gather(beta_kvt, k1, axis=0)
        for k2 in tf.range(model.num_topics):
            b = tf.gather(beta_kvt, k2, axis=0)
            cos_t = tfm.reduce_sum(a * b, axis=0) / tf.norm(a, axis=0) / tf.norm(b, axis=0)
            cosine_similarity = tf.tensor_scatter_nd_update(cosine_similarity, [[[k1, k2]]], cos_t[tf.newaxis, tf.newaxis, :])

    return cosine_similarity


def plot_similarity(similarity, title, xlabels, ylabels, xlab, ylab, path, col_rev=False, size=(9, 9)):
    plt.figure(figsize=size)
    plt.grid(False)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.xticks(tf.range(0.5, len(xlabels) * 1 + 0.5, 1), xlabels)
    plt.xlim(0, len(xlabels))
    plt.yticks(tf.range(0.5, len(ylabels) * 1 + 0.5, 1), ylabels)
    plt.ylim(0, len(ylabels))
    if col_rev:
        cmap = cm.gray_r
    else:
        cmap = cm.gray
    im = plt.pcolormesh(similarity, cmap=cmap, edgecolors='white', linewidths=1, antialiased=True)
    formatter = tlt.ScalarFormatter()
    formatter.set_powerlimits(lims=(-3, 4))
    plt.colorbar(im, format=formatter)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    # plt.show()
    plt.close()

def get_KL_dissimilarity_times(model):
    loc_kvt = model.ar_kv_varfam.location
    cov_kvtt, var_kvt = model.get_ar_cov(model.ar_kv_varfam)
    KL_dissimilarity = tf.zeros([model.num_times, model.num_topics])
    for t in tf.range(1, model.num_times):
        loc_t = tf.gather(loc_kvt, t, axis=-1)
        var_t = tf.gather(var_kvt, t, axis=-1)
        loc_t_1 = tf.gather(loc_kvt, t-1, axis=-1)
        var_t_1 = tf.gather(var_kvt, t-1, axis=-1)
        KL_k = 0.25 * tfm.reduce_sum(
            (tfm.square(var_t - var_t_1) + (var_t + var_t_1) * tfm.square(loc_t - loc_t_1)) / (var_t * var_t_1),
            axis=1
        )
        KL_dissimilarity = tf.tensor_scatter_nd_update(KL_dissimilarity, [[t]], KL_k[tf.newaxis, :])

    return KL_dissimilarity

def get_KL_dissimilarity_to_first(model):
    loc_kvt = model.ar_kv_varfam.location
    cov_kvtt, var_kvt = model.get_ar_cov(model.ar_kv_varfam)
    KL_dissimilarity = tf.zeros([model.num_times, model.num_topics])
    loc_0 = tf.gather(loc_kvt, 0, axis=-1)
    var_0 = tf.gather(var_kvt, 0, axis=-1)
    for t in tf.range(1, model.num_times):
        loc_t = tf.gather(loc_kvt, t, axis=-1)
        var_t = tf.gather(var_kvt, t, axis=-1)
        KL_k = 0.25 * tfm.reduce_sum(
            (tfm.square(var_t - var_0) + (var_t + var_0) * tfm.square(loc_t - loc_0)) / (var_t * var_0),
            axis=1
        )
        KL_dissimilarity = tf.tensor_scatter_nd_update(KL_dissimilarity, [[t]], KL_k[tf.newaxis, :])

    return KL_dissimilarity


def get_KL_dissimilarity_topics(model):
    loc_kvt = model.ar_kv_varfam.location
    cov_kvtt, var_kvt = model.get_ar_cov(model.ar_kv_varfam)
    KL_dissimilarity = tf.zeros([model.num_topics, model.num_topics, model.num_times])
    for k1 in tf.range(model.num_topics):
        loc_k1 = tf.gather(loc_kvt, k1, axis=0)
        var_k1 = tf.gather(var_kvt, k1, axis=0)
        for k2 in tf.range(model.num_topics):
            loc_k2 = tf.gather(loc_kvt, k2, axis=0)
            var_k2 = tf.gather(var_kvt, k2, axis=0)
            KL_t = 0.25 * tfm.reduce_sum(
                (tfm.square(var_k1 - var_k2) + (var_k1 + var_k2) * tfm.square(loc_k1 - loc_k2)) / (var_k1 * var_k2),
                axis=0
            )
            KL_dissimilarity = tf.tensor_scatter_nd_update(KL_dissimilarity, [[[k1, k2]]],
                                                           KL_t[tf.newaxis, tf.newaxis, :])

    return KL_dissimilarity



def get_fe_kvt(model, w=0.5):
    """Computes the frequency-exclusivity measure for each word, topic, time-period.
    The comparison set for exclusivity is the set of all topics.
    This version works with 3D objects that have high memory consumption!

    Args:
        model: A TPF model with .ar_kv_varfam.location of shape:
            [num_topics, num_words, num_times]
        w: A weight parameter between 0 and 1 for the compromise between frequency (0) and exclusivity (1).
            [float]

    Returns:
        fe_kvt: Frequency-Exclusivity measure of the same shape as the entry:
            [num_topics, num_words, num_times]
    """
    # Frequency
    # f_tkv = tfp.distributions.Binomial(10, probs=0.5).sample([5,3,10])
    beta_kvt = get_Ekvt(model, logscale=False)
    # f_tkv = tf.transpose(model.ar_kv_varfam.location, perm=[2, 0, 1])
    f_tkv = tf.transpose(beta_kvt, perm=[2, 0, 1])
    df = tfp.distributions.Empirical(f_tkv)
    f_ecdf_vtk = df.cdf(tf.transpose(f_tkv, perm=[2, 0, 1]))

    # Exclusivity
    e_tkv = f_tkv / tf.reduce_sum(f_tkv, axis=1)[:, tf.newaxis, :]
    de = tfp.distributions.Empirical(e_tkv)
    e_ecdf_vtk = de.cdf(tf.transpose(e_tkv, perm=[2, 0, 1]))

    fe_vtk = tfm.reciprocal(w/e_ecdf_vtk + (1-w)/f_ecdf_vtk)
    fe_kvt = tf.transpose(fe_vtk, perm=[2, 0, 1])

    return fe_kvt

def get_fe_kvt2(model, w=0.5):
    """Computes the frequency-exclusivity measure for each word, topic, time-period.
    The comparison set for exclusivity is the set of all topics.
    This version works with 2D objects for each time-period separately, less memory consumption.

    Args:
        model: A TPF model with .ar_kv_varfam.location of shape:
            [num_topics, num_words, num_times]
        w: A weight parameter between 0 and 1 for the compromise between frequency (0) and exclusivity (1).
            [float]

    Returns:
        fe_kvt: Frequency-Exclusivity measure of the same shape as the entry:
            [num_topics, num_words, num_times]
    """
    fe_tkv = tf.zeros([model.num_times, model.num_topics, model.num_words])
    # sample_kvt = tfp.distributions.Binomial(10, 0.5).sample([3, 10, 5])
    beta_kvt = get_Ekvt(model, logscale=False)
    for t in range(model.num_times):
        # Frequency
        # f_kv = tf.gather(sample_kvt, t, axis=-1)
        # f_kv = tf.gather(model.ar_kv_varfam.location, t, axis=-1)
        f_kv = tf.gather(beta_kvt, t, axis=-1)
        df = tfp.distributions.Empirical(f_kv)
        f_ecdf_vk = df.cdf(tf.transpose(f_kv))

        # Exclusivity
        beta_v = tf.reduce_sum(f_kv, axis=0)
        e_kv = f_kv / beta_v[tf.newaxis, :]
        de = tfp.distributions.Empirical(e_kv)
        e_ecdf_vk = de.cdf(tf.transpose(e_kv))

        fe_vk = tfm.reciprocal(w/e_ecdf_vk + (1-w)/f_ecdf_vk)
        fe_tkv = tf.tensor_scatter_nd_update(fe_tkv, tf.constant([[t]]), tf.transpose(fe_vk)[tf.newaxis, :, :])

    fe_kvt = tf.transpose(fe_tkv, perm=[1, 2, 0])
    return fe_kvt

def get_fe_kvt3(model, w=0.5):
    """Computes the frequency-exclusivity measure for each word, topic, time-period.
    The comparison set for exclusivity is the set of all topics.
    This version works with 2D objects for each time-period separately, less memory consumption.

    Args:
        model: A TPF model with .ar_kv_varfam.location of shape:
            [num_topics, num_words, num_times]
        w: A weight parameter between 0 and 1 for the compromise between frequency (0) and exclusivity (1).
            [float]

    Returns:
        fe_kvt: Frequency-Exclusivity measure of the same shape as the entry:
            [num_topics, num_words, num_times]
    """
    fe_tkv = tf.zeros([model.num_times, model.num_topics, model.num_words])
    fe_kv = tf.zeros([model.num_topics, model.num_words])
    beta_kvt = get_Ekvt(model, logscale=False)
    # sample_kvt = tfp.distributions.Binomial(10, 0.5).sample([3, 10, 5])
    for t in range(model.num_times):
        # f_kv = tf.gather(sample_kvt, t, axis=-1)
        # f_kv = tf.gather(model.ar_kv_varfam.location, t, axis=-1)
        # beta_kv = tfm.exp(f_kv)
        f_kv = tf.gather(beta_kvt, t, axis=-1)
        beta_v = tf.reduce_sum(f_kv, axis=0)
        for k in range(model.num_topics):
            f_v = tf.gather(f_kv, k, axis=0)
            # Frequency
            df = tfp.distributions.Empirical(f_v)
            f_ecdf_v = df.cdf(f_v)

            # Exclusivity
            betak_v = tf.gather(f_kv, k, axis=0)
            e_v = betak_v / beta_v[tf.newaxis, :]
            de = tfp.distributions.Empirical(e_v)
            e_ecdf_v = de.cdf(e_v)

            fe_v = tfm.reciprocal(w/e_ecdf_v + (1-w)/f_ecdf_v)
            fe_kv = tf.tensor_scatter_nd_update(fe_kv, tf.constant([[k]]), fe_v)
        fe_tkv = tf.tensor_scatter_nd_update(fe_tkv, tf.constant([[t]]), fe_kv[tf.newaxis, :, :])

    fe_kvt = tf.transpose(fe_tkv, perm=[1, 2, 0])
    return fe_kvt




def create_all_general_descriptive_figures(model, fig_dir: str, vocabulary, breaks):
    """ Create and save all the figures that describe the model output (histograms, barplots, wordclouds, ...).
    All the plots are completely general, no specific information or assumption about the dataset are required.
    Hence, these plots can be created for any dataset of speeches.

    Args:
        model: TPF class object
        fig_dir: directory where to save the figure
        vocabulary: str[num_words] all used words
        breaks: int[num_times+1] time-period breakpoints as integers of format 'YYYYMMDD'

    """
    ### Histograms:
    print("Min log theta mean: " + str(np.min(model.get_Eqmean(model.theta_varfam, log=True))))
    print("Max log theta mean: " + str(np.max(model.get_Eqmean(model.theta_varfam, log=True))))
    hist_log_theta_means(model, os.path.join(fig_dir, 'hist_log_theta_means.png'),
                         grid_from=-7, grid_to=5, size=(15, 15))

    # hist_max_theta_over_k(model, os.path.join(fig_dir, 'hist_max_theta_over_k.png'), size=(12, 5))

    print("Averaged ar_kv locations over k:")
    print(tfm.reduce_mean(model.ar_kv_varfam.location, axis=0))

    print("Averaged ar_kv locations over v:")
    print(tfm.reduce_mean(model.ar_kv_varfam.location, axis=1))

    print("Averaged ar_kv locations over kv:")
    print(tfm.reduce_mean(model.ar_kv_varfam.location, axis=[0, 1]))

    print("Averaged ar_kv_mean locations over k:")
    print(tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=0))

    print("Averaged ar_kv_mean locations over v:")
    print(tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=1))

    ### Barplot for theta rates
    if model.theta_rate_varfam.family == 'gamma':
        x = model.theta_rate_varfam.distribution.mean()
        barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_theta_rate.png'), size=(15, 5), do_label=False)

    ### Topic-specific AR(1) parameters
    # means
    x = tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=1)
    barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_k_mean.png'), size=(15, 5))
    # variances
    x = model.ar_kv_prec_varfam.rate / (model.ar_kv_prec_varfam.shape - 1.0)
    x = tfm.reduce_mean(x, axis=1)
    barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_k_prec_to_var.png'), size=(15, 5))
    # deltas
    if model.prior_choice['delta'] in ['AR', 'ART']:
        x = tfm.reduce_mean(model.ar_kv_delta_varfam.distribution.mean(), axis=1)
        barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_k_delta.png'), size=(15, 5))

    ### Word-specific AR(1) parameters
    # means
    x = tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=0)
    barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_v_mean.png'), size=(15, 5), do_label=False)
    # variances
    x = tfm.reduce_mean(model.ar_kv_prec_varfam.rate / (model.ar_kv_prec_varfam.shape - 1.0), axis=0)
    barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_v_prec_to_var.png'),
                               size=(15, 5), do_label=False)
    # deltas
    if model.prior_choice['delta'] in ['AR', 'ART']:
        x = tfm.reduce_mean(model.ar_kv_delta_varfam.distribution.mean(), axis=0)
        barplot_ordered_labels_top(x, '', os.path.join(fig_dir, 'barplot_ar_v_delta.png'),
                                   size=(15, 5), do_label=False)

    ### Topic evolution in time
    Ekvt = get_Ekvt(model, logscale=True)
    avg_kt = get_avg_rates(model)
    avg_k = tfm.reduce_mean(avg_kt, axis=-1)
    perc_kt = 100.0 * avg_kt / tfm.reduce_sum(avg_kt, axis=0)[tf.newaxis, :]
    perc_k = tfm.reduce_mean(perc_kt, axis=-1)
    for k in range(model.num_topics):
        # ar_kv based plot with mean
        hkt_mean = tf.gather(tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=1), k, axis=0)
        hkt = tf.concat([tf.reshape(hkt_mean, [1]),
                         tf.gather(tfm.reduce_mean(model.ar_kv_varfam.location, axis=1), k, axis=0)], axis=0)
        words = plot_topics_in_time(model, os.path.join(fig_dir, 'evolution_topic_with_mean_' + str(k+1) + '.png'),
                                    vocabulary, breaks, k, Ekvt, hkt, hline=hkt_mean, nwords=5,
                                    size=(10 + 0.5 * model.num_times, 8))

        # ar_k based plot without mean
        hkt -= hkt_mean
        words = plot_topics_in_time(model, os.path.join(fig_dir, 'evolution_topic_without_mean_' + str(k+1) + '.png'),
                                    vocabulary, breaks, k, Ekvt, hkt, hline=0.0, nwords=5,
                                    size=(10 + 0.5 * model.num_times, 8))

        # avg_kt based plot on original scale
        ak = tf.gather(avg_k, k, axis=0)
        akt = tf.concat([tf.reshape(ak, [1]), tf.gather(avg_kt, k, axis=0)], axis=0)
        words = plot_topics_in_time(model, os.path.join(fig_dir, 'evolution_avg_kt_' + str(k+1) + '.png'),
                                    vocabulary, breaks, k, Ekvt, akt, hline=ak, nwords=5,
                                    size=(10 + 0.5 * model.num_times, 8))

        # avg_kt based plot using percentages across
        pk = tf.gather(perc_k, k, axis=0)
        pkt = tf.concat([tf.reshape(pk, [1]), tf.gather(perc_kt, k, axis=0)], axis=0)
        words = plot_topics_in_time(model, os.path.join(fig_dir, 'evolution_perc_kt_' + str(k+1) + '.png'),
                                    vocabulary, breaks, k, Ekvt, pkt, hline=pk, nwords=5,
                                    size=(10 + 0.5 * model.num_times, 8))

        plot_words_in_time(model, os.path.join(fig_dir, 'evolution_words_for_topic_' + str(k+1) + '.png'),
                           'Evolution of top words for topic ' + str(k+1),
                           vocabulary, breaks, k, words, hkt, size=(10 + 0.5 * model.num_times, 8), how="ar")

    plt.scatter(tfm.reduce_mean(model.ar_kv_mean_varfam.location, axis=1), perc_k)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'perc_vs_ar_k_mean.png'))
    plt.close()



    # for iw in [1, 25, 50, 75, 99]:
    for iw in [50]:
        w = iw / 100
        fe_kvt = get_fe_kvt3(model, w)
        fe_kt_mean = tfm.reduce_mean(fe_kvt, axis=1)
        fe_k_mean = tfm.reduce_mean(fe_kvt, axis=[1, 2])
        for k in range(model.num_topics):
            fe_k = tf.gather(fe_k_mean, k, axis=0)
            fe_kt = tf.concat([tf.reshape(fe_k, [1]), tf.gather(fe_kt_mean, k, axis=0)], axis=0)
            words = plot_topics_in_time(model, os.path.join(fig_dir, 'fe_kvt_' + str(k) + '_w_' + str(iw) + '.png'),
                                        vocabulary, breaks, k, fe_kvt, fe_kt, hline=fe_k, nwords=5,
                                        size=(10 + 0.5 * model.num_times, 8))

    # chosen_words = ['acid rain', 'al qaeda', 'climate change', 'gun violence', 'soviet union', 'terrorist attack',
    #                 'trade center', 'web site']
    # words11 = ['acid rain', 'carbon pollution', 'clean energy', 'climate change', 'fossil fuel',
    #            'gas prices', 'global warming', 'nuclear waste', 'oil gas', 'renewable energy']
    # words11 = ['acid rain', 'clean energy', 'climate change', 'environmental protection', 'fossil fuel',
    #            'gas prices', 'global warming', 'nuclear waste', 'oil gas', 'renewable energy']
    # words11 = ['acid rain', 'carbon pollution', 'climate change', 'global warming', 'oil gas',
    #            'web site', 'exim bank', 'civil rights', 'war ii', 'wall street']
    # words11 = ['acid rain', 'carbon pollution', 'climate change', 'global warming', 'keystone xl', 'oil gas',
    #            'exim bank', 'gun violence', 'wall street', 'web site']
    #
    # numwords = []
    # for word in words11:
    #     v = tf.reshape(tf.where(vocabulary == word), [])
    #     numwords.append(v)

    numwords = [j*50 for j in range(10)]

    for iw in [50]:
        w = iw / 100
        fe_kvt = get_fe_kvt3(model, w)
        # for k in range(model.num_topics):
        #     pk = tf.gather(perc_k, k, axis=0)
        #     pkt = tf.concat([tf.reshape(pk, [1]), tf.gather(perc_kt, k, axis=0)], axis=0)
        #     words = plot_topics_in_time(model, os.path.join(fig_dir, 'perc_' + str(k+1) + '_ef_w_' + str(iw) + '.png'),
        #                                 vocabulary, breaks, k, fe_kvt, pkt, hline=pk, nwords=10,
        #                                 size=(10 + 0.5 * model.num_times, 8))

        # for k in [11]:
        for k in range(6):
            fevtk = tf.gather(fe_kvt, k, axis=0)
            fevtk_words = tf.gather(fevtk, numwords, axis=0)
            plot_words_in_time(model,
                               os.path.join(fig_dir, 'evolution_words_for_topic_'+str(k)+'_ef_w_' + str(iw) + '.png'),
                               None, #'Evolution of FREX for topic 12',
                               vocabulary, breaks, k, numwords, fevtk_words, size=(10, 6), how="fe")
            plot_words_in_time(model,
                               os.path.join(fig_dir, 'evolution_words_for_topic_'+str(k)+'_ar.png'),
                               None, #'Evolution of AR(1) sequence for topic 12',
                               vocabulary, breaks, k, numwords, fevtk_words, size=(10, 6), how="ar")

        # for word in chosen_words:
        #     v = tf.reshape(tf.where(vocabulary == word), [])
        #     fe_kt = tf.gather(fe_kvt, v, axis=1)
        #     val, topk = tfm.top_k(tfm.reduce_mean(fe_kt, axis=-1), 1)
        #     k = tf.reshape(topk, []).numpy()
        #     frex_kv_t = tf.gather(fe_kt, k, axis=0)
        #     plot_word_in_time(
        #         os.path.join(fig_dir, 'frex_word_' + word.replace(" ", "") + '_for_topic_' + str(k+1) + '.png'),
        #         frex_kv_t, word, breaks, k, size=(10 + 0.5 * model.num_times, 8)
        #     )
        #     ar_kv_t = tf.gather(tf.gather(model.ar_kv_varfam.location, k, axis=0), v, axis=0)
        #     plot_word_in_time(
        #         os.path.join(fig_dir, 'ar_word_' + word.replace(" ", "") + '_for_topic_' + str(k+1) + '.png'),
        #         ar_kv_t, word, breaks, k, size=(10 + 0.5 * model.num_times, 8)
        #     )

def stoy(s):
    return 1981 + 2 * (s - 97)
def create_all_figures_specific_to_data(model, data_name: str, fig_dir: str, vocabulary, breaks, time_periods):
    """ Create and save all the figures that describe the model output which is specific to the analysed data.
    So far, only datasets 'hein-daily', 'cze_senate' and 'pharma' are recognized.
    In case a new dataset is created, add additional elif for data_name and create your own plots specific to your data.

    Args:
        model: TBIP class object
        data_name: str[1] name of the dataset which selects what plots are created
        fig_dir: directory where to save the figure
        vocabulary: str[num_words] all used words
        breaks: int[num_times+1] time-period breakpoints as integers of format 'YYYYMMDD'
        time_periods: String that defines the set of dates that break the data into time-periods.

    """
    if (data_name == 'hein-daily') or (data_name[0:10] == "simulation"):
        dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in
                   breaks]
        middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(model.num_times)]

        ### Topic evolution in time
        # spineplot - Topical prevalence in time
        Ekvt = get_Ekvt(model, logscale=False)
        x = tfm.reduce_mean(Ekvt, axis=1)
        if time_periods == "sessions":
            xlabels_long = [str(stoy(i))[2:4] + '-' + str(stoy(i)+1)[2:4] + '\n(' + str(i) + ')' for i in range(97, 115)]
            xlabels_short = [str(i) for i in range(97, 115)]
        elif time_periods == "years":
            xlabels_long = xlabels_short = [str(i) for i in range(1981, 2017)]
        elif time_periods == "":
            xlabels_long = xlabels_short = [str(i) for i in range(len(breaks)-1)]
        else:
            xlabels_long = xlabels_short = [t.strftime("%Y-%m-%d") for t in middbreaks]
        ylabels = [str(k+1) for k in range(model.num_topics)]
        spineplot(x, xlabels_long, ylabels, 'Topic prevalence evolution',
                  os.path.join(fig_dir, 'spineplot_topics.png'), size=(10, 10))

        ### Averaging over rates
        avg_kt = get_avg_rates(model)
        spineplot(avg_kt, xlabels_long, ylabels, 'Topic prevalence evolution',
                  os.path.join(fig_dir, 'spineplot_avg_rates_topics.png'),
                  sort='unsorted', size=(10, 10))
        spineplot(avg_kt, xlabels_long, ylabels, 'Topic prevalence evolution',
                  os.path.join(fig_dir, 'spineplot_avg_rates_sorted_topics.png'),
                  sort='by_avg_rate', size=(10, 10))

        ### Cosine similarities heatmaps
        for what in ["beta", "log_beta", "fe"]:
            add = '_' + what
            # Topic similarities across time
            cosine_similarity_tk = get_cosine_similarity_times(model, what)
            plot_similarity(tf.transpose(cosine_similarity_tk),
                            title='Cosine similarities across consecutive time-periods',
                            xlabels=xlabels_short, ylabels=ylabels,
                            xlab=None, ylab='Topic',
                            path=os.path.join(fig_dir, 'cosine_similarity_in_time' + add + '.png'))

            # Similarities among topics each time separately
            cosine_similarity_kkt = get_cosine_similarity_topics(model, what)
            # for t in tf.range(model.num_times):
            #     plot_similarity(tf.gather(cosine_similarity_kkt, t, axis=-1),
            #                     title='Cosine similarities among topics, time-period=' + xlabels_short[t],
            #                     xlabels=ylabels, ylabels=ylabels,
            #                     xlab='Topic', ylab='Topic',
            #                     path=os.path.join(fig_dir,
            #                                       'cosine_similarity_topics_t_' + str(t.numpy()) + add + '.png'))

            # averaged over time
            plot_similarity(tf.reduce_mean(cosine_similarity_kkt, axis=-1),
                            title='Averaged cosine similarities among topics',
                            xlabels=ylabels, ylabels=ylabels,
                            xlab='Topic', ylab='Topic',
                            path=os.path.join(fig_dir, 'cosine_similarity_topics_averaged' + add + '.png'))

            # max over time
            plot_similarity(tf.reduce_max(cosine_similarity_kkt, axis=-1),
                            title='Maximal cosine similarities among topics',
                            xlabels=ylabels, ylabels=ylabels,
                            xlab='Topic', ylab='Topic',
                            path=os.path.join(fig_dir, 'cosine_similarity_topics_maximal' + add + '.png'))

        ### KL dissimilarity heatmaps
        # Topic dissimilarities across time
        KL_dissimilarity_tk = get_KL_dissimilarity_times(model)
        plot_similarity(tf.transpose(KL_dissimilarity_tk),
                        title=None, #'KL dissimilarity across consecutive time-periods',
                        xlabels=xlabels_short, ylabels=ylabels,
                        xlab='Session', ylab='Topic',
                        path=os.path.join(fig_dir, 'KL_dissimilarity_in_time.png'),
                        col_rev=True)

        # Topic dissimilarities compared to the first session (to measure gradual and complete change of the topic)
        KL_dissimilarity_tk = get_KL_dissimilarity_to_first(model)
        plot_similarity(tf.transpose(KL_dissimilarity_tk),
                        title=None, #'KL dissimilarity compared to the first session',
                        xlabels=xlabels_short, ylabels=ylabels,
                        xlab='Session', ylab='Topic',
                        path=os.path.join(fig_dir, 'KL_dissimilarity_to_first.png'),
                        col_rev=True)

        # # Dissimilarities among topics each time separately
        KL_dissimilarity_kkt = get_KL_dissimilarity_topics(model)
        # for t in tf.range(model.num_times):
        #     plot_similarity(tf.gather(KL_dissimilarity_kkt, t, axis=-1),
        #                     title=None, #'KL dissimilarity among topics, time-period=' + xlabels[t],
        #                     xlabels=ylabels, ylabels=ylabels,
        #                     xlab='Topic', ylab='Topic',
        #                     path=os.path.join(fig_dir, 'KL_dissimilarity_topics_t_' + str(t.numpy()) + '.png'),
        #                     col_rev=True)

        # averaged over time
        plot_similarity(tf.reduce_mean(KL_dissimilarity_kkt, axis=-1),
                        title=None, #'Averaged KL dissimilarities among topics',
                        xlabels=ylabels, ylabels=ylabels,
                        xlab='Topic', ylab='Topic',
                        path=os.path.join(fig_dir, 'KL_dissimilarity_topics_averaged.png'),
                        col_rev=True)

        # min over time
        plot_similarity(tf.reduce_min(KL_dissimilarity_kkt, axis=-1),
                        title=None, #'Minimal KL dissimilarities among topics',
                        xlabels=ylabels, ylabels=ylabels,
                        xlab='Topic', ylab='Topic',
                        path=os.path.join(fig_dir, 'KL_dissimilarity_topics_minimal.png'),
                        col_rev=True)
    else:
        raise ValueError('Unrecognized data_name.')


def hist_word_counts(countsSparse, countsNonzero, fig_dir, name_start="orig_"):
    """Create histograms that summarize the document-term matrix.

    Args:
        countsSparse: Document-term matrix containing word counts in a sparse format.
        countsNonzero: 0/1 indicators whether the word count is nonzero or not.
        fig_dir: Directory to save the plots.
        name_start: How should the plots be named at the beggining.
    """
    # Histogram of counts
    count_freq = tf.sparse.bincount(countsSparse)
    f0 = tf.cast(tfm.reduce_prod(countsSparse.shape) - countsSparse.indices.shape[0],
                 dtype=tf.int64)  # frequency of 0
    f = tf.concat(([f0], count_freq.values), axis=0)
    x = tf.concat(([0], count_freq.indices[:, 0]), axis=0)
    plt.bar(x.numpy(), np.log(f.numpy()))
    plt.title("Log-frequencies of word counts")
    plt.ylabel("Log-frequency")
    plt.xlabel("Word count in a document")
    plt.savefig(os.path.join(fig_dir, name_start + "log_hist_counts.png"))
    plt.close()

    # Word counts summed over documents
    plt.hist(tf.sparse.reduce_sum(countsSparse, axis=0), histtype='step', bins=100)  # summed over documents
    plt.title("Word counts across all documents")
    plt.xlabel("Word frequencies")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_counts_words.png"))
    plt.close()

    # Word counts summed over words.
    plt.hist(tf.sparse.reduce_sum(countsSparse, axis=1), histtype='step', bins=100)  # summed over words
    plt.title("Word counts across all words")
    plt.xlabel("Document length")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_counts_docs.png"))
    plt.close()

    # Number of different words used in the document
    x = tf.sparse.reduce_sum(countsNonzero, axis=1)
    plt.hist(x, histtype='step', bins=100)
    plt.title("Words in a document")
    plt.xlabel("Word count per document")
    plt.savefig(os.path.join(fig_dir, name_start + "hist_word_in_doc.png"))
    plt.close()

    # Number of documents in which the word appears
    x = tf.sparse.reduce_sum(countsNonzero, axis=0)
    plt.hist(x, histtype='step', bins=100)
    plt.title("Documents with given word")
    plt.xlabel("Document count per word")
    plt.savefig(os.path.join(fig_dir,  name_start + "hist_doc_having_word.png"))
    plt.close()


def sparse_select_indices(sp_input, indices, axis=0):
    # Only necessary if indices may have non-unique elements
    indices, _ = tf.unique(indices)
    n_indices = tf.size(indices)
    # Only necessary if indices may not be sorted
    indices, _ = tfm.top_k(indices, n_indices)
    indices = tf.reverse(indices, [0])
    # Get indices for the axis
    idx = sp_input.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))
    # Mask for selected values
    sel = tfm.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(sp_input.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tfm.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(sp_input.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis],
                             tf.expand_dims(idx_new, 1),
                             indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([sp_input.dense_shape[:axis],
                           [n_indices],
                           sp_input.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)

def summary_time_periods(countsSparse, countsNonzero, time_indices, author_indices, fig_dir, breaks, time_periods):
    num_times = len(breaks)-1
    num_docs = countsNonzero.shape[0]
    num_words = countsNonzero.shape[1]
    num_authors = max(author_indices)+1
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]
    middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(num_times)]
    counts = sparse.csr_matrix(
        (countsSparse.values, (countsSparse.indices[:, 0], countsSparse.indices[:, 1])), shape=countsSparse.dense_shape
    )
    counts_nonzero = sparse.csr_matrix(
        (countsNonzero.values, (countsNonzero.indices[:, 0], countsNonzero.indices[:, 1])),
        shape=countsNonzero.dense_shape
    )

    # Number of words used in each time period:
    words_in_speech = tf.sparse.reduce_sum(countsNonzero, axis=1)
    words_in_time = np.array([[]])
    time_word = np.zeros([num_times, num_words])
    for t in range(num_times):
        count_t = counts_nonzero[time_indices == t]
        count_t_sum = 1 * (count_t.sum(axis=0) > 0)
        time_word[t, :] = count_t_sum
        words_in_time = np.append(words_in_time, count_t_sum.sum())
    plt.bar(middbreaks, height=num_words - words_in_time,
            width=middbreaks[1]-middbreaks[0]-datetime.timedelta(days=30), bottom=0)
    plt.ylabel("Word count")
    plt.title("Number of words not used in time-periods (out of " + str(countsNonzero.shape[1]) + ")")
    plt.savefig(os.path.join(fig_dir, "words_not_used_in_time_" + time_periods + ".png"))
    plt.close()

    # In how many time periods does each word appear? excluding all sessions
    with_all = np.sum(time_word, axis=0)
    with_all_sum = tfm.unsorted_segment_sum(tf.ones(with_all.shape[0]),
                                            tf.constant(with_all, dtype=tf.int32), num_segments=num_times+1)
    # without_all = with_all[with_all != num_times]
    # without_all_sum = tfm.unsorted_segment_sum(tf.ones(without_all.shape[0]),
    #                                            tf.constant(without_all, dtype=tf.int32), num_segments=num_times)
    max_without_all_sum = max(with_all_sum[0:-1])

    # plt.hist(without_all, bins=range(0,num_times))
    plt.bar(range(num_times+1), with_all_sum, width=0.9)
    plt.text(num_times, max_without_all_sum, str(int(with_all_sum[num_times].numpy())),
             verticalalignment="bottom", horizontalalignment="center")
    plt.ylim((0, 1.05*max_without_all_sum))
    plt.xticks(range(num_times+1))
    plt.ylabel("Word count")
    plt.title("In how many time periods does each word appear?")
    plt.savefig(os.path.join(fig_dir, "words_in_times_" + time_periods + ".png"))
    plt.close()


    # Number of documents in each time period:
    speeches_in_time = tfm.unsorted_segment_sum(tf.ones(time_indices.shape), time_indices, num_segments=num_times)
    plt.bar(middbreaks, height=speeches_in_time,
            width=middbreaks[1] - middbreaks[0] - datetime.timedelta(days=30), bottom=0)
    plt.ylabel("Speech count")
    plt.title("Number of speeches in time-periods (out of " + str(num_docs) + ")")
    plt.savefig(os.path.join(fig_dir, "speeches_in_time_" + time_periods + ".png"))
    plt.close()

    # The average document length in each time period
    avg_word_per_speech = np.zeros([num_times])
    avg_dif_word_per_speech = np.zeros([num_times])
    for t in range(num_times):
        counts_t = counts[time_indices == t]
        counts_nonzero_t = counts_nonzero[time_indices == t]
        avg_word_per_speech[t] = counts_t.sum() / counts_t.shape[0]
        avg_dif_word_per_speech[t] = counts_nonzero_t.sum() / counts_t.shape[0]
    plt.bar(middbreaks, height=avg_word_per_speech,
            width=middbreaks[1] - middbreaks[0] - datetime.timedelta(days=30), bottom=0)
    plt.ylabel("Words per speech")
    plt.title("Average word count per speech")
    plt.savefig(os.path.join(fig_dir, "avg_word_per_speech_" + time_periods + ".png"))
    plt.close()
    plt.bar(middbreaks, height=avg_dif_word_per_speech,
            width=middbreaks[1] - middbreaks[0] - datetime.timedelta(days=30), bottom=0)
    plt.ylabel("Words per speech")
    plt.title("Average unique word count per speech")
    plt.savefig(os.path.join(fig_dir, "avg_dif_word_per_speech_" + time_periods + ".png"))
    plt.close()

    # Authors in each time period
    author_time = tf.reshape(
        tfm.unsorted_segment_sum(tf.ones(time_indices.shape),
                                 time_indices*num_authors + author_indices,
                                 num_segments=num_times*num_authors),
        [num_times, num_authors]
    )
    plt.bar(middbreaks, height=tfm.reduce_sum(tf.cast(author_time > 0, tf.int32), axis=1),
            width=middbreaks[1] - middbreaks[0] - datetime.timedelta(days=30), bottom=0)
    plt.ylabel("Author count")
    plt.title("Number of authors in each time period")
    plt.savefig(os.path.join(fig_dir, "author_counts_in_time_" + time_periods + ".png"))
    plt.close()

    # Author speech counts in time
    for a in range(num_authors):
        plt.plot(middbreaks, tfm.log(tf.gather(author_time, a, axis=1)+1))
    plt.ylabel("log(Speech count)")
    plt.title("Author speech counts in each session")
    plt.savefig(os.path.join(fig_dir, "author_speech_counts_in_time_" + time_periods + ".png"))
    plt.close()

    # Heatmap of author speeches in time
    plt.figure(figsize=(6, 15))
    plt.title("Author speech counts in each session", fontsize=18)
    sns.heatmap(tfm.log(tf.transpose(author_time)+1), cmap=sns.color_palette("Blues", as_cmap=True))
    plt.savefig(os.path.join(fig_dir, "author_speech_counts_heatmap_" + time_periods + ".png"))
    plt.close()




