# Import global packages
import os
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
import datetime


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

def get_Ekvt(model, logscale=True):
    if logscale:
        Ekvt = model.ar_kv_varfam.location
    else:
        # covariance = scale_tril @ scale_tril.T      is of shape: [num_topics, num_words, num_times, num_times]
        ar_kv_cov, ar_kv_var = model.get_ar_cov(model.ar_kv_varfam)
        ar_kv = model.ar_kv_varfam.location + 0.5 * ar_kv_var
        Ekvt = tfm.exp(ar_kv)
    return Ekvt


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
    # sample_kvt = tfp.distributions.Binomial(10, 0.5).sample([3, 10, 5])
    for t in range(model.num_times):
        # f_kv = tf.gather(sample_kvt, t, axis=-1)
        f_kv = tf.gather(model.ar_kv_varfam.location, t, axis=-1)
        beta_kv = tfm.exp(f_kv)
        beta_v = tf.reduce_sum(beta_kv, axis=0)
        for k in range(model.num_topics):
            f_v = tf.gather(f_kv, k, axis=0)
            # Frequency
            df = tfp.distributions.Empirical(f_v)
            f_ecdf_v = df.cdf(f_v)

            # Exclusivity
            betak_v = tf.gather(beta_kv, k, axis=0)
            e_v = betak_v / beta_v[tf.newaxis, :]
            de = tfp.distributions.Empirical(e_v)
            e_ecdf_v = de.cdf(e_v)

            fe_v = tfm.reciprocal(w/e_ecdf_v + (1-w)/f_ecdf_v)
            fe_kv = tf.tensor_scatter_nd_update(fe_kv, tf.constant([[k]]), fe_v)
        fe_tkv = tf.tensor_scatter_nd_update(fe_tkv, tf.constant([[t]]), fe_kv[tf.newaxis, :, :])

    fe_kvt = tf.transpose(fe_tkv, perm=[1, 2, 0])
    return fe_kvt


def get_vocabulary_intensity(model, how):
    ### Determine the top words
    if "frex" in how:
        weight = int(how.replace("frex_", "")) / 100
        intensity = get_fe_kvt3(model, w=weight)
    elif how == "beta":
        intensity = get_Ekvt(model, logscale=False)
    elif how == "log_beta":
        intensity = get_Ekvt(model, logscale=True)
    else:
        intensity = get_fe_kvt3(model, w=0.5)

    return intensity


def vocabulary_evolution(path, model, k, intensity, vocabulary, tlab, nword, ncol):
    """
    t=1 | t=2 | ... | t=ncol  <-- replace with tlab[t]
    ------------------------
    a   | a   | ... | b      __
    .   | .   | ... | .       |- nword rows
    c   | d   | ... | f      --

    Args:
        path: Where to save the latex table (including the name of the file).
        model: TPF class object
        k: topic number
        intensity: float[num_words, num_times] intensity for each term in vocabulary and time-period
        vocabulary: an array of the vocabulary terms
        tlab: str[num_times] labels of the time periods
        nword: number of top words to display
        ncol: number of columns
    """
    ### Determine the top words for topic k
    val, ind = tfm.top_k(tf.transpose(intensity), nword)
    nrow = np.int32(np.ceil(model.num_times / ncol))

    with open(path, 'w') as file:
        file.write('\\begin{tabular}{|' + 'l|'*ncol + '}\n')
        # file.write('\\multicolumn{'+str(ncol)+'}{c}{Top '+str(nword)+' words for topic '+str(k+1)+'}\\\\\n')
        file.write('\\toprule\n')
        for r in range(nrow):
            file.write("\\multicolumn{1}{c}{")
            file.write("} & \\multicolumn{1}{c}{".join(tlab[(r*ncol):((r+1)*ncol)]))
            file.write('} \\\\\n')
            file.write('\\midrule\n')
            for w in range(nword):
                winds = tf.gather(ind, w, axis=1)
                ts = range(r*ncol, tfm.minimum((r+1)*ncol, model.num_times))
                twinds = tf.gather(winds, ts, axis=0)
                file.write(" & ".join(vocabulary[twinds]))
                if len(ts) < ncol:
                    file.write(" & " * (ncol - len(ts)))
                file.write('\\\\\n')
            file.write('\\midrule\n')
        file.write('\\end{tabular}\n')


def gather_all_tables(num_topics, path):
    with open(path, 'w') as file:
        for k in range(num_topics):
            file.write('\\begin{table}\n')
            file.write('\t\\hspace*{-1em}\n')
            file.write('\t\\resizebox{1.03\\textwidth}{!}{\n')
            file.write('\t\t\\input{tab/TPF_AR_MVnormal_K25_sessions/vocabulary_evolution_frex_50_' + str(k+1) + '}\n')
            file.write('\t}\n')
            file.write('\t\\caption{Topic ' + str(k+1) + ', FREX measure, $w=0.5$.}\n')
            file.write('\t\\label{tab:AR_MVnormal_vocabulary_evolution_frex_50_' + str(k+1) + '}\n')
            file.write('\\end{table}\n\n')
            file.write('\\begin{table}\n')
            file.write('\t\\hspace*{-1em}\n')
            file.write('\t\\resizebox{1.03\\textwidth}{!}{\n')
            file.write('\t\t\\input{tab/TPF_AR_MVnormal_K25_sessions/vocabulary_evolution_beta_' + str(k+1) + '}\n')
            file.write('\t}\n')
            file.write('\t\\caption{Topic ' + str(k+1) + ', $\\exp\\{h^\\beta\\}$-based.}\n')
            file.write('\t\\label{tab:AR_MVnormal_vocabulary_evolution_beta_' + str(k+1) + '}\n')
            file.write('\\end{table}\n\n')


def topical_content_evolution(path, model, x, tlab, dig=3, ncol=9):
    """
    k | t=1 | t=2 | ... | t=ncol  <-- replace with tlab[t]
    ----------------------------
    1 | .   | .   | ... | .
    2 | .   | .   | ... | .
    . | .   | .   | ... | .
    K | .   | .   | ... | .
    ----------------------------
    k | t=1 | t=2 | ... | t=ncol
    1 | .   | .   | ... | .
    Args:
        path: Where to save the latex table (including the name of the file).
        model: a TPF model
        x: float[num_topics, num_times] topical content measure
        tlab: str[num_times] labels of the time periods
        ncol: number of columns
    """
    xsum = tf.math.reduce_sum(x, axis=0)
    xscaled = x / xsum[tf.newaxis, :]
    # Determine the top words for topic k
    nrow = np.int32(np.ceil(model.num_times / ncol))

    with open(path, 'w') as file:
        file.write('\\begin{tabular}{l|' + 'c|'*ncol + '}\n')
        # file.write('\\multicolumn{'+str(ncol)+'}{c}{Top '+str(nword)+' words for topic '+str(k+1)+'}\\\\\n')
        file.write('\\toprule\n')
        for r in range(nrow):
            file.write("\\multicolumn{1}{c}{Topic} & \\multicolumn{1}{c}{")
            file.write("} & \\multicolumn{1}{c}{".join(tlab[(r*ncol):((r+1)*ncol)]))
            file.write('} \\\\\n')
            file.write('\\midrule\n')
            for k in range(model.num_topics):
                xk = tf.gather(xscaled, k, axis=0)
                file.write(str(k+1) + " & ")
                ts = range(r * ncol, tfm.minimum((r + 1) * ncol, model.num_times))
                DIG = tf.cast(tf.pow(10, dig), tf.float32)
                xround = tf.math.floor(tf.math.round(DIG * tf.gather(xk, ts, axis=0))) / DIG
                file.write(" & ".join(map(str, xround.numpy())))
                if len(ts) < ncol:
                    file.write(" & " * (ncol - len(ts)))
                file.write('\\\\\n')
            file.write('\\midrule\n')
        file.write('\\end{tabular}\n')


def stoy(s):
    return 1981 + 2 * (s - 97)

def create_latex_tables(model, tab_dir: str, vocabulary, breaks, time_periods):
    """ Create and save latex tables into tab_dir directory.

    Args:
        model: TPF class object
        tab_dir: directory where to save the tex tables
        vocabulary: str[num_words] all used words
        breaks: int[num_times+1] time-period breakpoints as integers of format 'YYYYMMDD'
        time_periods: String that defines the set of dates that break the data into time-periods.

    """
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in
               breaks]
    middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(model.num_times)]
    if time_periods == 'sessions':
        tlab = [str(i) + ": " + str(stoy(i)) + "--" + str(stoy(i)+1) for i in range(97, 115)]
        tlab_short = [str(i) for i in range(97, 115)]
    elif time_periods == "years":
        tlab = tlab_short = [str(i) for i in range(1981, 2017)]
    elif time_periods == "":
        tlab = tlab_short = ["t="+str(i) for i in range(len(breaks)-1)]
    else:
        tlab = tlab_short = [t.strftime("%Y-%m-%d") for t in middbreaks]

    for how in ['frex_50', 'beta', 'log_beta']:
        intensity = get_vocabulary_intensity(model, how)
        for k in range(model.num_topics):
            vocabulary_evolution(os.path.join(tab_dir, 'vocabulary_evolution_' + how + "_" + str(k+1) + '.tex'),
                                 model, k, tf.gather(intensity, k, axis=0), vocabulary, tlab, nword=10, ncol=5) # 6 for hein-daily

    # topical content table
    avg_kt = get_avg_rates(model)
    topical_content_evolution(os.path.join(tab_dir, 'topical_content_evolution.tex'),
                              model, avg_kt, tlab_short, dig=3, ncol=10) # 9 for hein-daily

    gather_all_tables(model.num_topics, os.path.join(tab_dir, 'all_tables_frex_beta.tex'))

