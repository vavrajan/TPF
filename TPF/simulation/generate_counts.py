## Import global packages
import os
import time

import numpy as np
import scipy.sparse as sparse
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import string
import random
import datetime

from absl import app
from absl import flags

tfd = tfp.distributions
tfm = tf.math

flags.DEFINE_string("simulation_name", default="simulation", help="Name of the simulation.")
flags.DEFINE_string("addendum", default="", help="String to be added at the end of files.")
flags.DEFINE_integer("num_authors", default=1000, help="Number of authors.")
flags.DEFINE_integer("num_topics", default=6, help="Number of topics.")
flags.DEFINE_integer("num_words", default=500, help="Number of words.")
flags.DEFINE_integer("num_times", default=10, help="Number of time-periods.")
flags.DEFINE_integer("num_replications", default=10, help="Number of replications.")

flags.DEFINE_float("ar_kv_delta", default=0.5, help="Fixed value of ar_kv_delta.")
flags.DEFINE_float("ar_kv_prec", default=10.0, help="Fixed value of ar_kv_prec.")
flags.DEFINE_float("ar_kv_mean_start", default=-2.0, help="Starting position for ar_kv_mean.")
flags.DEFINE_float("vowel_word", default=0.1, help="What is the contribution of vowel per appearance in a word.")
flags.DEFINE_float("consonant_width", default=0.1, help="ar_kv_mean +- consonant_width/2 is 'n', "
                                                        "other consonants will be determined by this width.")

# the rest are not needed (for initial idea)
flags.DEFINE_float("Pois_rate", default=0.8, help="Fixed rate for Pois counts.")
flags.DEFINE_float("theta_shp", default=0.5, help="Fixed shape for Gamma distributed theta.")
flags.DEFINE_float("theta_rte", default=10.0, help="Fixed rate for Gamma distributed theta.")
flags.DEFINE_float("log_theta_loc", default=-1.0, help="Fixed location for Normal distributed log_theta.")
flags.DEFINE_float("log_theta_scl", default=0.3, help="Fixed scale for Normal distributed log_theta.")
flags.DEFINE_float("beta_shp", default=0.5, help="Fixed shape for Gamma distributed beta.")
flags.DEFINE_float("beta_rte", default=10.0, help="Fixed rate for Gamma distributed beta.")
flags.DEFINE_float("log_beta_start", default=-2.0, help="Starting position for log_beta.")
flags.DEFINE_float("consonant_word", default=0.05, help="Fixed rate for Gamma distributed beta.")

FLAGS = flags.FLAGS


def get_delta_matrix(delta, T):
    delta2 = tfm.square(delta)
    superdiag = subdiag = tf.repeat(-delta[:, :, tf.newaxis], T, axis=-1)
    diagdelta2 = tf.repeat(delta2[:, :, tf.newaxis], T - 1, axis=-1)
    replacement_slice = tf.zeros(tf.shape(diagdelta2)[:-1])
    maindiag = 1.0 + tf.concat([diagdelta2, replacement_slice[:, :, tf.newaxis]], axis=-1)
    prec = tf.linalg.LinearOperatorTridiag([superdiag, maindiag, subdiag], diagonals_format='sequence')
    return prec

def get_inverse_delta_matrix(delta, T):
    var = np.zeros([delta.shape[0], delta.shape[1], T, T])
    sum_delta_2i = tf.zeros(delta.shape)
    for i in range(T):
        sum_delta_2i += tfm.pow(delta, 2*i)
        for j in range(i, T):
            var[:, :, i, j] = var[:, :, j, i] = tfm.pow(delta, j - i) * sum_delta_2i
    return tf.constant(var, dtype=tf.float32)

def generate_word(n_couples = 5):
    letters = list(string.ascii_lowercase)
    vowels = ["a", "e", "i", "o", "u"]
    consonants = [letter for letter in letters if letter not in vowels]
    # one vowel preference
    preferred_vowel = random.choice(vowels)
    for i in range(4):
        vowels.append(preferred_vowel)

    word = ''
    for i in range(n_couples):
        word += random.choice(consonants) + random.choice(vowels)
    return word

def generate_word_vowels(n_couples = 5):
    letters = list(string.ascii_lowercase)
    vowels = ["a", "e", "i", "o", "u"]
    consonants = [letter for letter in letters if letter not in vowels]
    # one vowel preference
    preferred_vowel = random.choice(vowels)
    for i in range(4):
        vowels.append(preferred_vowel)

    word = ''
    for i in range(n_couples):
        word += random.choice(vowels)
    return word

def get_beta(how, num_authors, num_times, num_topics, vocabulary, FLAGS):
    num_words = len(vocabulary)
    if how == 'PF':
        beta_dist = tfp.distributions.Gamma(shape=tf.fill([num_topics, num_words], FLAGS.beta_shp),
                                             rate=tf.fill([num_topics, num_words], FLAGS.beta_rte))
        beta = tf.gather(beta_dist.sample(1), 0, axis=0)
        beta = tf.repeat(beta[tf.newaxis, :, :], num_authors*num_times, axis=0)
    elif how in ['DPF', 'TPF']:
        log_beta = tf.Variable(tf.fill([num_times, num_topics, num_words], FLAGS.log_beta_start))

        letters = list(string.ascii_lowercase)
        vowels = ["a", "e", "i", "o", "u"]
        consonants = [letter for letter in letters if letter not in vowels]

        # set initial values for the first time period
        for k in range(np.min(num_topics, len(vowels))):
            log_beta[0, k, :].assign(log_beta[0, k, :] + FLAGS.vowel_topic)
            for v in range(num_words):
                word = vocabulary[v]
                multiplier = word.count(vowels[k])
                log_beta[0, k, v].assign(log_beta[0, k, v] + FLAGS.vowel_word * multiplier)
        # now for any topic capture the evolution in time: (t-1) -> t
        for v in range(num_words):
            for vowel in vowels:
                word = word.replace(vowel, "")
            for t in range(1, num_times):
                con_t = consonants.index(word[t])
                con_t_1 = consonants.index(word[t-1])
                log_beta[t, :, v].assign(log_beta[t-1, :, v] + FLAGS.consonant_word * (con_t - con_t_1))

        beta = tf.repeat(tfm.exp(log_beta), np.repeat(num_authors, num_times), axis=0)
    else:
        raise ValueError("Unrecognized choice of how to generate counts.")
    return beta

def main(argv):
    del argv

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.simulation_name)
    data_dir = os.path.join(source_dir, 'clean')
    fig_dir = os.path.join(source_dir, 'fig')

    if not os.path.exists(source_dir):
        os.mkdir(source_dir)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    num_documents = FLAGS.num_authors * FLAGS.num_times

    # Generating vowels in vocabulary
    vocabulary_vowels = []
    for v in range(FLAGS.num_words):
        vocabulary_vowels.append(generate_word_vowels(FLAGS.num_times))
        # not_unique = True
        # while not_unique:
        #     word = generate_word_vowels(FLAGS.num_times)
        #     not_unique = (word in vocabulary)
        # vocabulary.append(word)

    letters = list(string.ascii_lowercase)
    vowels = ["a", "e", "i", "o", "u"]
    consonants = [letter for letter in letters if letter not in vowels]

    # Generating thetas
    available_thetas = tf.constant([0.8, 0.9, 1.0, 1.1, 1.2])
    theta_ind = tfd.Categorical(probs=tf.fill([5], 1/5)).sample([num_documents, FLAGS.num_topics])
    theta = tf.gather(available_thetas, theta_ind)

    # Generating ar_kv_mean
    ar_kv_mean = tf.Variable(tf.fill([FLAGS.num_topics, FLAGS.num_words], FLAGS.ar_kv_mean_start))
    for k in range(np.min([FLAGS.num_topics, len(vowels)])):
        for v in range(FLAGS.num_words):
            word_vowels = vocabulary_vowels[v]
            multiplier = word_vowels.count(vowels[k])
            ar_kv_mean[k, v].assign(ar_kv_mean[k, v] + FLAGS.vowel_word * multiplier / FLAGS.num_times)

    # Generating ar_kv
    covariance = get_inverse_delta_matrix(delta=tf.fill([FLAGS.num_topics, FLAGS.num_words], FLAGS.ar_kv_delta),
                                          T=FLAGS.num_times)
    scale_tril = tf.linalg.cholesky(covariance) / tfm.sqrt(FLAGS.ar_kv_prec)
    # the same ar_kv for all topics
    centered_ar_v = tfp.distributions.MultivariateNormalTriL(loc=tf.zeros([FLAGS.num_words, FLAGS.num_times]),
                                                             scale_tril=tf.gather(scale_tril, 0, axis=0)).sample()
    ar_kv = ar_kv_mean[:, :, tf.newaxis] + centered_ar_v[tf.newaxis, :, :]

    # Define vocabulary based on sampled ar_kv
    if FLAGS.ar_kv_delta == 1.0:
        width = FLAGS.consonant_width + 0.1
    else:
        width = FLAGS.consonant_width
    grid = [(c + 0.5) * width for c in range(-10, 10)]
    grid.append(99999999.9)
    vocabulary = []
    for v in range(FLAGS.num_words):
        word = ''
        for t in range(FLAGS.num_times):
            c = 0
            while centered_ar_v[v, t] > grid[c]:
                c += 1
            word += consonants[c] + vocabulary_vowels[v][t]
        vocabulary.append(word)
    vocabulary = np.array(vocabulary)

    # Sorting in alphabetical order
    ordering = vocabulary.argsort()
    vocabulary = vocabulary[ordering]
    ar_kv = tf.gather(ar_kv, ordering, axis=1)
    ar_kv_mean = tf.gather(ar_kv_mean, ordering, axis=1)
    centered_ar_v = tf.gather(centered_ar_v, ordering, axis=0)

    # breaks
    breaks = [(2000 + t) * 10000 + 101 for t in range(FLAGS.num_times + 1)]
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]
    middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(FLAGS.num_times)]

    # Plotting centered_ar_v
    plt.figure(figsize=(15, 10))
    plt.plot(middbreaks, tf.transpose(centered_ar_v))
    plt.legend(vocabulary, loc="upper right")
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'centered_ar_v' + FLAGS.addendum + '.png'))

    # Plotting ar_kv
    for k in range(FLAGS.num_topics):
        plt.figure(figsize=(15, 10))
        plt.plot(middbreaks, tf.transpose(tf.gather(ar_kv, k, axis=0)))
        plt.legend(vocabulary, loc="upper right")
        plt.savefig(os.path.join(fig_dir, 'ar_' + str(k + 1) + 'v' + FLAGS.addendum + '.png'))

    # Generating counts
    time_indices = tf.repeat([t for t in range(FLAGS.num_times)], FLAGS.num_authors)
    author_indices = [d % FLAGS.num_authors for d in range(num_documents)]
    author_time_indices = FLAGS.num_times * time_indices + author_indices
    selected_ar_kv = tf.transpose(tf.gather(ar_kv, time_indices, axis=-1),
                                  perm=[2, 0, 1])  # [num_documents, num_topics, num_words]
    rate = tfm.reduce_sum(theta[:, :, tf.newaxis] * tfm.exp(selected_ar_kv), axis=1)  # sum over topics
    print(rate)
    print("Total summary of rates - min, mean, max")
    print(tf.reduce_min(rate))
    print(tf.reduce_mean(rate))
    print(tf.reduce_max(rate))
    print("Total summary of rates over axis=0 - min, mean, max")
    print(tf.reduce_min(rate, axis=0))
    print(tf.reduce_mean(rate, axis=0))
    print(tf.reduce_max(rate, axis=0))
    print("Total summary of rates over axis=1 - min, mean, max")
    print(tf.reduce_min(rate, axis=1))
    print(tf.reduce_mean(rate, axis=1))
    print(tf.reduce_max(rate, axis=1))
    Poisdist = tfp.distributions.Poisson(rate=rate)
    counts_replicated = Poisdist.sample(FLAGS.num_replications)

    # Saving generated data
    for i in range(FLAGS.num_replications):
        sparse_counts = sparse.csr_matrix(tf.gather(counts_replicated, i, axis=0))
        sparse.save_npz(os.path.join(data_dir, "counts" + FLAGS.addendum + "_" + str(i) + ".npz"), sparse_counts)
    np.savetxt(os.path.join(data_dir, 'vocabulary' + FLAGS.addendum + '.txt'), vocabulary, fmt="%s")
    np.save(os.path.join(data_dir, "breaks" + FLAGS.addendum), breaks)
    np.save(os.path.join(data_dir, "time_indices" + FLAGS.addendum), time_indices)
    np.save(os.path.join(data_dir, "author_indices" + FLAGS.addendum), author_indices)
    np.save(os.path.join(data_dir, "author_time_indices" + FLAGS.addendum), author_time_indices)
    np.save(os.path.join(data_dir, "theta" + FLAGS.addendum), theta)
    np.save(os.path.join(data_dir, "ar_kv" + FLAGS.addendum), ar_kv)
    np.save(os.path.join(data_dir, "ar_kv_mean" + FLAGS.addendum), ar_kv_mean)

if __name__ == '__main__':
    app.run(main)
