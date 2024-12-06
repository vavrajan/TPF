# Import global packages
import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tensorflow as tf
tfm = tf.math
from DPF.code.plotting_functions import hist_word_counts, summary_time_periods


def build_input_pipeline_hein_daily(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    fit_dir=None,
                                    counts_transformation="nothing",
                                    pre_initialize_parameters="No",
                                    addendum='',
                                    time_periods='sessions',
                                    author='author'):
    """Load data and build iterator for minibatches.
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts_avt---.npy`, `author_detailed_info---.csv`, and `vocabulary---.txt`.
            --- = addendum + "_" + time_periods
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures.
        fit_dir: The directory with initial values.
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.

    Returns:
        dataset: Batched author_indices.
        permutation: Permutation of the documents.
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.

    """
    # Load data
    counts_av = np.load(os.path.join(data_dir, "counts_" + author + "v" + addendum + "_" + time_periods + ".npy"))
    speech_info = pd.read_csv(os.path.join(data_dir, "speech_info_" + author + addendum + "_" + time_periods + ".csv"))
    author_time_info = pd.read_csv(os.path.join(data_dir, "author_time_info_" + author + addendum + "_" + time_periods + ".csv"))
    breaks = np.load(os.path.join(data_dir, 'breaks' + addendum + '_' + time_periods + '.npy'))

    file = open(os.path.join(data_dir, "vocabulary" + addendum + '_' + time_periods + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')

    # Random shuffle of the documents
    num_author_times, num_words = counts_av.shape
    num_times = len(breaks)-1
    num_authors = len(author_time_info['author'].unique())
    author_times = random_state.permutation(num_author_times)
    shuffled_counts_av = counts_av[author_times]
    tf_counts_av = tf.sparse.from_dense(shuffled_counts_av)
    if counts_transformation == "nothing":
        count_values = tf_counts_av.values
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + tf_counts_av.values))
    else:
        raise ValueError("Unrecognized counts transformation.")
    tf_counts_av = tf.SparseTensor(
        indices=tf_counts_av.indices,
        values=tf.cast(count_values, "float32"),
        dense_shape=tf_counts_av.dense_shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=tf_counts_av.indices,
            values=tf.cast(count_values, "int64"), #count_values, #.astype(np.int64),
            dense_shape=tf_counts_av.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=tf_counts_av.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=tf_counts_av.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        # hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")
        # todo If required, construct similar plots
        # summary_time_periods(shuffled_countsSparse, shuffled_countsNonzero,
        #                      shuffled_speech_info["time"], shuffled_speech_info["author"],
        #                      fig_dir, breaks, time_periods)

    sub_author_time_info = author_time_info[author_time_info['presence']].reset_index(drop=True)
    dataset = tf.data.Dataset.from_tensor_slices(
        ({"author_time_indices": author_times,
          "author_indices": sub_author_time_info["author"].loc[author_times],
          "time_indices": sub_author_time_info["time"].loc[author_times]}, tf_counts_av))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ### Initilization by Poisson factorization
    ### Requires to run 'poisson_factorization.py' first to save document and topic shapes and rates.
    if pre_initialize_parameters == "PF":
        # Run 'poisson_factorization.py' first to store the initial values.
        theta_shape = np.load(fit_dir.replace("replace", "document_shape")).astype(np.float32)
        theta_rate = np.load(fit_dir.replace("replace", "document_rate")).astype(np.float32)
        topic_shape = np.load(fit_dir.replace("replace", "topic_shape")).astype(np.float32)
        topic_rate = np.load(fit_dir.replace("replace", "topic_rate")).astype(np.float32)
        # Average over the authors
        author_shape = tf.math.unsorted_segment_sum(theta_shape, speech_info["author"], num_segments=num_authors)
        author_rate = tf.math.unsorted_segment_sum(theta_rate, speech_info["author"], num_segments=num_authors)
        inits = {"ar_ak_mean": tf.math.digamma(author_shape) - tf.math.log(author_rate),
                 "ar_kv_mean": tf.math.digamma(topic_shape) - tf.math.log(topic_rate)}
    else:
        inits = {"ar_ak_mean": None, "ar_kv_mean": None}

    return dataset, author_times, vocabulary, author_time_info, breaks, inits

def build_input_pipeline_simulation(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    fit_dir=None,
                                    counts_transformation="nothing",
                                    pre_initialize_parameters="No",
                                    addendum='',
                                    time_periods='sessions',
                                    author='author'):
    """Load data and build iterator for minibatches.
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts_avt---.npy`, `author_detailed_info---.csv`, and `vocabulary---.txt`.
            --- = addendum + "_" + time_periods
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures.
        fit_dir: The directory with initial values.
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.

    Returns:
        dataset: Batched author_indices.
        permutation: Permutation of the documents.
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.

    """
    # Load data
    counts_av = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
    breaks = np.load(os.path.join(data_dir, "breaks.npy"))
    time_indices = np.load(os.path.join(data_dir, "time_indices.npy"))
    author_indices = np.load(os.path.join(data_dir, "author_indices.npy"))
    author_time_indices = np.load(os.path.join(data_dir, "author_time_indices.npy"))
    author_time_info = pd.DataFrame({'time': time_indices,
                                     'author': author_indices,
                                     'author_time': author_time_indices})

    file = open(os.path.join(data_dir, "vocabulary.txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')

    # Random shuffle of the documents
    num_author_times, num_words = counts_av.shape
    num_times = len(breaks)-1
    num_authors = len(author_time_info['author'].unique())
    author_times = random_state.permutation(num_author_times)
    shuffled_counts_av = counts_av[author_times]
    shuffled_author_time_info = author_time_info.loc[author_times]

    if counts_transformation == "nothing":
        count_values = shuffled_counts_av.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts_av.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts_av.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts_av.shape)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"author_time_indices": author_times,
          "author_indices": shuffled_author_time_info["author"],
          "time_indices": shuffled_author_time_info["time"]},
         shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ### Initilization by Poisson factorization
    ### Requires to run 'poisson_factorization.py' first to save document and topic shapes and rates.
    if pre_initialize_parameters == "PF":
        # Run 'poisson_factorization.py' first to store the initial values.
        theta_shape = np.load(fit_dir.replace("replace", "document_shape")).astype(np.float32)
        theta_rate = np.load(fit_dir.replace("replace", "document_rate")).astype(np.float32)
        topic_shape = np.load(fit_dir.replace("replace", "topic_shape")).astype(np.float32)
        topic_rate = np.load(fit_dir.replace("replace", "topic_rate")).astype(np.float32)
        # Average over the authors
        author_shape = tf.math.unsorted_segment_sum(theta_shape, author_time_info["author"], num_segments=num_authors)
        author_rate = tf.math.unsorted_segment_sum(theta_rate, author_time_info["author"], num_segments=num_authors)
        inits = {"ar_ak_mean": tf.math.digamma(author_shape) - tf.math.log(author_rate),
                 "ar_kv_mean": tf.math.digamma(topic_shape) - tf.math.log(topic_rate)}
    elif pre_initialize_parameters == "sim_true":
        theta = np.load(os.path.join(data_dir, "theta.npy")).astype(np.float32)
        theta_ak = tfm.unsorted_segment_mean(theta,
                                            author_time_info['author'],
                                            num_segments=num_authors)
        inits = {"ar_ak_mean": tf.math.log(theta_ak),
                 "ar_kv_mean": np.load(os.path.join(data_dir, "ar_kv_mean.npy")).astype(np.float32)}
    else:
        inits = {"ar_ak_mean": None, "ar_kv_mean": None}

    return dataset, author_times, vocabulary, author_time_info, breaks, inits

def build_input_pipeline_example(data_dir,
                                 batch_size,
                                 random_state,
                                 fig_dir=None,
                                 fit_dir=None,
                                 counts_transformation="nothing",
                                 pre_initialize_parameters="No",
                                 addendum='',
                                 time_periods='',
                                 author='author'):
    # Create data
    if addendum == '_01':
        counts_avt = np.array(
            [[[0, 1, 2],
              [2, 1, 0],
              [3, 0, 2],
              [0, 1, 1],
              [0, 2, 1],
              [3, 3, 3],
              [4, 0, 1]],
             [[4, 1, 2],
              [0, 3, 2],
              [0, 5, 1],
              [0, 0, 1],
              [2, 0, 0],
              [1, 0, 1],
              [3, 2, 1]],
             [[1, 1, 1],
              [2, 2, 1],
              [1, 0, 0],
              [0, 5, 0],
              [0, 0, 2],
              [0, 1, 0],
              [3, 0, 0]],
             [[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 2, 0],
              [1, 1, 0],
              [0, 0, 3],
              [1, 1, 1]]
             ])
    elif addendum == '_02':
        counts_avt = np.array(
            [[[0, 0, 2],
              [2, 0, 0],
              [3, 0, 2],
              [0, 0, 1],
              [0, 0, 1],
              [3, 0, 3],
              [4, 0, 1]],
             [[0, 1, 2],
              [0, 3, 2],
              [0, 5, 1],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 1],
              [0, 2, 1]],
             [[0, 0, 1],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, 3],
              [0, 0, 0],
              [0, 0, 5],
              [0, 0, 2]],
             [[1, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [0, 0, 0],
              [1, 0, 0]]
             ])
    elif addendum == '_03':
        counts_avt = np.zeros((4, 7, 3))
    elif addendum == '_04':
        # counts_avt = np.zeros((100, 200, 10))
        counts_avt = np.ones((100, 200, 10))
    elif addendum == '_05':
        counts_avt = random_state.binomial(1, 0.9, (100, 200, 10))
    else:
        raise ValueError("Unrecognized choice of addendum for example dataset.")
    author = [[a for t in range(counts_avt.shape[-1])] for a in range(counts_avt.shape[0])]
    time = [[t for a in range(counts_avt.shape[0])] for t in range(counts_avt.shape[-1])]
    author_time_info = pd.DataFrame({"author": np.resize(author, counts_avt.shape[0]*counts_avt.shape[-1]),
                                     "time": np.resize(time, counts_avt.shape[0]*counts_avt.shape[-1])})
    breaks = [(y + 2000) * 10000 + 101 for y in range(counts_avt.shape[-1]+1)]
    vocabulary = np.array(list(map(chr, range(97, 97 + counts_avt.shape[1]))))

    # a = [33, 34, 65, 81]
    # acounts = tf.gather(counts_avt, a, axis=0)
    # print("Counts of author " + str(a) + ":")
    # print(acounts)
    # print(tf.math.reduce_sum(acounts, axis=0))
    # print(tf.math.reduce_sum(acounts, axis=1))
    # print(tf.math.reduce_sum(acounts, axis=2))


    # Random shuffle of the documents
    num_authors, num_words, num_times = counts_avt.shape
    authors = random_state.permutation(num_authors)
    shuffled_counts_avt = counts_avt[authors]
    tf_counts_avt = tf.sparse.from_dense(shuffled_counts_avt)
    if counts_transformation == "nothing":
        count_values = tf_counts_avt.values
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + tf_counts_avt.values))
    else:
        raise ValueError("Unrecognized counts transformation.")
    tf_counts_avt = tf.SparseTensor(
        indices=tf_counts_avt.indices,
        values=tf.cast(count_values, "float32"),
        dense_shape=tf_counts_avt.dense_shape)
    dataset = tf.data.Dataset.from_tensor_slices(({"author_indices": authors}, tf_counts_avt))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # empty inits
    inits = {"ar_ak_mean": None, "ar_kv_mean" : None}

    return dataset, authors, vocabulary, author_time_info, breaks, inits

def build_input_pipeline(data_name, data_dir, batch_size, random_state, fig_dir, fit_dir, counts_transformation,
                         pre_initialize_parameters, addendum, time_periods, author):
    """Triggers the right build_input_pipeline depending on the current dataset.

    Args:
        data_name: String containing the name of the dataset. Important to choose build_input_pipeline function.
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts---.npz`, `speech_info---.csv`, `author_detailed_info---.csv`, and `vocabulary---.txt`.
            --- = addendum + "_" + time_periods
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        fit_dir: The directory with initial values.
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        pre_initialize_parameters: A string declaring which type of initialization should be used.
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.
        author: A string that defines who/what is considered to be an author.

    Returns:
        dataset: Batched author_time_indices.
        permutation: Permutation of the authors.
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.
        inits: A dictionary of initial values for some model parameters.

    """
    if (data_name == "hein-daily") or (data_name == "hein-daily_tv"):
        dataset, permutation, vocabulary, author_time_info, breaks, inits = build_input_pipeline_hein_daily(
            data_dir, batch_size, random_state, fig_dir, fit_dir, counts_transformation, pre_initialize_parameters,
            addendum, time_periods, author)
    elif data_name == "example":
        dataset, permutation, vocabulary, author_time_info, breaks, inits = build_input_pipeline_example(
            data_dir, batch_size, random_state, fig_dir, fit_dir, counts_transformation, pre_initialize_parameters,
            addendum, time_periods, author)
    elif data_name[0:10] == "simulation":
        dataset, permutation, vocabulary, author_time_info, breaks, inits = build_input_pipeline_simulation(
            data_dir, batch_size, random_state, fig_dir, fit_dir, counts_transformation, pre_initialize_parameters,
            addendum, time_periods, author)
    else:
        raise ValueError("Unrecognized dataset name in order to load the data. "
                         "Implement your own 'build_input_pipeline_...' for your dataset.")
    return dataset, permutation, vocabulary, author_time_info, breaks, inits

