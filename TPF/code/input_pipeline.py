# Import global packages
import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tensorflow as tf
from TPF.code.plotting_functions import hist_word_counts, summary_time_periods

def build_input_pipeline_hein_daily(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    counts_transformation="nothing",
                                    addendum='',
                                    time_periods='sessions'):
    """Load data and build iterator for minibatches.
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts---.npz`, `speech_info---.csv`, `author_detailed_info---.csv`, and `vocabulary---.txt`.
            --- = addendum + "_" + time_periods
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.

    Returns:
        dataset: Batched document_indices and author_indices.
        permutation: Permutation of the documents.
        shuffled_speech_info: Information about speeches after permutation (author, time, author_time indices, ...).
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.

    """
    # Load data
    counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + "_" + time_periods + ".npz"))
    speech_info = pd.read_csv(os.path.join(data_dir, "speech_info" + addendum + "_" + time_periods + ".csv"))
    author_time_info = pd.read_csv(os.path.join(data_dir, "author_time_info" + addendum + "_" + time_periods + ".csv"))
    breaks = np.load(os.path.join(data_dir, 'breaks' + addendum + '_' + time_periods + '.npy'))

    file = open(os.path.join(data_dir, "vocabulary" + addendum + '_' + time_periods + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocabulary = np.char.replace(voc, '\n', '')

    # Random shuffle of the documents
    num_documents, num_words = counts.shape
    documents = random_state.permutation(num_documents)
    shuffled_speech_info = speech_info.loc[documents]
    shuffled_counts = counts[documents]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")
        summary_time_periods(shuffled_countsSparse, shuffled_countsNonzero,
                             shuffled_speech_info["time"], shuffled_speech_info["author"],
                             fig_dir, breaks, time_periods)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": documents,
          "author_indices": shuffled_speech_info['author'],
          "time_indices": shuffled_speech_info['time'],
          "author_time_indices": shuffled_speech_info['author_time']}, shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, documents, shuffled_speech_info, vocabulary, author_time_info, breaks

def build_input_pipeline_simulation(data_dir,
                                    batch_size,
                                    random_state,
                                    fig_dir=None,
                                    counts_transformation="nothing",
                                    addendum='',
                                    time_periods='sessions'):
    """Load data and build iterator for minibatches.
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts---.npz`, `vocabulary.txt`, 'time_indices.npy' and 'author_indices.npy'.
            --- = addendum
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.

    Returns:
        dataset: Batched document_indices and author_indices.
        permutation: Permutation of the documents.
        shuffled_speech_info: Information about speeches after permutation (author, time, author_time indices, ...).
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.

    """
    # Load data
    counts = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
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
    num_documents, num_words = counts.shape
    documents = random_state.permutation(num_documents)
    shuffled_author_time_info = author_time_info.loc[documents]

    shuffled_counts = counts[documents]
    if counts_transformation == "nothing":
        count_values = shuffled_counts.data
    elif counts_transformation == "log":
        count_values = np.round(np.log(1 + shuffled_counts.data))
    else:
        raise ValueError("Unrecognized counts transformation.")
    shuffled_counts = tf.SparseTensor(
        indices=np.array(shuffled_counts.nonzero()).T,
        values=count_values,
        dense_shape=shuffled_counts.shape)
    if fig_dir is not None:
        shuffled_countsSparse = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=count_values.astype(np.int64),
            dense_shape=shuffled_counts.dense_shape)
        shuffled_countsNonzero = tf.SparseTensor(
            indices=shuffled_counts.indices,
            values=tf.fill(count_values.shape, 1),
            dense_shape=shuffled_counts.dense_shape)  # replace nonzero counts with 1 (indicator of non-zeroness)
        # counts = tf.sparse.to_dense(countsSparse)
        hist_word_counts(shuffled_countsSparse, shuffled_countsNonzero, fig_dir, name_start="orig_")
        summary_time_periods(shuffled_countsSparse, shuffled_countsNonzero,
                             shuffled_author_time_info['time'], shuffled_author_time_info['author'],
                             fig_dir, breaks, time_periods)

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"document_indices": documents,
          "author_indices": shuffled_author_time_info['author'],
          "time_indices": shuffled_author_time_info['time'],
          "author_time_indices": shuffled_author_time_info['author_time']},
         shuffled_counts))
    # dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size)
    # Prefetching to speed up computations
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset, documents, shuffled_author_time_info, vocabulary, author_time_info, breaks

def build_input_pipeline(data_name, data_dir, batch_size, random_state, fig_dir, counts_transformation,
                         addendum, time_periods):
    """Triggers the right build_input_pipeline depending on the current dataset.

    Args:
        data_name: String containing the name of the dataset. Important to choose build_input_pipeline function.
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts---.npz`, `speech_info---.csv`, `author_detailed_info---.csv`, and `vocabulary---.txt`.
            --- = addendum + "_" + time_periods
        batch_size: The batch size to use for training.
        random_state: A NumPy `RandomState` object, used to shuffle the data.
        fig_dir: The directory where to store figures. If None, do not draw any figures
        counts_transformation: A string indicating how to transform the counts. One of "nothing" or "log".
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.

    Returns:
        dataset: Batched document_indices and author_indices.
        permutation: Permutation of the documents.
        shuffled_speech_info: Information about speeches after permutation (author, time, author_time indices, ...).
        vocabulary: A vector of strings with the actual word-terms for word dimension.
        author_time_info: A pandas DataFrame containing author_time-level covariates.
        breaks: A list of dates as integers, e.g. '20150101', that define time-periods.

    """
    if (data_name == "hein-daily") or (data_name == "hein-daily_tv"):
        dataset, permutation, shuffled_speech_info, vocabulary, author_time_info, breaks = build_input_pipeline_hein_daily(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum, time_periods)
    elif data_name[0:10] == "simulation":
        dataset, permutation, shuffled_speech_info, vocabulary, author_time_info, breaks = build_input_pipeline_simulation(
            data_dir, batch_size, random_state, fig_dir, counts_transformation, addendum, time_periods)
    else:
        raise ValueError("Unrecognized dataset name in order to load the data. "
                         "Implement your own 'build_input_pipeline_...' for your dataset.")
    return dataset, permutation, shuffled_speech_info, vocabulary, author_time_info, breaks

