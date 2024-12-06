# Import global packages
import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import tensorflow as tf
import datetime

def get_closest_session_for_time(dsbreaks, dbreaks, sessions):
    num_times = len(dbreaks) - 1
    num_sessions = len(dsbreaks) - 1

    middbreaks = [dbreaks[t] + 0.5 * (dbreaks[t + 1] - dbreaks[t]) for t in range(num_times)]
    # midbreaks = [middbreaks[t].year * 10000 + middbreaks[t].month * 100 + middbreaks[t].day for t in range(num_times)]

    middsbreaks = [dsbreaks[s] + 0.5 * (dsbreaks[s + 1] - dsbreaks[s]) for s in range(num_sessions)]
    # midsbreaks = [middsbreaks[s].year * 10000 + middsbreaks[s].month * 100 + middsbreaks[s].day for s in range(num_sessions)]

    # session closest to a time-period
    closest_session_for_time = [
        sessions[np.argmin([abs(middbreaks[t] - middsbreaks[s]).total_seconds() for s in sessions])] for t in range(num_times)]
    return closest_session_for_time

def get_time_weights(session_breaks, breaks):
    dsbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in session_breaks]
    dbreaks = [datetime.datetime(year=int(str(b)[0:4]), month=int(str(b)[4:6]), day=int(str(b)[6:8])) for b in breaks]
    num_times = len(breaks)-1
    num_sessions = len(session_breaks)-1
    weights = np.zeros([num_times, num_sessions])
    for t in range(num_times):
        for s in range(num_sessions):
            low = max(dsbreaks[s], dbreaks[t])
            upp = min(dsbreaks[s+1], dbreaks[t+1])
            dist = max(upp - low, datetime.timedelta(0))
            weights[t, s] = dist / (dsbreaks[s+1] - dsbreaks[s])

    return weights, dsbreaks, dbreaks


def define_time_periods_hein_daily(data_dir, addendum='', time_periods='sessions', min_word_count=1,
                                   min_word_count_per_time=False):
    """Define the time-periods, trim counts, vocabulary and define author-time specific covariates.
    Save the changed files into the same name + addendum + "_" + time_periods
    Specific to hein-daily data.

    Args:
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `speech_info.csv`, `author_detailed_info.csv`, and `vocabulary.txt`.
            All also contain 'addendum' at the name of the file.
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.
        min_word_count: An integer giving the minimal number of word usage to be included in the dataset.
        min_word_count_per_time: A boolean declaring whether the minimal word count
            should be satisfied for each and every time period separately (True)
            or for all time periods combined.
    """
    session_years = 2*tf.range(990, 1009) + 1
    session_breaks = session_years.numpy()*10000 + 101
    if time_periods == 'sessions':
        breaks = session_breaks
    elif time_periods == 'years':
        years = tf.range(1981, 2018)
        breaks = years.numpy() * 10000 + 101
    elif time_periods == 'test':
        breaks = [20110101, 20120101, 20130701, 20150701, 20170101]
    else:
        raise ValueError("Unrecognized choice of time-periods, "
                         "update 'build_input_pipeline_hein_daily' for new type of time_periods!")
    num_times = len(breaks)-1

    counts_full = sparse.load_npz(os.path.join(data_dir, "counts" + addendum + ".npz"))
    speech_info_full = pd.read_csv(os.path.join(data_dir, "speech_info" + addendum + ".csv"))

    # Create subset of counts corresponding to the examined time-span (between first and last break)
    keep = (speech_info_full['date'] >= breaks[0]) & (speech_info_full['date'] < breaks[-1])
    counts = counts_full[keep]
    speech_info = speech_info_full[keep].reset_index(drop=True)
    # Eliminate words that appear at least min_word_count times ...
    speech_info['time'] = pd.cut(speech_info['date'], bins=breaks, right=False, labels=False)
    if min_word_count_per_time:
        # ... consider word counts for each time period separately
        word_counts = tf.zeros([0, counts.shape[1]])
        for t in range(num_times):
            sub_counts = counts[speech_info['time'] == t, :]
            sub_word_counts = sub_counts.sum(axis=0)
            word_counts = tf.concat([word_counts, sub_word_counts], 0)
        # word_counts = tf.math.unsorted_segment_sum(counts.todense(), speech_info['time'], num_segments=num_times)
        zeros, words = np.where(tf.reduce_all(word_counts >= min_word_count, axis=0))
    else:
        # ... consider word counts across all time periods combined
        word_counts = counts.sum(axis=0)
        zeros, words = np.where(word_counts >= min_word_count)
    counts = counts[:, words]

    # Load and prune vocabulary
    file = open(os.path.join(data_dir, "vocabulary" + addendum + ".txt"))
    lines = file.readlines()
    voc = np.array(lines)
    vocab = np.char.replace(voc, '\n', '')
    vocabulary = vocab[words]

    # Create index of author, time
    authors = sorted(speech_info['id'].unique())
    authors_index = pd.DataFrame(range(len(authors)), index=authors, columns=['ind'])
    speech_info['author'] = authors_index.loc[speech_info['id']].reset_index(drop=True)['ind'].to_numpy()
    speech_info['time'] = pd.cut(speech_info['date'], bins=breaks, right=False, labels=False)

    # Create author_time_info data_frame and author_time index
    weights, dsbreaks, dbreaks = get_time_weights(session_breaks, breaks)
    author_info = pd.read_csv(os.path.join(data_dir, "author_detailed_info" + addendum + ".csv"))
    author_time_info = pd.DataFrame()
    for ia in range(len(authors)):
        a = authors[ia]
        adata = author_info[author_info['id'] == a]
        asessions = adata['session'].to_numpy() - 97
        # aweights = weights[:, asessions]
        closest_session_for_time = get_closest_session_for_time(dsbreaks, dbreaks, asessions)
        for t in range(num_times):
            # todo Implement a weighted solution, for now we use the information from the closest session to this time t
            #  closest session = available session, midday of which is closest to the midday of the current time-period
            # w = aweights[t, :]
            # if sum(w) > 0:
            #     # we have some information on this author in this time-period --> add a row to author_time_info
            # else:
            #     # we do NOT have any information on this author in this time-period
            #     # use the closest information we have
            row = adata[adata['session'] == closest_session_for_time[t]+97]
            row['author'] = ia
            row['time'] = t
            author_time_info = pd.concat([author_time_info, row])
    author_time_info = author_time_info.reset_index(drop=True)
    speech_info['author_time'] = speech_info['author']*num_times + speech_info['time']

    # Save counts, vocabulary, speech_info, author_time_info, breaks
    sparse.save_npz(os.path.join(data_dir, 'counts'+addendum+'_'+time_periods+'.npz'),
                    sparse.csr_matrix(counts).astype(np.float32))
    np.savetxt(os.path.join(data_dir, 'vocabulary'+addendum+'_'+time_periods+'.txt'), vocabulary, fmt="%s")
    speech_info.to_csv(os.path.join(data_dir, 'speech_info'+addendum+'_'+time_periods+'.csv'), index=False)
    author_time_info.to_csv(os.path.join(data_dir, 'author_time_info'+addendum+'_'+time_periods+'.csv'), index=False)
    np.save(os.path.join(data_dir, 'breaks'+addendum+'_'+time_periods), breaks)



def define_time_periods(data_name, data_dir, addendum, time_periods, min_word_count, min_word_count_per_time):
    """Triggers the right define_time_periods depending on the current dataset.

    Args:
        data_name: String containing the name of the dataset. Important to choose build_input_pipeline function.
        data_dir: The directory where the data is located. There must be four
            files inside the rep: `counts.npz`, `speech_info.csv`, `author_detailed_info.csv`, and `vocabulary.txt`.
            All also contain 'addendum' at the name of the file.
        addendum: A string to be added to the end of files.
        time_periods: String that defines the set of dates that break the data into time-periods.
        min_word_count: an integer giving the minimal number of word usage to be included in the dataset.
        min_word_count_per_time: A boolean declaring whether the minimal word count
            should be satisfied for each and every time period separately (True)
            or for all time periods combined.
    """
    if data_name == "hein-daily":
        define_time_periods_hein_daily(data_dir, addendum, time_periods, min_word_count, min_word_count_per_time)
    else:
        raise ValueError("Unrecognized dataset name in order to load the data. "
                         "Implement your own 'define_time_periods_...' for your dataset.")


