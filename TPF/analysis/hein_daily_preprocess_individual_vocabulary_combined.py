import os
#import setup_utils as utils
import numpy as np
# import tensorflow as tf
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from collections import Counter
from collections import defaultdict

#Unabridged source code originally available at: https://github.com/keyonvafa/tbip
# difference to data quoting=3!!!

#data source : https://data.stanford.edu/congress_text#download-data
#Please download and unzip hein-daily.zip
#data diractory where Hein-Daily database is saved
data_name = 'hein-daily_tv'
data_name = 'hein-daily'

### My laptop
# project_dir = 'C:\\Users\\jvavra\\Documents\\TBIP_colab'
# project_dir = 'C:\\Users\\jvavra\\OneDrive - WU Wien\\Documents\\TBIP_colab'
# source_dir = os.path.join(project_dir, "data\\{}".format(data_name))
project_dir = os.getcwd()
source_dir = os.path.join(project_dir, "data", data_name)
### Cluster
# project_dir = ''
# project_dir = os.getcwd()
# source_dir = project_dir+'/data/'+data_name+'/'

### My laptop
data_dir = os.path.join(source_dir, "orig")
save_dir = os.path.join(source_dir, "clean")



#predefined set of stopwords
# stopwords = set(np.loadtxt(os.path.join(data_dir, "stopwords.txt"),
#                            dtype=str, delimiter="\n")) #to be changed approrpriately wherever stopwords are stored

file = open(os.path.join(data_dir, "stopwords.txt"))
lines = file.readlines()
sw = np.array(lines)
stopwords = np.char.replace(sw, '\n', '')

#stopwords available at: https://github.com/keyonvafa/tbip/blob/master/setup/stopwords/senate_speeches.txt
#to be downloaded and saved to data_dir as defined above

#Parameters

#minimum number of speeches given by a senator
#default value 24
min_speeches = 1
#minimum number of senators using a bigram
#default value 10
min_authors_per_word = 1

#parameters for count vectorizer
min_df = 0.001 #minimum document frequency
max_df = 0.3 #maximum document frequency
stop_words = stopwords.tolist()
ngram_range = (2, 2) #bigrams only
token_pattern = "[a-zA-Z]+"

#Helper function
#source code originally available at: https://github.com/keyonvafa/tbip
#Count number of occurrences of each value in array of non-negative integers
#documentation: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

def bincount_2d(x, weights):
    _, num_topics = weights.shape
    num_cases = np.max(x) + 1
    counts = np.array(
      [np.bincount(x, weights=weights[:, topic], minlength=num_cases)
       for topic in range(num_topics)])
    return counts.T

count_vectorizer = CountVectorizer(min_df=min_df,
                                   max_df=max_df,
                                   stop_words=stop_words,
                                   ngram_range=ngram_range,
                                   token_pattern=token_pattern)

# Empty vocabulary
vocabulary = np.array([])



# creating a complete vocabulary covering all the sessions
speech_data_combined = pd.DataFrame(columns=['speech_id', 'id', 'speakerid', 'date', 'speech', 'index'])
speaker_data_combined = pd.DataFrame(columns=['speakerid', 'name', 'party', 'state', 'gender', 'index'])
for i in range(97, 115):
    if (i < 100):
        stri = '0' + str(i)
    else:
        stri = str(i)

    speeches = pd.read_csv(os.path.join(data_dir, 'speeches_' + stri + '.txt'),
                           encoding="ISO-8859-1",
                           sep="|", quoting=3,
                           on_bad_lines='warn')
    description = pd.read_csv(os.path.join(data_dir, 'descr_' + stri + '.txt'),
                              encoding="ISO-8859-1",
                              sep="|")
    speaker_map = pd.read_csv(os.path.join(data_dir, stri + '_SpeakerMap.txt'),
                              encoding="ISO-8859-1",
                              sep="|")

    merged_df = speeches.merge(description,
                               left_on='speech_id',
                               right_on='speech_id')
    df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

    # Only look at senate speeches.
    # to select speakers with speeches in the senate (includes Senators and House Reps)
    senate_df = df[df['chamber_x'] == 'S']
    # to select ONLY Senators uncomment the next line
    senate_df = senate_df[senate_df['chamber_y'] == 'S'] ##  here 7.2

    # Rename or redefine ids
    senate_df['id'] = senate_df['speakerid'] - i*1000000
    senate_df['name'] = senate_df['firstname'] + ' ' + senate_df['lastname']
    senate_df = senate_df.rename(columns={'state_y': 'state', 'gender_y': 'gender'})

    # Create dataframe containing info about each speaker
    # speaker_data = senate_df.groupby(['id'])[['id', 'name', 'party', 'state', 'gender']].agg(pd.Series.mode)
    speaker_data = senate_df.groupby(['speakerid'])[['speakerid', 'id', 'name', 'party', 'state', 'gender']].agg(pd.Series.mode)
    speaker_data['index'] = range(speaker_data.shape[0])
    speaker_data['index'] += speaker_data_combined.shape[0]
    senate_df['index'] = speaker_data.loc[senate_df['speakerid']]['index'].to_numpy() # + speaker_data_combined.shape[0]

    # Learn initial document term matrix. This is only initial because we use it to
    # identify words to exclude based on author counts.
    counts = count_vectorizer.fit_transform(senate_df['speech'].astype(str))
    session_vocabulary = np.array([k for (k, v) in sorted(count_vectorizer.vocabulary_.items(),
                                                          key=lambda kv: kv[1])])
    vocabulary = np.union1d(vocabulary, session_vocabulary)  # already sorted

    # Keep only important columns and combine with data from previous sessions
    # speech_data = senate_df[['speech_id', 'id', 'index', 'date', 'speech']]
    speech_data = senate_df[['speech_id', 'id', 'speakerid', 'date', 'speech', 'index']]
    speech_data_combined = pd.concat([speech_data_combined, speech_data], ignore_index=True, sort=False)
    speaker_data_combined = pd.concat([speaker_data_combined, speaker_data], ignore_index=True, sort=False)
    print('done for session ' + stri)

# Remove senators who make less than min_speeches
unique_id, id_counts = np.unique(speech_data_combined['id'], return_counts=True)
present_speakers = unique_id[np.where(id_counts >= min_speeches)] # 7 authors have less than 24 speeches
speech_data_combined = speech_data_combined.loc[speech_data_combined['id'].isin(present_speakers)]
speaker_data_combined = speaker_data_combined.loc[speaker_data_combined['id'].isin(present_speakers)]

print(speech_data_combined['index'])
print(speaker_data_combined['index'])

# Define CountVectorizer
count_vectorizer2 = CountVectorizer(vocabulary=vocabulary)

# Learn initial document term matrix. This is only initial because we use it to
# identify words to exclude based on author counts.
counts = count_vectorizer2.fit_transform(speech_data_combined['speech'].astype(str))

# Remove bigrams spoken by less than 10 Senators.
speakers = np.unique(speech_data_combined['id'])
counts_per_author = tf.zeros([0, counts.shape[1]])
for s in speakers:
    sub_counts = counts[speech_data_combined['id'] == s, :]
    sub_word_counts = sub_counts.sum(axis=0)
    counts_per_author = tf.concat([counts_per_author, sub_word_counts], 0)
# counts_per_author = bincount_2d(speech_data_combined['id'], counts.toarray())
author_counts_per_word = np.sum(counts_per_author > 0, axis=0) # minimum of authors using this word is 67
acceptable_words = np.where(author_counts_per_word >= min_authors_per_word)[0]
vocabulary = vocabulary[acceptable_words]

# Fit final document-term matrix with modified vocabulary.
count_vectorizer3 = CountVectorizer(vocabulary=vocabulary)
counts2 = count_vectorizer3.fit_transform(speech_data_combined['speech'].astype(str))
# counts_dense = remove_cooccurring_ngrams(counts, vocabulary) #not required since only bigrams are being considered

# Remove speeches with no words.
existing_speeches = np.where(np.sum(counts2, axis=1) > 0)[0]
counts = counts2[existing_speeches]
speech_data_combined = speech_data_combined.reset_index(drop=True)
speech_data_combined = speech_data_combined.loc[existing_speeches]
speech_data_combined.shape


# np.sum(np.sum(counts, axis=0) == 0) # no word does not appear
# np.sum(np.sum(counts, axis=0) <= 10) # no word does appear less than 10-times
# np.sum(np.sum(counts, axis=1) == 0) # no empty documents
# np.sum(np.sum(counts, axis=1) <= 1)  # there are documents of just one bigram!!


### Adding information from the congress data
us_states = {
    'AL': {'name': 'Alabama', 'region': 'Southeast'},
    'AK': {'name': 'Alaska', 'region': 'West'},
    'AZ': {'name': 'Arizona', 'region': 'West'},
    'AR': {'name': 'Arkansas', 'region': 'South'},
    'CA': {'name': 'California', 'region': 'West'},
    'CO': {'name': 'Colorado', 'region': 'West'},
    'CT': {'name': 'Connecticut', 'region': 'Northeast'},
    'DE': {'name': 'Delaware', 'region': 'Northeast'},
    'FL': {'name': 'Florida', 'region': 'Southeast'},
    'GA': {'name': 'Georgia', 'region': 'Southeast'},
    'HI': {'name': 'Hawaii', 'region': 'West'},
    'ID': {'name': 'Idaho', 'region': 'West'},
    'IL': {'name': 'Illinois', 'region': 'Midwest'},
    'IN': {'name': 'Indiana', 'region': 'Midwest'},
    'IA': {'name': 'Iowa', 'region': 'Midwest'},
    'KS': {'name': 'Kansas', 'region': 'Midwest'},
    'KY': {'name': 'Kentucky', 'region': 'Southeast'},
    'LA': {'name': 'Louisiana', 'region': 'South'},
    'ME': {'name': 'Maine', 'region': 'Northeast'},
    'MD': {'name': 'Maryland', 'region': 'Northeast'},
    'MA': {'name': 'Massachusetts', 'region': 'Northeast'},
    'MI': {'name': 'Michigan', 'region': 'Midwest'},
    'MN': {'name': 'Minnesota', 'region': 'Midwest'},
    'MS': {'name': 'Mississippi', 'region': 'South'},
    'MO': {'name': 'Missouri', 'region': 'Midwest'},
    'MT': {'name': 'Montana', 'region': 'West'},
    'NE': {'name': 'Nebraska', 'region': 'Midwest'},
    'NV': {'name': 'Nevada', 'region': 'West'},
    'NH': {'name': 'New Hampshire', 'region': 'Northeast'},
    'NJ': {'name': 'New Jersey', 'region': 'Northeast'},
    'NM': {'name': 'New Mexico', 'region': 'West'},
    'NY': {'name': 'New York', 'region': 'Northeast'},
    'NC': {'name': 'North Carolina', 'region': 'Southeast'},
    'ND': {'name': 'North Dakota', 'region': 'Midwest'},
    'OH': {'name': 'Ohio', 'region': 'Midwest'},
    'OK': {'name': 'Oklahoma', 'region': 'South'},
    'OR': {'name': 'Oregon', 'region': 'West'},
    'PA': {'name': 'Pennsylvania', 'region': 'Northeast'},
    'RI': {'name': 'Rhode Island', 'region': 'Northeast'},
    'SC': {'name': 'South Carolina', 'region': 'Southeast'},
    'SD': {'name': 'South Dakota', 'region': 'Midwest'},
    'TN': {'name': 'Tennessee', 'region': 'Southeast'},
    'TX': {'name': 'Texas', 'region': 'South'},
    'UT': {'name': 'Utah', 'region': 'West'},
    'VT': {'name': 'Vermont', 'region': 'Northeast'},
    'VA': {'name': 'Virginia', 'region': 'Southeast'},
    'WA': {'name': 'Washington', 'region': 'West'},
    'WV': {'name': 'West Virginia', 'region': 'Southeast'},
    'WI': {'name': 'Wisconsin', 'region': 'Midwest'},
    'WY': {'name': 'Wyoming', 'region': 'West'}
}


def transform_name(name):
    commacount = name.count(', ')
    if commacount == 1:
        last_name, first_name = name.split(', ')
    elif commacount == 2:
        last_name, first_name, other = name.split(', ')
    else:
        last_name, first_name, other1, other2 = name.split(', ')
    #return f'{first_name.upper()} {last_name.upper()}'
    return f'{first_name.split()[0].upper()} {last_name.upper()}'

def get_surname(name):
    # first_name, last_name = name.split(' ')
    splitted_name = name.split(' ')
    last_name = splitted_name[-1].replace("'", "")
    return f'{last_name.split()[0].upper()}'

def get_n_surname(name):
    # Get initial from the first name and paste it with surname together
    splitted_name = name.split(' ')
    last_name = splitted_name[-1].replace("'", "")
    initial_name = splitted_name[0][0:1].upper()
    return f'{initial_name + "_" + last_name.split()[0].upper()}'



speaker_data_combined["region"] = np.array([us_states[state]["region"] for state in speaker_data_combined["state"]])

congress = pd.read_csv(os.path.join(data_dir, "data_aging_congress.csv"))
congress = congress[congress['congress'].isin(range(97, 115))]
congress['name'] = congress['bioname'].apply(transform_name)
congress['surname'] = congress['name'].apply(get_surname)
# congress.shape
# congress
senators = congress[congress['chamber'] == 'Senate']
# senators.shape

### Merging the datasets
print("Define names, surnames, unique ids.")
speaker_data_combined['surname'] = speaker_data_combined['name'].apply(get_surname)

# Corrections of author_info to match senators
id = speaker_data_combined.index[speaker_data_combined.name == 'THAD COCHRAN'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'WILLIAM COCHRAN'

senators = pd.concat([senators, congress[(congress.surname == 'BROYHILL') & (congress.congress == 99)]])

sur = 'MCCONNELL'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'MITCH MCCONNELL'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Addison Mitchell (Mitch) MCCONNELL'

sur = 'GRAMM'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'PHIL GRAMM'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'William Philip (Phil) GRAMM'

sur = 'GRAHAM'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'BOB GRAHAM'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Daniel Robert (Bob) GRAHAM'

sur = 'SANFORD'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'JAMES SANFORD'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'James Terry SANFORD'
id = senators.index[senators.bioname == 'SANFORD, (James) Terry'].tolist()
if len(id) > 0:
    senators['name'][id] = 'James Terry SANFORD'

senators = pd.concat([senators, congress[(congress.surname == 'AKAKA') & (congress.congress == 101)]])
senators = pd.concat([senators, congress[(congress.surname == 'COATS') & (congress.congress == 101)]])

sur = 'LOTT'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'TRENT LOTT'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Chester Trent LOTT'

sur = 'WYDEN'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'RON WYDEN'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Ronald Lee WYDEN'

sur = 'ROBERTS'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'PAT ROBERTS'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Charles Patrick (Pat) ROBERTS'

sur = 'ALLARD'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'WAYNE ALLARD'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'A. Wayne ALLARD'

sur = 'NELSON'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'BEN NELSON'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Earl Benjamin (Ben) NELSON'
id = speaker_data_combined.index[speaker_data_combined.name == 'BILL NELSON'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Clarence William (Bill) NELSON'

senators = pd.concat([senators, congress[(congress.surname == 'MENENDEZ') & (congress.congress == 109)]])

sur = 'CORKER'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'BOB CORKER'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Robert (Bob) CORKER'

senators = pd.concat([senators, congress[(congress.surname == 'WICKER') & (congress.congress == 110)]])

sur = 'HEITKAMP'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'HEIDI HEITKAMP'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Mary Kathryn (Heidi) HEITKAMP'

sur = 'CRUZ'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'TED CRUZ'].tolist()
if len(id) > 0:
    speaker_data_combined['name'][id] = 'Rafael Edward (Ted) CRUZ'

speaker_data_combined['n_sur'] = speaker_data_combined['name'].apply(get_n_surname)

congress[congress.surname == 'QUAYLE']
senators[senators.surname == 'QUAYLE']
speaker_data_combined[speaker_data_combined.surname == 'QUAYLE']
quayle = congress[(congress.surname == 'QUAYLE') & (congress.congress == 100)]
quayle['congress'] = 101
as_list = quayle.index.tolist()
as_list[0] = 21508
quayle.index = as_list
senators = pd.concat([senators, quayle])

congress[congress.name == 'ROBERT SMITH']
speaker_data_combined[speaker_data_combined.n_sur == 'R_SMITH']
senators = pd.concat([senators, congress.loc[[24586]]])

congress[congress.surname == 'DORGAN']
speaker_data_combined[speaker_data_combined.surname == 'DORGAN']
senators = pd.concat([senators, congress[(congress.surname == 'DORGAN') & (congress.congress == 102)]])

congress[congress.surname == 'GORE']
speaker_data_combined[speaker_data_combined.surname == 'GORE']
gore = congress[(congress.surname == 'GORE') & (congress.congress == 102)]
gore['congress'] = 103
as_list = gore.index.tolist()
as_list[0] = 10200
gore.index = as_list
senators = pd.concat([senators, gore])

congress[congress.surname == 'INHOFE']
speaker_data_combined[speaker_data_combined.surname == 'INHOFE']
senators = pd.concat([senators, congress[(congress.surname == 'INHOFE') & (congress.congress == 103)]])

congress[congress.surname == 'WYDEN']
speaker_data_combined[speaker_data_combined.surname == 'WYDEN']
senators = pd.concat([senators, congress[(congress.surname == 'WYDEN') & (congress.congress == 104)]])

senators.index
np.unique(senators.index).shape

### Define first letter + surname as an identifier
senators['n_sur'] = senators['name'].apply(get_n_surname)
print("Names, surnames, unique ids defined.")

# deal with duplicated keys:
sur = 'BROWN'
speaker_data_combined[speaker_data_combined.surname == sur]
senators["bioname"][senators.surname == sur]
congress["bioname"][congress.surname == sur]
id = speaker_data_combined.index[speaker_data_combined.name == 'SCOTT BROWN'].tolist()
if len(id) > 0:
    speaker_data_combined['n_sur'][id] = 'SC_BROWN'
id = speaker_data_combined.index[speaker_data_combined.name == 'SHERROD BROWN'].tolist()
if len(id) > 0:
    speaker_data_combined['n_sur'][id] = 'SH_BROWN'

id = senators.index[senators.bioname == 'BROWN, Scott P.'].tolist()
if len(id) > 0:
    senators['n_sur'][id] = 'SC_BROWN'
id = senators.index[senators.bioname == 'BROWN, Sherrod'].tolist()
if len(id) > 0:
    senators['n_sur'][id] = 'SH_BROWN'

# senators['name'].to_numpy()
# speaker_data_combined['name'].to_numpy()
# senators['surname'].to_numpy()
# speaker_data_combined['surname'].to_numpy()

# matching on name --> some are not matched properly
# merged = pd.merge(speaker_data_combined, senators, on='name', how='outer', indicator=True)
# merged.shape
# mismatched = merged[merged['_merge'] != 'both']
# mismatched

# matching on surname --> but what if some senators have the same surname?
# merged = pd.merge(speaker_data_combined, senators, on=['surname'], how='left', indicator=True)
# merged.shape
# merged.columns
# mismatched = merged[merged['_merge'] != 'both']
# mismatched.name
# merged[merged.surname_x.isin(["BYRD"])]
# speaker_data_combined[speaker_data_combined.surname.isin(["BYRD"])]
# senators[senators.surname.isin(["BYRD"])]

# matching on first letter of a name combined with surname: n_sur
# add session number after n_sur
speaker_data_combined['session'] = np.floor(speaker_data_combined['speakerid']/1000000).astype(str)
speaker_data_combined['n_sur_session'] = speaker_data_combined['n_sur'] + '_' + speaker_data_combined['session']
senators['n_sur_session'] = senators['n_sur'] + '_' + senators['congress'].astype(str)
merged = pd.merge(speaker_data_combined, senators, on=['n_sur_session'], how='left', indicator=True)
# Still has some issues (somebody in speaker_data_combined uses his/her second name instead of the first).
# i=98, William Thad, COCHRANE goes by Thad COCHRANE in speaker_data_combined not as William COCHRANE. --> creates NaNs
# merged.shape
# merged.columns
print("speaker_data_combined a senators merged.")
# Create some new columns:
bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 10), (10, 100)])
merged['exper_cong'] = pd.cut(merged['cmltv_cong'], right=True,
                              bins=bins)  # , labels=['Beginner', 'Advanced', 'Expert'])

bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 5), (5, 100)])
merged['exper_chamber'] = pd.cut(merged['cmltv_chamber'], right=True,
                                 bins=bins)  # , labels=['Beginner', 'Advanced', 'Expert'])
print("Exper_cong and exper_chamber defined.")

### Saving
mergedsub = merged[['speakerid', 'id', 'name_x', 'name_y', 'surname_x', 'surname_y', 'n_sur_session', 'session',
                    'gender', 'party', 'state', 'region',
                    'age_years', 'generation', 'cmltv_cong', 'cmltv_chamber', 'exper_cong', 'exper_chamber']]
print("Subset of merged dataset is saved.")



speech_data_combined = speech_data_combined[['speech_id', 'id', 'index', 'date']]

# saving input matrices for TVPF
sparse.save_npz(os.path.join(save_dir, 'counts_voc_combined.npz'), sparse.csr_matrix(counts).astype(np.float32))
np.savetxt(os.path.join(save_dir, 'vocabulary_voc_combined.txt'), vocabulary, fmt="%s")
speaker_data_combined.to_csv(os.path.join(save_dir, 'author_info_voc_combined.csv'), index=False)
mergedsub.to_csv(os.path.join(save_dir, 'author_detailed_info_voc_combined.csv'), index=False)
speech_data_combined.to_csv(os.path.join(save_dir, 'speech_info_voc_combined.csv'), index=False)
