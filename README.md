# Temporal Poisson Factorization
Source code for the paper: 
[Evolving Voices Based on Temporal Poisson Factorisation by Jan Vávra, Bettina Grün, and Paul Hofmarcher (2024)](https://arxiv.org/abs/2410.18486).

## Directories and main files overview:

* [TPF](TPF) - Tensorflow implementation of Temporal Poisson Factorisation (TPF)
* [DPF](DPF) - Tensorflow implementation of Dynamic Poisson Factorisation (DPF)
*  both TPF and DPF have analogous directory structure and files
* [analysis](TPF/analysis) - 
contains the scripts for performing the pre-processing, estimation and post-processing TPF 
([DPF](DPF/analysis))
  * [hein_daily_preprocess_individual_vocabulary_combined](TPF/analysis/hein_daily_preprocess_individual_vocabulary_combined.py) - 
  load `hein-daily` original texts from all sessions 97-114, 
  process them separately with 
  [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
  create a combined vocabulary for all sessions 
  and save into [data/hein-daily/clean](data/hein-daily/clean) 
  * [define_time_periods_cluster](TPF/analysis/define_time_periods_cluster.py) - 
  trigger [define_time_periods](TPF/code/define_time_periods.py) script on computational cluster
  * [tpf_cluster](TPF/analysis/tpf_cluster.py), [dpf_cluster](DPF/analysis/dpf_cluster.py) - 
  estimates TPF or DPF model on computational cluster 
  * [table_ELBO](TPF/analysis/table_ELBO.py), 
  [models_VIC](TPF/analysis/models_VIC.py) 
  create tex tables for different settings of TPF containing ELBO, VIC and other characteristics
* [code](TPF/code) - source code `.py` files defining TPF 
([DPF](DPF/code))
  * [define_time_periods](TPF/code/define_time_periods.py) - run first to define the division into time-periods
  * [poisson_factorization](TPF/code/poisson_factorization.py) - run second for initialization
  * [tpf_model](TPF/code/tpf_model.py), [dpf_model](DPF/code/dpf_model.py) - the main file containing the definition of the TPF and DPF model
  * [check_prior](TPF/code/check_prior.py), 
  [input_pipeline](TPF/code/input_pipeline.py) - prepare inputs for TPF (DPF)
  * [train_step](TPF/code/train_step.py), 
  [information_criteria](TPF/code/information_criteria.py), 
  [var_and_prior_families](TPF/code/var_and_prior_family.py) - used for estimation of TPF (DPF)
  * [plotting_functions](TPF/code/plotting_functions.py) - contains functions to create descriptive plots using the latest TPF (DPF) model parameter values 
    * `create_all_general_descriptive_figures` - for any dataset (histograms, barplots, wordclouds, ...)
    * `create_all_figures_specific_to_data` - specific to each dataset (similarities, evolutions, ...)
  * [plotting_functions](TPF/code/create_latex_tables.py) - contains functions to create basic tex tables: vocabulary and content evolution in time
* [create_slurm_files](TPF/create_slurm_files) - `.py` files to create `.slurm` files for submitting jobs on computational cluster,
these files are specific to the computing environment used and are included for documentation (and inspiration) purposes 
* [data](data) - contains data in separate folders
  * [hein-daily](data/hein-daily) - contains data from [Hein-Daily](https://data.stanford.edu/congress_text) (here only session 114)
    * [orig](data/hein-daily/orig) - original `hein-daily` data for session 114
      * [stopwords](data/hein-daily/orig/stopwords.txt) - 
      list of stopwords used to process the speeches
      * `sss_SpeakerMap.txt`, `byparty_2gram_sss.txt`, `byspeaker_2gram_sss.txt`, `descr_sss.txt`, `speeches_sss.txt` - 
      data from [Hein-Daily](https://data.stanford.edu/congress_text) (not here on GitHub)
      where `sss` stands for the session number 
      * [data_aging_congress.csv](data/hein-daily/orig/data_aging_congress.csv) - 
      congress demographics data 
      [Congress today is older than it’s ever been by Skelley G. (2023)](https://fivethirtyeight.com/features/aging-congress-boomers/)
    * [clean](data/hein-daily/clean) - 
      * string `addendum='_voc_combined'` is added to the end of the file name 
      to specify a different version of pre-processing speeches such as definition of total vocabulary
      * string `time_periods='sessions'` joins after `addendum + '_'` and declares the way how time-periods have been defined
      by, in this case, time-period corresponds to session
      * in the following `_-_-` stands for `addendum + '_' + time_periods`
      * `author_time_info_-_-.csv` - information about the author in each time-period
      * `breaks_-_-.npy` - the dividing points (dates) for the time-periods 
      * `counts_-_-.npz` - 2D document-term matrix in sparse format
      * `speech_info_-_-.csv` - information about each speech including the author and time indices
      * `vocabulary_-.txt` - each row corresponds to one of the terms in the total vocabulary
    * [pf-fits](data/hein-daily/pf-fits) - initial values created by 
    [poisson_factorization](TPF/code/poisson_factorization.py) initial values, `-_-_-` abbreviates `str(num_topics) + addendum + '_' + time_periods`
      * `document_shape_K-_-_-.npy`, `document_rate_K-_-_-.npy` - PF-estimated shapes and rates for thetas (D×K)
      * `topic_shape_K-_-_-.npy`, `topic_rate_K-_-_-.npy` - PF-estimated shapes and rates for betas (K×V)
    * [fits](data/hein-daily/fits), 
    [figs](data/hein-daily/figs), 
    [txts](data/hein-daily/txts), 
    [tabs](data/hein-daily/tabs) - directories for TPF (DPF) estimated parameters and checkpoints, 
    figures, text files (influential speeches) and tables. 
    Contains directories for specific model settings - 
    defined by the `name` in [create_slurm_files](TPF/create_slurm_files) files.
* [err](err) - directory for error files, properly structured
* [out](out) - directory for output files, properly structured
* [slurm](slurm) - directory for `.slurm` files that submit jobs on cluster (very specific to the gpu cluster used for computations, 
user needs to adjust these), properly structured

## Adding a new dataset

First, create a new subdirectory in [data](data) named by `your_data_name` and add all the necessary folders. 
You can supply the same format of the data as in [data/hein-daily/clean](data/hein-daily/clean) 
created analogously to [preprocessing speeches](TPF/analysis/hein_daily_preprocess_individual_vocabulary_combined.py), 
then you only need to replace `data == 'hein-daily'` 
with `data_name in ['hein-daily', 'your_data_name']` in [input_pipeline](TPF/code/input_pipeline.py). 
This will expect the same input files as for `hein-daily`.

Another `data_name` sensitive function is in 
[plotting_functions](TPF/code/plotting_functions.py). 
In case you want to create your own plots, implement and add them for your `data_name`.
You can find some examples of these tweaks already implemented for other than `hein-daily` dataset.

## Pre-processing speech data

We follow similar steps to Keyon Vafa 
[Text-Based Ideal Points (2020)](https://github.com/keyonvafa/tbip),
but adjust them since wee need to merge multiple sessions together.
All is performed by [this](TPF/analysis/hein_daily_preprocess_individual_vocabulary_combined.py) 
`.py` file. First, for each session we:
1. Load and merge speeches with descriptions.
2. Select only speeches by senators given in Senate.
3. Create mapping between names and IDs and create a data frame of author-specific covariates.
4. Use [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
from [scikit-learn](https://scikit-learn.org/1.5/index.html) 
library to eliminate stopwords, select n-gram range (here bigrams only)
and set the minimal and maximal word-in-speech-appearance frequencies (0.001 and 0.3).
5. Get vocabulary specific to each session.
6. Stack all speech data into one dataframe for all sessions.

Obviously, the vocabularies will be different for these sessions, so we need to combined them. 
To do so we:
1. Eliminate senators who have given less than `min_speeches = 1` speeches.
2. Call [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
with the combined vocabulary to the dataframe containing all speeches.
3. Eliminate bigrams spoken by less than `min_authors_per_word = 1` Senators.
4. Recall [CountVectorizer](https://scikit-learn.org/1.5/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) 
with shortened vocabulary.
5. Remove empty speeches without any bigram included (row sums are zero).
6. Save sparse `counts_voc_combined.npz` into [data/hein-daily/clean](data/hein-daily/clean), 
where `_voc_combined` plays the role of `addendum` (it describes the preprocessing approach - many others have been tried before).

Moreover, we need to combine the information about senators into one dataframe and merge it 
with [data_aging_congress.csv](data/hein-daily/orig/data_aging_congress.csv).
The merging is not as straightforward due to inconsistent naming and numbering conventions.
These are resolve case by case. 
More important is to save the information about each of the speeches including 
the author index and the date when the speech has been given,
[speech_info_voc_combined.csv](data/hein-daily/clean/speech_info_voc_combined.csv)

We wanted to make the definition of the time-periods flexible. 
To divide speeches into different time-periods we use 
[define_time_periods](TPF/code/define_time_periods.py).
In paper, we simply use the division by sessions (2 years beginning Jan 01),
however, different break points can be supplied (e.g. important event dates).
We eliminate words in the vocabulary that do not have 
the minimal total appearance count `min_word_count = 1`.
We provide an option to satisfy this condition to each time-period, 
however, that would severely reduce the vocabulary size, and we do not recommend this option.

For DPF the story is a bit different. 
The speeches have to be aggregated into one document per some unit that repeatedly appears 
throughout the time-periods.
(Hundreds of speeches are pasted into one long document that covers all topics.)
This unit could be an author (speaker), but then it does not appear regularly in each time-period.
Then it would be DPF with missing data on authors. 
This is possible with our implementation since we do not strictly work with 
`counts[author, word, time]` but with `counts[author_time, word]`, where not all combinations
of author and time in `author_time` index have to exist. 
Strictly speaking, repetition of `author_time` index is not prohibited either, 
so no aggregation could also be an option, however, this was not tested!
Alternative definition of a unit could be the state or one of the two mandates per state. 
This would strictly lead to DPF where all combinations of unit, word and time exist for hein-daily data.


## Model definition

The implementation is very flexible and allows for many models to be fitted. 
You can 
choose different priors, 
adjust their hyperparameters, 
choose variational family for the autoregressive sequence,
specify tuning parameters for the estimation process,
specify whether the parts of ELBO should be evaluated exactly or approximated with Monte Carlo.
The same holds not only for TPF, but for DPF as well (different optimizer used then in the original paper).


### The choice of the prior distribution

There are two important inputs to TPF and DPF that define the structural choice of the estimated model.
* `prior_choice` - a dictionary that defines which prior structures should be used,
* `prior_hyperparameter` -  a dictionary containing the fixed values of hyperparameters of prior distributions.

Both are be defined from `FLAGS` argument with functions in [check_prior](TPF/code/check_prior.py). 
The choices and their meanings are all enumerated in details in 
[tpf_cluster](TPF/analysis/tpf_cluster.py) and [dpf_cluster](DPF/analysis/dpf_cluster.py). 

* `prior_choice["theta"]` determines whether 
Gamma with fixed hyperparameters (`Gfix`),
Gamma with flexible author-specific rates (`Garte`) or
Gamma with flexible document-specific rates (`Gdrate`)
should be used for `theta` parameter. 
This option is relevant only for TPF, DPF does not have this parameter.
* `prior_choice["delta"]` determines whether delta
unrestricted normal distribution (both prior and variational) (`AR`),
truncated normal distribution to `[-1, 1]` (both prior and variational) (`ART`) or
deterministic distribution (equal to 1, `RW`)
should be used for the autoregressive coefficient `delta`.


Unless you change some parameters the default values for the `FLAGS` will be used to create the dictionary
`prior_hyperparameter`. You can find more details in 
[tpf_cluster](TPF/analysis/tpf_cluster.py), [dpf_cluster](DPF/analysis/dpf_cluster.py) and 
corresponding function `get_and_check_prior_hyperparameter` from `check_prior.py`.



### The choice of the variational family

By default, variational families are chosen to be either Normal or Gamma to allow for CAVI updates.
The variational family for autoregressive coefficient `delta` matches the same choice for the prior (mixing them up is pointless).
Hence, there is only one choice to be made - the variational distribution for the AR sequence:
* `varfam_choice["ar_kv"]` - `ar_kv` stands for AR sequence that has also topic (k) and word (v) dimension
  * `normal` - independent univariate Normal distribution,
  * `MVnormal` - Multivariate Normal over the time indices.
For DPF, `theta` is replaced with `ar_ak` sequence with analogous prior and variational structure.
The original formulation of DPF has the mean directly included in the formula for Poisson rates.
We rather include it as a prior location parameter for the AR sequence.
This yields equivalent DPF model but the structure is now more similar to TPF.
It enabled us to simply modify the code for TPF to obtain DPF. 



### Other model tuning parameters

Moreover, there are other parameters that define the way TPF, DPF is estimated.

* `batch_size`: The batch size.
* `RobMon_exponent`: Exponent in [-1, -0.5) satisfying Robbins-Monroe condition to 
create convex-combinations of old and a new value.
* `exact_reconstruction`: Should the exact reconstruction be computed (True) 
or approximated with Monte Carlo (False)? 
* `exact_log_prior`: Should the exact entropy be computed (True) 
or approximated with Monte Carlo (False)?
* `exact_entropy`: Should the exact entropy be computed (True) 
or approximated with Monte Carlo (False)?
* `geom_approx`: Should the expected ideological term Edkv be approximated by the geometric mean (True) 
or should it be computed exactly (False)?
* `aux_prob_sparse`: Should the counts and auxiliary proportions be worked with 
as with sparse matrices (True/False)? 
From experience, strangely, sparse format does not lead to faster computation with Tensorflow.

The exact entropy is easily obtained from the variational family with the corresponding method.
This cannot be done for reconstruction and log-prior since the variational means of model parameters
have to be inserted at certain places. 
Hence, these two are implemented manually. 
Log-prob truly goes through all documents-specific parameters. 
However, for reconstruction we only have counts for the current batch. 
The contributions to reconstruction then have to be properly rescaled to match the size of the whole dataset.

### Monitoring the estimation process

When STBS is estimated with our combination of CAVI updates and SVI, the values of
ELBO, reconstruction, log_prior and entropy are saved into `model_state.csv` file.
It contains these values for each epoch and each step. 
After reaching the maximal number of epochs, traceplots of these quantities are plotted
in the range of between the starting and last epoch. 

If `computeIC_every > 0`, then ELBO is approximated using all batches of documents (not just a single batch).
On top of that 
[VAIC and VBIC](https://onlinelibrary.wiley.com/doi/full/10.1111/anzs.12063) 
are computed alongside this thorough approximation.
Since this approximation takes some non-negligible computation time,
we recommend to compute it only one a while (after every `computeIC_every` epoch). 
Results of this approximation are saved into `epoch_data.csv` including the computation times
for estimation of the epoch as well as evaluation of the information criteria. 
Similarly, traceplots are automatically created afterwards.

# TODO - the text below needs adjustments

## Post-processing the results

Some post-processing is done already in [estimate_STBS_cluster](analysis/estimate_STBS_cluster.py) where after the 
last epoch many useful plots (including barplots, histograms, wordclouds) are created using 
`create_all_general_descriptive_figures` and `create_all_figures_specific_to_data` from 
[plotting_functions](code/plotting_functions.py).
Then, if `num_top_speeches` > 0 then the most influential speeches are found using 
`find_most_influential_speeches` function from 
[influential_speeches](code/influential_speeches.py). 
For `hein-daily` data we decided to first select a batch (of `batch_size`) of documents
with the highest posterior mean of thetas (`shp / rte`) for each topic separately. Then,
`num_top_speeches` documents with the highest log-likelihood-ratio-like test statistic are 
saved into [txts](data/hein-daily/txts) subdirectory as the most influential speeches for the topic. 

Some post-analysis has to be performed by external `.py` or `.R` scripts.

### Model comparisons

What cannot be done immediately after estimation of an STBS is comparison of two STBS outputs.
First, both different settings have to be estimated. Then, we only load the parameters of interest
instead of the whole STBS structure.

We wish to compare the ideological positions estimated by the classical TBIP and estimated STBS
model and also compute the correlation coefficient between them to demonstrate the similarity.
We plot the results together with under many variations using 
[compare_TBIP_with_STBS](analysis/compare_TBIP_with_STBS.py).
The differences are in the way we create the group means for political parties, 
they can be either taken from the regression coefficients iota or just (weighted) averages
of ideological positions of speakers within the respective groups.
The topics are then ordered (and plotted in this order) by the difference between these two party means. 

Next, we also wish to compare the variability of ideological space between STBS model with
fixed and with topic-specific ideological positions. 
Script [compare_variability_of_ideal_term](analysis/compare_variability_of_ideal_term.py)
provides several aspects, in which these two could be topic-wise compared.
In the end, we decided for the variability induced by both 
ideological corrections eta and ideological positions, named with `eta_ideal_variability`. 
It is computed by multiplying location estimates for eta and ideal and reducing this 3D tensor
along author and word axis into variances for each topic. 
A nice barplots for this comparison including labels for the topics is created by 
[barplot_eta_ideal_variability](R/barplot_eta_ideal_variability.R). 
These labels were assigned after exploration of the wordclouds containing the most relevant terms.

### Regression summary plots using R

Base R plotting devices allow us to be more creative with regression summary plots than python environment.
Therefore, we create these plots (and regression summary tables) with `.R` scripts that can be found in `R` folder.
Functions are tailored for `hein-daily` but with some changes it could be used for other datasets as well.
We provide functions for both regression set-ups (additive and party-interaction) 
that are usable regardless of topic-specificity of ideological positions.
First, we have written a function to plot the results vertically to create thin plots.
In the end, its transposed version proved to be better for both paper and slides. 
