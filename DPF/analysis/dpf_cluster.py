## Import global packages
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from absl import app
from absl import flags


## Import local modules
# Necessary to import local modules from specific directory
# todo Can the appropriate directory containing STBIP be added to sys list of paths by some other means?
import sys
# first directory here is the one where analysis_cluster is located
# this is ./DPF/analysis/dpf_cluster
# So add ./ to the list so that it can find ./DPF.code....
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from DPF.code.check_prior import get_and_check_prior_choice, get_and_check_varfam_choice, \
    get_and_check_prior_hyperparameter
from DPF.code.input_pipeline import build_input_pipeline
from DPF.code.dpf_model import DPF
from DPF.code.train_step import train_step
from DPF.code.information_criteria import get_variational_information_criteria
from DPF.code.plotting_functions import create_all_general_descriptive_figures, create_all_figures_specific_to_data
from DPF.code.create_latex_tables import create_latex_tables

## FLAGS
flags.DEFINE_string("data_name", default="hein-daily", help="Data source being used.")
flags.DEFINE_string("addendum", default="", help="String to be added to data name."
                                                 "For example, for senate speeches the session number.")
flags.DEFINE_string("time_periods", default="sessions",
                    help="String that defines the set of dates that break the data into time-periods.")
flags.DEFINE_enum("author", default="author", enum_values=['author', 'state', 'state2'],
                    help="String that defines what is the label of author to be used:"
                         "author = each and every individual speaker by id,"
                         "state = state label is used instead of author id --> 50 authors,"
                         "state2 = each state has 2 senators in each session --> 100 authors.")
flags.DEFINE_string("checkpoint_name", default="checkpoint_name", help="Directory for saving checkpoints.")
flags.DEFINE_boolean("load_checkpoint", default=True,
                     help="Should checkpoints be loaded? If not, then the existed are overwritten.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "binary", "sqrt", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_enum("pre_initialize_parameters", default="No",
                  enum_values=["No", "PF", "sim_true", "TPF"],
                  help="Whether to use pre-initialized document and topic intensities: "
                       "No = do not use any initial values and start from scratch,"
                       "sim_true = initialize with the true values from simulation,"
                       "PF = initialize by values found by Poisson Factorization,"
                       "TPF = initialize by values from previous iterations of TPF.")
flags.DEFINE_integer("seed", default=123456789, help="Random seed to be used.")
flags.DEFINE_integer("num_epochs", default=1000, help="Number of epochs to perform.")
flags.DEFINE_integer("save_every", default=5, help="How often should we save checkpoints?")
flags.DEFINE_integer("computeIC_every", default=0,
                     help="How often should we compute more precise approximation of the ELBO "
                          "and compute variational Information Criteria as a by-product?"
                          "If <=0, then do not compute at all.")
flags.DEFINE_integer("max_steps", default=1000000, help="Number of training steps to run.")
flags.DEFINE_integer("print_steps", default=500, help="Number of steps to print and save results.")

# Method of estimation flags:
flags.DEFINE_integer("num_topics", default=30, help="Number of topics.")
flags.DEFINE_float("learning_rate", default=0.01, help="Adam learning rate.")
flags.DEFINE_float("RobMon_exponent", default=-0.51, lower_bound=-1, upper_bound=-0.5,
                   help="Robbins-Monro algorithm for stochastic gradient optimization requires the coefficients a_n "
                        "for convex combination of new update and old value to satisfy the following conditions:"
                        "(a) sum a_n = infty,"
                        "(b) sum a_n^2 < infty."
                        "We consider a_n = n^RobMon_exponent. Then,"
                        "(a) holds if RobMon_exponent  >= -1 and"
                        "(b) holds if 2*RobMon_exponent < -1,"
                        "which yields the restriction RobMon_exponent in [-1, -0.5). "
                        "It holds: n^{-1} < n^{RobMon_exponent} < n^{-0.5}."
                        "Therefore, value close to -1 puts higher strength on the old value."
                        "Value close to -0.5 prefers the newly learned direction. "
                        "For example, n=50 --> "
                        "a_n = 0.020 if exponent=-1,"
                        "a_n = 0.065 if exponent=-0.7,"
                        "a_n = 0.141 if exponent=-0.5")
flags.DEFINE_integer("batch_size", default=10, help="Batch size.")
flags.DEFINE_integer("num_samples", default=1, help="Number of samples to use for ELBO approximation.")
flags.DEFINE_integer("num_samplesIC", default=1,
                     help="Number of samples to use for detailed ELBO approximation + IC evaluation.")
flags.DEFINE_boolean("exact_entropy", default=False,
                     help="If True, entropy is calculated precisely instead of Monte Carlo approximation. ")
flags.DEFINE_boolean("exact_log_prior", default=False,
                     help="If True, log_prior is calculated precisely instead of Monte Carlo approximation. ")
flags.DEFINE_boolean("exact_reconstruction", default=False,
                     help="If True, reconstruction is calculated precisely instead of Monte Carlo approximation. ")


# Prior structure of the model setting:
flags.DEFINE_enum("delta", default="AR", enum_values=["AR", "ART", "RW"],
                  help="How should time-specific parameters depend on previous values:"
                       "AR=AR(1) process: mean = delta*previous with delta unrestricted,"
                       "ART= same as AR, but delta has Truncated [-1,1] normal distribution (also for var. family),"
                       "RW=random walk: mean = previous, that is delta=1 by default.")

# Variational family to be used for parameters
flags.DEFINE_enum("varfam_ar_ak", default="MVnormal", enum_values=["MVnormal", "normal"],
                  help="MVnormal= Multivariate normal distribution over the last (time) dimension, "
                       "    variational family captures covariances between different time points."
                       "normal= Univariate normal distributions, "
                       "    observations across different times are independent in the variational family.")
flags.DEFINE_enum("varfam_ar_kv", default="MVnormal", enum_values=["MVnormal", "normal"],
                  help="MVnormal= Multivariate normal distribution over the last (time) dimension, "
                       "    variational family captures covariances between different time points."
                       "normal= Univariate normal distributions, "
                       "    observations across different times are independent in the variational family.")

# Model prior hyperparameters:
flags.DEFINE_float("ar_ak_mean_loc", default=0.0, help="AR_ak(1) mean prior location")
flags.DEFINE_float("ar_ak_mean_scl", default=10.0, help="AR_ak(1) mean prior scale")
flags.DEFINE_float("ar_kv_mean_loc", default=0.0, help="AR_kv(1) mean prior location")
flags.DEFINE_float("ar_kv_mean_scl", default=10.0, help="AR_kv(1) mean prior scale")

flags.DEFINE_float("ar_ak_delta_loc", default=0.5, help="AR_ak(1) coefficient prior location")
flags.DEFINE_float("ar_ak_delta_scl", default=0.2, help="AR_ak(1) coefficient prior scale")
flags.DEFINE_float("ar_kv_delta_loc", default=0.5, help="AR_kv(1) coefficient prior location")
flags.DEFINE_float("ar_kv_delta_scl", default=0.2, help="AR_kv(1) coefficient prior scale")

flags.DEFINE_float("ar_ak_prec_shp", default=0.3, help="AR_ak(1) precision prior shape")
flags.DEFINE_float("ar_ak_prec_rte", default=0.3, help="AR_ak(1) precision prior rate")
flags.DEFINE_float("ar_kv_prec_shp", default=0.3, help="AR_kv(1) precision prior shape")
flags.DEFINE_float("ar_kv_prec_rte", default=0.3, help="AR_kv(1) precision prior rate")

FLAGS = flags.FLAGS


### Used memory
def memory():
    import os
    import psutil
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse)


def main(argv):
    del argv
    # Hyperparameters for triple gamma prior for eta adjustments:
    prior_choice = get_and_check_prior_choice(FLAGS)
    varfam_choice = get_and_check_varfam_choice(FLAGS)
    prior_hyperparameter = get_and_check_prior_hyperparameter(FLAGS)

    tf.random.set_seed(FLAGS.seed)
    random_state = np.random.RandomState(FLAGS.seed)

    ### Setting up directories
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data', FLAGS.data_name)
    fit_dir = os.path.join(source_dir, 'pf-fits')
    dpf_fit_dir = os.path.join(fit_dir, FLAGS.checkpoint_name)
    data_dir = os.path.join(source_dir, 'clean')
    save_dir = os.path.join(source_dir, 'fits', FLAGS.checkpoint_name)
    fig_dir = os.path.join(source_dir, 'figs', FLAGS.checkpoint_name)
    tab_dir = os.path.join(source_dir, 'tabs', FLAGS.checkpoint_name)
    txt_dir = os.path.join(source_dir, 'txts', FLAGS.checkpoint_name)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(fit_dir):
        os.mkdir(fit_dir)
    if not os.path.exists(dpf_fit_dir):
        os.mkdir(dpf_fit_dir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    if not os.path.exists(tab_dir):
        os.mkdir(tab_dir)
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)

    if os.path.exists(os.path.join(fig_dir, 'orig_log_hist_counts.png')):
        # If the descriptive histograms of the document-term matrix exist
        # --> do not plot them again, because it takes some time.
        use_fig_dir = None
    else:
        # Create the descriptive histograms of the document-term matrix
        # when loading the data within 'build_input_pipeline()'.
        use_fig_dir = fig_dir

    print("DPF analysis of " + FLAGS.data_name + FLAGS.addendum + " dataset with time_periods: " + FLAGS.time_periods)

    ### Import clean datasets
    (dataset, permutation, vocabulary, author_time_info, breaks, inits) = build_input_pipeline(
        FLAGS.data_name, data_dir, FLAGS.batch_size, random_state, use_fig_dir,
        os.path.join(fit_dir, "replace_K" + str(FLAGS.num_topics) + FLAGS.addendum + "_" + FLAGS.time_periods + ".npy"),
        FLAGS.counts_transformation, FLAGS.pre_initialize_parameters,
        FLAGS.addendum, FLAGS.time_periods, FLAGS.author
    )
    num_author_times = len(permutation)
    num_authors = len(author_time_info['author'].unique())
    num_words = len(vocabulary)
    num_times = len(breaks)-1

    print("Number of plausible authors-time combinations: " + str(num_author_times))
    print("Number of authors: " + str(num_authors))
    print("Number of words: " + str(num_words))
    print("Number of time-periods: " + str(num_times))

    ### Model initialization
    optim = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    model = DPF(num_authors,
                num_author_times,
                FLAGS.num_topics,
                num_words,
                num_times,
                FLAGS.num_samples,
                inits=inits,
                prior_hyperparameter=prior_hyperparameter,
                prior_choice=prior_choice,
                varfam_choice=varfam_choice,
                batch_size=FLAGS.batch_size,
                RobMon_exponent=FLAGS.RobMon_exponent,
                exact_entropy=FLAGS.exact_entropy,
                exact_log_prior=FLAGS.exact_log_prior,
                exact_reconstruction=FLAGS.exact_reconstruction
                )

    ### Model training preparation
    # Add start epoch so checkpoint state is saved.
    model.start_epoch = tf.Variable(-1, trainable=False)

    if os.path.exists(checkpoint_dir) and FLAGS.load_checkpoint:
        pass
    else:
        # If we're not loading a checkpoint, overwrite the existing directory with saved results.
        if os.path.exists(save_dir):
            print("Deleting old log directory at {}".format(save_dir))
            tf.io.gfile.rmtree(save_dir)

    # We keep track of the seed to make sure the random number state is the same whether or not we load the model.
    _, seed = tfp.random.split_seed(FLAGS.seed)
    checkpoint = tf.train.Checkpoint(optimizer=optim,
                                     net=model,
                                     seed=tf.Variable(seed))
    manager = tf.train.CheckpointManager(checkpoint,
                                         checkpoint_dir,
                                         max_to_keep=1)

    checkpoint.restore(manager.latest_checkpoint)

    if manager.latest_checkpoint:
        # Load from saved checkpoint, keeping track of the seed.
        seed = checkpoint.seed
        # Since the dataset shuffles at every epoch and we'd like the runs to be
        # identical whether or not we load a checkpoint, we need to make sure the
        # dataset state is consistent. This is a hack but it will do for now.
        # Here's the issue: https://github.com/tensorflow/tensorflow/issues/48178
        for e in range(model.start_epoch.numpy() + 1):
            _ = iter(dataset)
            if FLAGS.computeIC_every > 0:
                if e % FLAGS.computeIC_every == 0:
                    _ = iter(dataset)

        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    summary_writer = tf.summary.create_file_writer(save_dir)
    summary_writer.set_as_default()
    start_time = time.time()
    start_epoch = model.start_epoch.numpy()

    if os.path.exists(os.path.join(save_dir, 'model_state.csv')):
        model_state = pd.read_csv(os.path.join(save_dir, 'model_state.csv'), index_col=False)
        # in case there are saved more epochs than the last saved checkpoint (can happen if save_every > 1)
        model_state = model_state[model_state['epoch'] <= start_epoch]
    else:
        model_state = pd.DataFrame({'ELBO': [], 'entropy': [], 'log_prior': [], 'reconstruction': [],
                                    'epoch': [], 'batch': [], 'step': []})

    batches_per_epoch = len(dataset)

    if FLAGS.computeIC_every > 0:
        if os.path.exists(os.path.join(save_dir, 'epoch_data.csv')):
            epoch_data = pd.read_csv(os.path.join(save_dir, 'epoch_data.csv'), index_col=False)
            for y in ["sec/epoch", "sec/ELBO"]:
                if y not in epoch_data.keys():
                    epoch_data[y] = 0.0
        else:
            epoch_data = pd.DataFrame({'ELBO': [], 'entropy': [], 'log_prior': [], 'reconstruction': [],
                                       'reconstruction_at_Eqmean': [], 'effective_number_of_parameters': [],
                                       'VAIC': [], 'VBIC': [], "sec/epoch": [], "sec/ELBO": [], 'epoch': []})

    step = 0
    for epoch in range(start_epoch + 1, FLAGS.num_epochs):
        for batch_index, batch in enumerate(iter(dataset)):
            step = batches_per_epoch * epoch + batch_index
            inputs, outputs = batch
            (total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed) = train_step(
                model, inputs, outputs, optim, seed, tf.constant(step))
            checkpoint.seed.assign(seed)
            state_step = {'ELBO': [-total_loss.numpy()], 'entropy': [-entropy_loss.numpy()],
                          'log_prior': [-log_prior_loss.numpy()], 'reconstruction': [-reconstruction_loss.numpy()],
                          'epoch': [epoch], 'batch': [batch_index], 'step': [step]}
            # model_state = model_state.append(state_step, ignore_index=True)
            model_state = pd.concat([model_state, pd.DataFrame(state_step)], ignore_index=True)
            model_state.to_csv(os.path.join(save_dir, 'model_state.csv'), index=False)
            # model_state['ELBO'].append(-total_loss.numpy())
            # model_state['entropy'].append(-entropy_loss.numpy())
            # model_state['log_prior'].append(-log_prior_loss.numpy())
            # model_state['reconstruction'].append(-reconstruction_loss.numpy())
            # model_state['epoch'].append(epoch)
            # model_state['batch'].append(batch_index)
            # model_state['step'].append(step)

        sec_per_step = (time.time() - start_time) / (step + 1)
        sec_per_epoch = (time.time() - start_time) / (epoch - start_epoch)
        print(f"Epoch: {epoch} ELBO: {-total_loss.numpy()}")
        print(f"Entropy: {-entropy_loss.numpy()} Log-prob: {-log_prior_loss.numpy()} "
              f"Reconstruction: {-reconstruction_loss.numpy()}")
        print("({:.3f} sec/step, {:.3f} sec/epoch)".format(sec_per_step, sec_per_epoch))
        memory()

        if FLAGS.computeIC_every > 0:
            if epoch % FLAGS.computeIC_every == 0:
                ELBOstart = time.time()
                ## Using decorated @tf.function (for some reason requires too much memory)
                ELBO, log_prior, entropy, reconstruction, reconstruction_at_Eqmean, effective_number_of_parameters, VAIC, VBIC, seed = get_variational_information_criteria(
                    model, dataset, seed=seed, nsamples=FLAGS.num_samplesIC)
                ## Using a method of TBIP model
                # ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean, effective_number_of_parameters, VAIC, VBIC, seed = model.get_variational_information_criteria(
                #     dataset, seed=seed, nsamples=FLAGS.num_samplesIC)
                ELBOtime = time.time() - ELBOstart
                epoch_line = {'ELBO': [ELBO.numpy()], 'entropy': [entropy.numpy()],
                              'log_prior': [log_prior.numpy()], 'reconstruction': [reconstruction.numpy()],
                              'reconstruction_at_Eqmean': [reconstruction_at_Eqmean.numpy()],
                              'effective_number_of_parameters': [effective_number_of_parameters.numpy()],
                              'VAIC': [VAIC.numpy()], 'VBIC': [VBIC.numpy()],
                              "sec/epoch": [sec_per_epoch], "sec/ELBO": [ELBOtime],
                              'epoch': [epoch]}
                # epoch_data = epoch_data.append(epoch, ignore_index=True)
                epoch_data = pd.concat([epoch_data, pd.DataFrame(epoch_line)], ignore_index=True)
                epoch_data.to_csv(os.path.join(save_dir, 'epoch_data.csv'), index=False)
                # epoch_data['ELBO'].append(ELBO.numpy())
                # epoch_data['entropy'].append(entropy.numpy())
                # epoch_data['log_prior'].append(log_prior.numpy())
                # epoch_data['reconstruction'].append(reconstruction.numpy())
                # epoch_data['reconstruction_at_Eqmean'].append(reconstruction_at_Eqmean.numpy())
                # epoch_data['effective_number_of_parameters'].append(effective_number_of_parameters.numpy())
                # epoch_data['VAIC'].append(VAIC.numpy())
                # epoch_data['VBIC'].append(VBIC.numpy())
                # epoch_data['epoch'].append(epoch)

                print(f"Epoch: {epoch} ELBO: {ELBO.numpy()}")
                print(
                    f"Entropy: {entropy.numpy()} Log-prob: {log_prior.numpy()} Reconstruction: {reconstruction.numpy()}")
                print(
                    f"Reconstruction at Eqmean: {reconstruction_at_Eqmean.numpy()} Effective number of parameters: {effective_number_of_parameters.numpy()}")
                print(f"VAIC: {VAIC.numpy()} VBIC: {VBIC.numpy()}")
                print("({:.3f} sec/ELBO and IC evaluation".format(ELBOtime))

        # Log to tensorboard at the end of every `save_every` epochs.
        if epoch % FLAGS.save_every == 0:
            tf.summary.scalar("loss", total_loss, step=step)
            tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
            tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
            tf.summary.scalar("elbo/count_log_likelihood", -reconstruction_loss, step=step)
            tf.summary.scalar("elbo/elbo", -total_loss, step=step)
            summary_writer.flush()

            # Save checkpoint too.
            model.start_epoch.assign(epoch)
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

            # All model parameters can be accessed by loading the checkpoint, similar
            # to the logic at the beginning of this function. Since that may be
            # too much hassle, we also save the ideal point model parameters to a
            # separate file. You can save additional model parameters if you'd like.
            param_save_dir = os.path.join(save_dir, "params/")
            if not os.path.exists(param_save_dir):
                os.makedirs(param_save_dir)

            # # ar_ak, ar_kv
            # pd.DataFrame(model.ar_ak_varfam.location.numpy()).to_csv(os.path.join(param_save_dir, "ar_ak_loc.csv"))
            # pd.DataFrame(model.ar_kv_varfam.location.numpy()).to_csv(os.path.join(param_save_dir, "ar_kv_loc.csv"))
            #
            # np.save(os.path.join(param_save_dir, "ar_ak_scale_tril"), model.ar_ak_varfam.scale_tril.numpy())
            # # pd.DataFrame(model.ar_ak_varfam.scale_tril.numpy()).to_csv(
            # #     os.path.join(param_save_dir, "ar_ak_scale_tril.csv"), index=False)
            # np.save(os.path.join(param_save_dir, "ar_kv_scale_tril"), model.ar_kv_varfam.scale_tril.numpy())
            # # pd.DataFrame(model.ar_kv_varfam.scale_tril.numpy()).to_csv(
            # #     os.path.join(param_save_dir, "ar_kv_scale_tril.csv"), index=False)
            np.save(os.path.join(param_save_dir, "ar_ak_loc"), model.ar_ak_varfam.location.numpy())
            if model.ar_ak_varfam.family == 'MVnormal':
                np.save(os.path.join(param_save_dir, "ar_ak_scale_tril"), model.ar_ak_varfam.scale_tril.numpy())
            else:
                np.save(os.path.join(param_save_dir, "ar_ak_scl"), model.ar_ak_varfam.scale.numpy())

            np.save(os.path.join(param_save_dir, "ar_kv_loc"), model.ar_kv_varfam.location.numpy())
            if model.ar_kv_varfam.family == 'MVnormal':
                np.save(os.path.join(param_save_dir, "ar_kv_scale_tril"), model.ar_kv_varfam.scale_tril.numpy())
            else:
                np.save(os.path.join(param_save_dir, "ar_kv_scl"), model.ar_kv_varfam.scale.numpy())

            # ar_ak_mean, ar_kv_mean
            pd.DataFrame(model.ar_ak_mean_varfam.location.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_ak_mean_loc.csv"))
            pd.DataFrame(model.ar_ak_mean_varfam.scale.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_ak_mean_scl.csv"))
            pd.DataFrame(model.ar_kv_mean_varfam.location.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_kv_mean_loc.csv"))
            pd.DataFrame(model.ar_kv_mean_varfam.scale.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_kv_mean_scl.csv"))

            # ar_ak_delta, ar_kv_delta
            if prior_choice["delta"] == "AR":
                pd.DataFrame(model.ar_ak_delta_varfam.location.numpy()).to_csv(
                    os.path.join(param_save_dir, "ar_ak_delta_loc.csv"))
                pd.DataFrame(model.ar_ak_delta_varfam.scale.numpy()).to_csv(
                    os.path.join(param_save_dir, "ar_ak_delta_scl.csv"))
                pd.DataFrame(model.ar_kv_delta_varfam.location.numpy()).to_csv(
                    os.path.join(param_save_dir, "ar_kv_delta_loc.csv"))
                pd.DataFrame(model.ar_kv_delta_varfam.scale.numpy()).to_csv(
                    os.path.join(param_save_dir, "ar_kv_delta_scl.csv"))

            # ar_ak_prec, ar_kv_prec
            pd.DataFrame(model.ar_ak_prec_varfam.shape.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_ak_prec_shp.csv"))
            pd.DataFrame(model.ar_ak_prec_varfam.rate.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_ak_prec_rte.csv"))
            pd.DataFrame(model.ar_kv_prec_varfam.shape.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_kv_prec_shp.csv"))
            pd.DataFrame(model.ar_kv_prec_varfam.rate.numpy()).to_csv(
                os.path.join(param_save_dir, "ar_kv_prec_rte.csv"))

    ### Plotting the ELBO evolution in time plots
    for var in ['ELBO', 'entropy', 'log_prior', 'reconstruction']:
        sub_model_state = model_state[model_state['epoch'] >= start_epoch]
        # All steps
        plt.plot(sub_model_state['step'], sub_model_state[var])
        plt.ylabel(var)
        plt.xlabel('Step')
        plt.savefig(os.path.join(fig_dir, 'batch_' + var + '.png'))
        plt.close()
        # Averages over epochs
        avg = sub_model_state[var].to_numpy()
        avg = avg.reshape((FLAGS.num_epochs - max(start_epoch, 0), batches_per_epoch))
        avg = np.mean(avg, axis=1)
        plt.plot(range(max(start_epoch, 0), FLAGS.num_epochs), avg)
        plt.ylabel(var)
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(fig_dir, 'avg_batch_' + var + '.png'))
        plt.close()

    if FLAGS.computeIC_every > 0:
        for var in ['ELBO', 'entropy', 'log_prior', 'reconstruction', 'reconstruction_at_Eqmean',
                    'effective_number_of_parameters', 'VAIC', 'VBIC', 'sec/epoch', 'sec/ELBO']:
            sub_epoch_data = epoch_data[epoch_data['epoch'] >= start_epoch]
            plt.plot(sub_epoch_data['epoch'], sub_epoch_data[var])
            plt.ylabel(var)
            plt.xlabel('Epoch')
            # plt.show()
            plt.savefig(os.path.join(fig_dir, 'full_approx_' + var.replace('/', '_per_') + '.png'))
            plt.close()

    ### Other figures
    create_all_general_descriptive_figures(model, fig_dir, vocabulary, breaks)

    create_all_figures_specific_to_data(model, FLAGS.data_name, fig_dir, vocabulary, breaks, FLAGS.time_periods)

    ### Create LaTeX tables
    create_latex_tables(model, tab_dir, vocabulary, breaks, FLAGS.time_periods)



if __name__ == '__main__':
    app.run(main)
