# Import global packages
import time
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
import scipy.sparse as sparse
import warnings
import math

# Import local modules
from DPF.code.var_and_prior_family import VariationalFamily, PriorFamily, cov_from_AR1_prec_matrix

prior_hyperparameter = {
    "ar_ak_mean": {"location": 0.0, "scale": 1.0},
    "ar_kv_mean": {"location": 0.0, "scale": 1.0},
    "ar_ak_delta": {"location": 0.5, "scale": 0.2},
    "ar_kv_delta": {"location": 0.5, "scale": 0.2},
    "ar_ak_prec": {"shape": 0.3, "rate": 0.3},
    "ar_kv_prec": {"shape": 0.3, "rate": 0.3},
}

prior_choice = {
    "delta": "AR",              # AR=N(,) leading to AR(1), RW=1 leading to random walk
}

varfam_choice = {
    "ar_ak": "MVnormal",        # 'MVnormal' or 'normal' variational family
    "ar_kv": "MVnormal",        # 'MVnormal' or 'normal' variational family
}

class DPF(tf.keras.Model):

    """Tensorflow implementation of the Dynamic Poisson Factorization"""

    def __init__(self,
                 num_authors: int,
                 num_author_times: int,
                 num_topics: int,
                 num_words: int,
                 num_times: int,
                 num_samples: int,
                 inits: dict,
                 prior_hyperparameter: dict = prior_hyperparameter,
                 prior_choice: dict = prior_choice,
                 varfam_choice: dict = varfam_choice,
                 batch_size: int = 1,
                 RobMon_exponent: float = -0.7, # should be something in [-1, -0.5)
                 exact_entropy: bool = False,
                 exact_log_prior: bool = False,
                 exact_reconstruction: bool = False):
        """Initialize Dynamic Poisson Factorization model.

        Args:
            num_authors: The number of authors in the corpus.
            num_author_times: The number of pairs author-time in the corpus.
            num_topics: The number of topics used for the model.
            num_words: The number of words in the vocabulary.
            num_times: The number of time-periods in the corpus.
            num_samples: The number of Monte-Carlo samples to use to approximate the ELBO.
            inits: A dictionary with inital values for some model parameters.
            prior_hyperparameter: Dictionary of all relevant fixed prior hyperparameter values.
            prior_choice: Dictionary of indicators declaring the chosen hierarchical prior.
            varfam_choice: Dictionary of indicators of variational families that should be used,
                if more options available.
            RobMon_exponent: Exponent in [-1, -0.5) satisfying Robbins-Monroe condition to create
                convex-combinations of old and a new value.
            batch_size: The batch size (over author dimension).
            exact_entropy: Should we compute the exact entropy (True) or approximate it with Monte Carlo (False)?
            exact_log_prior: Should we compute the exact log_prior (True) or approximate it with Monte Carlo (False)?
            exact_reconstruction: Should we compute the exact reconstruction (True) or approximate it with Monte Carlo?
        """
        super(DPF, self).__init__()
        self.num_authors = num_authors
        self.num_author_times = num_author_times
        self.num_topics = num_topics
        self.num_words = num_words
        self.num_times = num_times
        self.num_samples = num_samples
        self.prior_hyperparameter = prior_hyperparameter
        self.prior_choice = prior_choice
        self.varfam_choice = varfam_choice
        self.step_size = 1.0
        self.RobMon_exponent = RobMon_exponent
        self.exact_entropy = exact_entropy
        self.exact_log_prior = exact_log_prior
        self.exact_reconstruction = exact_reconstruction
        self.batch_size = batch_size
        # batch_size = tf.shape(counts)[0]
        self.minibatch_scaling = tf.dtypes.cast(self.num_author_times / batch_size, tf.float32)
        self.log_prior_constant = 0.0

        # ar_ak_mean, ar_kv_mean = Mean of the AR sequences
        self.ar_ak_mean_varfam = VariationalFamily('normal', [num_authors, num_topics], cavi=True, name="ar_ak_mean",
                                                   fitted_location=inits["ar_ak_mean"])
        self.ar_kv_mean_varfam = VariationalFamily('normal', [num_topics, num_words], cavi=True, name="ar_kv_mean",
                                                   fitted_location=inits["ar_kv_mean"])
        self.ar_ak_mean_prior = PriorFamily('normal', num_samples=self.num_samples,
            location=tf.fill([num_authors, num_topics], prior_hyperparameter["ar_ak_mean"]["location"]),
            scale=tf.fill([num_authors, num_topics], prior_hyperparameter["ar_ak_mean"]["scale"]))
        self.ar_kv_mean_prior = PriorFamily('normal', num_samples=self.num_samples,
            location=tf.fill([num_topics, num_words], prior_hyperparameter["ar_kv_mean"]["location"]),
            scale=tf.fill([num_topics, num_words], prior_hyperparameter["ar_kv_mean"]["scale"]))
        self.log_prior_constant -= (0.5 * tfm.log(2.0 * math.pi) + tfm.log(
            self.prior_hyperparameter["ar_ak_mean"]["scale"])) * tf.cast(tfm.reduce_prod(
            self.ar_ak_mean_varfam.location.shape), float)
        self.log_prior_constant -= (0.5 * tfm.log(2.0 * math.pi) + tfm.log(
            self.prior_hyperparameter["ar_kv_mean"]["scale"])) * tf.cast(tfm.reduce_prod(
            self.ar_kv_mean_varfam.location.shape), float)

        # ar_ak_delta, ar_kv_delta = Autoregression coefficients (without any restriction) for AR(1)
        if self.prior_choice["delta"] in ["AR", "ART"]:
            if self.prior_choice["delta"] == "AR":
                dist = 'normal'
                log_normalizing_constant_ak = 0.0
                log_normalizing_constant_kv = 0.0
            else:
                dist = 'Tnormal'
                aux_ak = (tf.constant([-1.0, 1.0]) - prior_hyperparameter["ar_ak_delta"]["location"]) / \
                         prior_hyperparameter["ar_ak_delta"]["scale"]
                aux_kv = (tf.constant([-1.0, 1.0]) - prior_hyperparameter["ar_kv_delta"]["location"]) / \
                         prior_hyperparameter["ar_kv_delta"]["scale"]
                standN = tfp.distributions.Normal(loc=0.0, scale=1.0)
                Phiaux_ak = standN.cdf(aux_ak)
                Phiaux_kv = standN.cdf(aux_kv)
                log_normalizing_constant_ak = tfm.log(Phiaux_ak[1] - Phiaux_ak[0])
                log_normalizing_constant_kv = tfm.log(Phiaux_kv[1] - Phiaux_kv[0])
            self.ar_ak_delta_varfam = VariationalFamily(dist, [num_authors, num_topics],
                                                        cavi=True, name="ar_ak_delta")
            self.ar_kv_delta_varfam = VariationalFamily(dist, [num_topics, num_words],
                                                        cavi=True, name="ar_kv_delta")

            self.ar_ak_delta_prior = PriorFamily(dist, num_samples=self.num_samples,
                location=tf.fill([num_authors, num_topics], prior_hyperparameter["ar_ak_delta"]["location"]),
                scale=tf.fill([num_authors, num_topics], prior_hyperparameter["ar_ak_delta"]["scale"]))
            self.ar_kv_delta_prior = PriorFamily(dist, num_samples=self.num_samples,
                location=tf.fill([num_topics, num_words], prior_hyperparameter["ar_kv_delta"]["location"]),
                scale=tf.fill([num_topics, num_words], prior_hyperparameter["ar_kv_delta"]["scale"]))
            self.log_prior_constant -= (0.5 * tfm.log(2.0 * math.pi) + log_normalizing_constant_ak + tfm.log(
                self.prior_hyperparameter["ar_ak_delta"]["scale"])) * tf.cast(tfm.reduce_prod(
                self.ar_ak_delta_varfam.location.shape), float)
            self.log_prior_constant -= (0.5 * tfm.log(2.0 * math.pi) + log_normalizing_constant_kv + tfm.log(
                self.prior_hyperparameter["ar_kv_delta"]["scale"])) * tf.cast(tfm.reduce_prod(
                self.ar_kv_delta_varfam.location.shape), float)
        elif self.prior_choice["delta"] == "RW":
            self.ar_ak_delta_varfam = VariationalFamily('deterministic', [num_authors, num_topics], cavi=None,
                                                        fitted_location=tf.ones([num_authors, num_topics]),
                                                        name="ar_ak_delta")
            self.ar_kv_delta_varfam = VariationalFamily('deterministic', [num_topics, num_words], cavi=None,
                                                        fitted_location=tf.ones([num_topics, num_words]),
                                                        name="ar_kv_delta")
            self.ar_ak_delta_prior = PriorFamily('deterministic', num_samples=self.num_samples,
                                                 location=tf.ones([num_authors, num_topics]))
            self.ar_kv_delta_prior = PriorFamily('deterministic', num_samples=self.num_samples,
                                                 location=tf.ones([num_topics, num_words]))
        else:
            raise ValueError("Unrecognized prior choice for delta.")

        # ar_ak_prec, ar_kv_prec = precision parameters of AR(1) sequences
        # shapes have fixed CAVI update
        self.ar_ak_prec_varfam = VariationalFamily('gamma', [num_authors, num_topics], cavi=True,
            fitted_shape=tf.fill([num_authors, num_topics],
                                 prior_hyperparameter["ar_ak_prec"]["shape"] + 0.5 * num_times),
            name="ar_ak_prec")
        self.ar_kv_prec_varfam = VariationalFamily('gamma', [num_topics, num_words], cavi=True,
            fitted_shape=tf.fill([num_topics, num_words],
                                 prior_hyperparameter["ar_kv_prec"]["shape"] + 0.5 * num_times),
            name="ar_kv_prec")

        self.ar_ak_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                                            shape=tf.fill([num_authors, num_topics],
                                                          prior_hyperparameter["ar_ak_prec"]["shape"]),
                                            rate=tf.fill([num_authors, num_topics],
                                                         prior_hyperparameter["ar_ak_prec"]["rate"]))
        self.ar_kv_prec_prior = PriorFamily('gamma', num_samples=self.num_samples,
                                            shape=tf.fill([num_topics, num_words],
                                                          prior_hyperparameter["ar_kv_prec"]["shape"]),
                                            rate=tf.fill([num_topics, num_words],
                                                         prior_hyperparameter["ar_kv_prec"]["rate"]))
        self.log_prior_constant += self.prior_hyperparameter["ar_ak_prec"]["shape"] * tfm.log(
            self.prior_hyperparameter["ar_ak_prec"]["rate"]) * tf.cast(tfm.reduce_prod(
            self.ar_ak_prec_varfam.rate.shape), float)
        self.log_prior_constant += self.prior_hyperparameter["ar_kv_prec"]["shape"] * tfm.log(
            self.prior_hyperparameter["ar_kv_prec"]["rate"]) * tf.cast(tfm.reduce_prod(
            self.ar_kv_prec_varfam.rate.shape), float)

        # ar_ak, ar_kv = centered AR(1) sequences
        self.ar_ak_varfam = VariationalFamily(self.varfam_choice["ar_ak"], [num_authors, num_topics, num_times],
                                              cavi=False, name="ar_ak")
        self.ar_kv_varfam = VariationalFamily(self.varfam_choice["ar_kv"], [num_topics, num_words, num_times],
                                              cavi=False, name="ar_kv")

        self.ar_ak_prior = PriorFamily('ARnormal', num_samples=self.num_samples,
            location=tf.repeat(self.ar_ak_mean_varfam.location[:, :, tf.newaxis], num_times, axis=-1),
            shape=self.ar_ak_delta_varfam.location,
            rate=tf.fill([num_authors, num_topics],
                         prior_hyperparameter["ar_ak_prec"]["shape"] / prior_hyperparameter["ar_ak_prec"]["rate"]))
        self.ar_kv_prior = PriorFamily('ARnormal', num_samples=self.num_samples,
            location=tf.repeat(self.ar_kv_mean_varfam.location[:, :, tf.newaxis], num_times, axis=-1),
            shape=self.ar_kv_delta_varfam.location,
            rate=tf.fill([num_topics, num_words],
                         prior_hyperparameter["ar_kv_prec"]["shape"] / prior_hyperparameter["ar_kv_prec"]["rate"]))
        self.log_prior_constant -= 0.5 * tfm.log(2.0 * math.pi) * tf.cast(tfm.reduce_prod(
            self.ar_ak_varfam.location.shape), float)
        self.log_prior_constant -= 0.5 * tfm.log(2.0 * math.pi) * tf.cast(tfm.reduce_prod(
            self.ar_kv_varfam.location.shape), float)


    def get_log_prior(self, samples, nsamples=None):
        """Compute log prior of samples, which are stored in a dictionary of samples.

        Args:
            samples: Dictionary of samples, e.g. samples["ar_ak"] are samples of ar_ak.
            nsamples: Number of independent samples.

        Returns:
            log_prior: Monte-Carlo estimate of the log prior. A tensor with shape [num_samples].
        """
        if nsamples is None:
            nsamples = samples["ar_ak_mean"].shape[0]
        log_prior = tf.zeros([nsamples])

        ### AR mean contribution
        log_prior += self.ar_ak_mean_prior.get_log_prior(samples["ar_ak_mean"])
        log_prior += self.ar_kv_mean_prior.get_log_prior(samples["ar_kv_mean"])

        ### Delta (ARcoef) contribution
        log_prior += self.ar_ak_delta_prior.get_log_prior(samples["ar_ak_delta"])
        log_prior += self.ar_kv_delta_prior.get_log_prior(samples["ar_kv_delta"])

        ### Tau (prec) contribution
        log_prior += self.ar_ak_prec_prior.get_log_prior(samples["ar_ak_prec"])
        log_prior += self.ar_kv_prec_prior.get_log_prior(samples["ar_kv_prec"])

        ### AR(1) contribution
        log_prior += self.ar_ak_prior.get_log_prior(samples["ar_ak"])
        log_prior += self.ar_kv_prior.get_log_prior(samples["ar_kv"])

        return log_prior

    def get_E_delta_matrix(self, delta_varfam):
        delta = delta_varfam.distribution.mean()
        delta2 = tfm.square(delta) + delta_varfam.distribution.variance()

        superdiag = subdiag = tf.repeat(-delta[:, :, tf.newaxis], self.num_times, axis=-1)
        diagdelta2 = tf.repeat(delta2[:, :, tf.newaxis], self.num_times - 1, axis=-1)
        replacement_slice = tf.zeros(tf.shape(diagdelta2)[:-1])
        maindiag = 1.0 + tf.concat([diagdelta2, replacement_slice[:, :, tf.newaxis]], axis=-1)
        prec = tf.linalg.LinearOperatorTridiag([superdiag, maindiag, subdiag], diagonals_format='sequence')
        return prec

    def get_ar_cov(self, ar_varfam, indices=None):
        if ar_varfam.family == "MVnormal":
            if indices is None:
                scl_tril = ar_varfam.scale_tril
            else:
                scl_tril = tf.gather(ar_varfam.scale_tril, indices, axis=0)
            ar_cov = scl_tril @ tf.transpose(scl_tril, perm=[0, 1, 3, 2])
            ar_var = tf.linalg.diag_part(ar_cov)
        elif ar_varfam.family == "normal":
            if indices is None:
                ar_var = tfm.square(ar_varfam.scale)
            else:
                ar_var = tfm.square(tf.gather(ar_varfam.scale, indices, axis=0))
            ar_cov = tf.linalg.diag(ar_var)
        else:
            raise ValueError("Unrecognized variational family choice for ar_ak. Choose either MVnormal or normal.")

        return ar_cov, ar_var

    def get_exact_log_prior(self, ):
        log_prior = tf.constant(self.log_prior_constant)

        ### AR mean contribution
        ar_ak_mean_var = tfm.square(self.ar_ak_mean_varfam.scale)
        log_prior -= 0.5 * tfm.reduce_sum(
            tfm.square(self.ar_ak_mean_varfam.location - self.prior_hyperparameter["ar_ak_mean"]["location"]) +
            ar_ak_mean_var
        ) / tfm.square(self.prior_hyperparameter["ar_ak_mean"]["scale"])

        ar_kv_mean_var = tfm.square(self.ar_kv_mean_varfam.scale)
        log_prior -= 0.5 * tfm.reduce_sum(
            tfm.square(self.ar_kv_mean_varfam.location - self.prior_hyperparameter["ar_kv_mean"]["location"]) +
            ar_kv_mean_var
        ) / tfm.square(self.prior_hyperparameter["ar_kv_mean"]["scale"])

        ### Delta (ARcoef) contribution
        if self.ar_ak_delta_varfam.family == "normal":
            log_prior -= 0.5 * tfm.reduce_sum(
                tfm.square(self.ar_ak_delta_varfam.location - self.prior_hyperparameter["ar_ak_delta"]["location"]) +
                tfm.square(self.ar_ak_delta_varfam.scale)
            ) / tfm.square(self.prior_hyperparameter["ar_ak_delta"]["scale"])
        elif self.ar_ak_delta_varfam.family == "Tnormal":
            Eq_delta = self.ar_ak_delta_varfam.distribution.mean()
            Varq_delta = self.ar_ak_delta_varfam.distribution.variance()
            log_prior -= 0.5 * tfm.reduce_sum(
                tfm.square(self.prior_hyperparameter["ar_ak_delta"]["location"] - Eq_delta) + Varq_delta
            ) / tfm.square(self.prior_hyperparameter["ar_ak_delta"]["scale"])

        if self.ar_kv_delta_varfam.family == "normal":
            log_prior -= 0.5 * tfm.reduce_sum(
                tfm.square(self.ar_kv_delta_varfam.location - self.prior_hyperparameter["ar_kv_delta"]["location"]) +
                tfm.square(self.ar_kv_delta_varfam.scale)
            ) / tfm.square(self.prior_hyperparameter["ar_kv_delta"]["scale"])
        elif self.ar_kv_delta_varfam.family == "Tnormal":
            Eq_delta = self.ar_kv_delta_varfam.distribution.mean()
            Varq_delta = self.ar_kv_delta_varfam.distribution.variance()
            log_prior -= 0.5 * tfm.reduce_sum(
                tfm.square(self.prior_hyperparameter["ar_kv_delta"]["location"] - Eq_delta) + Varq_delta
            ) / tfm.square(self.prior_hyperparameter["ar_kv_delta"]["scale"])

        ### Tau (prec) contribution
        sum_E_log_ar_ak_prec = tfm.reduce_sum(
            tfm.digamma(self.ar_ak_prec_varfam.shape) - tfm.log(self.ar_ak_prec_varfam.rate)
        )
        E_ar_ak_prec = self.ar_ak_prec_varfam.shape / self.ar_ak_prec_varfam.rate
        log_prior += (self.prior_hyperparameter["ar_ak_prec"]["shape"] - 1.0) * sum_E_log_ar_ak_prec
        log_prior -= self.prior_hyperparameter["ar_ak_prec"]["rate"] * tfm.reduce_sum(E_ar_ak_prec)

        sum_E_log_ar_kv_prec = tfm.reduce_sum(
            tfm.digamma(self.ar_kv_prec_varfam.shape) - tfm.log(self.ar_kv_prec_varfam.rate)
        )
        E_ar_kv_prec = self.ar_kv_prec_varfam.shape / self.ar_kv_prec_varfam.rate
        log_prior += (self.prior_hyperparameter["ar_kv_prec"]["shape"] - 1.0) * sum_E_log_ar_kv_prec
        log_prior -= self.prior_hyperparameter["ar_kv_prec"]["rate"] * tfm.reduce_sum(E_ar_kv_prec)

        ### AR(1) contribution
        E_delta_ak = self.get_E_delta_matrix(self.ar_ak_delta_varfam)
        ar_ak_cov, ar_ak_var = self.get_ar_cov(self.ar_ak_varfam)
        dif_ar_ak = self.ar_ak_varfam.location - self.ar_ak_mean_varfam.location[:, :, tf.newaxis]
        # log_prior += 0.5 * tfm.log(1.0) # determinant of the delta matrix is 1 --> no contribution
        log_prior += 0.5 * self.num_times * sum_E_log_ar_ak_prec
        log_prior -= 0.5 * tfm.reduce_sum(
            E_ar_ak_prec * (
                tf.linalg.trace(tf.linalg.matmul(E_delta_ak, ar_ak_cov))
                + tfm.reduce_sum(E_delta_ak.diagonals, axis=[0, -1]) * ar_ak_mean_var
                + tfm.reduce_sum(dif_ar_ak * tf.linalg.matvec(E_delta_ak, dif_ar_ak), axis=-1)
            )
        )

        E_delta_kv = self.get_E_delta_matrix(self.ar_kv_delta_varfam)
        ar_kv_cov, ar_kv_var = self.get_ar_cov(self.ar_kv_varfam)
        dif_ar_kv = self.ar_kv_varfam.location - self.ar_kv_mean_varfam.location[:, :, tf.newaxis]
        # log_prior += 0.5 * tfm.log(1.0) # determinant of the delta matrix is 1 --> no contribution
        log_prior += 0.5 * self.num_times * sum_E_log_ar_kv_prec
        log_prior -= 0.5 * tfm.reduce_sum(
            E_ar_kv_prec * (
                tf.linalg.trace(tf.linalg.matmul(E_delta_kv, ar_kv_cov))
                + tfm.reduce_sum(E_delta_kv.diagonals, axis=[0, -1]) * ar_kv_mean_var
                + tfm.reduce_sum(dif_ar_kv * tf.linalg.matvec(E_delta_kv, dif_ar_kv), axis=-1)
            )
        )

        return log_prior


    def get_entropy(self, samples, nsamples=None, exact=False):
        """Compute entropies of samples, which are stored in a dictionary of samples.
        Samples have to be from variational families to work as an approximation of entropy.

        Args:
            samples: Dictionary of samples, e.g. samples["ar_ak"] are samples of ar_ak.
            nsamples: Number of independent samples.
            exact: [boolean] True --> exact entropy is computed using .entropy()
                            False --> entropy is approximated using the given samples (from varfam necessary!)

        Returns:
            entropy: Monte-Carlo estimate of the entropy. A tensor with shape [num_samples].
        """
        if nsamples is None:
            nsamples = samples["ar_ak_mean"].shape[0]
        entropy = tf.zeros([nsamples])

        ### AR mean contribution
        entropy += self.ar_ak_mean_varfam.get_entropy(samples["ar_ak_mean"], exact)
        entropy += self.ar_kv_mean_varfam.get_entropy(samples["ar_kv_mean"], exact)

        ### Delta (ARcoef) contribution
        entropy += self.ar_ak_delta_varfam.get_entropy(samples["ar_ak_delta"], exact)
        entropy += self.ar_kv_delta_varfam.get_entropy(samples["ar_kv_delta"], exact)

        ### Tau (prec) contribution
        entropy += self.ar_ak_prec_varfam.get_entropy(samples["ar_ak_prec"], exact)
        entropy += self.ar_kv_prec_varfam.get_entropy(samples["ar_kv_prec"], exact)

        ### AR(1) contribution
        entropy += self.ar_ak_varfam.get_entropy(samples["ar_ak"], exact)
        entropy += self.ar_kv_varfam.get_entropy(samples["ar_kv"], exact)

        return entropy


    def get_empty_samples(self):
        """Creates an empty dictionary for samples of the model parameters."""
        samples = {"ar_ak_mean": None}
        samples["ar_kv_mean"] = None
        samples["ar_ak_delta"] = None
        samples["ar_kv_delta"] = None
        samples["ar_ak_prec"] = None
        samples["ar_kv_prec"] = None
        samples["ar_ak"] = None
        samples["ar_kv"] = None

        return samples


    def get_samples_and_update_prior_customized(self, samples, seed=None, varfam=True, nsamples=1):
        """
        Follow the structure of the model to sample all model parameters.
        Sample from the most inner prior distributions first and then go up the hierarchy.
        Update the priors simultaneously.
        Compute contributions to log_prior and entropy along the way.
        Return samples needed to recover the Poisson rates for word counts.

        Args:
            samples: Dictionary of samples, e.g. samples["ar_ak"] are samples of ar_ak.
                Some may be empty to be filled, some may be given. If given then not sample.
            seed: Random seed to set the random number generator.
            varfam: True --> sample from variational family
                   False --> sample from prior family
            nsamples: When sampling from variational family, number of samples can be specified.
                        Usually self.num_samples, but an arbitrary number can be supplied.

        Returns:
            samples: Dictionary of samples, e.g. samples["ar_ak"] are samples of ar_ak.
            seed: Random seed to set the random number generator.
        """
        if varfam:
            num_samples = nsamples
        else:
            num_samples = self.num_samples

        ### AR mean
        if samples["ar_ak_mean"] is None:
            if varfam:
                samples["ar_ak_mean"], seed = self.ar_ak_mean_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_ak_mean"], seed = self.ar_ak_mean_prior.sample((), seed=seed)

        if samples["ar_kv_mean"] is None:
            if varfam:
                samples["ar_kv_mean"], seed = self.ar_kv_mean_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_kv_mean"], seed = self.ar_kv_mean_prior.sample((), seed=seed)

        ### Delta (ARcoef)
        if samples["ar_ak_delta"] is None:
            if varfam:
                samples["ar_ak_delta"], seed = self.ar_ak_delta_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_ak_delta"], seed = self.ar_ak_delta_prior.sample((), seed=seed)

        if samples["ar_kv_delta"] is None:
            if varfam:
                samples["ar_kv_delta"], seed = self.ar_kv_delta_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_kv_delta"], seed = self.ar_kv_delta_prior.sample((), seed=seed)

        ### Tau (prec)
        if samples["ar_ak_prec"] is None:
            if varfam:
                samples["ar_ak_prec"], seed = self.ar_ak_prec_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_ak_prec"], seed = self.ar_ak_prec_prior.sample((), seed=seed)

        if samples["ar_kv_prec"] is None:
            if varfam:
                samples["ar_kv_prec"], seed = self.ar_kv_prec_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_kv_prec"], seed = self.ar_kv_prec_prior.sample((), seed=seed)

        ### AR(1) sequences
        # Update prior with new deltas and taus
        if self.ar_ak_delta_varfam.family != "deterministic":
            self.ar_ak_prior.ARcoef.assign(samples["ar_ak_delta"])
        if self.ar_kv_delta_varfam.family != "deterministic":
            self.ar_kv_prior.ARcoef.assign(samples["ar_kv_delta"])

        self.ar_ak_prior.location.assign(tf.repeat(samples["ar_ak_mean"][:, :, :, tf.newaxis], self.num_times, axis=-1))
        self.ar_ak_prior.prec.assign(samples["ar_ak_prec"])
        self.ar_ak_prior.covariance.assign(
            cov_from_AR1_prec_matrix(delta=self.ar_ak_prior.ARcoef, tau=self.ar_ak_prior.prec, T=self.num_times))
        self.ar_ak_prior.scale_tril.assign(tf.linalg.cholesky(self.ar_ak_prior.covariance))

        self.ar_kv_prior.location.assign(tf.repeat(samples["ar_kv_mean"][:, :, :, tf.newaxis], self.num_times, axis=-1))
        self.ar_kv_prior.prec.assign(samples["ar_kv_prec"])
        self.ar_kv_prior.covariance.assign(
            cov_from_AR1_prec_matrix(delta=self.ar_kv_prior.ARcoef, tau=self.ar_kv_prior.prec, T=self.num_times))
        self.ar_kv_prior.scale_tril.assign(tf.linalg.cholesky(self.ar_kv_prior.covariance))

        # create samples
        if samples["ar_ak"] is None:
            if varfam:
                samples["ar_ak"], seed = self.ar_ak_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_ak"], seed = self.ar_ak_prior.sample((), seed=seed)

        if samples["ar_kv"] is None:
            if varfam:
                samples["ar_kv"], seed = self.ar_kv_varfam.sample(num_samples, seed=seed)
            else:
                samples["ar_kv"], seed = self.ar_kv_prior.sample((), seed=seed)

        return samples, seed

    def sakt_to_sbk(self, tensor, author_indices, time_indices):
        tensor_sbkt = tf.gather(tensor, author_indices, axis=1)
        tensor_sbkb = tf.gather(tensor_sbkt, time_indices, axis=-1)
        tensor_skb = tf.linalg.diag_part(tf.transpose(tensor_sbkb, perm=[0, 2, 1, 3]))
        tensor_sbk = tf.transpose(tensor_skb, perm=[0, 2, 1])
        return tensor_sbk

    def get_rates(self, samples, author_indices, time_indices):
        """
        Given samples of ar_ak and ar_kv compute the rates for DPF.

        Args:
            samples: A dictionary of samples including ar_ak and ar_kv.
            author_indices: int[batch_size] of author indices.
            time_indices: int[batch_size] of time indices.

        Returns:
            rate: float[num_samples, batch_size, num_words]
        """
        # Start with subsetting required samples to documents included in current batch
        ar_ak_sbk = self.sakt_to_sbk(samples["ar_ak"], author_indices, time_indices)
        ar_kv_sbkv = tf.transpose(tf.gather(samples["ar_kv"], time_indices, axis=-1), perm=[0, 3, 1, 2])
        exp_ar = tfm.exp(ar_ak_sbk[:, :, :, tf.newaxis] + ar_kv_sbkv)
        # [num_samples, batch_size, num_topics, num_words]

        rate = tfm.reduce_sum(exp_ar, axis=2)  # sum over all topics (3rd dimension)
        # [num_samples, batch_size, num_words]
        return rate


    def get_gamma_distribution_Eqmean_subset(self, distribution, indices, log=False):
        """Returns mean of the variational family (Eqmean) for gamma-distributed parameter,
            e.g. ar_ak_prec
        First takes only a subset of shapes and rates corresponding to given document indices.

        Args:
            distribution: gamma distribution to work with
            log: [boolean] Should we compute E_q [ X ] (False) or E_q [ log(X) ] (True)?
            indices: Indices for the first dimension to subset. A tensor with shape [batch_size].

        Returns:
            Eqmean: [batch_size, :other dimensions:]
        """
        shp = tf.gather(distribution.shape, indices, axis=0)
        rte = tf.gather(distribution.rate, indices, axis=0)
        if log:
            Eqmean = tfm.digamma(shp) - tfm.log(rte)
        else:
            Eqmean = shp / rte
        return Eqmean

    def get_Eqmean(self, distribution, log=False):
        """Returns mean of the variational family (Eqmean).

        Args:
            distribution: VariationalFamily probability distribution to work with.
            log: [boolean] Should we compute E_q [ X ] (False) or E_q [ log(X) ] (True)?

        Returns:
            Eqmean: variational mean or log-scale mean.

        """
        if log:
            if distribution.family == 'deterministic':
                Eqmean = tfm.log(distribution.location)
            elif distribution.family == 'lognormal':
                Eqmean = distribution.location
            elif distribution.family == 'gamma':
                Eqmean = tfm.digamma(distribution.shape) - tfm.log(distribution.rate)
            elif distribution.family in ['normal', 'Tnormal', 'MVnormal']:
                raise ValueError("Cannot compute E_q log(X) if X ~ Normal.")
            else:
                raise ValueError("Unrecognized distributional family.")
        else:
            Eqmean = distribution.distribution.mean()
        return Eqmean


    def check_and_print_non_finite(self, p, info_string, name=''):
        """Checks given parameter for NaN and infinite values and prints them if so."""
        nfp = tfm.logical_not(tfm.is_finite(p))
        if tfm.reduce_any(nfp):
            indices = tf.where(nfp)
            print(info_string)
            print("Found " + str(tf.shape(indices)[0]) + " NaN or infinite values for parameter: " + name + ".")
            print("Indices: ")
            print(indices)
            print("Values:")
            print(tf.gather_nd(p, indices))


    def print_non_finite_parameters(self, info_string):
        """Checks which model parameters have NaN values and prints them and indices."""
        # AR means
        self.check_and_print_non_finite(self.ar_ak_mean_varfam.location, info_string, self.ar_ak_mean_varfam.location.name)
        self.check_and_print_non_finite(self.ar_ak_mean_varfam.scale, info_string, self.ar_ak_mean_varfam.scale.name)
        self.check_and_print_non_finite(self.ar_kv_mean_varfam.location, info_string, self.ar_kv_mean_varfam.location.name)
        self.check_and_print_non_finite(self.ar_kv_mean_varfam.scale, info_string, self.ar_kv_mean_varfam.scale.name)

        # Delta (ARcoef) parameters
        if self.ar_ak_delta_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.ar_ak_delta_varfam.location, info_string,
                                            self.ar_ak_delta_varfam.location.name)
            self.check_and_print_non_finite(self.ar_ak_delta_varfam.scale, info_string,
                                            self.ar_ak_delta_varfam.scale.name)
        if self.ar_kv_delta_varfam.family != "deterministic":
            self.check_and_print_non_finite(self.ar_kv_delta_varfam.location, info_string,
                                            self.ar_kv_delta_varfam.location.name)
            self.check_and_print_non_finite(self.ar_kv_delta_varfam.scale, info_string,
                                            self.ar_kv_delta_varfam.scale.name)

        # Tau (prec) parameters
        self.check_and_print_non_finite(self.ar_ak_prec_varfam.shape, info_string, self.ar_ak_prec_varfam.shape.name)
        self.check_and_print_non_finite(self.ar_ak_prec_varfam.rate, info_string, self.ar_ak_prec_varfam.rate.name)
        self.check_and_print_non_finite(self.ar_kv_prec_varfam.shape, info_string, self.ar_kv_prec_varfam.shape.name)
        self.check_and_print_non_finite(self.ar_kv_prec_varfam.rate, info_string, self.ar_kv_prec_varfam.rate.name)

        # AR(1) sequences
        self.check_and_print_non_finite(self.ar_ak_varfam.location, info_string, self.ar_ak_varfam.location.name)
        if self.varfam_choice["ar_ak"] == "MVnormal":
            self.check_and_print_non_finite(self.ar_ak_varfam.scale_tril, info_string, self.ar_ak_varfam.scale_tril.name)
        elif self.varfam_choice["ar_ak"] == "normal":
            self.check_and_print_non_finite(self.ar_ak_varfam.scale, info_string, self.ar_ak_varfam.scale.name)
        else:
            raise ValueError("Unrecognized variational family choice for ar_ak. Choose either MVnormal or normal.")

        self.check_and_print_non_finite(self.ar_kv_varfam.location, info_string, self.ar_kv_varfam.location.name)
        if self.varfam_choice["ar_kv"] == "MVnormal":
            self.check_and_print_non_finite(self.ar_kv_varfam.scale_tril, info_string, self.ar_kv_varfam.scale_tril.name)
        elif self.varfam_choice["ar_kv"] == "normal":
            self.check_and_print_non_finite(self.ar_kv_varfam.scale, info_string, self.ar_kv_varfam.scale.name)
        else:
            raise ValueError("Unrecognized variational family choice for ar_kv. Choose either MVnormal or normal.")

    def get_expected_ar_terms(self, ):
        """Compute variational mean of
            exp_ar = E [exp(ar_ak + ar_kv)],
            centered_ar_ak_2 = E [(ar_ak - ar_ak_mean)^2],
            centered_ar_kv_2 = E [(ar_kv - ar_kv_mean)^2],
            sum_centered_ar_ak_lag = sum_t E [(ar_ak[t]-ar_ak_mean) * (ar_ak[t-1]-ar_ak_mean)],
            sum_centered_ar_kv_lag = sum_t E [(ar_kv[t]-ar_kv_mean) * (ar_kv[t-1]-ar_kv_mean)].
        ..._lag are already summed over time dimension, because they will be used only in this form anyway.

        More specifically, we need to compute:
            E exp(ar_ak) = exp( ar_ak_loc + 0.5 * ar_ak_var )
            E ar_ak * ar_ak = ar_ak_var + ar_ak_loc * ar_ak_loc
            E ar[t] ar[t-1] = ar_cov[t, t-1] + ar_loc[t] * ar_loc[t-1]
            E ar_ak_mean^2 = ar_ak_mean_var + ar_ak_mean_loc^2
            E ar_kv_mean^2 = ar_kv_mean_var + ar_kv_mean_loc^2
            ... and sum them up properly

        Returns:
            exp_ar: float[num_times, batch_size, num_topics, num_words]
            ar_ak_2: float[batch_size, num_topics, num_times]
            ar_kv_2: float[num_topics, num_words, num_times]
            sum_ar_ak_lag: float[batch_size, num_topics]
            sum_ar_kv_lag: float[num_topics, num_words]
        """
        ar_ak_cov, ar_ak_var = self.get_ar_cov(self.ar_ak_varfam)
        # shape: [batch_size, num_topics, num_times]
        ar_kv_cov, ar_kv_var = self.get_ar_cov(self.ar_kv_varfam)
        # shape: [num_topics, num_words, num_times]
        # transpose so that times dimension is the first one (to be replaced with document dimension)
        exp_ar = tfm.exp(
            self.ar_ak_varfam.location[:, :, tf.newaxis, :] + 0.5 * ar_ak_var[:, :, tf.newaxis, :] +
            self.ar_kv_varfam.location[tf.newaxis, :, :, :] + 0.5 * ar_kv_var[tf.newaxis, :, :, :]
        ) # shape: [num_authors, num_topics, num_words, num_times]

        ar_ak_2 = tfm.square(self.ar_ak_varfam.location) + ar_ak_var
        ar_kv_2 = tfm.square(self.ar_kv_varfam.location) + ar_kv_var

        ar_ak_mean_2 = tfm.square(self.ar_ak_mean_varfam.location) + tfm.square(self.ar_ak_mean_varfam.scale)
        ar_kv_mean_2 = tfm.square(self.ar_kv_mean_varfam.location) + tfm.square(self.ar_kv_mean_varfam.scale)

        centered_ar_ak_2 = ar_ak_2 - 2.0 * self.ar_ak_varfam.location * \
                           self.ar_ak_mean_varfam.location[:, :, tf.newaxis] + ar_ak_mean_2[:, :, tf.newaxis]
        centered_ar_kv_2 = ar_kv_2 - 2.0 * self.ar_kv_varfam.location * \
                           self.ar_kv_mean_varfam.location[:, :, tf.newaxis] + ar_kv_mean_2[:, :, tf.newaxis]

        ## An example of how extracting upper diagonal and creating lags works:
        # A = tf.reshape(-tf.range(1* 3 * 4 * 4), [1, 3, 4, 4])
        # loc = tf.reshape(tf.range(1 * 3 * 4), [1, 3, 4])
        # mask = tf.linalg.band_part(tf.ones([4, 4]), 0, 1) - tf.linalg.diag(tf.ones([4]))
        # ar_ak_super_diag = tf.boolean_mask(A, mask, axis=2)
        # ar_ak_upper = tf.gather(loc, range(1, 4), axis=-1)
        # ar_ak_lower = tf.gather(loc, range(0, 4 - 1), axis=-1)
        # sum_ar_ak_lag = tfm.reduce_sum(ar_ak_super_diag + ar_ak_upper * ar_ak_lower, axis=-1)
        mask = tf.linalg.band_part(tf.ones([self.num_times, self.num_times]), 0, 1) - tf.linalg.diag(tf.ones([self.num_times]))

        ar_ak_super_diag = tf.boolean_mask(ar_ak_cov, mask, axis=2)
        ar_ak_upper = tf.gather(self.ar_ak_varfam.location, range(1, self.num_times), axis=-1)
        ar_ak_lower = tf.gather(self.ar_ak_varfam.location, range(0, self.num_times - 1), axis=-1)
        sum_ar_ak_lag = tfm.reduce_sum(ar_ak_super_diag + ar_ak_upper * ar_ak_lower, axis=-1)
        sum_ar_ak_upp_low = tfm.reduce_sum(ar_ak_lower + ar_ak_upper, axis=-1)
        sum_centered_ar_ak_lag = sum_ar_ak_lag - self.ar_ak_mean_varfam.location * sum_ar_ak_upp_low + \
                                 (self.num_times - 1.0) * ar_ak_mean_2

        ar_kv_super_diag = tf.boolean_mask(ar_kv_cov, mask, axis=2)
        ar_kv_upper = tf.gather(self.ar_kv_varfam.location, range(1, self.num_times), axis=-1)
        ar_kv_lower = tf.gather(self.ar_kv_varfam.location, range(0, self.num_times - 1), axis=-1)
        sum_ar_kv_lag = tfm.reduce_sum(ar_kv_super_diag + ar_kv_upper * ar_kv_lower, axis=-1)
        sum_ar_kv_upp_low = tfm.reduce_sum(ar_kv_lower + ar_kv_upper, axis=-1)
        sum_centered_ar_kv_lag = sum_ar_kv_lag - self.ar_kv_mean_varfam.location * sum_ar_kv_upp_low + \
                                 (self.num_times - 1.0) * ar_kv_mean_2

        return exp_ar, centered_ar_ak_2, centered_ar_kv_2, sum_centered_ar_ak_lag, sum_centered_ar_kv_lag


    def get_cavi_auxiliary_proportions(self, aux_prob_log):
        """Perform CAVI update for auxiliary proportion variables.

        Args:
            aux_prob_log: update for auxiliary proportions on log-scale.
                int[batch_size, num_topics, num_words]

        Returns:
            auxiliary_proportions: The updated auxiliary proportions. The tensor is normalized across topics,
                so it can be interpreted as the proportion of each topic belong to each word.
                float[batch_size, num_topics, num_words]
        """
        # Before we proceed with exp() and normalizing over topics we perform trick ensuring numerical stability first.
        # Compute maximum over topics (second dimension) - shape: [batch_size, num_words]
        # Rescale to 3D - shape: [batch_size, num_topics, num_words]
        # Subtract the maxima over topics to obtain non-positive values to be exponentiated.
        aux_prob_log -= tfm.reduce_max(aux_prob_log, axis=1)[:, tf.newaxis, :]  # -aux_prob_log_max

        # Now we can finally call the exp() function
        auxiliary_numerator = tf.exp(aux_prob_log)

        # Quantities are proportional across topics, rescale to sum to 1 over topics
        auxiliary_proportions = auxiliary_numerator / tfm.reduce_sum(auxiliary_numerator, axis=1)[:, tf.newaxis, :]
        return auxiliary_proportions


    def get_cavi_ar_delta(self, ar_2, sum_ar_lag, tau_Eqmean, mu, scl):
        """Get CAVI updates for delta parameters (AR(1) coefficients) given needed quantities.

        Args:
            ar_2: Variational mean of squared AR(1) sequence.
                float[:, num_times]
            sum_ar_lag: Summed variational mean of ar[t]*ar[t-1] over time t.
                float[:]
            tau_Eqmean: Variational mean of precision parameter of AR(1) sequence.
                float[:]
            mu: prior mean value for delta.
                float[1]
            scl: prior scale value for delta.
                float[1]

        Returns:
            update_delta_loc: CAVI update for location parameter
                float[:]
            update_delta_scl: CAVI update for scale parameter
                float[:]
        """
        # sum from t = 1 to just T-1, not T!
        prec = tfm.reciprocal(tfm.square(scl))
        update_delta_prec = prec + tau_Eqmean * tfm.reduce_sum(tf.gather(ar_2, tf.range(self.num_times-1), axis=-1), axis=-1)
        update_delta_var = tfm.reciprocal(update_delta_prec)
        update_delta_scl = tfm.sqrt(update_delta_var)
        update_delta_loc = update_delta_var * (mu * prec + tau_Eqmean * sum_ar_lag)
        return update_delta_loc, update_delta_scl

    def cavi_update_ar_delta(self, ar_ak_2, ar_kv_2, sum_ar_ak_lag, sum_ar_kv_lag):
        """ Perform CAVI update for delta parameters (AR(1) coefficients).
        Updates location and scale parameters for both AK-and-KV-specific AR(1) sequences.

        Args:
            ar_ak_2: Variational mean of squared AR(1) ak-specific sequence.
                float[num_authors, num_topics, num_times]
            ar_kv_2: Variational mean of squared AR(1) kv-specific sequence.
                float[num_topics, num_words, num_times]
            sum_ar_ak_lag: Summed variational mean of ar_ak[t]*ar_ak[t-1] over time t.
                float[num_authors, num_topics]
            sum_ar_kv_lag: Summed variational mean of ar_kv[t]*ar_kv[t-1] over time t.
                float[num_topics, num_words]
        """
        # AK-specific AR(1) sequences - local updates
        update_ak_delta_loc, update_ak_delta_scl = self.get_cavi_ar_delta(
            ar_ak_2, sum_ar_ak_lag, self.ar_ak_prec_varfam.distribution.mean(),
            self.prior_hyperparameter["ar_ak_delta"]["location"], self.prior_hyperparameter["ar_ak_delta"]["scale"]
        )
        global_ak_delta_loc = self.step_size * update_ak_delta_loc + (1 - self.step_size) * self.ar_ak_delta_varfam.location
        global_ak_delta_scl = self.step_size * update_ak_delta_scl + (1 - self.step_size) * self.ar_ak_delta_varfam.scale
        self.ar_ak_delta_varfam.location.assign(global_ak_delta_loc)
        self.ar_ak_delta_varfam.scale.assign(global_ak_delta_scl)

        # KV-specific AR(1) sequences - global updates
        update_kv_delta_loc, update_kv_delta_scl = self.get_cavi_ar_delta(
            ar_kv_2, sum_ar_kv_lag, self.ar_kv_prec_varfam.distribution.mean(),
            self.prior_hyperparameter["ar_kv_delta"]["location"], self.prior_hyperparameter["ar_kv_delta"]["scale"]
        )
        global_kv_delta_loc = self.step_size * update_kv_delta_loc + (1 - self.step_size) * self.ar_kv_delta_varfam.location
        global_kv_delta_scl = self.step_size * update_kv_delta_scl + (1 - self.step_size) * self.ar_kv_delta_varfam.scale
        self.ar_kv_delta_varfam.location.assign(global_kv_delta_loc)
        self.ar_kv_delta_varfam.scale.assign(global_kv_delta_scl)


    def get_cavi_ar_prec(self, ar_2, sum_ar_lag, Eq_delta, Eq_delta_2, rate):
        """Get CAVI updates for rate of precision parameters of AR(1) given needed quantities.

        Args:
            ar_2: Variational mean of squared AR(1) sequence.
                float[:, num_times]
            sum_ar_lag: Summed variational mean of ar[t]*ar[t-1] over time t.
                float[:]
            Eq_delta: Variational mean (location for normal) of delta parameter of AR(1) sequence.
                float[:]
            Eq_delta_2: Second variational moment (mean^2 + var) of delta parameter of AR(1) sequence.
                float[:]
            rate: prior mean value for delta.
                float[1]

        Returns:
            update_prec_rte: CAVI update for rate parameter
                float[:]
        """
        # sum from t = 1 to just T-1, not T!
        sum_ar_2 = (1.0 + Eq_delta_2) * tfm.reduce_sum(tf.gather(ar_2, tf.range(self.num_times-1), axis=-1), axis=-1)
        # the last time T is multiplied by just (1.0 + 0.0) and not Eq_delta_2
        update_prec_rte = rate + 0.5 * (sum_ar_2 + tf.gather(ar_2, self.num_times-1, axis=-1)) - Eq_delta * sum_ar_lag
        return update_prec_rte

    def cavi_update_ar_prec(self, ar_ak_2, ar_kv_2, sum_ar_ak_lag, sum_ar_kv_lag,
                            Eq_ar_ak_delta, Eq_ar_kv_delta,
                            Eq_ar_ak_delta_2, Eq_ar_kv_delta_2):
        """ Perform CAVI update for precision parameters of AR(1) sequences.
        Updates rate parameters for both AK-and-KV-specific AR(1) sequences.

        Args:
            ar_ak_2: Variational mean of squared AR(1) ak-specific sequence.
                float[num_authors, num_topics, num_times]
            ar_kv_2: Variational mean of squared AR(1) kv-specific sequence.
                float[num_topics, num_words, num_times]
            sum_ar_ak_lag: Summed variational mean of ar_ak[t]*ar_ak[t-1] over time t.
                float[num_authors, num_topics]
            sum_ar_kv_lag: Summed variational mean of ar_kv[t]*ar_kv[t-1] over time t.
                float[num_topics, num_words]
            Eq_ar_ak_delta: Variational mean (location for normal) of ar_ak_delta parameter of AR(1) sequence.
                float[:]
            Eq_ar_kv_delta: Variational mean (location for normal) of ar_kv_delta parameter of AR(1) sequence.
                float[:]
            Eq_ar_ak_delta_2: Second variational moment of ar_ak_delta (mean^2 + var).
                float[num_authors, num_topics]
            Eq_ar_kv_delta_2: Second variational moment of ar_kv_delta (mean^2 + var).
                float[num_topics, num_words]
        """
        # AK-specific AR(1) sequences - local updates
        update_ak_prec_rte = self.get_cavi_ar_prec(
            ar_ak_2, sum_ar_ak_lag, Eq_ar_ak_delta, Eq_ar_ak_delta_2,
            self.prior_hyperparameter["ar_ak_prec"]["rate"]
        )
        global_ak_prec_rte = self.step_size * update_ak_prec_rte + (1 - self.step_size) * self.ar_ak_prec_varfam.rate
        self.ar_ak_prec_varfam.rate.assign(global_ak_prec_rte)

        # KV-specific AR(1) sequences - global updates
        update_kv_prec_rte = self.get_cavi_ar_prec(
            ar_kv_2, sum_ar_kv_lag, Eq_ar_kv_delta, Eq_ar_kv_delta_2,
            self.prior_hyperparameter["ar_kv_prec"]["rate"]
        )
        global_kv_prec_rte = self.step_size * update_kv_prec_rte + (1 - self.step_size) * self.ar_kv_prec_varfam.rate
        self.ar_kv_prec_varfam.rate.assign(global_kv_prec_rte)


    def get_cavi_ar_mean(self, prior_loc, prior_scl, Eq_tau, Eq_delta, Eq_delta_2, Eq_ar):
        """Get CAVI updates for AR(1) means (location and scale) given needed quantities.

        Args:
            prior_loc: Prior mean of the AR mean.
                float[1]
            prior_scl: Prior scale of the AR mean.
                float[1]
            Eq_tau: Variational mean (shp/rte) of precision parameter of AR(1) sequence.
                float[:]
            Eq_delta: Variational mean (location for normal) of delta parameter of AR(1) sequence.
                float[:]
            Eq_delta_2: Second variational moment (mean^2 + var) of delta parameter of AR(1) sequence.
                float[:]
            Eq_ar: Variational mean (location) of the AR(1) sequence.
                float[:, num_times]

        Returns:
            update_mean_loc: CAVI update for location parameter
                float[:]
            update_mean_scl: CAVI update for scale parameter
                float[:]
        """
        prior_prec = tfm.reciprocal(tfm.square(prior_scl))
        delta_squared = 1.0 - 2.0 * Eq_delta + Eq_delta_2
        prec_shift = 1.0 + (self.num_times - 1.0) * delta_squared
        loc_shift  = (1.0 - Eq_delta + Eq_delta_2) * tf.gather(Eq_ar, 0, axis=-1)
        loc_shift += delta_squared * tfm.reduce_sum(tf.gather(Eq_ar, tf.range(1, self.num_times-1), axis=-1), axis=-1)
        loc_shift += (1.0 - Eq_delta) * tf.gather(Eq_ar, self.num_times-1, axis=-1)

        update_mean_prec = prior_prec + Eq_tau * prec_shift
        update_mean_var = tfm.reciprocal(update_mean_prec)
        update_mean_scl = tfm.sqrt(update_mean_var)
        update_mean_loc = update_mean_var * (prior_loc * prior_prec + Eq_tau * loc_shift)
        return update_mean_loc, update_mean_scl

    def cavi_update_ar_mean(self, Eq_ar_ak_delta, Eq_ar_kv_delta, Eq_ar_ak_delta_2, Eq_ar_kv_delta_2):
        """ Perform CAVI update for AR(1) means.
        Updates location and scale parameters for both AK-and-KV-specific AR(1) sequences.

        Args:
            Eq_ar_ak_delta: Variational mean (location for normal) of ar_ak_delta parameter of AR(1) sequence.
                float[:]
            Eq_ar_kv_delta: Variational mean (location for normal) of ar_kv_delta parameter of AR(1) sequence.
                float[:]
            Eq_ar_ak_delta_2: Second variational moment of ar_ak_delta (loc^2 + var).
                float[num_authors, num_topics]
            Eq_ar_kv_delta_2: Second variational moment of ar_kv_delta (loc^2 + var).
                float[num_topics, num_words]
        """
        # AK-specific AR(1) sequences - local updates
        update_ak_mean_loc, update_ak_mean_scl = self.get_cavi_ar_mean(
            self.prior_hyperparameter["ar_ak_mean"]["location"],
            self.prior_hyperparameter["ar_ak_mean"]["scale"],
            self.ar_ak_prec_varfam.distribution.mean(),
            Eq_ar_ak_delta, Eq_ar_ak_delta_2,
            self.ar_ak_varfam.location
        )
        global_ak_mean_loc = self.step_size * update_ak_mean_loc + (1 - self.step_size) * self.ar_ak_mean_varfam.location
        global_ak_mean_scl = self.step_size * update_ak_mean_scl + (1 - self.step_size) * self.ar_ak_mean_varfam.scale
        self.ar_ak_mean_varfam.location.assign(global_ak_mean_loc)
        self.ar_ak_mean_varfam.scale.assign(global_ak_mean_scl)

        # KV-specific AR(1) sequences - global updates
        update_kv_mean_loc, update_kv_mean_scl = self.get_cavi_ar_mean(
            self.prior_hyperparameter["ar_kv_mean"]["location"],
            self.prior_hyperparameter["ar_kv_mean"]["scale"],
            self.ar_kv_prec_varfam.distribution.mean(),
            Eq_ar_kv_delta, Eq_ar_kv_delta_2,
            self.ar_kv_varfam.location
        )
        global_kv_mean_loc = self.step_size * update_kv_mean_loc + (1 - self.step_size) * self.ar_kv_mean_varfam.location
        global_kv_mean_scl = self.step_size * update_kv_mean_scl + (1 - self.step_size) * self.ar_kv_mean_varfam.scale
        self.ar_kv_mean_varfam.location.assign(global_kv_mean_loc)
        self.ar_kv_mean_varfam.scale.assign(global_kv_mean_scl)


    def perform_cavi_updates(self, inputs, outputs, step):
        """Perform CAVI updates for document intensities and objective topics.

        Args:
            inputs: A dictionary of input tensors.
            outputs: A sparse tensor containing word counts.
            step: The current training step.
        """
        self.batch_size = tf.shape(outputs)[0]
        self.print_non_finite_parameters("At start of perform_cavi_updates for step " + str(step))

        # start = time.time()
        # counts = tf.sparse.to_dense(outputs)
        # auxiliary_proportions = self.get_cavi_auxiliary_proportions(inputs['author_indices'])
        # aux_counts = auxiliary_proportions * counts[:, tf.newaxis, :, :]
        # end = time.time()
        # print(end - start)

        # get variational means of quantities formed out of AR(1) sequences
        exp_ar, ar_ak_2, ar_kv_2, sum_ar_ak_lag, sum_ar_kv_lag = self.get_expected_ar_terms()
        self.check_and_print_non_finite(exp_ar, "After updating exp_ar for step " + str(step), name='exp_ar')

        # We scale to account for the fact that we're only using a minibatch to
        # update the variational parameters of a global latent variable.
        self.minibatch_scaling = tf.cast(self.num_author_times / self.batch_size, tf.dtypes.float32)
        self.step_size = tfm.pow(tf.cast(step, tf.dtypes.float32) + 1, self.RobMon_exponent)
        # print(self.step_size)

        ## Delta (ARcoef) - autoregressive coefficients for AR(1) sequences
        if self.prior_choice["delta"] != "RW":
            self.cavi_update_ar_delta(ar_ak_2, ar_kv_2, sum_ar_ak_lag, sum_ar_kv_lag)
            self.print_non_finite_parameters("After updating ar_delta parameters for step " + str(step))

        Eq_ar_ak_delta = self.ar_ak_delta_varfam.distribution.mean()
        Eq_ar_ak_delta_2 = tfm.square(Eq_ar_ak_delta)
        Eq_ar_ak_delta_2 += self.ar_ak_delta_varfam.distribution.variance()

        Eq_ar_kv_delta = self.ar_kv_delta_varfam.distribution.mean()
        Eq_ar_kv_delta_2 = tfm.square(Eq_ar_kv_delta)
        Eq_ar_kv_delta_2 += self.ar_kv_delta_varfam.distribution.variance()

        ## Tau (prec) - precision parameters for AR(1) sequences
        self.cavi_update_ar_prec(ar_ak_2, ar_kv_2, sum_ar_ak_lag, sum_ar_kv_lag,
                                 Eq_ar_ak_delta, Eq_ar_kv_delta,
                                 Eq_ar_ak_delta_2, Eq_ar_kv_delta_2)
        self.print_non_finite_parameters("After updating ar_prec parameters for step " + str(step))

        ## AR means
        self.cavi_update_ar_mean(Eq_ar_ak_delta, Eq_ar_kv_delta, Eq_ar_ak_delta_2, Eq_ar_kv_delta_2)
        self.print_non_finite_parameters("After updating ar_mean parameters for step " + str(step))


    def get_reconstruction_at_Eqmean(self, inputs, outputs):
        """Evaluates the log probability of word counts given the variational means (Eqmean)
        of the model parameters.

        Args:
            inputs: document and author indices of shape [batch_size]
            outputs: sparse notation of [batch_size, num_words] matrix of word counts.

        Returns:
            reconstruction: [float] a single value of Poisson log-prob evaluated at Eqmean
        """
        # Start with subsetting required samples to authors included in current batch
        ar_ak_bk = self.akt_to_bk(self.ar_ak_varfam.location, inputs['author_indices'], inputs['time_indices'])
        ar_kv_bkv = tf.transpose(tf.gather(self.ar_kv_varfam.location, inputs['time_indices'], axis=-1),
                                 perm=[2, 0, 1])
        exp_ar = tfm.exp(ar_ak_bk[:, :, tf.newaxis] + ar_kv_bkv)
        # [batch_size, num_topics, num_words, num_times]

        rate = tfm.reduce_sum(exp_ar, axis=1)  # sum over all topics (2nd dimension)
        # [batch_size, num_words, num_times]

        count_distribution = tfp.distributions.Poisson(rate=rate)
        # reconstruction = log-likelihood of the word counts
        reconstruction = count_distribution.log_prob(tf.sparse.to_dense(outputs))
        reconstruction = tfm.reduce_sum(reconstruction)

        return reconstruction

    def get_variational_information_criteria(self, dataset, seed=None, nsamples=10):
        """Performs thorough approximation of the individual components of the ELBO.
        Then computes several variational versions of known information criteria.

        Args:
            dataset: sparse notation of [num_documents, num_words] matrix of word counts. Iterator enabled.
            seed: random generator seed for sampling the parameters needed for MC approximation
            nsamples: number of samples per parameters to be sampled,
                        high values result in more precise approximations,
                        but take more time to evaluate

        Returns:
            ELBO_MC: Monte Carlo approximation of the Evidence Lower BOund
            log_prior_MC: Monte Carlo approximation of the log_prior
            entropy_MC: Monte Carlo approximation of the entropy
            reconstruction_MC: Monte Carlo approximation of the reconstruction
            reconstruction_at_Eqmean: reconstruction evaluated at variational means
            effective_number_of_parameters: effective number of parameters
            VAIC: Variational Akaike Information Criterion
            VBIC: Variational Bayes Information Criterion
            seed: seed for random generator
        """
        ### First we need to approximate the ELBO and all its components.
        # Get individual Monte Carlo approximations of rates and log-likelihoods.
        # To spare memory, we have to do it batch by batch.
        entropy = []
        log_prior = []
        reconstruction = []
        reconstruction_at_Eqmean = []

        for step, batch in enumerate(iter(dataset)):
            inputs, outputs = batch
            empty_samples = self.get_empty_samples()
            samples, seed = self.get_samples_and_update_prior_customized(empty_samples, seed=seed, varfam=True,
                                                                         nsamples=nsamples)
            log_prior_batch = self.get_log_prior(samples, nsamples=nsamples)
            entropy_batch = self.get_entropy(samples, nsamples=nsamples, exact=self.exact_entropy)
            rate_batch = self.get_rates(samples, inputs['author_indices'], inputs['time_indices'])
            # Create the Poisson distribution with given rates
            count_distribution = tfp.distributions.Poisson(rate=rate_batch)
            # reconstruction = log-likelihood of the word counts
            reconstruction_batch = count_distribution.log_prob(tf.sparse.to_dense(outputs))

            entropy.append(tfm.reduce_mean(entropy_batch).numpy())
            log_prior.append(tfm.reduce_mean(log_prior_batch).numpy())
            reconstruction.append(tfm.reduce_mean(tfm.reduce_sum(reconstruction_batch, axis=[1, 2])).numpy())
            reconstruction_at_Eqmean.append(self.get_reconstruction_at_Eqmean(inputs, outputs))

        # Entropy and log_prior is computed several times, but it is practically the same, just different samples.
        log_prior_MC = tfm.reduce_mean(log_prior)                # mean over the same quantities in each batch
        entropy_MC = tfm.reduce_mean(entropy)                    # mean over the same quantities in each batch
        reconstruction_MC = tfm.reduce_sum(reconstruction)       # sum over all batches
        ELBO_MC = log_prior_MC + entropy_MC + reconstruction_MC

        # Reconstruction at Eqmean - sum over all batches
        reconstruction_at_Eqmean_sum = tfm.reduce_sum(reconstruction_at_Eqmean)

        # Effective number of parameters
        effective_number_of_parameters = 2.0 * (reconstruction_at_Eqmean_sum - reconstruction_MC)

        ## Variational Akaike Information Criterion = VAIC
        #  AIC = -2*loglik(param_ML)             + 2*number_of_parameters
        #  DIC = -2*loglik(param_posterior_mean) + 2*effective_number_of_parameters
        # VAIC = -2*loglik(param_Eqmean)         + 2*effective_number_of_parameters
        VAIC = -2.0 * reconstruction_at_Eqmean_sum + 2.0 * effective_number_of_parameters

        ## Variational Bayes Information Criterion = VBIC
        #  BIC = -2*loglik(param_ML)             + number_of_parameters * log(sample_size)
        #  BIC = -2*loglik() (param integrated out) + 2*log_prior(param_ML) +...+ O(1/sample_size) (for linear regression)
        # VBIC = -2*ELBO + 2*log_prior
        # VBIC = -2*reconstruction - 2*entropy
        VBIC = -2.0 * ELBO_MC + 2.0 * log_prior_MC
        # todo Question the reasonability of VBIC!

        return ELBO_MC, log_prior_MC, entropy_MC, reconstruction_MC, reconstruction_at_Eqmean_sum, effective_number_of_parameters, VAIC, VBIC, seed

    def akt_to_bk(self, tensor, author_indices, time_indices):
        tensor_bkt = tf.gather(tensor, author_indices, axis=0)
        tensor_bkb = tf.gather(tensor_bkt, time_indices, axis=-1)
        tensor_kb = tf.linalg.diag_part(tf.transpose(tensor_bkb, perm=[1, 0, 2]))
        tensor_bk = tf.transpose(tensor_kb)
        return tensor_bk

    def call(self, inputs, outputs, seed, nsamples):
        """Approximate terms in the ELBO with Monte-Carlo samples.

        Args:
            inputs: A dictionary of input tensors.
            outputs: Sparse batched (author-time)-word counts.
            seed: A seed for the random number generator.
            nsamples: A number of samples to approximate variational means with by Monte Carlo.

        Returns:
            rate: Sampled word count rates of shape [num_samples, batch_size, num_words, num_times].
            log_prior_loss: The negative log prior averaged over samples of shape [1].
            entropy_loss: The negative entropy averaged over samples of shape [1].
            seed: The updated seed.
        """
        # Sample only if at least one of entropy, log_prior, reconstruction is not exact
        if self.exact_reconstruction and self.exact_entropy and self.exact_log_prior:
            samples = self.get_empty_samples()
            nsamples = 1
        else:
            empty_samples = self.get_empty_samples()
            samples, seed = self.get_samples_and_update_prior_customized(empty_samples, seed=seed, varfam=True,
                                                                         nsamples=nsamples)
        # reconstruction
        if self.exact_reconstruction:
            ar_ak_bk = self.akt_to_bk(self.ar_ak_varfam.location, inputs['author_indices'], inputs['time_indices'])
            ar_kv_bkv = tf.transpose(tf.gather(self.ar_kv_varfam.location, inputs['time_indices'], axis=-1),
                                     perm=[2, 0, 1])
            aux_prob_log = ar_ak_bk[:, :, tf.newaxis] + ar_kv_bkv
            # aux_prob_log of shape [batch_size, num_topics, num_words]
            auxiliary_proportions = self.get_cavi_auxiliary_proportions(aux_prob_log)
            ar_ak_cov, ar_ak_var = self.get_ar_cov(self.ar_ak_varfam)
            ar_ak_var_bk = self.akt_to_bk(ar_ak_var, inputs['author_indices'], inputs['time_indices'])
            ar_kv_cov, ar_kv_var = self.get_ar_cov(self.ar_kv_varfam) # [K, V, T]
            ar_kv_var_bkv = tf.transpose(tf.gather(ar_kv_var, inputs['time_indices'], axis=-1), perm=[2, 0, 1])
            reconstruction = tfm.reduce_sum(
                tf.sparse.to_dense(outputs)[:, tf.newaxis, :] * auxiliary_proportions * aux_prob_log -
                tfm.exp(aux_prob_log + 0.5 * ar_ak_var_bk[:, :, tf.newaxis] + 0.5 * ar_kv_var_bkv)
            )
            count_log_likelihood = tf.repeat(reconstruction, nsamples)
        else:
            rate = self.get_rates(samples, author_indices=inputs['author_indices'], time_indices=inputs['time_indices'])
            # shape: [num_samples, batch_size, num_words]
            count_dist = tfp.distributions.Poisson(rate=rate)
            count_log_likelihood = tfm.reduce_sum(count_dist.log_prob(tf.sparse.to_dense(outputs)), axis=[1, 2])

        # log_prior
        if self.exact_log_prior:
            log_prior = tf.repeat(self.get_exact_log_prior(), nsamples)
        else:
            log_prior = self.get_log_prior(samples, nsamples=nsamples)

        # entropy
        entropy = self.get_entropy(samples, nsamples=nsamples, exact=self.exact_entropy)

        # Adjust for the fact that we're only using a minibatch.
        reconstruction_loss = -tfm.reduce_mean(count_log_likelihood) # * self.minibatch_scaling
        log_prior_loss = -tfm.reduce_mean(log_prior)
        entropy_loss = -tfm.reduce_mean(entropy)
        return reconstruction_loss, log_prior_loss, entropy_loss, seed
