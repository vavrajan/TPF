# Import global packages
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def train_step(model, inputs, outputs, optim, seed, step=None):
    """Perform a single training step.

    Args:
        model: The DPF model.
        inputs: A dictionary of input tensors.
        outputs: A sparse tensor containing word counts.
        optim: An optimizer.
        seed: The random seed.
        step: The current step.

    Returns:
        total_loss: The total loss for the minibatch (the negative ELBO, sampled with Monte-Carlo).
        reconstruction_loss: The reconstruction loss (negative log-likelihood), sampled for the minibatch.
        log_prior_loss: The negative log prior.
        entropy_loss: The negative entropy.
    """
    # Perform CAVI updates.
    model.perform_cavi_updates(inputs, outputs, step)
    # Approximate the ELBO and tape the gradients.
    with tf.GradientTape() as tape:
        reconstruction_loss, log_prior_loss, entropy_loss, seed = model(inputs, outputs, seed, model.num_samples)
        total_loss = reconstruction_loss + log_prior_loss + entropy_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    model.print_non_finite_parameters("After applying stochastic gradient updates for step " + str(step))

    return total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed