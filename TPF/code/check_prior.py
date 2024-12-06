def get_and_check_prior_choice(FLAGS):
    """Create a dictionary of choices of hierarchical priors.
    Also check whether there are no contradiction in the choices.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    prior_choice = {
        "theta": FLAGS.theta,
        "delta": FLAGS.delta,
    }

    if prior_choice['theta'] == 'Gfix':
        raise Warning('The model does not adjust for any verbosity at all. '
                      'Consider more hierarchical prior for theta (Garte, Gdrte) instead.')

    return prior_choice

def get_and_check_varfam_choice(FLAGS):
    """Create a dictionary of choices of variational families (if choosable, otherwise fixed).
    Also check whether there are no contradictions in the choices.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    varfam_choice = {
        "ar_kv": FLAGS.varfam_ar_kv,
    }

    if varfam_choice["ar_kv"] not in ["MVnormal", "normal"]:
        raise ValueError("Unrecognized variational family choice for ar_kv. Choose either MVnormal or normal.")

    return varfam_choice

def get_and_check_prior_hyperparameter(FLAGS):
    """Create a dictionary of hyperparameters for hierarchical priors.
    Also check whether their values fit the necessary conditions.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    prior_hyperparameter = {
        "theta": {"shape": FLAGS.theta_shp, "rate": FLAGS.theta_rte},
        "theta_rate": {"shape": FLAGS.theta_rate_shp, "rate": FLAGS.theta_rate_shp / FLAGS.theta_rate_mean},
        "ar_kv_mean": {"location": FLAGS.ar_kv_mean_loc, "scale": FLAGS.ar_kv_mean_scl},
        "ar_kv_delta": {"location": FLAGS.ar_kv_delta_loc, "scale": FLAGS.ar_kv_delta_scl},
        "ar_kv_prec": {"shape": FLAGS.ar_kv_prec_shp, "rate": FLAGS.ar_kv_prec_rte},
    }

    # theta
    if prior_hyperparameter['theta']['shape'] <= 0:
        raise ValueError('Hyperparameter theta:shp is not positive.')
    if prior_hyperparameter['theta']['rate'] <= 0:
        raise ValueError('Hyperparameter theta:rte is not positive.')
    if prior_hyperparameter['theta_rate']['shape'] <= 0:
        raise ValueError('Hyperparameter theta_rate:shp is not positive.')
    if prior_hyperparameter['theta_rate']['rate'] <= 0:
        raise ValueError('Hyperparameter theta_rate:rte is not positive.')

    # ar_kv_mean
    if prior_hyperparameter['ar_kv_mean']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_kv_mean:scl is not positive.')

    # ar_kv_delta
    if abs(prior_hyperparameter['ar_kv_delta']['location']) > 1:
        raise Warning('Prior location for AR coefficient ar_kv_delta:loc is not in [-1, 1].')
    if prior_hyperparameter['ar_kv_delta']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_kv_delta:scl is not positive.')

    # ar_kv_prec
    if prior_hyperparameter['ar_kv_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter ar_kv_prec:shp is not positive.')
    if prior_hyperparameter['ar_kv_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter ar_kv_prec:rte is not positive.')

    return prior_hyperparameter