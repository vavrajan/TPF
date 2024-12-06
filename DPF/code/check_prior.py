def get_and_check_prior_choice(FLAGS):
    """Create a dictionary of choices of hierarchical priors.
    Also check whether there are no contradictions in the choices.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    prior_choice = {
        "delta": FLAGS.delta,
    }

    if prior_choice["delta"] not in ["AR", "ART", "RW"]:
        raise ValueError("Unrecognized prior distribution for delta (ARcoef parameters). Choose either AR or RW.")

    return prior_choice

def get_and_check_varfam_choice(FLAGS):
    """Create a dictionary of choices of variational families (if choosable, otherwise fixed).
    Also check whether there are no contradictions in the choices.

    Args:
        FLAGS: A dictionary of flags passed as and argument to analysis python script.
    """
    varfam_choice = {
        "ar_ak": FLAGS.varfam_ar_ak,
        "ar_kv": FLAGS.varfam_ar_kv,
    }

    if varfam_choice["ar_ak"] not in ["MVnormal", "normal"]:
        raise ValueError("Unrecognized variational family choice for ar_ak. Choose either MVnormal or normal.")
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
        "ar_ak_mean": {"location": FLAGS.ar_ak_mean_loc, "scale": FLAGS.ar_ak_mean_scl},
        "ar_kv_mean": {"location": FLAGS.ar_kv_mean_loc, "scale": FLAGS.ar_kv_mean_scl},
        "ar_ak_delta": {"location": FLAGS.ar_ak_delta_loc, "scale": FLAGS.ar_ak_delta_scl},
        "ar_kv_delta": {"location": FLAGS.ar_kv_delta_loc, "scale": FLAGS.ar_kv_delta_scl},
        "ar_ak_prec": {"shape": FLAGS.ar_ak_prec_shp, "rate": FLAGS.ar_ak_prec_rte},
        "ar_kv_prec": {"shape": FLAGS.ar_kv_prec_shp, "rate": FLAGS.ar_kv_prec_rte},
    }

    # ar_ak_mean, ar_kv_mean
    if prior_hyperparameter['ar_ak_mean']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_ak_mean:scl is not positive.')
    if prior_hyperparameter['ar_kv_mean']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_kv_mean:scl is not positive.')

    # ar_ak_delta, ar_kv_delta
    if abs(prior_hyperparameter['ar_ak_delta']['location']) > 1:
        raise Warning('Prior location for AR coefficient ar_ak_delta:loc is not in [-1, 1].')
    if abs(prior_hyperparameter['ar_kv_delta']['location']) > 1:
        raise Warning('Prior location for AR coefficient ar_kv_delta:loc is not in [-1, 1].')
    if prior_hyperparameter['ar_ak_delta']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_ak_delta:scl is not positive.')
    if prior_hyperparameter['ar_kv_delta']['scale'] <= 0:
        raise ValueError('Hyperparameter ar_kv_delta:scl is not positive.')

    # ar_ak_prec, ar_kv_prec
    if prior_hyperparameter['ar_ak_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter ar_ak_prec:shp is not positive.')
    if prior_hyperparameter['ar_ak_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter ar_ak_prec:rte is not positive.')
    if prior_hyperparameter['ar_kv_prec']['shape'] <= 0:
        raise ValueError('Hyperparameter ar_kv_prec:shp is not positive.')
    if prior_hyperparameter['ar_kv_prec']['rate'] <= 0:
        raise ValueError('Hyperparameter ar_kv_prec:rte is not positive.')

    return prior_hyperparameter