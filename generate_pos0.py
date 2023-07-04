"""
Generate initial points (pos0) around the maximum likelihood estimate from the analysis
of GW150914 assuming that GR is correct (i.e. from the initial dynesty analysis).
"""

import numpy as np
import bilby
from scipy.stats import truncnorm

def log_prior_probability(sample):
    log_posterior_probability = np.log(res.posterior_probability(sample=sample))
    return log_posterior_probability - max_log_likelihood + res.log_evidence

if __name__ == '__main__':

    pos0 = dict()

    ntemps = 5
    nwalkers = 250
    shape = ntemps, nwalkers 

    res = bilby.core.result.read_in_result('/home/noah.wolfe/bgr_source_model/paper_runs/gw150914/hybrid/hybrid-gw150914/result/hybrid-gw150914_data0_1126259462-391_analysis_H1L1_result.json')
    posterior = res.posterior
    max_like_idx = posterior["log_likelihood"].argmax()
    max_like_params = dict(posterior.iloc[max_like_idx])
    max_log_likelihood  = posterior.iloc[max_like_idx]["log_likelihood"]

    for key in res.search_parameter_keys:
        if key in ["mass_ratio", "a_1", "a_2"]:
            mu = max_like_params[key]
            sigma = np.abs(mu) * 0.01
            pos0[key] = truncnorm.rvs(
                -sigma, (1 - mu) / sigma,
                loc=mu, scale=sigma,
                size = shape
            )
        elif key == "H1_time":
            pos0[key] = np.random.uniform(
                low = 1126259462.2910001,
                high = 1126259462.491,
                size = shape
            )
        else:
            mu = max_like_params[key]
            sigma = np.abs(mu) * 0.01
            pos0[key] = truncnorm.rvs(
                -sigma, sigma,
                loc=mu, scale=sigma,
                size = shape
            )

    mu = 0
    sigma = 0.001
    pos0['d_phi_2'] = truncnorm(
        -sigma, sigma,
        loc=mu, scale=sigma
    ).rvs(size = shape)

    np.savez("./config/gw150914_maxlikelihood_pos0.npz", **pos0)