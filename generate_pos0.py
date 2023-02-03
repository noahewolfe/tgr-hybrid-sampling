import numpy as np
import bilby
from scipy.stats import truncnorm

pos0 = dict()

ntemps = 5
nwalkers = 250
shape = ntemps, nwalkers 

res = bilby.core.result.read_in_result('/home/noah.wolfe/bgr_source_model/paper_runs/gw150914/hybrid/hybrid-gw150914/result/hybrid-gw150914_data0_1126259462-391_analysis_H1L1_result.json')
posterior = res.posterior
max_like_idx = posterior["log_likelihood"].argmax()
max_like_params = dict(posterior.iloc[max_like_idx])
max_log_likelihood  = posterior.iloc[max_like_idx]["log_likelihood"]

def log_prior_probability(sample):
    log_posterior_probability = np.log(res.posterior_probability(sample=sample))
    return log_posterior_probability - max_log_likelihood + res.log_evidence

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
        #pos0[key] = np.tile(max_like_params[key], shape)
        #mu = max_like_params[key]
        #sigma = .1 #np.abs(mu) * 0.01
        pos0[key] = np.random.uniform(low = 1126259462.2910001, high = 1126259462.491, size = shape)
        #truncnorm.rvs(
        #    ((1126259462.2910001) - mu) / sigma, ((1126259462.491) - mu) / sigma,
        #    loc=mu, scale=sigma,
        #    size = shape
        #)
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

#for t in range(ntemps):
#    for w in range(nwalkers):
#        sample = { k : pos0[k][t,w] for k in res.search_parameter_keys }
#        if np.isinf(log_prior_probability(sample)):
#            print(f"Sample is inf! {sample}")

#print(pos0['d_phi_2'])
#print(pos0['H1_time'])
np.savez("./gw150914_maxlikelihood_pos0.npz", **pos0)