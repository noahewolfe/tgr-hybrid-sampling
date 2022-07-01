import bilby
import numpy as np
from scipy.special import gamma as gamma_func, gammaln, loggamma, gdtrix
from scipy.special import logsumexp as logsumexp

import matplotlib as mpl
import matplotlib.pyplot as plt

class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data):
        """
        A very simple Gaussian likelihood.

        Parameters
        ==========
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={'mu': None, 'sigma': None})
        self.data = data
        self.N = len(data)

    def log_likelihood(self):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        res = self.data - mu
        return ( -0.5 * (np.sum((res / sigma)**2) +
                       self.N * np.log(2 * np.pi * sigma**2)) )

class GeneralizedNormalLikelihood(bilby.Likelihood):
    def __init__(self, data):
        """
        A generalized normal distribution: 
        https://en.wikipedia.org/wiki/Generalized_normal_distribution

        Parameters
        ==========
        data: array_like
            The data to analyse
        """
        super().__init__(parameters={'mu': None, 'alpha': None, 'beta': None})
        self.data = data
        self.N = len(data)

    def log_likelihood(self):
        mu = self.parameters['mu']
        alpha = self.parameters['alpha']
        beta = self.parameters['beta']
        
        res = self.data - mu
        
        return ( -1 * np.sum( ( np.abs(res) / alpha )**beta ) 
                + self.N * ( np.log(beta / (2 * alpha)) - loggamma(1/beta) ) )

def ggd_pdf(x, mu, alpha, beta):
    # TODO: docstring
    return (
        (beta / (2 * alpha * gamma_func(1/beta))) * 
        np.exp( -((np.abs(x - mu) / alpha))**beta ) )

def sgd_pdf(x, mu, sigma):
    # TODO: docstring
    return(
        1 / (sigma * np.sqrt(2 * np.pi)) * 
        np.exp( -1/2 * (( (x - mu) / sigma))**2 ) )

def sample_generalized_gaussian(mu, alpha, beta, N):
    """
    Draws N samples from a generalized gaussian distribution specified by
    mean mu, scale alpha, and shape beta.
    See https://en.wikipedia.org/wiki/Generalized_normal_distribution.

    In this implementation, we draw N samples from a uniform distribution on
    [0,1], and then use the inverse cumulative distribution function (CDF) of
    the generalized gaussian to convert this into a generalized gaussian
    distribution of samples.

    Parameters
    ==========
    mu : float
        Mean of the disitribution, like the mean of a standard Gaussian distribution.
    alpha : float
        Scale of the distribution; in the case of a standard Gaussian distribution
        (beta = 2), alpha is related to the standard deviation sigma as 
        sigma = alpha / sqrt(2).
    beta : float
        Shape of the distribution; for example, when beta = 1 we have the
        Laplace distribution, and when beta = 2, we have a standard Gaussian
        distribution.
    N : int
        Number of samples to generate from the generalized gaussian distribution
        specified by mu, alpha, and beta.

    Returns
    =======
    np.ndarray (N,) : saples from a generalized gaussian distribution specified
    by mu, alpha, and beta.
    """

    # Uses the inverse CDF to generate a generalized gaussian distribution
    # Note: scipy and wikipedia switch the order of the shape, scale parameters!
    # https://en.wikipedia.org/wiki/Generalized_normal_distribution
    # https://en.wikipedia.org/wiki/Gamma_distribution
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gdtrix.html#scipy.special.gdtrix
    rand_draws = np.random.uniform(0,1,size=N)
    return np.sign( rand_draws - 0.5 ) * gdtrix( 
        1/(alpha**beta), 
        1/beta, 
        2 * np.abs(rand_draws - 0.5) )**(1/beta) + mu

def alpha_to_sigma(alpha, beta=2):
    """
    Convert from the generalized Gaussian scale parameter, alpha, to that of the
    standard Gaussian, the standard deviation sigma.

    Parameters
    ==========
    alpha : float
        Generalized Gaussian scale parameter.

    Returns
    =======
    float : scale sigma of a standard Gaussian under alpha and beta.
    """
    return np.exp(gammaln(3 / beta) - gammaln(1 / beta))**0.5 * alpha

def sigma_to_alpha(sigma, beta=2):
    # TODO: docstring
    return sigma / np.exp(gammaln(3 / beta) - gammaln(1 / beta))**0.5

def temper_posterior_weights(nlive, log_likelihood, beta_T):
    # TODO: docstring

    # temper the initial samples in {mu, sigma} from dynesty,
    # to prepare them as initial points for ptemcee
    nnest = len(log_likelihood)
    ntemps = len(beta_T)

    # TODO: this code needs some explanation, plus some math in a cell above it!
    tempered_posterior_weights = np.zeros((ntemps, nnest))
    for j in range(ntemps):
        log_likelihood_beta = log_likelihood * beta_T[j]

        i = np.arange(0, nnest, 1)
        logw = -1 * i / nlive + np.log( np.exp(1/nlive) - 1 )
            
        a = log_likelihood_beta + logw
        logZ = logsumexp( a )
        logp = log_likelihood_beta + logw - logZ
        p = np.exp(logp)

        tempered_posterior_weights[j,:] = p

    return tempered_posterior_weights

def generate_ptemcee_pos0(ntemps, nwalkers, ndims, tempered_posterior_weights, nested_samples):
    # TODO: docstring
    
    nnest  = len(nested_samples)
    pos0 = np.zeros((ntemps, nwalkers, ndims))

    for j in range(ntemps):
        rand_sample_idxs = np.random.choice(
            nnest, size=nwalkers, 
            p=tempered_posterior_weights[j] / np.sum(tempered_posterior_weights[j])
        )

        pos0[j,:,0] = nested_samples["mu"][rand_sample_idxs]
        pos0[j,:,1] = sigma_to_alpha( nested_samples["sigma"][rand_sample_idxs] )
        pos0[j,:,2] = np.random.normal(loc=2, scale=0.1, size=nwalkers)

    return pos0

def plot_tempered_posterior_weights(beta_T, tempered_posterior_weights):
    # TODO: docstring
    ntemps = len(beta_T)

    # plot the tempered posterior weights
    # generates Figure XX from the report
    fig, ax = plt.subplots()
    ax.set_xlabel("Sample Number")
    ax.set_ylabel(r"Tempered Posterior Weights")

    cmap = mpl.cm.plasma
    norm = mpl.colors.LogNorm(vmin = min(beta_T), vmax = max(beta_T))
    colors = cmap(norm(beta_T))

    for j in range(ntemps):
        ax.plot( tempered_posterior_weights[j,:], color=colors[j] )

    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = norm))
    cbar.ax.set_ylabel(r"$\beta_T$")
    cbar.ax.tick_params()

    return fig, ax

def calc_posterior_predictive_distribution(x, samples, pdf):
    """
    # TODO: this!

    Parameters
    ==========
    x : array_like
        Test values over which to evaluate the probability density function at
        each sample from the posterior distribution.
    samples : array_like, shaped (nsamples, ndims)
        Samples from a joint posterior distribution.
    pdf : callable
        Probability density function to evaluate at each sample in the joint
        posterior distribution.
    """

    nsamples = len(samples)
    ntest = len(x)
    ppd_samples = np.zeros((nsamples, ntest))

    for i, sample in enumerate(samples):
        ppd_samples[i] = pdf(x, *sample)

    return ppd_samples

def plot_posterior_predictive_distribution(truth, data, hybrid_dynesty_samples, hybrid_ptemcee_samples, dynesty_samples):
    x = np.linspace(-10, 16, 1000)

    ppd_dynesty        = calc_posterior_predictive_distribution(x, dynesty_samples, ggd_pdf)
    ppd_hybrid_dynesty = calc_posterior_predictive_distribution(x, hybrid_dynesty_samples, sgd_pdf)
    ppd_hybrid_ptemcee = calc_posterior_predictive_distribution(x, hybrid_ptemcee_samples, ggd_pdf)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    fig, ax = plt.subplots( figsize=(12,8), dpi=300 )

    # dynesty (generalized)
    y1 = np.quantile( ppd_dynesty.T, 0.05, axis=1 )
    y2 = np.quantile( ppd_dynesty.T, 0.95, axis=1 )
    ax.fill_between(x, y1, y2, alpha=0.5, color=colors[2])

    # dynesty (normal)
    y1 = np.quantile( ppd_hybrid_dynesty.T, 0.05, axis=1 )
    y2 = np.quantile( ppd_hybrid_dynesty.T, 0.95, axis=1 )
    ax.fill_between(x, y1, y2, alpha=0.5, color=colors[0])

    # ptemcee (generalized)
    y1 = np.quantile( ppd_hybrid_ptemcee.T, 0.05, axis=1 )
    y2 = np.quantile( ppd_hybrid_ptemcee.T, 0.95, axis=1 )
    ax.fill_between(x, y1, y2, alpha=0.5, color=colors[1])

    ax.plot( x, ggd_pdf(x, *truth), label="True", linewidth=3.5, color=colors[3])

    ax.plot( x, ggd_pdf(x, *np.mean(dynesty_samples, axis=0)),        label="Generalized - only dynesty", linewidth=3.5, color=colors[2])
    ax.plot( x, sgd_pdf(x, *np.mean(hybrid_dynesty_samples, axis=0)), label="Gaussian - hybrid step 1", linewidth=3.5, color=colors[0])
    ax.plot( x, ggd_pdf(x, *np.mean(hybrid_ptemcee_samples, axis=0)), label="Generalized - hybrid step 2", linewidth=3.5, color=colors[1])

    ax.set_yticklabels([])
    ax.tick_params(labelsize=17.5)

    ax.hist(data, density=True, histtype="step", bins=10, label="Data")

    plt.legend(fontsize=15)

    ax.set_ylim(0)

    ax.set_xlabel(r"$x$", fontsize=17.5)
    ax.set_ylabel(r"$P(x)$", fontsize=17.5)

    return fig, ax

    