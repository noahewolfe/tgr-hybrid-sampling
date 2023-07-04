import bilby
import matplotlib.pyplot as plt
import matplotlib as mpl
from ptemcee import default_beta_ladder
import pandas as pd
import numpy as np

if __name__ == '__main__':
    plt.style.use("tgr-hybrid-sampling.mplstyle")

    names = [
        'hybrid',
        'maxlike',
        'prior'
    ]

    labels = [
        'Hybrid',
        'Maximum\nLikelihood',
        'Prior'
    ]

    hybrid_resume = pd.read_pickle(
        "./data/GW150914_results/150914_d_phi_2_no-overlap_resume.pickle"
    )
    _, _, ndims = hybrid_resume['chain_array'].shape

    with np.load('./data/GW150914_results/150914_ptemcee_initialization_comparison_logposteriorarray.npz') as logp_file:
        maxlike_logp = logp_file['maxlikelihood']
        prior_logp = logp_file['prior']
        maxlike_last_it = logp_file['maxlikelihood_last_iteration']
        prior_last_it = logp_file['prior_last_iteration']

    log_posteriors = [
        hybrid_resume['log_posterior_array'],
        maxlike_logp,
        prior_logp
    ]

    last_iterations = [
        hybrid_resume['iteration'],
        maxlike_last_it,
        prior_last_it
    ]

    for (name, label, log_posterior, last_it) in zip(names, labels, log_posteriors, last_iterations):
        ntemps, _, _ = log_posterior.shape

        mean_log_posterior = np.mean(log_posterior[:,:,:last_it], axis = 1)
        mean_log_posterior -= np.tile( mean_log_posterior[:,-1,None], last_it )

        fig, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$\langle \ln~p(\theta | d) \rangle$ (arbitrary offset)")

        temperatures = default_beta_ladder(ndims, ntemps = ntemps)
        cmap = mpl.cm.plasma
        norm = mpl.colors.LogNorm(vmin = min(temperatures), vmax = max(temperatures))

        for i, temp in enumerate(temperatures):
            yoff = 10 - 2 * i
            ax.plot(
                mean_log_posterior[i,:] + yoff - np.mean(mean_log_posterior[i,-100:]),
                color = cmap(norm(temp))
            )
            ax.axhline(y = yoff, color = 'grey', alpha = 0.5, linewidth = 1 )

        ax.set_ylim(0, 11.2)


        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = norm), ticks = [0.3, 0.4, 0.6, 1.0], format='%.1f')
        cbar.ax.set_ylabel(r"$\beta_T$")

        ax.legend(
            [ mpl.lines.Line2D([], [], linestyle='') ],
            [label],
            handlelength=0,
            handletextpad=0,
            loc = 'upper left',
        )

        plt.tight_layout()

        fig.savefig(f"./figures/meanlogposterior_{name}.pdf")