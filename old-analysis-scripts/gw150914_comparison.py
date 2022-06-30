import bilby
import corner
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams[""]

print("Loading bilby results...")
gr_nest = bilby.core.result.read_in_result("/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914/result/hybrid_pure-gr_GW150914_data0_1126259462-391_analysis_H1L1_dynesty_result.json")
dphi2_pt = bilby.core.result.read_in_result("/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914/result/hybrid_pure-gr_GW150914_data0_1126259462-391_analysis_H1L1_dynesty_d_phi_2_no-overlap_result.json")
dalpha2_pt = bilby.core.result.read_in_result("/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914/result/hybrid_pure-gr_GW150914_data0_1126259462-391_analysis_H1L1_dynesty_d_alpha_2_no-overlap_result.json")

bilby.core.result.plot_multiple(
    [gr_nest, dphi2_pt],
    labels=["step 1 (dynesty)", "step 2 (ptemcee)"],
    parameters=["chirp_mass", "mass_ratio", "azimuth", "zenith"]
)


print("Plotting dynesty-step (GR result)...")

# corner w/ plot kwargs stolen from bilby docs
fig = corner.corner(
    np.column_stack(( gr_nest.samples[:,0], gr_nest.samples[:,1] )), 
    **dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

print("Plotting hybrid dphi2 result against GR result...")

fig = corner.corner(
    np.column_stack(( dphi2_pt.samples[:,0], dphi2_pt.samples[:,1] )),
    **dict(
        fig=fig, bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#FF8C00',
        truth_color='black', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

fig.savefig("./dphi2_gr_comp.png")
plt.close()

print("Plotting hybrid dalpha2 result against GR result...")

fig = corner.corner(
    np.column_stack(( gr_nest.samples[:,0], gr_nest.samples[:,1] )), 
    **dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

fig = corner.corner(
    np.column_stack(( dalpha2_pt.samples[:,0], dalpha2_pt.samples[:,1] )),
    **dict(
        fig=fig, bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#FF8C00',
        truth_color='black', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

fig.savefig("./dalpha2_gr_comp.png")
plt.close()

print("Plotting hybrid results together...")

fig = corner.corner(
    np.column_stack(( dphi2_pt.samples[:,0], dphi2_pt.samples[:,1] )), 
    **dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#0072C1',
        truth_color='tab:orange', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

fig = corner.corner(
    np.column_stack(( dalpha2_pt.samples[:,0], dalpha2_pt.samples[:,1] )),
    **dict(
        fig=fig, bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16), color='#FF8C00',
        truth_color='black', quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
)

fig.savefig("./dphi2_dalpha2_comp.png")