import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HYBRID_COLOR = "#3a0ca3"
HYBRID_OVERLAP0_COLOR = "#b5179e"
COMP_COLOR = "#4895ef"

# nice shades of purple that end in HYBRID_COLOR
HYBRID_EVOL_COLORS = ["#DEB1FC", "#BC63F8", "#933FF3", "#6C1AEF", "#3A0CA3"]
HYBRID_INIT_COLOR = "#f72585"

def dpi_key_to_label(key):
    """ Helper method to convert dpi names in bilby labels, prior files, etc. 
    to a latex label; e.g. d_phi_2 -> $\\delta \\varphi_2$. """

    return (
        "$" + key.replace('d_', '\\delta \\')
        .replace('phi', 'varphi')
        .replace('5L', '{5l}')
        .replace('6L', '{6l}') + "$"
    )

def get_bilby_longname(label, trigger_times=[0]):
    """ Compute the full label used by bilby to name output files. """

    longnames = []
    for i,trigger_time in enumerate(trigger_times):
        trigger_time_str = "-".join(str(float(trigger_time)).split("."))
        longnames.append( f"{label}_data{i}_{trigger_time_str}_analysis_H1L1" )

    if len(longnames) == 1:
        return longnames[0]
    else:
        return longnames

def plot_single_evolution(
    hybrid_result, hybrid_resume, xparam, yparam, xlabel, ylabel, xtrue=None, ytrue=None, 
    pos0=None, iterations=None, xidx_override=None):
    """ Plot ptemcee sampling of the joint posterior of yparam vs. xparam
    over successive sampling iterations. """
    
    xidx = hybrid_result.search_parameter_keys.index(xparam) if xidx_override is None else xidx_override
    y_hybrid_idx = hybrid_result.search_parameter_keys.index(yparam)
    
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex="col", sharey="row")

    if iterations is None:
        iterations = np.array([1, 2, 4, 8, 16, 128]) - 1
    
    axes[0, 1].axis('off')
    
    hybrid_colors = np.array([HYBRID_INIT_COLOR] + HYBRID_EVOL_COLORS[:len(iterations) - 1], dtype="object")
        
    hybrid_chain = hybrid_resume["chain_array"]
    
    ### top hists ###
    ax = axes[0, 0]
    for color, it in zip(hybrid_colors, iterations):
        if it == 0 and pos0 is not None:
            xdata = pos0[0,:,xidx]
        else:
            xdata = hybrid_chain[:,it,xidx]
        ax.hist(xdata, density=True, color=color, histtype="step", zorder=0)

    if xtrue is not None:
        ax.axvline(x=xtrue, linestyle="--", color="black", zorder=-1)
            
    ### left hists ###
    ax = axes[1, 1]
    for color,it in zip(hybrid_colors,iterations):
        if it == 0 and pos0 is not None:
            ydata = pos0[0,:,y_hybrid_idx]
        else:
            ydata = hybrid_chain[:,it,y_hybrid_idx]
        ax.hist(
            ydata, density=True, color=color, histtype="step",
            orientation="horizontal",
            zorder=0,
        )

    if ytrue is not None:
        ax.axhline(y=ytrue, linestyle="--", color="black", zorder=-1)
    
    ### interior plots ###
    ax = axes[1, 0]
    #if pos0 is None:
    #    ax.scatter(
    #        xdata, ydata,
    #        s=2,
    #        c=color,
    #        zorder=0
    #    )
    #else:
    #    ax.scatter( pos0[0,:,xidx], pos0[0,:,y_hybrid_idx], s=2, c=color, zorder=0 )

    for color, it in zip(hybrid_colors,iterations):
        xdata = hybrid_chain[:,it,xidx]
        ydata = hybrid_chain[:,it,y_hybrid_idx]
        
        if xtrue is not None:
            ax.axvline(x=xtrue, linestyle="--", color="black", zorder=-1)

        if ytrue is not None:
            ax.axhline(y=ytrue, linestyle="--", color="black", zorder=-1)

        ax.scatter(
            xdata, ydata,
            s=2, c=color,
            zorder=1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig

def violinplot(runs, param_key, truth, param_label,
    square_y = True, share_y=False, 
    colors = [HYBRID_COLOR, HYBRID_OVERLAP0_COLOR, COMP_COLOR],
    labels = [ "Hybrid", "Hybrid (no overlap)", "Dynesty" ],
    linewidths = [ 6, 3, 1 ],
    linestyles = [ "-", "--", "-" ],
    fig_kwargs=dict(), 
    dpi_panels = [["d_phi_0"], ["d_phi_1"], ["d_phi_2","d_phi_3"], ["d_phi_4","d_phi_5L", "d_phi_6"], ["d_phi_6L", "d_phi_7"], ["d_alpha_2", "d_alpha_3", "d_alpha_4", "d_beta_2", "d_beta_3"] ]):
    """ Make a violinplot collating results from multiple runs / postprocessing 
    jobs estimating beyond-GR deviation parameters $\delta p_i$.
    
    Parameters
    ==========
    runs : dict
        Dictionary with tuple keys of (dpi, overlap, runtype), all strings,
        where dpi indicates the beyond-GR deviation parameter estimated in that
        run, overlap indicates the overlap cut applied (probably "0.0" or "0.9"),
        and runtype is either "hybrid" or "comp" for a hybrid or a comparison run
        (to the hybrid run), respectively.
        The value associated with this tuple key should be the corresponding
        `bilby.core.result.Result object`.

    param_key : str
        "dpi" or the name of another search parameter estimated during the plotted runs;
        if "dpi", for each deviation parameter run, the violinplot will reflect
        the posterior estimated for that deviation parameter.
        if another parameter, e.g. "chirp mass", the violinplots will reflect
        the posterior estimated for that parameter from each deviation parameter
        run.

    truth : dict or float 
        If dict, a dictionary whose keys are deviation parameters dpi, 
        indicating the true value of the parameter plotted for each dpi-estimating run.
        Otherwise, the same truth value is plotted for each run.

    param_label : str
        String (preferrably in LaTeX style) with which to label the y-axis,
        denoting the value associated with `param_key`.

    square_y : bool, optional
        Whether to square the axes y-limits about zero. (good for plotting dpi)
    
    share_y : bool, optional
        Whether to share the same y-axis values among subplots, or not. 
        Use this instead of `sharey` in `fig_kwargs`.

    colors : list of str, optional
        Colors to plot for the hybrid run w/ overlap >= 0.9 results,
        the hybrid run w/ overlap == 0.0 results, and the comparison run
        results. Must match length of labels, and both must be <= 3 in length,
        if labels is not None.

    labels : list of str, optional
        Labels for the hybrid run w/ overlap >= 0.9 results,
        the hybrid run w/ overlap == 0.0 results, and the comparison run
        results. Must match length of colors, and both must be <= 3 in length.
        If None, no legend will be generated for the figure.

    fig_kwargs : dict, optional
        Additional keyword arguments for the `plt.subplots` call.

    dpi_panels : list (of lists of str), optional
        List of lists of str, where each sublist represents a subplot in the
        overall figure, and the strings correspond to runs estimating that
        deviation parameter, and noting their position in the subplot.

    Returns
    =======
    fig, axes : `matplotlib.figure.Figure object` and list of `matplotlib.axes.Ax`
    objects in that figure.
    """

    gs = mpl.gridspec.GridSpec(
        1, len(dpi_panels),
        width_ratios = [ len(panel) for panel in dpi_panels ], 
    )

    fig = plt.figure(**fig_kwargs)
    axes = [ fig.add_subplot(gs[i]) for i in range(len(dpi_panels)) ]
    if share_y:
        for i in range(1, len(dpi_panels)):
            axes[i-1].get_shared_y_axes().join(axes[i-1], axes[i])

    for run, result in reversed(runs.items()): # reversed plots overlap0 on top of 0.9 overlap runs
        dpi, overlap, runtype = run

        _param_key = dpi if param_key == "dpi" else param_key
        samples = result.samples[ :, result.search_parameter_keys.index(_param_key) ]

        # set plotting colors and other parameters based on how the result was generated
        _truth = truth[dpi] if type(truth) == dict else truth
        truth_color = "#d3d3d3" if _truth == 0 else "black"

        if runtype == "comp":
            facecolor = colors[2]
            edgecolor = colors[2]
            alpha = 0.75
            linewidth = linewidths[2]
            linestyle = linestyles[2]
        else:
            facecolor = "none"
            edgecolor = colors[0] if float(overlap) > 0 else colors[1]
            linewidth = linewidths[0] if float(overlap) > 0 else linewidths[1]
            linestyle = linestyles[0] if float(overlap) > 0 else linestyles[1]
            alpha = 1


        # select the ax object to plot in based on list in dpi_panels that dpi is in
        # use the integer index of dpi in that list as the x-axis position of the violin
        pidx, x = [ (i, panel.index(dpi)) for i,panel in enumerate(dpi_panels) if dpi in panel ][0]
        ax = axes[pidx]

        # truth line
        ax.hlines(
            _truth, x - 0.5, x + 0.5, 
            colors=[truth_color], linestyles=["--"], linewidths=[5], alpha=[0.3] 
        )
        
        # vertical line, in light gray, to guide the eye (to correlate dpi label to plot)
        ax.axvline(x = x, color = "#d3d3d3", zorder = -1)

        # plot and format violin
        dists = ax.violinplot(samples, [x], quantiles=[0.05, 0.95], showextrema=False)
        for partname in ('cquantiles',):
            dists[partname].set_edgecolor(edgecolor)

        for pc in dists['bodies']:
            pc.set_facecolor(facecolor)
            pc.set_edgecolor(edgecolor)
            pc.set_linewidth(linewidth)
            pc.set_linestyle(linestyle)
            pc.set_alpha(alpha)

    # center y-axis on zero, label x-axis of each subplot
    for i,(panel, ax) in enumerate(zip(dpi_panels, axes)):
        if square_y:
            ymax = np.max(np.abs( list(ax.get_ylim()) ))
            ax.set_ylim(-ymax, ymax)

        # don't add tick labels if we're not the leftmost subplot, when share_y
        if share_y and i > 0:
            ax.set_yticklabels([])

        ax.set_xticks([ i for i in range(len(panel)) ])
        ax.set_xticklabels([ dpi_key_to_label(dpi_key) for dpi_key in panel ])

    axes[0].set_ylabel(param_label)

    if labels is not None:
        axes[-1].legend(
            [ 
                mpl.patches.Patch(facecolor=color, linestyle=lstyle, linewidth=lwidth) 
                if (i == 2) else mpl.patches.Patch(facecolor="white", edgecolor=color, linestyle=lstyle, linewidth=lwidth) 
                for i,(color, lstyle, lwidth) in enumerate(zip(colors, linestyles, linewidths))
            ],
            labels
        )
        
    plt.tight_layout()

    return fig, axes


def plot_multiple_lower_dim(high_dim_results, low_dim_result, **kwargs):
    import bilby
    import corner
    from copy import deepcopy

    ndim = len(kwargs["parameters"])

    init_kwargs = deepcopy(kwargs)
    init_kwargs["colours"] = init_kwargs["colours"][:-1]
    init_kwargs["labels"]  = init_kwargs["labels"][:-1]

    fig = bilby.core.result.plot_multiple(
        high_dim_results,
        **init_kwargs
    )

    extra_dim_range = fig.get_axes()[-1].get_xlim()

    # stolen from bilby
    low_dim_kwargs = dict(
        bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
        titles=False, quantiles=None,
        truth_color="black",
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False, plot_datapoints=True, fill_contours=True,
        max_n_ticks=3, hist_kwargs=dict(density=True)
    )
    low_dim_kwargs["color"] = kwargs["colours"][-1]

    xs = low_dim_result.posterior[kwargs["parameters"]].values
    fig = corner.corner(xs, fig = fig, **low_dim_kwargs)
    axes = fig.get_axes()

    axes[-1].set_xlim(extra_dim_range)
    for k in range(1, ndim):
        axes[-1 - k].set_ylim(extra_dim_range)
    
    axes[ndim - 1].legend(
        [ 
            mpl.lines.Line2D([], [], color=c)
            for c in kwargs["colours"]
        ],
        kwargs["labels"]
    )

    return fig