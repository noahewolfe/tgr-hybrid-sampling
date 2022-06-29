import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    hybrid_result, hybrid_resume, xparam, yparam, xlabel, ylabel, xtrue, ytrue, 
    pos0=None, iterations=None):
    """ Plot ptemcee sampling of the joint posterior of yparam vs. xparam
    over successive sampling iterations. """
    
    xidx = hybrid_result.search_parameter_keys.index(xparam)
    y_hybrid_idx = hybrid_result.search_parameter_keys.index(yparam)
    
    fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, sharex="col", sharey="row")

    if iterations is None:
        iterations = np.array([1, 2, 4, 8, 16, 128]) - 1
    
    axes[0, 1].axis('off')
    
    hybrid_colors = np.array(["#f72585"] + [ "#B450F7", "#9B15F4", "#7D19F0", "#6210E5", "#3a0ca3" ], dtype="object")
        
    hybrid_chain = hybrid_resume["chain_array"]
    
    ### top hists ###
    ax = axes[0, 0]
    for color, it in zip(hybrid_colors, iterations):
        if it == 0 and pos0 is not None:
            xdata = pos0[0,:,xidx]
        else:
            xdata = hybrid_chain[:,it,xidx]
        ax.hist(xdata, density=True, color=color, histtype="step", zorder=0)

    ax.axvline(x=xtrue, linestyle="--", color="black", zorder=-1)
            
    ### left hists ###
    ax = axes[1, 1]
    for color,it in zip(hybrid_colors[:-1],iterations[:-1]):
        if it == 0 and pos0 is not None:
            ydata = pos0[0,:,y_hybrid_idx]
        else:
            ydata = hybrid_chain[:,it,y_hybrid_idx]
        ax.hist(
            ydata, density=True, color=color, histtype="step",
            orientation="horizontal",
            zorder=0,
        )

    ax.axhline(y=ytrue, linestyle="--", color="black", zorder=-1)
    
    ### interior plots ###
    ax = axes[1, 0]
    if pos0 is None:
        ax.scatter(
            xdata, ydata,
            s=2,
            c=color,
            zorder=0
        )
    else:
        ax.scatter( pos0[0,:,xidx], pos0[0,:,y_hybrid_idx], s=2, c=color, zorder=0 )
    for color, it in zip(hybrid_colors,iterations):
        xdata = hybrid_chain[:,it,xidx]
        ydata = hybrid_chain[:,it,y_hybrid_idx]
        
        ax.axvline(x=xtrue, linestyle="--", color="black", zorder=-1)
        ax.axhline(y=ytrue, linestyle="--", color="black", zorder=-1)
        ax.scatter(
            xdata, ydata,
            s=2, c=color,
            zorder=1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, axes

def violinplot(runs, param_key, truth, param_label, comp_label, square_y = True, share_y=False, hybrid_color="#3a0ca3", hybrid_overlap0_color="#b5179e", comp_color = "#4895ef", fig_kwargs=dict(), dpi_panels = [["d_phi_0"], ["d_phi_1"], ["d_phi_2","d_phi_3"], ["d_phi_4","d_phi_5L", "d_phi_6"], ["d_phi_6L", "d_phi_7"], ["d_alpha_2", "d_alpha_3", "d_alpha_4", "d_beta_2", "d_beta_3"] ]):
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

    comp_label : str
        Label on figure legend to refer to the comparison run results.

    square_y : bool, optional
        Whether to square the axes y-limits about zero. (good for plotting dpi)
    
    share_y : bool, optional
        Whether to share the same y-axis values among subplots, or not. 
        Use this instead of `sharey` in `fig_kwargs`.

    hybrid_color : str, optional
        Color to plot hybrid run (where overlap >= 0.9) results with.

    hybrid_overlap0_color : str, optional
        Color to plot hybrid run (where overlap = 0.0) results with.
    
    comp_color : str, optional
        Color to plot comparison run of hybrid results with.

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
            facecolor = comp_color
            edgecolor = comp_color
            linewidth = 1
            alpha = 0.75
        else:
            facecolor = "none"
            edgecolor = hybrid_color if float(overlap) > 0 else hybrid_overlap0_color
            linewidth = 3
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
            pc.set_alpha(alpha)

    # center y-axis on zero, label x-axis of each subplot
    for panel, ax in zip(dpi_panels, axes):
        if square_y:
            ymax = np.max(np.abs( list(ax.get_ylim()) ))
            ax.set_ylim(-ymax, ymax)

        ax.set_xticks([ i for i in range(len(panel)) ])
        ax.set_xticklabels([ dpi_key_to_label(dpi_key) for dpi_key in panel ])

    axes[0].set_ylabel(param_label)
    axes[-1].legend(
        [ 
            mpl.patches.Patch(facecolor=hybrid_color), 
            mpl.patches.Patch(facecolor=hybrid_overlap0_color), 
            mpl.patches.Patch(facecolor=comp_color) 
        ],
        [ "Hybrid", "Hybrid (no overlap)", comp_label ]
    )
        
    plt.tight_layout()

    return fig, axes