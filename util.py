import numpy as np
import matplotlib.pyplot as plt

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