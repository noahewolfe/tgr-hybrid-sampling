import os

import bilby
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import dill

from tqdm import tqdm

dphi = ["d_phi_0", "d_phi_1", "d_phi_2", "d_phi_3", "d_phi_4", "d_phi_5L", "d_phi_6", "d_phi_6L", "d_phi_7"]
dalpha = ["d_alpha_2", "d_alpha_3", "d_alpha_4"]
dbeta = ["d_beta_2", "d_beta_3"]

all_dpi = dphi + dalpha + dbeta

dphi_labels = [
    "$" + s.replace("d_", "\\delta \\var").replace("5L", "{5L}").replace("6L", "{6L}") + "$" for s in dphi
]

dpi_labels = dphi_labels + [
    "$" + s.replace("d_", "\\delta \\") + "$" for s in (dalpha + dbeta)
]

def combo_violinplot(data, labels, colors, dpi_vals=[0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]):
    fig, axes = plt.subplots(figsize=(20,8),dpi=80, nrows=1,ncols=5)

    for i,v in enumerate([
        [0], [1], [2,3,4,5,6], [7,8], [9,10,11,12,13]
    ]):
        for j in v:
            for color,d in zip(colors, data):
                dist = axes[i].violinplot(d[j], [j])

                for partname in ('cbars','cmins','cmaxes'):
                    dist[partname].set_edgecolor(color)

                for pc in dist['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                
        axes[i].hlines( dpi_vals[j], j - 0.2, j + 0.2, colors=["black"], linestyle=["--"] )

        axes[i].set_xticks(v)
        axes[i].set_xticklabels( dpi_labels[v[0]:v[-1]+1] ) 

        axes[i].tick_params(labelsize=17.5)

    fig.legend(
        [ mpl.patches.Patch(facecolor=color) for color in colors ],
        labels,
        fontsize=15
    )
        
    plt.tight_layout()
    fig.set_figwidth(20)
    fig.set_figheight(10)
    
    return fig, axes


def load_combo_results(parentdir, parentlabel, dpi=all_dpi, extra_hybrid_label=None, extra_hybrid_no_overlap_label="no-overlap"):
    hybrid_label = f"{parentlabel}_hybrid"
    hybrid_results = load_hybrid_results(
        os.path.join(parentdir, hybrid_label), hybrid_label, dpi=dpi, extra_label=extra_hybrid_label
    )
    
    hybrid_overlap0_results = load_hybrid_results(
        os.path.join(parentdir, hybrid_label), hybrid_label, dpi=dpi, extra_label=extra_hybrid_no_overlap_label
    )
    
    nest_results = load_nest_results(parentdir, parentlabel, dpi=dpi)
    nest_overlap0_results = load_nest_results(
        parentdir, parentlabel, extra_label = "no-overlap", dpi=dpi
    )
    
    return hybrid_results, nest_results, nest_overlap0_results, hybrid_overlap0_results

def load_nest_results(parentdir, parentlabel, extra_label=None, dpi=all_dpi):
    nest_results = []

    if extra_label is None:
        nest_dir_fmt = os.path.join(
               parentdir, f"{parentlabel}_%s_only-dynesty"
        )
        nest_res_fmt = f"{parentlabel}_%s_only-dynesty_data0_0-0_analysis_H1L1_dynesty_result.json"
    else:
        nest_dir_fmt = os.path.join(
               parentdir, f"{parentlabel}_%s_only-dynesty_{extra_label}"
        )
        nest_res_fmt = f"{parentlabel}_%s_only-dynesty_{extra_label}_data0_0-0_analysis_H1L1_dynesty_result.json"

    for d in tqdm(dpi):

        nest_dir = nest_dir_fmt % d
        result_path = os.path.join(
            nest_dir,
            "result",
            nest_res_fmt % d
        )

        if os.path.isfile(result_path):
            nest_results.append(
                bilby.core.result.read_in_result(result_path)
            )
        else:
            nest_results.append(None)
            
    return nest_results

    
def load_hybrid_results(outdir, label, dpi=all_dpi, extra_label=None):
    resultdir = os.path.join(outdir, "result")
    result_prefix = f"{label}_data0_0-0_analysis_H1L1_dynesty"
    results = []

    for d in tqdm(dpi):
        if extra_label is None:
            result_path = os.path.join(resultdir, f"{result_prefix}_{d}_result.json")
        else:
            result_path = os.path.join(resultdir, f"{result_prefix}_{d}_{extra_label}_result.json")

        if os.path.isfile(result_path):
            results.append(
                bilby.core.result.read_in_result(result_path)
            )
        else:
            results.append(None)
            
    return results

def load_dpi_resume_data(rundir, result_prefix, search_dpi_idx, dpi = ["d_phi_0", "d_phi_1", "d_phi_2", "d_phi_3", "d_phi_4", "d_phi_5L", "d_phi_6", "d_phi_6L", "d_phi_7", "d_alpha_2", "d_alpha_3", "d_alpha_4", "d_beta_2", "d_beta_3"], extra_label = None):
    
    resultdir = os.path.join(rundir, "result")
    
    resume_data = []

    for d in tqdm(dpi):
        if extra_label is None:
            resume_path = os.path.join(resultdir, f"{result_prefix}_{d}_checkpoint_resume.pickle")
        else:
            resume_path = os.path.join(resultdir, f"{result_prefix}_{d}_{extra_label}_checkpoint_resume.pickle")
        with open(resume_path, "rb") as f:
            resume_data.append(dill.load(f))
        
    nmin = 10
    nmax = 1E10 # a ridiculous number
    for rd in resume_data:
        n = rd["chain_array"].shape[1]
        if n < nmax:
            nmax = n
            
    dpi_data = [
        rd["chain_array"][:,nmin:nmax,search_dpi_idx].T
        for rd in resume_data
    ]
    
    plot_dpi_data = np.array([d.flatten() for d in dpi_data])
    return plot_dpi_data
