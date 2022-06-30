import os
import bilby

rundir = "/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_delta-phi2_GW150914/"
result_dir = os.path.join(rundir, "result")

dynesty_result_path = os.path.join(
    result_dir,
    "hybrid_pure-gr_delta-phi2_GW150914_data0_1126259462-391_analysis_H1L1_dynesty_result.json"
)

dphi_result_path = os.path.join(
    result_dir,
    "hybrid_pure-gr_delta-phi2_GW150914_data0_1126259462 391_analysis_H1L1_dynesty_delta-phi3_result.json"
)

res_nest = bilby.core.result.read_in_result(dynesty_result_path)
res_pt = bilby.core.result.read_in_result(dphi_result_path)
res_pt.plot_corner(parameters=["chirp_mass", "mass_ratio", "luminosity_distance", "delta-phi3"], filename="test.png")