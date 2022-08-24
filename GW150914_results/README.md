`150914_hybrid_step1_time-maximized.hdf5` : posterior from the first hybrid step with `dynesty`, assuming only GR, and providing the initial points for sampling with `ptemcee` (in the second step, we maximize over time when calculating the overlap, but the time marginalization doesn't come in during the first step).

`150914_hybrid_step2_time-maximized.hdf5` : posterior from the second hybrid step with `ptemcee`, allowing for beyond-GR deviations, initialized with the (tempered) posterior found in `150914_hybrid_step1_time-maximized.hdf5`. When the overlap is calculated during this second step, we maximize over time.

`150914_d_phi_2_no-overlap_time-maximized_resume.pickle` : Resume file from `bilby`; a pickled `python` dictionary, where the value of `chain_array` contains samples from the `ptemcee` chain, during the second step of hybrid sampling, during which we apply no overlap cut. Technically, if the overlap were to be calculated during this step, we would maximize over the time when it is calculated, though we don't calculate this overlap when the cut is at 0.

Source run (for above files): `/home/noah.wolfe/bgr_source_model/paper_runs/gw150914/hybrid/hybrid-gw150914`

---

`150914_hybrid_step1.hdf5`, `150914_hybrid_step2.hdf5` : same as the posterior files above, but we only maximize over phase when calculating the overlap during the second step. Also, the step2 posterior only contains samples of dphi2.

Source run (for above files): `/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914`

---

`150914_hybrid_step2_resume.hdf5` : saved the 800 - 1500th samples from the chain_array (flattened across all walkers) of the resume files found in `/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914`, as not all of these runs wrote result files