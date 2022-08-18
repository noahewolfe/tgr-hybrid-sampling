`150914_hybrid_step1_time-marginalized.hdf5` : posterior from the first hybrid step with `dynesty`, assuming only GR, and providing the initial points for sampling with `ptemcee` (in the second step, we marginalize over time when calculating the overlap, but the time marginalization doesn't come in during the first step).

`150914_hybrid_step2_time-marginalized.hdf5` : posterior from the second hybrid step with `ptemcee`, allowing for beyond-GR deviations, initialized with the (tempered) posterior found in `150914_hybrid_step1_time-marginalized.hdf5`. When the overlap is calculated during this second step, we marginalize over time.

`150914_d_phi_2_no-overlap_time-marginalized_resume.pickle` : Resume file from `bilby`; a pickled `python` dictionary, where the value of `chain_array` contains samples from the `ptemcee` chain, during the second step of hybrid sampling, during which we apply no overlap cut. Technically, if the overlap were to be calculated during this step, we would marginalize over the time when it is calculated, though we don't calculate this overlap when the cut is at 0.

Source run (for above files): `/home/noah.wolfe/bgr_source_model/paper_runs/gw150914/hybrid/hybrid-gw150914`

---

`150914_hybrid_step1.hdf5`, `150914_hybrid_step2.hdf5` : same as the posterior files above, but we only marginalize over phase when calculating the overlap during the second step. Also, the step2 posterior only contains samples of dphi2.

Source run (for above files): `/home/noah.wolfe/MA499_report_runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914`

