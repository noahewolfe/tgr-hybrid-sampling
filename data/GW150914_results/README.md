`150914_hybrid_step1_time-maximized.hdf5` : posterior from the first hybrid step with `dynesty`, assuming only GR, and providing the initial points for sampling with `ptemcee` (in the second step, we maximize over time when calculating the overlap, but the time marginalization doesn't come in during the first step). The only key is `'gr'`.

`150914_hybrid_step2_time-maximized.hdf5` : posterior from the second hybrid step with `ptemcee`, allowing for beyond-GR deviations, initialized with the (tempered) posterior found in `150914_hybrid_step1_time-maximized.hdf5`. When the overlap is calculated during this second step, we maximize over time. The keys are `'d_phi_2'` and `'d_phi_2_no_overlap'`.

Source run (for above files): `/home/noah.wolfe/tgr/tgr-hybrid-sampling_paper-runs/paper-runs/gw150914/hybrid/hybrid-gw150914_C`

---

`150914_hybrid.hdf5`: same as the posterior files above, but we only maximize over phase
when calculating the overlap during the second step. Keys are `'gr'` (for the first step analysis with `dynesty`, assuming GR is correct), `'d_phi_2'` (second step analysis with `ptemcee`, allowing for non-zero `d_phi_2`, and with an overlap cut of 0.9), and `'d_phi_2_no_overlap'` (same as previous but no overlap cut).

`150914_d_phi_2_no-overlap_resume.pickle`: resume file, including all 54 dimensions (including calibration parameters), for the `ptemcee` stage of analaysis allowing `d_phi_2` to vary with no overlap cut applied.

Source run (for above files): `/home/noah.wolfe/tgr/tgr-hybrid-sampling_paper-runs/MA499-report-runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914`

---

`150914_hybrid_step2_resume.hdf5` : saved the 800 - 1500th samples from the chain_array (flattened across all walkers) of the resume files found in `/home/noah.wolfe/tgr/tgr-hybrid-sampling_paper-runs/MA499-report-runs/real/pure-gr/GW150914/hybrid_pure-gr_GW150914`, as not all of these runs wrote result files

--

`ptemcee-gw150914-max-likelihood-pos0-resume.pickle`: resume file for analysis of GW150914 with `ptemcee` initialized with random samples from narrow distributions around the maximum likelihood estimate from a `dynesty` analysis of the same event, assuming that GR is correct. Keys in this file are `'maxlikelihood'` and `'prior'`.

`ptemcee-gw150914-prior-pos0-resume.pickle`: resume file for analysis of GW150914 with `ptemcee` initialized with random samples from the prior.

`150914_ptemcee_initialization_comparison_posterior.hdf5`: posterior for the same two analyses described above. Keys are `'maxlikelihood'` and `'prior'`.

Run initializing from the prior can be found at `/home/noah.wolfe/tgr/tgr-hybrid-sampling_paper-runs/initialization-comparison/ptemcee-gw150914`

Run initializing near the maximum likelihood estimate can bne found at `/home/noah.wolfe/tgr/tgr-hybrid-sampling_paper-runs/initialization-comparison/ptemcee-gw150914-max-likelihood-pos0-test`