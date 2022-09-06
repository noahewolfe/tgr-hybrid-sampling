# tgr-hybrid-sampling

Repository with data, analysis, and plots for [arXiv:2208.12872](https://arxiv.org/abs/2208.12872), "Accelerating Tests of General Relativity with Gravitational-Wave Signals using Hybrid Sampling" (Wolfe, Talbot, & Golomb 2022).

## Code

In this work, we used a [lightly-modified fork of `bilby`](https://git.ligo.org/noah.wolfe/bilby). Beyond-GR gravitational waveforms were generated with a [custom source model](https://git.ligo.org/noah.wolfe/bilby-tgr).

Hybrid sampling is available as a ready-to-use pipeline on top of `bilby_pipe`; pipeline code to launch both steps of hybrid sampling (sampling first in `dynesty`, and then `ptemcee`), as well the postprocessing executable to launch `ptemcee` on a `dynesty` result file can be found [here](https://git.ligo.org/noah.wolfe/bgr_source_model).

## Manifest

`GW150914_results/`: Posteriors for each step of the hybrid analysis of GW150914, with and without a time-maximized overlap cut; see README.md therein.

`bgr_results/`: Posteriors for each step of the hybrid analysis of an injected beyond-GR signal; see README.md therein.

`toy-model_misspecified.ipynb`, `toy-model_well-specified.ipynb`: Hybrid sampling analysis of the generalized Gaussian distribution when the model in the first step is incorrectly specified or correctly specified, respectively.

`gw150914.ipynb`, `high-snr-injection.ipynb`: Plots generated based on the posteriors and sampling history of our hybrid analysis of GW150914 and an injected beyond-GR signal, respectively.

`config_gw150914.ini`, `config_injection.ini`: Configuration files for `bilby_pipe`, with additional arguments read by our hybrid sampling pipeline.

`modified_GW150914_4s.prior`, `modified.prior`: Prior files for our analysis fo GW150914 and an injected beyond-GR signal, respectively.

`toy.py`: Utility statistical and plotting functions for the hybrid analysis of the generalized Gaussian distribution.

`util.py`: Utility functions for the analysis of GW150914 and an injected signal, including plotting functions and common formatting definitions.

`tgr-hybrid-sampling.mplstyle`: Matplotlib style file for plot creation.
