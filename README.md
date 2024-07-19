# 2x2_Strange

This repository comprises my PhD analysis looking for strange matter production in the DUNE 2x2 experiment. A reconstruction trained for 2x2 is performed using [spine](https://github.com/DeepLearnPhysics/spine), prediction, among other things, semantic segmentation and clustering of energy depositions into particles. For this analysis, and at the energy regimes produced in collisions at 2x2, the lower energy Kaons found in this detector will very often be stopping tracks, and very often 2-body decay into a visible 540 cm muon. Using the [spine](https://github.com/DeepLearnPhysics/spine) add-ons for HIP/MIP prediction as well as the analysis in this repository, coupled with this characteristic signature of low energy kaon decays, reconstruction of the $\nu \rightarrow K^+ +\textbf{X}$ cross-section is computed. A separate analysis looking for $\nu \rightarrow \Lambda^0 +\textbf{X}$ will also be performed using the same tools.

![Full chain](https://github.com/DeepLearnPhysics/spine/blob/develop/docs/source/_static/img/spine-chain-alpha.png)

## Installation

This package depends primarily on:

* `spine`
* standard Python scientific libraries.
*TBD

## Usage
TBD

## Repository Structure

* `utils` contains scripts for creating simulation files with HIP/MIP semantic segmentation predictions
* `analysis` contains functions relevant to analysis (cuts, etc.)

## How to run
TBD