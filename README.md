# Specialization Project 1: Optimization of Binning Scheme for CP Violation Search with $D^0 \rightarrow \pi^+ \pi^- \pi^0$ Decays at LHCb

This repository contains all relevant scripts for this project.

## Project Overview

Charge-parity ($CP$) violation in the charm sector provides a unique and highly sensitive probe for New Physics (NP) beyond the Standard Model. In multibody charm decays, such as $D^0 \to \pi^+\pi^-\pi^0$, these $CP$ asymmetries are not uniform but manifest dynamically across the Dalitz phase space due to the interference of intermediate resonant states. While adaptive binning techniques have been applied to extract other charm observables, an equipopulated, adaptive binning strategy explicitly optimized to maximize localized $CP$ sensitivity has not yet been established for this decay. In this work, we develop and validate a novel, three-phase adaptive binning algorithm inspired by cellular automata. Tested against artificial $CP$-violation models, this approach successfully isolates regions of similar-sign asymmetry, demonstrating significant potential for high-precision $CP$ violation searches in upcoming large-statistics datasets.

## Repository Structure

### Optimization Algorithm Scripts

1. **Seeding Phase (`global_seed.py`)**: Selects optimal initial seed bins considering all artificial $CPV$ model scenarios, which subsequently grow into the desired clusters.
2. **Growth Phase (`global_phase1.py`)**: Expands seed bins into initial continuous clusters based on local geometry and the sign of the $CP$ asymmetry.
3. **Optimization Phase (`global_phase2.py`)**: Iteratively exchanges boundary bins between neighboring clusters to maximize an overall sensitivity and population-balance score across all models.

* **Execution Order:** The scripts must strictly be executed in the sequential order listed above (`global_seed.py` $\rightarrow$ `global_phase1.py` $\rightarrow$ `global_phase2.py`). However, because the intermediate outputs of each stage are saved separately, downstream scripts can be modified and re-run without re-executing earlier steps.
* **Documentation & Support:** In-line comments throughout the codebase are self-explanatory regarding how each script operates, though please do not hesitate to contact me if anything further is required.

### Analysis Scripts

* **`cp_compare.py`**: Generates raw visualizations of the $CP$ asymmetry on a resonance-aligned $11 \times 11$ binning grid, comparing toy models with injected artificial $CPV$ against the standard baseline $D^0$ model.
* **`recons_effi_ampgen.py`**: Calculates a relative estimate of the reconstruction efficiency using reconstructed Monte Carlo (MC) events as the numerator and the full baseline `AmpGen` model as the denominator. The resulting efficiency map is utilized in the optimization process.
* **`bin_stats.py`**: Evaluates and compares the $A_{CP}$ significance and bin population distributions across different numbers of bins resulting from the optimization process.
