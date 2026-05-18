README — Paper 3: PINN Gramian Verification
Version 20 — 18 May 2026  (FINAL SUBMISSION VERSION)
======================================================

PAPER:
"Physics-Informed Neural Network Verification of Kalman-Hewer
Controllability Gramians in Singular Bilinear Periodic Matrix
Differential Systems"
Authors: Sri Venkata Durga Sudarsan Madhyannapu &
         Sravanam Pradheep Kumar
Journal: Neural Networks (Elsevier, ISSN 0893-6080)

FILES IN THIS ZIP:
------------------
main.tex                            — Main manuscript (LaTeX, 1839 lines)
elsarticle.cls                      — Elsevier document class
cover_letter_NN.tex                 — Cover letter (with V20 revisions)
highlights_NN.tex                   — Research highlights (5 bullets)
consolidated_abstract_NN.tex        — Structured abstract
declarations_NN.tex                 — Author declarations
graphical_abstract_NN.tex           — Graphical abstract
suggested_reviewers_NN.tex          — Suggested reviewers
Fig1_training_loss.pdf              — Figure 1: Training loss curve
Fig2_pinn_vs_reference.pdf          — Figure 2: PINN vs reference Gramian
Fig3_approximation_error.pdf        — Figure 3: Approximation error
Paper3_PINN_Gramian_Verification.m  — MATLAB script (V20: NO data1/data2)
PAPER 3 PINN Gramian Verification V20.txt — MATLAB output log
README_V20.txt                      — This file

V20 CHANGES (vs V19) — FINAL MICRO-FIXES:
------------------------------------------
[V20-A] MATLAB figure data1/data2 legend artifacts ELIMINATED
        - All xline/yline/annotation calls: 'HandleVisibility','off'
        - legend() called with explicit handle arrays only
        - No data1/data2 appears in any of the 3 figures
        - Apologetic "Note: data1/data2..." captions removed from
          all three figure captions in main.tex
        - Figure captions rewritten as clean, professional descriptions

[V20-B] Abstract tightened
        - Removed ~2 lines of redundancy around PSD manifold sentence
        - Abstract now reads crisply: 10 sentences, all load-bearing

[V20-C] DeepONet citation CORRECTED
        - chen2021solving (Neural ODEs, Chen et al. 2018) was being
          used incorrectly for DeepONet citations
        - New reference added: lu2021deeponet
          Lu et al., "Learning nonlinear operators via DeepONet..."
          Nature Machine Intelligence 3, 218-229 (2021)
          DOI: 10.1038/s42256-021-00302-5
        - All body-text DeepONet citations updated to lu2021deeponet
        - chen2021solving retained correctly for Neural ODEs reference

[V20-D] Theorem 4.5 wording softened
        - "adapts the Adam convergence analysis of [kingma2015]"
          -> "extends standard nonconvex Adam convergence arguments
              to the composite PINN loss setting"
        - Applied in both the motivation paragraph and the proof

[V20-E] Conclusion future work expanded
        - Added: "Future work will further investigate fully parametric
          operator-learning across varying system families and
          uncertainty regimes, broadening the practical applicability
          of the Gramian-PINN framework to real-time monitoring,
          robust control design, and online controllability
          certification in engineering applications."

KEY NUMERICAL VALUES (MATLAB-verified, ALL OK):
-------------------------------------------------
e_rel = 6.3e-4                           OK
rank = 2                                 OK
eps*_hat (GPU, float32) = 0.169         OK  [error vs analytical: 0.001]
eps*_hat (MATLAB, float64) = 0.166      OK  [within tolerance]
eps* analytical = 0.170                  OK
Threshold error = 0.001 (GPU)            OK
Training iterations = 5000               OK
Training time (NVIDIA A100) = ~47 sec    OK
tau_r = 0.0935, delta_0 = 2.29e-04 < tau_r: TRUE — rank verified

GPU: NVIDIA A100 (80 GB HBM2e), float32 training
MATLAB: R2024b, float64 verification

COMPILATION:
pdflatex main.tex
(Bibliography is inline — no BibTeX required)

CODE REPOSITORY:
https://github.com/msvdsudarsan/PINN-Gramian-Controllability

SUBMISSION STATUS: READY FOR Neural Networks (Elsevier)
