# Reproducibility notes

The Phase-1 CSV archive is retained exactly at the numerical-result level, with public method labels canonicalized for naming consistency. The implementation in this repository is the corrected Core-Norm inverse that uses the same numerical stabilizer in forward and inverse scaling.

For publication-grade claims, rerun a fresh benchmark from a version-controlled harness on external datasets, record environment lockfiles, dataset versions/hashes, random seeds, preprocessing order, and confidence intervals, and preregister the primary comparison where practical.
