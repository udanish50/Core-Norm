# Limitations and responsible claims

Core-Norm is experimental research software.

1. **No universal dominance.** A scaler that is useful under contaminated or heterogeneous feature distributions may be unnecessary or inferior on clean distributions that strongly match another transform's assumptions.
2. **Feature dimension doubles.** Each input numeric feature creates two output coordinates. This changes memory use, first-layer parameter count, and distance geometry.
3. **Distance-based models require care.** KNN and related algorithms may react to the additional residual coordinates differently from linear or kernel models.
4. **No imputation.** NaN values are deliberately preserved; Core-Norm is not a missing-data method.
5. **No concept- or covariate-shift correction.** A static fitted scaler cannot generally correct a different test-time data-generating process.
6. **Finite robust-statistic breakdown.** Medians and quartiles are robust, not invulnerable. Large adversarial contamination can move fitted location/scale statistics.
7. **Target scaling is a separate design choice.** For ordinary supervised learning, Core-Norm is primarily intended for input features. If a target is encoded with Core-Norm, a predictive model must output a valid two-coordinate target representation before inverse transformation.
8. **Preliminary benchmark.** Included results support further study; they are not a substitute for external multi-dataset validation, uncertainty analysis, or peer review.

Recommended paper language: "Core-Norm provides a bounded, asymmetric and invertible representation designed to preserve tail magnitude while limiting extreme numerical leverage." Avoid "always better," "outlier-proof," or "state of the art" until those claims are independently supported.
