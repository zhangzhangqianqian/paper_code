# Stage 7.6 unified aggregation and acceptance

## Goal

Read the completed Stage 7.3, 7.4 and 7.5 formal output directories and
produce reproducible comparison tables without retraining or changing the
frozen experiment configuration.

## Inputs and outputs

The runner validates the three passed stage manifests, indexes every run
directory, recomputes test metrics from the saved prediction archives, and
aggregates the five fixed-seed results with mean and sample standard deviation.
It also records links to the validation-only Stage 6.4 gate diagnostics and
Stage 6.5 transfer analysis. Outputs include a raw run index, long-form
per-task/per-horizon/per-season metrics, overall comparison, resource summary,
diagnostic linkage, and an acceptance manifest.

## Acceptance rules

- Stage 7.3/7.4/7.5 root manifests must report all expected runs as passed.
- Every run must contain a test prediction archive and run manifest.
- Stage 6.4 and 6.5 diagnostic manifests must confirm that the test set was not
  accessed.
- The script is read-only with respect to source result directories and does
  not select a new model or hyperparameter.
