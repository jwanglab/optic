# Open Pediatric Tissue Classifier (OPTiC)

Hierarchical methylation-based classifier focused on pediatric hematologic, (ex-CNS) solid, and CNS tumors.
Runs on nanopore modBAMs (MM/ML tags), modkit pileups, or beta values.

```bash
cargo build --release optic
```

## Model

```
input: 353,232 CpGs (+1 methylated / -1 unmethylated / 0 uncalled)
      -> Linear(353232 -> 256, no bias) -> Linear(256 -> 1024) -> LN -> SiLU -> 2 x residual MLP blocks
      -> MLP (1024 -> 512 -> k) -> softmax with coverage-dependent temperature
```

## Model directory

```
lineage.json        the hierarchy, ordered class names, thresholds
optic.safetensors   weights
probe_subset.bed    353,232 input CpG features
```

## Output

The model walks the hierarchy in `lineage.json` from the root, always taking the branch with the
highest total weight. Each step checks the depth's calibrated threshold
from `descent_thresholds` (default 0.5).

```
  d1  Hematolymphoid                                0.751  >=  0.50
  d2  B-cell lymphoid (incl. plasma cell, Hodgkin)  0.687  >=  0.50
  d3  Mature B-cell neoplasms                       0.466   <  0.50   <- below threshold, call stops above
  d4  Large B-cell lymphomas                        0.279   <  0.80
  d5  ALK-positive large B-cell lymphoma            0.120   <  0.80

  call : B-cell lymphoid (incl. plasma cell, Hodgkin)  (depth 2, p 0.687)
```

`--json` includes every class probability

## Run

```bash
./target/release/optic bam    -i sample.bam --threads 16
./target/release/optic pileup -i sample.bed
./target/release/optic beta   -i betas.csv
./target/release/optic --json beta -i betas.csv

# --model DIR to use a model directory other than ./model
```

BAMs must be aligned to T2T-CHM13v2.0 (RefSeq `NC_0609xx.1` or `chr` contig names) with 5mC MM/ML tags.


## Federated learning structure

The model's cross-entropy loss means one round-trip is required for each epoch.
To make that as easy as possible, and because the model is pre-trained on _a lot_ of broad data, the federated backprop and merge goes back to the second layer, leaving the first (largest) 350k x 256 layer fixed. The per-round gradients and weights that have to be shared back and forth are ~12 MB.
Data is not shared, only aggregate gradients from a single pass/epoch on a labeled dataset.

```bash
optic contribute --labels cohort.tsv --out site_A.safetensors
```

`contribute` writes gradients for every parameter above `first.0` at `--out` and a matched `[prefix].json` (ex `site_A.json`)
including `n_samples`, `label_counts`, `class_names`, `loss`, `grad_norm`, and the sha256 of the weights they were computed against.
That is 3M parameters, ~12 MB.

`cohort.tsv` should be in the format: 
```
path    bam|pileup|beta    label
path    bam|pileup|beta    label
```

Where the label is any node of the tree. Training at non-terminal nodes spreads the weight over all its descendents.

The central merge from multiple gradient sets weights each set's gradient by the number of contributing samples, applies a uniform learning rate for the epoch, and runs a single weight update.
`merge` requires both safetensors (gradients) and json metadata (number of samples and checksums) and checks that the sha256 matches the base weights and the class names match the model tree. Currently, class weighting is not applied on the federated merge.

```bash
optic --model v1_DIR merge --updates site_A.safetensors site_B.safetensors --lr 0.01 --out v1.1_DIR
```

## Distributing a round

`--patch` writes only the tensors the merge changed (all but the first layer), so each back and forth is only the 12 MB:

```bash
# hub
optic merge --updates site_A.safetensors site_B.safetensors --lr 0.01 --out v1.1_DIR --patch round1.safetensors

# every site
optic --model v1_DIR apply --patch round1.safetensors --out v1.1_DIR
```
