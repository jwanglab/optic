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
      -> MLP (1024 -> 512 -> k) -> softmax
```

## Model directory

Specified with `--model` (default `model/`):

```
lineage.json        tree hierarchy, ordered class names, and thresholds
optic.safetensors   weights
probes.T2T.bed      353,232 input CpG features (T2T-CHM13v2.0), in order
probes.hg38.bed     the same probes on GRCh38
```

## Output

The model walks the hierarchy in `lineage.json` from the root, always taking the branch with the
highest total weight. Each step checks the depth's calibrated threshold from `descent_thresholds` (default 0.5).

```
  d1  Hematolymphoid                                0.751  >=  0.50
  d2  B-cell lymphoid (incl. plasma cell, Hodgkin)  0.687  >=  0.50
  d3  Mature B-cell neoplasms                       0.466   <  0.75   <- below threshold, call stops above
  d4  Large B-cell lymphomas                        0.279   <  0.99
  d5  ALK-positive large B-cell lymphoma            0.120   <  0.99

  call : B-cell lymphoid (incl. plasma cell, Hodgkin)  (depth 2, p 0.687)
```

`--json` shows every class probability.

## Run

```bash
./target/release/optic bam    -i sample.bam --threads 16
./target/release/optic pileup -i sample.bed
./target/release/optic beta   -i betas.csv
./target/release/optic --json beta -i betas.csv

# --model DIR to use a model directory other than ./model
```

BAMs must be aligned to T2T-CHM13v2.0 or GRCh38 and include 5mC MM/ML tags. Contigs are
recognized by length, so their names don't matter. Modkit pileups are matched against both probe maps. `optic bam` and `optic counts` first run the same checks as `optic inspect -i sample.bam`: reference
by contig length, naming, index, and MM/ML tags in the first 2,000 reads. They stop with the reason and
proposed fix for an unaligned or misaligned BAM, a stale index, or missing tags.

### Split a BAM across processes

```bash
for i in {0..3}; do
  optic counts -i sample.bam --shard $i/4 -o part$i.tsv
done
optic classify -i part*.tsv
```

`counts` piles up reads into per-probe counts (`probe_id, n_mod, n_unmod, fraction`).
With an index, a shard contains whole chromosomes, balanced by probe count. Without one, a shard is a compressed byte range:
it starts at the first BGZF block after size i/n and finds its first record by parsing a chain of three complete,
well-formed records. It also writes `part0.tsv.range`, the virtual positions of its first record and of the record it
stopped at. `classify` refuses a set of parts whose ranges don't chain end to end (a missing or repeated shard). Either way
the shards sum exactly to the whole BAM, and `classify` gives the same result as `optic bam` on the whole file.

### WebAssembly

```bash
rustup target add wasm32-wasip1
cargo build --release -p optic --target wasm32-wasip1 # -> target/wasm32-wasip1/release/optic.wasm (1.0 MB)
wasmtime run --dir model::/model --dir data::/data target/wasm32-wasip1/release/optic.wasm --model /model bam -i /data/x.bam
```

## Training

`optic-train` (CUDA) trains the model from scratch. Each training profile is a methylation beta vector with a target
distribution over the tree's leaves. Every epoch each profile is seen at several coverage levels: probes are subsampled
and turned into simulated nanopore reads (per-site read depth, per-read call error). Tumor rows are mixed with normal-tissue background at random purity. The loss
is a KL divergence to the target, plus a KL term at every level of the tree. A consistency term builds a
shallow-coverage subsampling of each sample, trained toward its prediction distribution at full depth.
AdamW with a cosine decay

Inputs and parameters are defined in `spec.json` including network size, optimizer, coverage levels, read model,
row weighting, augmentations. plus paths to four input files:

```
beta_u8      rows x probes uint8 matrix: 0..250 = beta (c + 0.5) / 250, 255 = missing
probes.tsv   one line per beta column, in beta_u8 column order; the model's inputs are in this order
labels.tsv   row, target, in_training (row weights are computed from the targets)
             target = leaf:<leaf> | node:<node> | mix:<node>=<mass>;<node>=<mass>
tree.json    leaves (output order), nodes (child -> parent), leaf_domain
```

`probes.tsv`
```
col	probe_id	chrom	start	end
0	cg06355467	NC_060925.1	26251	26253
1	cg08258224	NC_060925.1	293676	293678
...
```

`labels.tsv`
```
row	sample_id	domain	target	label_source	in_training
0	sample0	heme	leaf:AML RUNX1-RUNX1T1	curated	1
1	sample1	heme	node:AML with NPM1 mutation	clinical_annotation	1
2	sample2	heme	mix:AML KMT2A-r, HOXA/B=0.8000;AML with KMT2A-r=0.2000	clinical_annotation	1
...
```

Build (needs an NVIDIA driver R580+ and the CUDA 13 toolkit for libdevice; no nvcc or cuBLAS):

```bash
# once: cuda-oxide's pinned toolchain and cargo plugin
rustup toolchain install 1.99.1 --profile minimal -c rust-src -c rustc-dev -c llvm-tools
cargo +1.99.1 install --locked --git https://github.com/NVIDIA/cuda-rust.git --rev 2718c23 cargo-oxide

cargo build --release -p optic-train # CUDA_COMPUTE_CAP=120 etc. if nvidia-smi cannot report the GPU
```

Train:

```bash
optic-train check --spec SPEC_DIR
optic-train train --spec SPEC_DIR --out OUT_DIR
```

`SPEC_DIR` is provided instead of the spec.json because the spec defines the other input files relative to this path (just put them all in this folder)

`optic.safetensors`, `config.json`, and `log.json` will be written to `OUT_DIR`


## Federated learning structure

The model's cross-entropy loss means one round-trip is required for each epoch.
To make that as easy as possible, and because the model is pre-trained on _a lot_ of broad data, the federated backprop and merge goes back to the second layer, leaving the first (largest) 350k x 256 layer fixed. The per-round gradients and weights that have to be shared back and forth are ~12 MB.
Data is not shared, only aggregate gradients from a single pass/epoch on a labeled dataset.

```bash
optic contribute --labels cohort.tsv --out site_A.safetensors
```

`contribute` writes gradients for every parameter above `first.0` at `--out` and a matched `[prefix].json` extension (ex `site_A.json`)
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
