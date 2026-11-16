# MAD-FM: Multi-omics Analysis and Dissection via Foundation Models

We present MAD-FM (Multi-omics Analysis and Dissection via Foundation Models), a framework that couples a sparse weighted bilevel optimization algorithm (Module 1) with biological foundation models (Module 2). Module 1 identifies trait-associated genes from unaligned genotype, transcriptome, translatome, and proteome data and assigns each gene functional omics layers. Module 2 re-ranks candidate genes using ESM-2 protein representations and five classes of gene-gene relationship networks.

## Framework

### Module 1: multi-omics association and functional-layer dissection

Module 1 evaluates each gene using any available combination of genotype, transcriptome, translatome and proteome measurements. A nested optimization procedure selects the supported omics-layer combination, while repeated cross-validation and phenotype permutation establish gene-level statistical significance. The final `voted_subnet` vector follows the order `[genotype, transcriptome, translatome, proteome]`; `1` indicates a selected layer, `0` an available but unselected layer, and `-1` an unavailable layer.

### Module 2: foundation-model-guided candidate-gene re-ranking

Module 2 starts from fixed 480-dimensional ESM-2 protein representations and learns five relation-specific spaces supervised by GO functional similarity, gene co-expression, Hi-C chromatin proximity, protein-protein interactions and TF regulatory relationships. Candidate genes are scored against known trait genes in each space, ranked independently and combined by equal-weight reciprocal-rank fusion (RRF).

The explicit interface between the modules is:

```text
Module 1: significant_genes.csv
                    |
                    v
Module 2: 02_rank_candidate_genes.py
                    |
                    v
          madfm_ranked_candidates.csv
```

The integrated output retains the Module 1 p-value and omics-layer assignment alongside the five Module 2 relation scores, relation-specific ranks, RRF score and final rank.

## Repository structure

```text
MAD-FM/
├── README.md
├── module1/
│   ├── 01_train_model.py
│   ├── 02_permutation_test.py
│   ├── 03_compute_pvalues.py
│   └── 04_bonferroni_correction.py
└── module2/
    ├── 01_train_relation_model.py
    ├── 02_rank_candidate_genes.py
    ├── 03_cross_species_analysis.py
    ├── training_core.py
    ├── ranking_core.py
    ├── ranking_grid.py
    ├── species_comparison.py
    ├── species_helpers.py
    └── workflow_utils.py
```

This source-only repository does not contain example data, processed biological datasets, ESM-2 embeddings or trained checkpoints.

## Installation

Python 3.10 or later is recommended.

```bash
pip install numpy pandas pyarrow scipy scikit-learn joblib tqdm matplotlib pillow torch
```

Install a CUDA-enabled PyTorch build separately when GPU execution is required.

## Module 1 inputs

Module 1 accepts the following user-provided tables.

| Argument | Required content |
|---|---|
| `--phenotype` | CSV table whose selected column contains one phenotype value per sample |
| `--genes` | CSV table whose selected column contains genes to be tested |
| `--genotype` | SNP-by-sample CSV; the third column contains the gene ID and sample genotypes begin at column 10 |
| `--transcriptome` | Sample-by-gene normalized expression CSV |
| `--translatome` | Sample-by-gene normalized translation CSV |
| `--proteome` | Sample-by-gene normalized protein-abundance CSV |

The three molecular-omics arguments and the genotype argument are independently optional, but every tested gene must be represented in at least one layer. Rows must follow the same sample order as the phenotype table; genes need not be measured in every omics layer.

## Run Module 1

### 1. Fit the observed multi-omics models

```bash
python module1/01_train_model.py \
  --phenotype /path/to/phenotype.csv \
  --genes /path/to/genes.csv \
  --genotype /path/to/genotype.csv \
  --transcriptome /path/to/transcriptome.csv \
  --translatome /path/to/translatome.csv \
  --proteome /path/to/proteome.csv \
  --output-dir results/module1
```

Outputs: `observed_results.csv` and `omics_contributions.csv`.

### 2. Build the phenotype-permutation null distribution

```bash
python module1/02_permutation_test.py \
  --phenotype /path/to/phenotype.csv \
  --genes /path/to/genes.csv \
  --genotype /path/to/genotype.csv \
  --transcriptome /path/to/transcriptome.csv \
  --translatome /path/to/translatome.csv \
  --proteome /path/to/proteome.csv \
  --permutations 20 \
  --output-dir results/module1
```

Output: `permuted_results.csv`.

### 3. Compute uncorrected gene-level p-values

```bash
python module1/03_compute_pvalues.py \
  --observed results/module1/observed_results.csv \
  --permuted results/module1/permuted_results.csv \
  --output results/module1/uncorrected_pvalues.csv
```

### 4. Apply Bonferroni correction

```bash
python module1/04_bonferroni_correction.py \
  --input results/module1/uncorrected_pvalues.csv \
  --threshold 0.001 \
  --output-dir results/module1
```

The principal Module 1 output is `results/module1/significant_genes.csv`.

## Module 2 data requirements

Module 2 training uses three aligned processed files:

- a Parquet gene table containing an `id` column;
- an `N x 480` NumPy matrix of fixed ESM-2 protein embeddings;
- a JSON Lines relationship table with `anchor`, `positive` and `source` fields, where `source` is one of `go`, `coexp`, `hic`, `string` (PPI) or `tf`.

Candidate re-ranking requires a trained Module 2 checkpoint, a text file containing known trait genes, a text file containing the gene IDs of an embedding catalog, and the aligned `N x 480` embedding matrix. Module 1 results provide the candidate genes.

## Run Module 2

### 1. Train the five-relation model

```bash
python module2/01_train_relation_model.py \
  --pairs /path/to/pairs_train_v2.jsonl \
  --genes /path/to/genes.parquet \
  --protein-embeddings /path/to/protein_emb.npy \
  --output-dir results/module2/model
```

The default configuration uses a shared residual architecture, 512-dimensional relation embeddings, a 256-dimensional bottleneck, dropout 0.1, 49 filtered negatives per positive pair, learning rate `2e-4`, batch size 32 and a maximum of 150 epochs.

### 2. Re-rank Module 1 candidate genes

```bash
python module2/02_rank_candidate_genes.py \
  --module1-results results/module1/significant_genes.csv \
  --known-genes /path/to/known_gene_ids.txt \
  --embedding-ids /path/to/esm2_gene_ids.txt \
  --embeddings /path/to/esm2_embeddings.npy \
  --model /path/to/protein_multihead.pt \
  --output-dir results/module2/ranking
```

The script extracts candidate genes from the Module 1 table, matches both candidate and known genes to the ESM-2 catalog, evaluates the reference-gene and similarity-aggregation grid, and writes `madfm_ranked_candidates.csv`. Genes without ESM-2 embeddings are reported separately rather than silently discarded.

### 3. Cross-species transfer and target-species fine-tuning

```bash
python module2/03_cross_species_analysis.py \
  --task all \
  --data-root /path/to/cross_species_data \
  --maize-model /path/to/protein_multihead.pt \
  --output-dir results/module2/cross_species
```

Supported task aliases are `arabidopsis`, `brca`, and `luad`. Each task directory under `--data-root` must contain:

```text
<task>/
├── training/
│   ├── genes.parquet
│   ├── protein_emb_esm2_480.npy
│   └── pairs_train_v2.jsonl
└── ranking/
    ├── known_genes.parquet
    ├── known_protein_emb_esm2_480.npy
    ├── candidate_genes.parquet
    ├── candidate_protein_emb_esm2_480.npy
    └── dataset_manifest.json
```

The default `--mode all` performs target-species five-relation fine-tuning, maize-model transfer evaluation and a matched comparison of self-fine-tuned RRF, maize-transfer RRF, raw ESM-2 and uniform-random ranking. Use `--mode finetune` or `--mode transfer` to run an individual stage.

## Principal outputs

| Output | Description |
|---|---|
| `significant_genes.csv` | Module 1 trait-associated genes and selected functional omics layers |
| `protein_multihead.pt` | Module 2 five-relation projection model |
| `madfm_ranked_candidates.csv` | Integrated Module 1 evidence and Module 2 biological re-ranking |
| `four_strategy_comparison/` | Cross-species self-fine-tuned, maize-transfer, raw ESM-2 and random comparison |

## Reproducibility

The default random seed is 2025 for Module 1 and 42 for Module 2. Module 2 uses relation-wise 80%/10%/10% edge splits, independently filtered negatives and repeated known-gene holdout evaluation. Hyperparameter-grid selection is exploratory and should be followed by nested or external validation when an unbiased performance estimate is required.

## Data and model availability

No data or model weights are stored in this repository. Processed relationship data, ESM-2 embeddings and trained checkpoints should be distributed as a separate versioned archive and supplied through the command-line paths documented above.

## Citation

Please cite the MAD-FM manuscript when using this software. Full citation details will be added upon publication.

