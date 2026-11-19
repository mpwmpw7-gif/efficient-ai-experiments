# Reproducing the results

The repository runs from its root. The default commands need Python, NumPy and SciPy. They use only local artifacts and CPU compute. The Windows verification used Python 3.13.5, NumPy 2.5.4 and SciPy 1.18.1.

| Command | What it recomputes | Inputs |
|---|---|---|
| `python -m experiments.decision_first` | Correctness, completeness and grouped bootstrap contrasts | Archived generated text and synthetic references |
| `python -m experiments.relation_binding` | Signed edge counts, governing swaps and executed outcomes | Locked test text and archived model predictions |
| `python -m experiments.visual_evidence` | Fresh logistic fits, development temperature and held-out belief replay | Archived 56-feature arrays, scenario IDs and reference outcomes |
| `python -m unittest discover -s tests -v` | Fixture consistency, replay metrics, controls and optional PyTorch checks | The files checked in here |

Historical measurements are stored in [results](../results/). The automated tests compare the scores computed here with those measurements. Visual replay reports binary Brier score. The archived two-class summed Brier score is twice that value. The fixed-budget logistic refit can vary slightly with BLAS kernels. Tests compare log loss within 0.005 bits per outcome and check the decisive-neutral and correct-direction counts exactly. The bootstrap contrasts use group means and quantify variation across sampled groups for a fixed set of predictions.

For the visual test, the fit only uses [training arrays](../data/visual/train.npz). Development selects a temperature. No model or threshold selection uses confirmation labels. The export omits the raw simulator frames and the original collector, but includes the fixed feature definitions, group IDs and aligned/unaligned feature matrices. Geometry probabilities are archived. The [source manifest](../results/provenance.json) records SHA-256 hashes of original inputs before the public export. The original input files belong to the separate research archive.

## PyTorch training

The optional path uses the same two-action synthetic documents, candidate names and sentence spans for both arms. Use a local copy of all-MiniLM-L6-v2 at the revision in [the configuration](../configs/relations.json). The reference model files are available upstream under their own terms. This repository contains the head source and training schedule. It leaves the backbone weights upstream. [Pinned model snapshot](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)

```sh
python -m pip install -r requirements-relation.txt
python -m experiments.train_relations --model-dir ./models/all-MiniLM-L6-v2 --device cpu --smoke
python -m experiments.train_relations --model-dir ./models/all-MiniLM-L6-v2 --device cuda --seed 32
python -m experiments.train_relations --model-dir ./models/all-MiniLM-L6-v2 --device cuda --seed 43
```

Use a CUDA-compatible PyTorch build for `--device cuda`. For CPU training, use `--device cpu`. The smoke check uses one matched 24-update window and 32 test documents. Its metrics are pipeline checks on a reduced run. The full commands use the original 360-update schedules and locked test. The local verification used PyTorch 2.7.1+cu126 and Transformers 5.22.2 on an RTX 4060 Laptop GPU. Both full GPU runs reproduced every reported confirmation count for both arms. Other hardware and library versions may produce numerical differences. [Verification record](../results/verification.json)

The generated-state benchmark is an evaluation export. Reproducing its autoregressive generation requires the historical adapted scorer and decoder snapshots, which are outside the release. The exact outputs used for the reported text counts are available here.

## Artifact and model terms

The text fixtures are synthetic project data. The visual archives contain numeric features derived from project simulator runs. Model weights, third-party corpora, private notes and working directories are outside the export. The upstream MiniLM model card identifies Apache-2.0 terms for those weights. No separate open-source license is asserted for this repository. [Pinned MiniLM model page](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
