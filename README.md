# Independent AI Research — Small-Model Efficiency & Reasoning Experiments

I use small, controlled experiments to study where a model's decisions break down. This repository follows three questions. Can a small language model answer more reliably when a separate scorer chooses the facts? Does a learned relation model understand which statement governs an action? Can a compact visual model tell when an observation should change its belief?

The experiments use synthetic text and simulated transitions, with paired counterfactuals, explicit baselines and held-out shifts. The repository includes the results I would use to make engineering decisions, including the approaches I dropped.

## Selected experiments

| Experiment | What the comparison showed | What you can run |
|---|---|---|
| [Decision-first generation](docs/decision-first.md) | A MiniLM scorer followed by SmolLM2 generated **51/80** correct complete states versus **16/80** for the decoder alone. A tested mid-generation checkpoint fell to **14/80**. | Re-score the actual generated answers and compare formatting controls on the same selected facts. |
| [Relation binding and batching](docs/relation-binding.md) | Mixed batches improved exact forward edges in both seeds. Correct signed reverse claims went **7 → 72** in one seed and **73 → 22** in the other, out of 512 each. | Audit saved predictions, or retrain the matched PyTorch heads with a local MiniLM copy. |
| [Visual evidence under shift](docs/visual-evidence.md) | Two 57-parameter spatial readouts separated **40/48** historical feature collisions. On a new obstacle/sensor family, the aligned readout made **704/920** neutral events appear at least 3:1 decisive, versus **128/920** for the original features. | Refit the three logistic heads from archived feature matrices and replay their belief updates. |

## Getting started

Run these commands from the repository root with Python 3.12 or 3.13. The default evaluations use NumPy and SciPy on CPU and require no model downloads.

```sh
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux. Then run the offline evaluations and tests.

```sh
python -m pip install -r requirements.txt
python -m experiments.decision_first
python -m experiments.relation_binding
python -m experiments.visual_evidence
python -m unittest discover -s tests -v
```

The [reproduction guide](docs/reproducibility.md) covers the optional PyTorch training run, model revision, test status and artifact boundaries.

## Engineering details

- [Text scoring](experiments/decision_first.py) separates answer completeness from factual accuracy. Scenario-group resampling keeps related examples together.
- [PyTorch model](experiments/relation_model.py) pools frozen token features by role and mention order. The [trainer](experiments/train_relations.py) checks exposure before running matched schedules. The [evaluator](experiments/relation_binding.py) scores governing edges separately from the final decision.
- [Visual features](experiments/visual_features.py) use two RGB frames and an action. The [evidence replay](experiments/visual_evidence.py) tests both raw and temperature-calibrated likelihoods with a fixed belief update.
- [Configurations](configs/) hold the fit and split settings. [Results](results/) preserve reference measurements and source hashes. [Tests](tests/) verify the key controls and re-score archived outcomes.

## Scope

The text benchmark supplies a finite candidate schema. The relation conclusions are limited to the supplied candidate records and generated wording families. The visual results are for a controlled simulator and fixed disturbances. They support decisions about these implementations. An open-ended reasoning or deployment claim would require broader data and independent tests.

The next visual comparison should hold the physical probe fixed and vary only capture, to distinguish sensor sensitivity from unfamiliar obstacle geometry. The relation results call for a test that varies forward and reverse polarity independently. [Visual study](docs/visual-evidence.md) · [Relation study](docs/relation-binding.md)
