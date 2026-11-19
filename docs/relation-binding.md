# Relation binding and batching

Can a small classifier identify the record that governs an action, including direction and polarity? The baseline reads each document with frozen all-MiniLM-L6-v2 and pools token features by action mention, record mention and mention order. A 224,549-parameter PyTorch head produces five signed edge classes. An explicit executor uses the selected forward edges and stated record statuses to predict the two actions.

The batching test compared single-group batches with mixed-group batches. Both started at the same seed-specific weights and used the same AdamW settings, 360 updates and 2,880 document presentations. They see each document equally often in every 24-update window. The intervention also changes example order and when paraphrases meet in a batch, so it cannot isolate a gradient-interference mechanism.

| New wording, 384 documents | Grouped → mixed, seed 32 | Grouped → mixed, seed 43 |
|---|---:|---:|
| Exact forward edges / 768 | 262 → 351 | 245 → 308 |
| Signed reverse claims / 512 | 7 → 72 | 73 → 22 |
| Complete governing swaps / 192 | 28 → 41 | 27 → 36 |
| Both actions correct / 384 | 100 → 143 | 78 → 146 |

Final actions can hide structural mistakes. For example, 65 of 146 correct mixed-batch states at seed 43 contained a wrong governing edge. The executor solved all 384 documents with the reference edges. The six held-out wording pairs share procedural rules, and reverse polarity is coupled to the forward clause by construction. Forward improvements and better training fit therefore provide limited evidence about transferable directional reasoning.

The two arms use the same inference code. The historical warm timings include tokenisation, inference and execution for supplied actions, spans and candidate records. Their small timing difference is not evidence of an architectural speedup. Batching improved some measured outcomes while failing the key cross-seed direction test.

```sh
python -m experiments.relation_binding
```

That audit re-scores archived predictions and re-executes edges on CPU. To train the same head, follow the [reproduction guide](reproducibility.md#pytorch-training). [Reference measurements](../results/relation_binding.json) · [Head and token pooling](../experiments/relation_model.py) · [Training](../experiments/train_relations.py)
