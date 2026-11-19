# Decision-first generation

Can a small decoder answer more reliably when a separate model selects the factual conclusion? This comparison used a retained MiniLM scorer with a LoRA adapter and a frozen SmolLM2 decoder. The scorer ranked a predefined pool of colour/location states, then the decoder generated a response conditioned on the chosen state. The native control generated directly. A checkpoint arm generated six provisional tokens before scoring a decision and continuing.

| Same 80 state prompts | Correct complete answers | Parsed complete or abstained |
|---|---:|---:|
| Native decoder | 16 | 52 |
| Decision first | 51 | 80 |
| During-generation checkpoint | 14 | 25 |
| Gold conclusion, diagnostic | 76 | 76 |

The decision-first output followed its selected conclusion even when the conclusion was wrong. The pool supplied complete candidate states, making this a test of selection and verbalisation under a known schema. The 80 prompts come from 12 related synthetic scenario groups. The test reused inspected scenarios and a single model run. The parser uses a fixed vocabulary and can miss valid paraphrases.

## A formatting control

A follow-up kept independently selected attributes fixed across two serializers on 64 prompts. Field-form prompts generated 4/64 correct answers. The existing sentence format reached 32/64. With gold plans, the change was 4/64 to 60/64, isolating a decoder interface issue. Whole-answer selection reached 39/64 on the same prompts. The sentence-format test was an exploratory follow-up. All four genuinely unknown state cases still failed under the tested free-form variants.

The original timings were warm, amortised and unreplicated. This portfolio makes no general inference-speed claim from that comparison.

## Run the evidence check

```sh
python -m experiments.decision_first
```

The command parses actual [generated answers](../data/decision_outputs.json) and recomputes correctness, completeness and paired intervals. It performs no generation. Decoder checkpoints, external logic examples and unrelated experiments stay outside the export. [Reference counts](../results/decision_first.json) · [Scorer](../experiments/decision_first.py)
