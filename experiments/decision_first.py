"""Re-score archived colour/location generation with its original task grammar."""
import argparse
import re

from .common import ROOT, grouped_difference, print_json, read_json


OBJECTS = ("book", "coin", "mug", "box")
COLORS = ("red", "blue", "green", "yellow", "orange", "purple")
LOCATIONS = ("garden", "kitchen", "library", "workshop", "atrium", "storeroom")
UNKNOWN = "unknown"


def mentioned(sentence, vocabulary):
    return [word for word in vocabulary if re.search(r"\b" + re.escape(word) + r"\b", sentence)]


def first_sentence(text):
    return re.split(r"[.!?]", text.strip(), maxsplit=1)[0].lower()


def parse_state(text):
    sentence = first_sentence(text)
    abstains = re.search(
        r"(?:state|outcome|answer|colou?r|location).*(?:unknown|uncertain|undetermined)"
        r"|cannot (?:be )?determin|not enough information|does not determine",
        sentence,
    )
    if abstains:
        return None if mentioned(sentence, COLORS + LOCATIONS) else UNKNOWN
    if re.search(r"\b(?:not|never|neither|nor|but|except|instead)\b", sentence):
        return None
    found = [mentioned(sentence, vocab) for vocab in (OBJECTS, COLORS, LOCATIONS)]
    if any(len(values) != 1 for values in found):
        return None
    obj, colour, place = (values[0] for values in found)
    if not re.search(
        rf"\b(?:{colour}\s+{obj}|{obj}\s+(?:is|was|will be)\s+{colour}"
        rf"|colou?r\s+of\s+(?:the\s+)?{obj}\s+is\s+{colour})\b", sentence
    ):
        return None
    if not re.search(rf"\b(?:in|at)\s+(?:the\s+)?{place}\b", sentence):
        return None
    return {"object": obj, "color": colour, "location": place}


def parse_fields(text):
    """Use the historical attribute metric on the serialization experiment."""
    if parse_state(text) == UNKNOWN:
        return {"color": UNKNOWN, "location": UNKNOWN}, True
    sentence = first_sentence(text)
    fields = {}
    for name, vocab, label in [("color", COLORS, r"colou?r"), ("location", LOCATIONS, "location")]:
        values = mentioned(sentence, vocab)
        explicit_unknown = re.search(label + r"\s*(?:is|:|=)?\s*(?:unknown|undetermined)", sentence)
        fields[name] = UNKNOWN if explicit_unknown else values[0] if len(values) == 1 else None
    return fields, False


def assess(data, *, fields):
    stats, correct = {}, {}
    for method in data["methods"]:
        checks, complete, unknown, tokens = [], [], [], []
        for row in data["cases"]:
            answer = row["outputs"][method]
            truth = row["truth"]
            if fields:
                predicted, abstains = parse_fields(answer["text"])
                missing = UNKNOWN in truth.values()
                ok = missing if abstains else predicted == truth
                parsed = abstains or all(value is not None for value in predicted.values())
            else:
                predicted = parse_state(answer["text"])
                missing = truth is None
                ok = predicted == (UNKNOWN if missing else truth)
                parsed = predicted is not None
            checks.append(int(ok))
            complete.append(int(parsed))
            unknown.append(int(ok) if missing else None)
            tokens.append(answer["tokens"])
        correct[method] = checks
        stats[method] = {
            "cases": len(checks),
            "correct": sum(checks),
            "complete": sum(complete),
            "unknown_correct": sum(v for v in unknown if v is not None),
            "unknown_cases": sum(v is not None for v in unknown),
            "mean_final_tokens": sum(tokens) / len(tokens),
        }
    groups = [row["group"] for row in data["cases"]]
    method = "decision_first" if not fields else "independent_sentence"
    base = "native" if not fields else "independent_fields"
    return {"methods": stats, "paired_difference": grouped_difference(correct[method], correct[base], groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=ROOT / "data" / "decision_outputs.json")
    args = parser.parse_args()
    fixture = read_json(args.data)
    print_json({
        "state_generation": assess(fixture["state_generation"], fields=False),
        "plan_serialization": assess(fixture["plan_serialization"], fields=True),
    })


if __name__ == "__main__":
    main()
