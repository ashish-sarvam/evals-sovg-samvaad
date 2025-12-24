import json
from argparse import ArgumentParser


def extract_wrong_golden_responses(results: dict) -> list[str]:
    wrong_golden_responses = []
    for sample in results["sample_results"]:
        for model_name, model_result in sample["model_results"].items():
            if model_name == "golden_response":
                if model_result["score"] < 1.0:
                    wrong_golden_responses.append(model_result)

    return wrong_golden_responses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument(
        "--output_file", type=str, required=False, default="wrong_golden_responses.json"
    )
    args = parser.parse_args()

    with open(args.results_file, "r") as f:
        results = json.load(f)

    wrong_golden_responses = extract_wrong_golden_responses(results)
    json.dump(wrong_golden_responses, open(args.output_file, "w"), indent=4)
