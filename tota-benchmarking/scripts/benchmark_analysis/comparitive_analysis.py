import json
from argparse import ArgumentParser


def get_model_samples(benchmark, model_name):
    model_name_samples = {}
    for sample_id, sample in benchmark["samples"].items():
        model_results = sample["model_results"]
        model_name_results = model_results[model_name]
        model_name_samples[sample_id] = model_name_results

    return model_name_samples


def get_analysis_results(model_1_samples, model_2_samples):
    both_failed = []
    model_1_failed = []
    model_2_failed = []
    both_passed = []

    for sample_id, model_1_result in model_1_samples.items():
        model_1_has_failed = model_1_result["has_failed"]
        model_2_has_failed = model_2_samples[sample_id]["has_failed"]

        if model_1_has_failed and model_2_has_failed:
            both_failed.append(sample_id)
        elif model_1_has_failed:
            model_1_failed.append(sample_id)
        elif model_2_has_failed:
            model_2_failed.append(sample_id)
        else:
            both_passed.append(sample_id)

    return {
        "both_failed": both_failed,
        "model_1_failed": model_1_failed,
        "model_2_failed": model_2_failed,
        "both_passed": both_passed,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark_dir_1", type=str, required=True)
    parser.add_argument("--benchmark_dir_2", type=str, required=True)
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, required=True)
    args = parser.parse_args()

    try:
        b1 = json.load(open(args.benchmark_dir_1, "r"))
    except Exception as e:
        print(f"Error loading benchmark 1: {e}")
        import logging
        logging.exception(e)
        exit(1)

    try:
        b2 = json.load(open(args.benchmark_dir_2, "r"))
    except Exception as e:
        print(f"Error loading benchmark 2: {e}")
        import logging
        logging.exception(e)
        exit(1)

    b1_samples = b1["samples"]
    b2_samples = b2["samples"]

    model_1_samples = get_model_samples(b1, args.model_1)
    model_2_samples = get_model_samples(b2, args.model_2)

    analysis_results = get_analysis_results(model_1_samples, model_2_samples)

    print("### Evaluation Results ###")
    print(f'Both Failed: {len(analysis_results["both_failed"])}')
    print(f'Model 1 Failed: {len(analysis_results["model_1_failed"])}')
    print(f'Model 2 Failed: {len(analysis_results["model_2_failed"])}')
    print(f'Both Passed: {len(analysis_results["both_passed"])}')

    with open("comparitive_analysis_results.json", "w") as f:
        json.dump(analysis_results, f, indent=2)
