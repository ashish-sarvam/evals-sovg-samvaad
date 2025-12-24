import json
from argparse import ArgumentParser


def convert_samples_to_dict(judge_eval):
    match_counter = 0
    match_keys = []
    sample_results = judge_eval["sample_results"]
    sample_results_dict = {}
    for sample_result in sample_results:
        if sample_result["thread_str"] in sample_results_dict:
            match_keys.append(sample_result["thread_str"])
            match_counter += 1
        sample_results_dict[sample_result["thread_str"]] = sample_result

    match_keys = list(set(match_keys))
    for key in match_keys:
        sample_results_dict.pop(key)

    print(f"Match counter: {match_counter}")

    return sample_results_dict


def check_match(judge_1_samples_dict, judge_2_samples_dict):
    keys_1 = list(judge_1_samples_dict.keys())
    keys_2 = list(judge_2_samples_dict.keys())

    for key_1 in keys_1:
        if key_1 not in keys_2:
            print(f"Thread {key_1} not in judge_2")

    for key_2 in keys_2:
        if key_2 not in keys_1:
            print(f"Thread {key_2} not in judge_1")


def get_model_results(judge_samples_dict, model_name):
    model_results = {}
    for key in judge_samples_dict.keys():
        try:
            model_results[key] = judge_samples_dict[key]["model_results"][model_name]
        except Exception as e:
            # print(f"Error getting model results for key {key}: {e}")
            continue

    return model_results


def analyze_model_results(judge_1_model_results, judge_2_model_results):
    same_results = 0
    different_results = 0

    different_results_dict_dict = {}

    for key in judge_1_model_results.keys():
        try:
            judge_1_sample = judge_1_model_results[key]
            judge_2_sample = judge_2_model_results[key]
        
            if judge_1_sample["score"] == judge_2_sample["score"]:
                same_results += 1
            else:
                different_results_dict_dict[key] = {
                    "judge_1_sample": judge_1_sample,
                    "judge_2_sample": judge_2_sample,
                }
                different_results += 1

        except Exception as e:
            # print(f"Error analyzing model results for key {key}: {e}")
            continue
    
    json.dump(different_results_dict_dict, open("different_results_analysis.json", "w"))

    print(f"Same results: {same_results}")
    print(f"Different results: {different_results}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--judge_eval_file_1", type=str, required=True)
    parser.add_argument("--judge_eval_file_2", type=str, required=True)
    parser.add_argument("--model_1", type=str, required=True)
    parser.add_argument("--model_2", type=str, required=True)
    args = parser.parse_args()

    judge_eval_file_1 = json.load(open(args.judge_eval_file_1))
    judge_eval_file_2 = json.load(open(args.judge_eval_file_2))

    judge_1_samples_dict = convert_samples_to_dict(judge_eval_file_1)
    judge_2_samples_dict = convert_samples_to_dict(judge_eval_file_2)

    check_match(judge_1_samples_dict, judge_2_samples_dict)

    judge_1_model_results = get_model_results(judge_1_samples_dict, args.model_1)
    judge_2_model_results = get_model_results(judge_2_samples_dict, args.model_2)

    analyze_model_results(judge_1_model_results, judge_2_model_results)
