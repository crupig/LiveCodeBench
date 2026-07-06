# ./lcb_runner/runner/main_gen_tests.py
import os
import json

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from lcb_runner.runner.runner_utils import build_runner
from lcb_runner.utils.path_utils import get_output_path
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    combine_results,
    sort_and_extract_save_results,
)
from transformers import AutoTokenizer


def main():
    args = get_args()

    model = LanguageModelStore[args.model]
    benchmark, format_prompt = build_prompt_benchmark(args)
    if args.debug:
        print(f"Running with {len(benchmark)} instances in debug mode")
        benchmark = benchmark[:15]

    output_path = get_output_path(model.model_repr, args)
    eval_file = output_path.replace(".json", "_eval.json")
    eval_all_file = output_path.replace(".json", "_eval_all.json")

    if args.continue_existing or args.continue_existing_with_eval:
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                old_save_results = json.load(f)
        elif os.path.exists(eval_all_file):
            with open(eval_all_file, "r") as f:
                old_save_results = json.load(f)
        else:
            print(
                f"File {output_path} does not exist in --continue_existing, starting from scratch"
            )
            old_save_results = []

        old_save_results = [
            instance
            for instance in old_save_results
            if instance["output_list"] and [x for x in instance["output_list"] if x]
        ]
        old_save_results_question_ids = [
            instance["question_id"] for instance in old_save_results
        ]
        remaining_benchmark = [
            instance
            for instance in benchmark
            if instance.question_id not in old_save_results_question_ids
        ]
        print(
            f"Found {len(old_save_results)} existing generations, continuing with {len(remaining_benchmark)} remaining"
        )
    else:
        old_save_results = []
        remaining_benchmark = benchmark

    if len(remaining_benchmark) > 0:
        ids_to_keep = json.load(open(args.all_ids_dict))['LiveCodeBench'][args.split] if args.split != "all" else None
        runner = build_runner(args, model)
        if ids_to_keep is not None:
            remaining_benchmark = [instance for instance in remaining_benchmark if instance.question_id in ids_to_keep]
        results = runner.run_main(remaining_benchmark, format_prompt)
    else:
        results = []
        prompts = []

    combined_results = combine_results(
        args.scenario, results, model, args.cot_code_execution
    )

    
    save_results = []
    for instance, (outputs_list, extracted_list) in zip(
        remaining_benchmark, combined_results
    ):
        save_result_instance = instance.insert_output(
            outputs_list, extracted_list
        )
        save_result_instance["generated_by"] = args.model.split("/")[-1]
        # rename fields for uniformity
        save_results.append(save_result_instance)

    if args.continue_existing or args.continue_existing_with_eval:
        save_results += old_save_results

    save_results, combined_results = sort_and_extract_save_results(
        args.scenario, save_results
    )

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=4)


if __name__ == "__main__":
    main()
