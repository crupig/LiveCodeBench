# ./lcb_runner/runner/main_eval_tests.py
import os
import json
import re

from lcb_runner.runner.parser import get_args
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.runner.scenario_router import (
    build_prompt_benchmark,
    get_metrics,
)

def main():
    args = get_args()

    with open(args.generations_path) as f:
        save_results = json.load(f)
    
    generated_by = save_results[0]["generated_by"] if len(save_results) > 0 else "unknown_model"
    combined_results = [
        (save_result_instance["solution_idx"], save_result_instance["code_list"], save_result_instance["test_idx"])
        for save_result_instance in save_results
    ]

    input_outputs_ids = [
        (save_result_instance["test_input"], save_result_instance["test_output"], save_result_instance["task_idx"])
        for save_result_instance in save_results
    ]

    benchmark, _ = build_prompt_benchmark(args)
    eval_file = args.generations_path.replace("/test-4-execution/", "/test-execution/").replace(".json", "_eval.json")
    eval_all_file = args.generations_path.replace("/test-4-execution/", "/test-execution/").replace(".json", "_eval_results.json")

    if args.continue_existing_with_eval and os.path.exists(eval_all_file):
        with open(eval_all_file) as fp:
            old_eval_all_results = json.load(fp)

        if os.path.exists(eval_file):
            with open(eval_file) as fp:
                old_eval_results = json.load(fp)
        else:
            old_eval_results = None

        old_eval_results_question_ids = [
            instance["question_id"] for instance in old_eval_all_results
        ]
        remaining_indices = [
            idx
            for idx in range(len(benchmark))
            if benchmark[idx].question_id not in old_eval_results_question_ids
        ]
        benchmark = [benchmark[idx] for idx in remaining_indices]
        combined_results = [combined_results[idx] for idx in remaining_indices]

        old_eval_size = len(old_eval_results_question_ids)
        new_eval_size = len(benchmark)

        if new_eval_size == 0:
            return

        print(f"Found {old_eval_size}, running evals for {new_eval_size} problems")

        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])

        if old_eval_results:
            for key in metrics[0]:
                if key in old_eval_results[0]:
                    if key != "detail":
                        metrics[0][key] = (
                            old_eval_size * old_eval_results[0][key]
                            + new_eval_size * metrics[0][key]
                        )
                        metrics[0][key] /= old_eval_size + new_eval_size

            for key in metrics[0]["detail"]:
                if key in old_eval_results[0]["detail"]:
                    metrics[0]["detail"][key] = {
                        **metrics[0]["detail"][key],
                        **old_eval_results[0]["detail"][key],
                    }
            metrics[1] = {**metrics[1], **old_eval_results[1]}
        else:
            print("Old eval file not present, cannot update eval file")
            metrics = {}

    else:
        metrics = get_metrics(args.scenario, args, benchmark, combined_results, input_outputs_ids)
        graded = extract_instance_results(metrics[1])
        old_eval_all_results = []
        old_eval_results = []

    if args.scenario == Scenario.codegeneration:
        if metrics:
            metadatas = metrics[2]
            test_idxs = metrics[3]
            input_outputs = metrics[4]
        else:
            metadatas = [[] for _ in benchmark]
        
        # go through test execution outputs and append to save_eval_results
        save_eval_results = []
        for (outputs_list, extracted_list, _), graded_list, meta in zip(combined_results, graded, metadatas):
            toapp = {
                "solution_idx_list": outputs_list,
                "code_list": extracted_list,
                "graded_list": graded_list,
                "metadata": meta,
            }
            save_eval_results.append(toapp)
            
        # add test_idx to each instance in save_eval_results
        for ser, test_idx, io in zip(save_eval_results, test_idxs, input_outputs):
            ser["test_idx"] = test_idx
            ser["test_input_output"] = io

        if metrics and old_eval_results:
            old_eval_results
            metrics[2] = old_eval_results[2] + metrics[2]
        
    elif args.scenario == Scenario.selfrepair:
        metadatas = metrics[2]
        with open(
            f"output/{model.model_repr}/{Scenario.codegeneration}_{args.codegen_n}_{args.temperature}_eval_all.json"
        ) as f:
            code_gen_evals = json.load(f)
        original_code_lists = [
            code_gen_eval["code_list"] for code_gen_eval in code_gen_evals
        ]

        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list,
                extracted_list,
                graded_list,
                metadata=meta,
                original_code_list=original_code_list,
            )
            for instance, (
                outputs_list,
                extracted_list,
            ), graded_list, meta, original_code_list in zip(
                benchmark, combined_results, graded, metadatas, original_code_lists
            )
        ]

    else:
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list
            )
            for instance, (outputs_list, extracted_list), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]

    save_eval_results = old_eval_all_results + save_eval_results

    # add generated_by to each instance in save_eval_results
    save_eval_results_ = []
    for instance in save_eval_results:
        instance["generated_by"] = generated_by
        save_eval_results_.append(instance)
    save_eval_results = save_eval_results_

    with open(eval_all_file, "w") as f:
        json.dump(save_eval_results, f, indent=4)


if __name__ == "__main__":
    main()
