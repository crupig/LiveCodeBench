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

def postprocess_code_generation_output(output: str):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # comment any line starting with ```
    output = re.sub(r"^```.*$", lambda m: "# " + m.group(0), output, flags=re.MULTILINE)
    return output

def main():
    args = get_args()

    with open(args.generations_path) as f:
        save_results = json.load(f)
    
    generated_by = save_results[0]["generated_by"] if len(save_results) > 0 else "unknown_model"
    ids_in_results = [instance["question_id"] for instance in save_results]

    combined_results = [
        (save_result_instance["output_list"], save_result_instance["code_list"], save_result_instance["log_probabilities"], save_result_instance["question_id"])
        for save_result_instance in save_results
    ]

    # post-process generations
    code_list = [extracted for _, extracted, _, _ in combined_results]
    postprocessed_code_list = [
                                [postprocess_code_generation_output(code) for code in code_l]
                                for code_l in code_list
                            ]
    combined_results = [
        (output_list, postprocessed_code, log_probabilities, question_id)
        for (output_list, _, log_probabilities, question_id), postprocessed_code in zip(combined_results, postprocessed_code_list)
    ]
    # end post-processing

    benchmark, _ = build_prompt_benchmark(args)
    eval_file = args.generations_path.replace("/generations/", "/generations-tested/").replace(".json", "_eval.json")
    eval_all_file = args.generations_path.replace("/generations/", "/generations-tested/").replace(".json", "_eval_results.json")

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
        #########
        # IDS_TO_KEEP = ['abc342_c', '3580', '2868', 'abc318_e', 'abc301_d', 'abc397_f', 'abc304_d', 'abc356_e', '3608', '1899_D', 'abc377_f', 'abc331_c', 'abc392_g', 'abc341_e', '2952', '3344', 'abc372_c', '3362', '3306', 'abc367_f', '3788', '3700', 'arc185_e', 'abc355_c', '3213', 'abc371_d', 'abc369_c', 'abc388_c', 'abc361_c', 'abc367_d', '3583', 'abc390_e', 'abc362_d', 'abc331_e', 'abc324_e', 'abc344_c', 'abc340_e', '2915', '3329', 'abc369_d', '3750', '3025', 'abc358_e', '3382', 'abc305_e', 'abc315_c', '3725', '3801', 'abc366_f', 'abc322_e', '3696', 'abc353_c', '3243', 'abc322_c', 'abc368_c', 'arc182_e', '3316', 'abc393_f', 'abc362_e', '3032', 'abc306_e', 'abc388_f', '3240', 'abc351_f', '2845', '3699', '3223', 'abc376_e', 'abc370_e', 'abc348_e', '2833', '3466', 'abc334_d', 'abc343_d', 'abc391_g', 'abc375_d', 'abc364_c', '3688', 'abc355_d', 'abc344_e', 'abc377_d', '1899_C', '3000', '2779', 'abc304_e', 'abc373_f', 'abc372_g', '2784', '3438', 'abc346_d', 'abc302_d', 'abc326_c', 'abc373_c', 'arc181_a', 'abc359_d', 'abc352_e', 'abc321_d', '3479', '3680', 'abc367_g', 'arc181_d', 'abc311_e', 'abc391_f', 'abc308_e', '2884', '3104', 'abc372_d', 'abc375_g', '3777', '3337', 'abc370_d', 'abc396_f', 'arc182_d', 'abc325_c', 'abc366_e', 'abc364_d', 'abc396_g', 'abc379_e', 'abc304_c', '3416', 'abc330_e', '3423', 'abc356_d', 'abc330_b', '2953', 'abc365_e', 'abc306_c', 'abc350_c', 'abc329_d', 'abc353_d', '3603', '3634', 'abc371_e', '3528', '3192', 'arc184_c', 'abc329_f', 'abc331_d', '3759', 'abc382_d', '3363', 'abc347_e', '3482', 'abc346_e', 'abc395_c', 'abc328_c', 'arc196_a', '3507', 'abc396_c']
        # remaining_indices = [
        #     idx
        #     for idx in range(len(benchmark))
        #     if benchmark[idx].question_id in ids_in_results
        # ]
        remaining_indices = []
        for iir in ids_in_results:
            for idx in range(len(benchmark)):
                if benchmark[idx].question_id == iir:
                    remaining_indices.append(idx)
                    break
        benchmark = [benchmark[idx] for idx in remaining_indices]
        # combined_results = [combined_results[idx] for idx in remaining_indices]
        #########
        metrics = get_metrics(args.scenario, args, benchmark, combined_results)
        graded = extract_instance_results(metrics[1])
        old_eval_all_results = []
        old_eval_results = []

    if args.scenario == Scenario.codegeneration:
        if metrics:
            metadatas = metrics[2]
        else:
            metadatas = [[] for _ in benchmark]
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, log_probabilities, metadata=meta
            )
            for instance, (outputs_list, extracted_list, log_probabilities, question_id), graded_list, meta in zip(
                benchmark, combined_results, graded, metadatas
            )
        ]

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
                log_probabilities,
                metadata=meta,
                original_code_list=original_code_list,
            )
            for instance, (
                outputs_list,
                extracted_list,
                log_probabilities,
                question_id,
            ), graded_list, meta, original_code_list in zip(
                benchmark, combined_results, graded, metadatas, original_code_lists
            )
        ]

    else:
        save_eval_results = [
            instance.insert_output_evaluation(
                outputs_list, extracted_list, graded_list, log_probabilities
            )
            for instance, (outputs_list, extracted_list, log_probabilities, question_id), graded_list in zip(
                benchmark, combined_results, graded
            )
        ]

    save_eval_results = old_eval_all_results + save_eval_results

    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=4)

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
