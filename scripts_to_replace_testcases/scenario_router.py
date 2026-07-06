# ./lcb_runner/runner/scenario_router.py
import pandas as pd
import json
from typing import Union

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModel
from lcb_runner.evaluation import (
    codegen_metrics,
    test_output_metrics,
    code_execution_metrics,
)

from lcb_runner.prompts import (
    format_prompt_generation,
    format_prompt_test_output,
    format_prompt_execution,
    format_prompt_execution_cot,
    format_prompt_self_repair,
)
from lcb_runner.utils.extraction_utils import (
    extract_code,
    extract_test_output_code,
    extract_execution_code,
)

from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    TestOutputPredictionProblem,
    CodeExecutionProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
    load_test_prediction_dataset,
    load_code_execution_dataset,
)

# BenchMarkType = list[CodeGenerationProblem | TestOutputPredictionProblem]
BenchMarkType = list[
    Union[CodeGenerationProblem, CodeExecutionProblem, TestOutputPredictionProblem]
]


def build_prompt_benchmark(
    args,
) -> tuple[
    list[CodeExecutionProblem]
    | list[CodeGenerationProblem]
    | list[TestOutputPredictionProblem],
    callable,
]:
    scenario: Scenario = args.scenario

    if scenario == Scenario.codegeneration:
        not_fast: bool = args.not_fast
        if not_fast:
            benchmark = load_code_generation_dataset_not_fast(args.release_version)
        else:
            benchmark = load_code_generation_dataset(
                args.release_version,
                start_date=args.start_date,
                end_date=args.end_date
            )
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_generation
    elif scenario == Scenario.testoutputprediction:
        benchmark = load_test_prediction_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: (x.question_id, x.test_id))
        format_prompt = format_prompt_test_output
    elif scenario == Scenario.selfrepair:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_self_repair
    elif scenario == Scenario.codeexecution:
        cot_code_execution: bool = args.cot_code_execution
        benchmark = load_code_execution_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: int(x.id.split("_")[1]))
        if cot_code_execution:
            format_prompt = format_prompt_execution_cot
        else:
            format_prompt = format_prompt_execution
    else:
        raise ValueError(f"Scenario {scenario} not implemented")
    return benchmark, format_prompt


def combine_results(
    scenario: Scenario,
    results: list[list[str]],
    model: LanguageModel,
    cot_code_execution: bool = False,
):
    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.selfrepair:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.codeexecution:
        combined_results = [
            (
                outputs_list,
                [
                    extract_execution_code(
                        output, model.model_style, cot=cot_code_execution
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results


def sort_and_extract_save_results(scenario: Scenario, save_results: list[dict]):
    if scenario == Scenario.codegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]

    elif scenario == Scenario.testoutputprediction:
        save_results = sorted(
            save_results, key=lambda x: (x["question_id"], x["test_id"])
        )
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.selfrepair:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.codeexecution:
        save_results = sorted(save_results, key=lambda x: int(x["id"].split("_")[1]))
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return save_results, combined_results


def get_metrics(
    scenario: Scenario,
    args,
    benchmark: list[
        CodeGenerationProblem | CodeExecutionProblem | TestOutputPredictionProblem
    ],
    combined_results,
    input_outputs_ids
):
    IDS_TYPE_CALL = json.load(open("../../../../constants/ids_train_val_test.json", "r"))["LiveCodeBench"]["test_call"]
    IDS_TYPE_STDIO = json.load(open("../../../../constants/ids_train_val_test.json", "r"))["LiveCodeBench"]["test_stdio"]
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]

    generations = [extracted for _, extracted, _ in combined_results]
    test_idxs = [test_idx for _, _, test_idx in combined_results]
    inputs = [inp for inp, _, _ in input_outputs_ids]
    outputs = [out for _, out, _ in input_outputs_ids]
    task_idxs = [task_idx for _, _, task_idx in input_outputs_ids]
    eval_samples_new = []

    quid_list = [json.loads(e['input_output']) for e in eval_samples]
    quid_df = pd.DataFrame(quid_list)
    for inp, out, _id in zip(inputs, outputs, task_idxs):
        inp = f"{inp}"
        # treat the input and output based on the type of the problem (CALL or STDIO)
        # if CALL type
        if _id in IDS_TYPE_CALL:
            if _id in ["2892"]: # seems that these ids are not consistent. They ask for a list of ints in the problem description, but then they pass a list of lists in the tests...
                inp = f"[{inp}]"

            if isinstance(out, str) and "true" in out.lower():
                out = out.replace("True", "true")
            if isinstance(out, str) and "false" in out.lower():
                out = out.replace("False", "false")
        
        # if STDIO type
        elif _id in IDS_TYPE_STDIO:
            inp = inp.replace("\"", "")
            inp = inp.replace("\r", "")
            inp = inp.replace("\\n", "\n")
            out = out.replace("\"", "")
            out = out.replace("\r", "")
            out = out.replace("\\n", "\n")
        
        fn_name = quid_df[quid_df['question_id'] == _id]['fn_name'].iloc[0]
        es = {"inputs" : [inp], "outputs" : [out], "fn_name" : fn_name}
        eval_samples_new.append({"input_output": json.dumps(es)})

    if scenario == Scenario.codegeneration or scenario == Scenario.selfrepair:
        metrics = codegen_metrics(
            eval_samples_new,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )

    elif args.scenario == Scenario.testoutputprediction:
        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=[1, 5],
        )

    elif args.scenario == Scenario.codeexecution:
        metrics = code_execution_metrics(
            eval_samples,
            generations,
        )

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    print(metrics[0]["pass@1"])
    metrics.append(test_idxs)
    metrics.append(eval_samples_new)

    return metrics
