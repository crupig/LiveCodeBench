import json
import pandas as pd
import numpy as np
import os
import re
import sys
import ast
from collections import Counter
import re
from tqdm import tqdm

EXTRACT_OUTPUT_PATTERNS = [
        r'\*?\*?Output\s*\d*:?\*?\*?\s*`{1,3}\s*(.*?)\s*`{1,3}',
        r'\*?\*?Expected [Oo]utput\s*\d*:?\*?\*?\s*`{1,3}\s*(.*?)\s*`{1,3}',
        r'Output:\s*`{1,3}\s*(.*?)\s*`{1,3}',

        r'output_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'output_string_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'output_str_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'sample_output_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'expected_output_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'output_expected_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'expected_?\d*\s*=\s*\"{3}(.*?)\"{3}',

        r'output_?\d*\s*=\s*\"(.*?)\"',
        r'output_string_?\d*\s*=\s*\"(.*?)\"',
        r'output_str_?\d*\s*=\s*\"(.*?)\"',
        r'sample_output_?\d*\s*=\s*\"(.*?)\"',
        r'expected_output_?\d*\s*=\s*\"(.*?)\"',
        r'output_expected_?\d*\s*=\s*\"(.*?)\"',
        r'expected_?\d*\s*=\s*\"(.*?)\"',

        r'Test [Oo]utput\s*\d*:?\s*(.*?)$',
        r'-?\*?\*?\s*[Oo]utput:?\*?\*?:?\s*`{1,3}(.*?)`{1,3}',
        r'Output\s*\d*:?\s*(.*?)$',
        r'Output\s*\d*:?\s*(.*?)\s*`{1,3}',
        r'`{1,3}[Oo]utput\s*(.*?)\s*`{1,3}',
        r'Output:\s*(.*)$',
    ]

EXTRACT_INPUT_PATTERNS = [
        r'\*?\*?Input:?\*?\*?\s*`{1,3}\s*(.*?)\s*`{1,3}',
        r'Input:\s*`{1,3}\s*(.*?)\s*`{1,3}',
        r'Input:\s*(.*)\s*Expected [Oo]utput:?',
        r'Input:\s*(.*)\s*[Oo]utput:?',
        r'\*\*Input:?\*\*\s*(.*)\s*\*\*[Oo]utput:?\*\*',
        
        r'input_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'input_data\s*=\s*\"{3}(.*?)\"{3}',
        r'input_str_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'input_string_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'sample_input_?\d*\s*=\s*\"{3}(.*?)\"{3}',
        r'input_?\d*\s*=\s*\"{3}(.*?)\"{3}',

        r'input_?\d*\s*=\s*\"(.*?)\"',
        r'input_data\s*=\s*\"(.*?)\"',
        r'input_str_?\d*\s*=\s*\"(.*?)\"',
        r'input_string_?\d*\s*=\s*\"(.*?)\"',
        r'sample_input_?\d*\s*=\s*\"(.*?)\"',
        r'input_?\d*\s*=\s*\"(.*?)\"',
        
        r'`{1,3}[Ii]nput\s*(.*?)\s*`{1,3}',
    ]

CALL_ASSERT_PATTERNS = [
        r'^assert Solution\(\)\.\w+\((.*?)\)\s*==',
        r'^assert Solution\.\w+\((.*?)\)\s*==',
        r'^assert solution\.\w+\((.*?)\)\s*==',
        r'^assert solution\d+\.\w+\((.*?)\)\s*==',
        r'^assert self\.\w+\((.*?)\)\s*==',
        r'^assert sol\.\w+\((.*?)\)\s*==',
        r'^assert setup\.\w+\((.*?)\)\s*==',
        r'^assert s\.\w+\((.*?)\)\s*==',
        r'^assert class_solution\.\w+\((.*?)\)\s*==',
        r'^assert \(?\w+\((.*?)\)\)?\s*=='
    ]

STDIO_SPLIT_PATTERNS = [
            r"#?\s*Test\s+[Cc]ase\s+\d+:?",
            r"\*\*Test\s+[Cc]ase\s+\d+:?\*\*",
            r"#{3}\s*Test\s+[Cc]ase\s+\d+:?",
            r"\d+\.\s*\*\*Test\s+[Cc]ase\s+\d+:?\*\*",
            r"\d+\.\s*Test\s+[Cc]ase\s+\d+:?",
            r"#?\s*Sample [Tt]est [Cc]ase\s*\d*:?",

            r"#?\s*Test\s+[Ii]nput\s+\d+:?",
            r"\*\*Test\s+[Ii]nput\s+\d+:?\*\*",
            r"#{3}\s*Test\s+[Ii]nput\s+\d+:?",
            r"\d+\.\s*\*\*Test\s+[Ii]nput\s+\d+:?\*\*",
            r"\d+\.\s*Test\s+[Ii]nput\s+\d+:?",
            
            r"\d+\.\s*\*\*Input:?\*\*",
            r"\d+\.\s*\n\s*Input:?",
            r"\*\*Test\s+Input \d+:?\*\*",
            r"Test\s+Input:?",
            r"`{1,3}\s*Input\s*\d*:?",
            r"Input\s*\d*:?",
        ]


def extract_input(row):
    if row['type'] == 'CALL':
        
        for pattern in CALL_ASSERT_PATTERNS:
            match = re.match(pattern, row['test_statement'])
            if match:
                return match.group(1).strip()
    
    elif row['type'] == 'STDIO':
        for pattern in EXTRACT_INPUT_PATTERNS:
            match = re.search(pattern, row['test_statement'], re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # if no match: try with ```(.*?)``` followed by any output pattern
        for pattern in EXTRACT_OUTPUT_PATTERNS:
            match = re.search(r'`{1,3}(.*?)`{1,3}.*' + pattern, row['test_statement'], re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # if still no match: try with (.*?) followed by any output pattern
        for i, pattern in enumerate(EXTRACT_OUTPUT_PATTERNS):
            match = re.search(r'(.*?)' + pattern, row['test_statement'], re.DOTALL)
            if match:
                return match.group(1).strip()
    return '-'

def extract_output(row):
    if row['type'] == 'CALL':
        out = row['test_statement'].split("==")[-1].strip()
        patterns = [
            ", \"Test case",
            ", f\"Test case",
            ", \"Example",
            ]
        for pattern in patterns:
            if pattern in out:
                out = out.split(pattern)[0].strip()
                return out
        
        return out
    
    elif row['type'] == 'STDIO':
        for pattern in EXTRACT_OUTPUT_PATTERNS:
            match = re.search(pattern, row['test_statement'], re.DOTALL)
            if match:
                return match.group(1).strip()
    
    return '-'

def split_function_args_with_newline_CALL(arg_string: str) -> str:
    """
    Split top-level function arguments onto separate lines.

    Examples:
        '"abc", 0'
        -> '"abc"\\n0'

        '[1, 1, 1, 1, 1], [1]'
        -> '[1, 1, 1, 1, 1]\\n[1]'

        '"abcdefghij", [[1, 2, 3, 4], [2, 3, 4, 5]]'
        -> '"abcdefghij"\\n[[1, 2, 3, 4], [2, 3, 4, 5]]'
    """

    result = []
    current = []

    depth = 0
    in_string = False
    escape = False
    quote_char = None

    for ch in arg_string:
        # Handle escaping inside strings
        if escape:
            current.append(ch)
            escape = False
            continue

        if in_string:
            current.append(ch)

            if ch == "\\":
                escape = True
            elif ch == quote_char:
                in_string = False

            continue

        # Enter string
        if ch in ('"', "'"):
            in_string = True
            quote_char = ch
            current.append(ch)
            continue

        # Track nesting depth
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1

        # Split only on top-level commas
        if ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    # Add final argument
    if current:
        result.append("".join(current).strip())

    return "\n".join(result)

def clean_from_stange_formats(arg_string: str) -> str:
    # if the input is like "4 8 4\n1 5\n3 2\n4 1\n5 3"
    # strip quotes
    arg_string = arg_string.strip('`')

    if arg_string.startswith('"') and arg_string.endswith('"'):
        arg_string = arg_string.strip('"')

    # if the input is like |
    # |5|
    # |##...|
    # |, string like "5\n##...\n"
    if arg_string.startswith("|") and arg_string.endswith("|"):
        lines = arg_string.strip().split("\n")
        split_lines = ["\n".join(line.strip("|").split()) for line in lines]
        arg_string = "\n".join(split_lines)
    
    # split by newlines and strip each line, then join with newlines again
    lines = arg_string.split("\n")
    arg_string = "\n".join([line.strip() for line in lines if line.strip()])

    return arg_string

def split_function_args_with_newline(row):
    if row['type'] == 'CALL':
        return split_function_args_with_newline_CALL(row['test_input'])
    else:
        return clean_from_stange_formats(row['test_input'])

def get_test_statement(row):
    if row["type"] == "CALL":
        return '\n'.join([line.strip() for line in row["raw_output"].splitlines() if line.strip().startswith('assert') or line.strip().startswith('self.assert')])
    elif row["type"] == "STDIO":
        return row["raw_output"]

def split_io_pairs(text: str, patterns: list) -> list:
    '''
    takes a string and splits it according to the following regex
    does not include the split patter in the output
    also, iteratively applies the patterns until no more splits can be made
    '''
    parts = [text]

    changed = True
    while changed:
        changed = False
        new_parts = []

        for part in parts:
            split_occurred = False

            for pattern in patterns:
                if re.search(pattern, part):
                    split_parts = re.split(pattern, part)

                    # remove empty chunks
                    split_parts = [p.strip() for p in split_parts if p.strip()]

                    new_parts.extend(split_parts)
                    split_occurred = True
                    changed = True
                    break

            if not split_occurred:
                new_parts.append(part.strip())

        parts = new_parts

    return parts

def split_by_independent_test_case_and_deduplicate(row):
    if row["type"] == "CALL":
        # split by lines starting with assert and keep only unique asserts (removing comments)
        return np.unique([y.split('#')[0].strip() if '#' in y else y.strip() for y in row["test_statement_raw"].splitlines()])
    elif row["type"] == "STDIO":
        return split_io_pairs(row["test_statement_raw"], STDIO_SPLIT_PATTERNS)

if __name__ == "__main__":
    tqdm.pandas()

    root_dir = "../test-generations"
    # walk through all subdirectories
    overall_df = pd.DataFrame()
    pattern = r'_\d+_1\.0-\d+\.json$'
    for subdir, dirs, files in os.walk(root_dir):
        for file in sorted(files):

            if re.search(pattern, file):
                file_path = os.path.join(subdir, file)
                print(f"Processing file: {os.path.relpath(file_path)}")

                with open(file_path) as f:
                    results = json.load(f)

                postp = []
                for instance in results:
                    generated_by = os.path.relpath(file_path).split("/")[2].split('--')[-1]
                    base = {
                        "task_idx": instance["question_id"],
                        "generated_by": instance["generated_by"],
                        "prompt": instance["question_content"],
                        
                        "starter_code": instance["starter_code"],
                    }
                    for raw_out, out in zip(
                        instance["output_list"],
                        instance["code_list"],
                    ):
                        row = base.copy()
                        row["method"] = out
                        row["raw_output"] = raw_out
                        postp.append(row)
                
                overall_df = pd.concat([overall_df, pd.DataFrame(postp)])
    
    
    # import type of problem type (either CALL or STDIO)
    taskid_type = pd.read_csv("../constants/taskid_type.csv")
    overall_df = overall_df.merge(taskid_type, on="task_idx", how="left")

    overall_df['test_statement_raw'] = overall_df.apply(get_test_statement, axis=1)
    overall_df = overall_df.groupby(['task_idx', 'generated_by'], as_index=False).agg({
            'prompt': 'first',
            'type': 'first',
            'test_statement_raw': lambda x: '\n'.join(x)
        }).reset_index()

    # remove comments from assert statements and keep only unique asserts
    overall_df['test_statement'] = overall_df.apply(split_by_independent_test_case_and_deduplicate, axis=1)

    overall_df = overall_df.explode('test_statement').reset_index(drop=True)

    # CLEANING
    num_task_ids = overall_df['task_idx'].nunique()
    # remove the lines with test_statement that do not contain "output"
    overall_df = overall_df[(overall_df['test_statement'].str.lower().str.contains("output", case=False, na=False)) | (overall_df['type'] == "CALL")]
    # remove lines with test_statement that has more than one "output" or "input"
    overall_df = overall_df[~(
        (overall_df['test_statement'].str.lower().str.contains(r"(output.*){2,}|(input.*){2,}", case=False, na=False)) &
        (overall_df['type'] == "STDIO")
        )]
    
    overall_df["counter"] = overall_df.groupby(["generated_by", "task_idx"]).cumcount()
    overall_df['test_idx'] = overall_df.apply(lambda row: f"Python--LiveCodeBench--TaskID::{row['task_idx']}--GeneratedBy::{row['generated_by']}--TestID::{row['counter']:02d}", axis=1)

    overall_df = overall_df[[
        'task_idx',
        'generated_by',
        'prompt',
        'test_statement_raw',
        'test_statement',
        'test_idx', 
        'type',
    ]]

    # extract test input and output
    print("Extracting test input and output...")
    overall_df["test_input"] = overall_df.progress_apply(extract_input, axis=1)
    overall_df["test_output"] = overall_df.progress_apply(extract_output, axis=1)
    
    # CLEANING
    
    # CALL cleaning
    overall_df = overall_df.loc[~((overall_df['type'] == "CALL") & (overall_df['test_statement'].isin(["", "assert"])))]
    overall_df = overall_df.loc[~((overall_df['type'] == "CALL") & (overall_df['test_statement'].str.startswith(('assert True', 'assert False'))))]
    overall_df = overall_df.loc[(overall_df['test_statement'].str.contains('==')) | (overall_df['type'] == "STDIO")]
    overall_df = overall_df.loc[~((overall_df['type'] == "CALL") & (overall_df['test_statement'].str.contains(' and ')))]
    
    # split test input arguments onto separate lines (do this only when type is CALL)
    overall_df["test_input"] = overall_df.apply(split_function_args_with_newline, axis=1)
    # postprocess test output
    overall_df["test_output"] = overall_df["test_output"].apply(clean_from_stange_formats)

    # STDIO cleaning
    overall_df = overall_df.loc[~overall_df["test_input"].isin(["-", ""])]
    overall_df = overall_df.loc[~overall_df["test_output"].isin(["-", ""])]
    overall_df = overall_df.loc[~overall_df["test_input"].str.contains(r"plain|output|`", case=False, na=False)]
    overall_df = overall_df.loc[~overall_df["test_output"].str.contains(r"plain|input|`", case=False, na=False)]
    overall_df = overall_df.loc[~overall_df["test_input"].str.contains(r"expected", case=False, na=False)]
    overall_df = overall_df.loc[~overall_df["test_output"].str.contains(r"explanation|sample|final answer|calculate|here", case=False, na=False)]
    print(overall_df.shape)
    
    # no more than XX unique asserts per task ID per model
    overall_df['task_idx_genby'] = overall_df['test_idx'].apply(lambda x: x.split('--TestID::')[0])
    overall_df['counter'] = overall_df.groupby('task_idx_genby').cumcount()
    overall_df = overall_df[overall_df['counter'] < 10].drop(columns=['task_idx_genby', 'counter'])
    print(overall_df.shape)
    # END CLEANING

    
    g = overall_df.groupby(['generated_by', 'task_idx'], as_index=False).count()

    for generated_by in g['generated_by'].unique():
        print(f"\n################## {generated_by} ##################")
        print(f"Max number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g[g['generated_by'] == generated_by]['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {num_task_ids}):\t{num_task_ids - g[g['generated_by'] == generated_by]['task_idx'].nunique()}")
        print()
    
        save_dir = "../knowlbase-tests"
        authorized = input(f"Do you want to save the tests extracted for {generated_by} to '{save_dir}'? (y/n): ")
        if authorized.lower() in ['y', 'yes']:
            save_path = os.path.join(save_dir, f"{generated_by}_knowlbase_tests_livecodebench.jsonl")
            genby = overall_df[overall_df['generated_by'] == generated_by]
            genby.to_json(save_path, orient='records', lines=True)
            print(f"\nSaved overall extracted tests to '{save_path}'")

            genby.sample(min(genby.shape[0], 1000)).to_csv(os.path.join(save_dir, f"{generated_by}_sample.csv"), index=False)

        else:
            print("Skipped.")