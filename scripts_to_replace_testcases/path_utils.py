# ./lcb_runner/utils/path_utils.py
import pathlib
import os

from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    model_name = model_repr.split("--")[-1]
    path = f"test-generations/{model_repr}/{model_name}_{n}_{temperature}-{os.getpid()}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"
    return path
