# ./lcb_runner/prompts/code_generation.py
import json

try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT = None
    AI_PROMPT = None

from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."

    SYSTEM_MESSAGE_GEMINITHINK = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    SYSTEM_MESSAGE_QWEN_QWQ = f"<|im_start|>system\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n<|im_start|>user"

    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

    TASK_ID_METHOD_SIGNATURE = json.load(open("../../constants/taskid_methodsignature.json"))
    ALL_IDS = json.load(open("../../../../constants/ids_train_val_test.json"))

def get_generic_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_oaireason_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: Implement a function called `main()` which orchastrates the solution by reading inputs from stdin and writing the answer to stdout. Feel free to use additional functions as necessary. Next do NOT forget to call `main` function at the end of the program otherwise you will not be awarded any points.\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_geminithinking_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        )
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Response:\n\n"
    return prompt


def get_qwen_question_template_answer(question: CodeGenerationProblem):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/abacus/models/Qwen1.5-72B-Chat/", padding_side="left", use_fast=False
    )
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"

    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt


def get_codeqwen_question_template_answer(question: CodeGenerationProblem):
    # import ALL_IDS and differentiate the prompt based on the type of the problem (CALL vs STDIO)
    call_ids = PromptConstants.ALL_IDS["LiveCodeBench"]["test_call"]
    stdio_ids = PromptConstants.ALL_IDS["LiveCodeBench"]["test_stdio"]
    if question.question_id in stdio_ids:
        # prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
        # prompt = "You will be given a question (problem specification). Provide up to 10 assert statements in Python aimed at testing the correctness of the coding problem. The assert statements should be based on the problem specification provided below and should cover various edge cases. You will NOT return anything except for the assert statements.\n\n"
        # prompt = "You will be given a question (problem specification). Assume that a function named \"solve\" provides a candidate solution to the problem. Provide up to 10 assert statements in Python aimed at testing the correctness of the candidate solution. The assert statements should be based on the problem specification provided below and should cover various edge cases. You will NOT return anything except for the assert statements.\n\n"
        prompt = "You will be given a question (problem specification). Provide up to 10 input/output test pairs aimed at testing the correctness of the Python coding problem. Remember: the solution program reads input from stdin and writes output to stdout. For each test case, provide:\n* the exact raw input string passed to stdin;\n* the exact expected raw output string from stdout.\nFor each pair EXPLICITLY specify what is the input and what is the output.\nYou will NOT return anything except for the test input/outputs pairs.\n\n"
        prompt += f"Problem specification:\n{question.question_content}\n\n"
        prompt += f"```python\n# YOUR TEST CASES HERE\n```\n\n"
    
    elif question.question_id in call_ids:
        prompt = "You will be given a question (problem specification). Provide up to 10 assert statements in Python aimed at testing the correctness of the coding problem. The assert statements should be based on the problem specification provided below and should cover various edge cases. You will NOT return anything except for the assert statements.\n\n"
        prompt += f"Signature: <SIGNATURE>\n\nProblem specification:\n{question.question_content}\n\n"
    else:
        raise ValueError(f"Question ID {question.question_id} not found in either call_ids or stdio_ids")
    
    prompt += f"<|im_start|>assistant\n"
    return prompt


def get_qwen_qwq_question_template_answer(question: CodeGenerationProblem):
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt


def get_deepseek_r1_question_template_answer(question: CodeGenerationProblem):
    # Following modifications from: https://github.com/fanqiwan/FuseAI/blob/main/FuseO1-Preview/code_evaluation/lcb_runner_cq/prompts/code_generation.py
    prompt = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"<｜Assistant｜>"
    return prompt


with open("lcb_runner/prompts/few_shot_examples/generation/func.json") as f:
    func = json.load(f)

with open("lcb_runner/prompts/few_shot_examples/generation/stdin.json") as f:
    stdin = json.load(f)


def get_base_model_question_template_answer(question: CodeGenerationProblem):
    # import ALL_IDS and differentiate the prompt based on the type of the problem (CALL vs STDIO)
    call_ids = PromptConstants.ALL_IDS["LiveCodeBench"]["test_call"]
    stdio_ids = PromptConstants.ALL_IDS["LiveCodeBench"]["test_stdio"]
    if question.question_id in stdio_ids:
        prompt = "You will be given a question (problem specification). Provide up to 10 input/output test pairs aimed at testing the correctness of the Python coding problem. Remember: the solution program reads input from stdin and writes output to stdout. For each test case, provide:\n* the exact raw input string passed to stdin;\n* the exact expected raw output string from stdout.\nFor each pair EXPLICITLY specify what is the input and what is the output.\nYou will NOT return anything except for the test input/outputs pairs.\n\n"
        prompt += "### Problem specification\n"
        prompt += question.question_content
        prompt += "\n\n"
        prompt += "### YOUR TEST CASES HERE\n\n"
    
    elif question.question_id in call_ids:
        prompt = "You will be given a question (problem specification). Provide up to 10 assert statements in Python aimed at testing the correctness of the coding problem. The assert statements should be based on the problem specification provided below and should cover various edge cases. You will NOT return anything except for the assert statements.\n\n"
        prompt += "### Signature\n<SIGNATURE>\n\n### Problem specification\n"
        prompt += question.question_content
        prompt += "\n\n"
        prompt += "### YOUR ASSERTS HERE\n\n"
    else:
        raise ValueError(f"Question ID {question.question_id} not found in either call_ids or stdio_ids")
    
    return prompt


def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle
) -> str:
    if LanguageModelStyle in [
        LMStyle.OpenAIChat,
        LMStyle.DeepSeekAPI,
        LMStyle.TogetherAI,
        LMStyle.CohereCommand,
    ]:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        return chat_messages
    elif LanguageModelStyle in [LMStyle.OpenAIReasonPreview, LMStyle.Grok]:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC
                + "\n\n"
                + get_generic_question_template_answer(question),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.OpenAIReason:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC
                + "\n\n"
                + get_oaireason_question_template_answer(question),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )

    if LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n"
        prompt += f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n"
        prompt += f"{get_generic_question_template_answer(question).rstrip()}\n"
        prompt += f"{AI_PROMPT}"
        return prompt

    if LanguageModelStyle in [LMStyle.Claude3, LMStyle.Claude3Thinking]:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question).rstrip(),
            }
        ]
        return system, prompt

    if LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GEMINI}\n"
        prompt += f"{get_generic_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.GeminiThinking:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GEMINITHINK}\n"
        prompt += f"{get_geminithinking_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
            {
                "role": "user",
                "content": get_generic_question_template_answer(question),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n"
        prompt += f"{get_deepseekcode_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question)}"
        prompt = prompt.replace("<SIGNATURE>", PromptConstants.TASK_ID_METHOD_SIGNATURE[question.question_id])
        return prompt

    if LanguageModelStyle == LMStyle.QwQ:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_QWEN_QWQ}\n\n"
        prompt += f"{get_qwen_qwq_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}"
        prompt += f"{get_deepseek_r1_question_template_answer(question)}"
        return prompt

    if LanguageModelStyle == LMStyle.GenericBase:
        prompt = get_base_model_question_template_answer(question)
        prompt = prompt.replace("<SIGNATURE>", PromptConstants.TASK_ID_METHOD_SIGNATURE[question.question_id])
        return prompt

    raise NotImplementedError(
        f"LanguageModelStyle {LanguageModelStyle} not implemented"
    )


def test():
    import pathlib

    base_dir = "logs/example_prompts/generation"
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    for lmstyle in LMStyle:
        generation_problem = CodeGenerationProblem(
            "title",
            "question-content",
            "leetcode",
            "question_id",
            "contest_id",
            "contest_date",
            "",
            "easy",
            "[]",
            "[]",
            "{}",
        )
        prompt1 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_1.txt", "w") as f:
            try:
                f.write(prompt1)
            except TypeError:
                f.write(json.dumps(prompt1))

        generation_problem.starter_code = "starter code"
        prompt2 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_2.txt", "w") as f:
            try:
                f.write(prompt2)
            except TypeError:
                f.write(json.dumps(prompt2))


if __name__ == "__main__":
    test()
