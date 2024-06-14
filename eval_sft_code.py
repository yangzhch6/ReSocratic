import os
import json
import argparse
from io import StringIO
from contextlib import redirect_stdout
from colorama import Fore, Back, Style
from vllm import LLM, SamplingParams
from utils import *
import numpy as np
import copy
import subprocess
import time
import io


def run_code(code, timelimit=3):
    # 将输入的代码包装在一个函数中
    exec_code = code
    
    # 使用subprocess运行代码
    start_time = time.time()
    try:
        output = subprocess.check_output(['python', '-c', exec_code], stderr=subprocess.STDOUT, timeout=timelimit)
    except subprocess.TimeoutExpired:
        # 如果代码执行超过3秒
        return False, "The code runs over time limit."
    except:
        # 如果代码出错
        return False, "An error in the code."
    else:
        # 如果代码正常执行
        end_time = time.time()
        if end_time - start_time > timelimit:
            return True, "The code runs over time limit."
        else:
            return True, output.decode('utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/hpc2hdd/JH_DATA/share/xingjiantao/xingjiantao_llama3/Llama-3-8B-Instruct-hf/")
    parser.add_argument("--data_path", type=str, default="data/test_v1.json")
    parser.add_argument("--gencode_path", type=str, default="eval_results/optcode_llama3_8b_instruct_sft.json")
    parser.add_argument("--output_path", type=str, default="eval_results/llama3_8b_instruct_sft.json")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # load data
    eval_data = load_json(args.data_path)
    print(len(eval_data))

    gencode_data = load_json(args.gencode_path)
    print(len(gencode_data))

    assert len(eval_data) == len(gencode_data)

    # load model
    sampling_params = SamplingParams(temperature=0, max_tokens=3000, stop=["<|eot_id|>"])
    resulting_params = SamplingParams(temperature=0, max_tokens=3000, stop=["* ", "<|eot_id|>"])
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)

    # set prompt template
    DIALOG_PROMPT = [
        {"role": "system", "content": "Please use python code with pyscipopt to solve the given optimization question."}
    ]

    inference_results = []
    acc_num = 0
    numercial_content = """Accoding to the code output, please give your final answer for the following query. (The answer should be boxed in '\\boxed{}', and only in numerical form, and round it to 5 decimal places, such as '\\boxed{27.00000}', '\\boxed{3.20000}', and '\\boxed{0.23334}')."""

    for eval_line, gencode_line in zip(eval_data, gencode_data):
        
        if "```python\n" in gencode_line["llm_code"]:
            llm_match_code = match_response_code(gencode_line["llm_code"])
        else:
            llm_match_code = match_response_code("```python\n" + gencode_line["llm_code"])
        code_correct, llm_code_output = run_code(llm_match_code)

        numercial_line_content = "```code_output\n{}\n```\n\n\n".format(llm_code_output) + numercial_content

        source_dialog = DIALOG_PROMPT + [{"role": "user", "content": eval_line["question"]}, {"role": "assistant", "content": gencode_line["llm_code"]}, {"role": "user", "content": numercial_line_content}]

        numercial_query = dialog_to_text_llama3(source_dialog)
    
        if not code_correct or "The problem could not be solved to optimality" in llm_code_output:
            line_correct = False
            llm_results_dict = {}
        else:
            llm_results_dict = copy.deepcopy(eval_line["results"])
            for key in llm_results_dict:
                llm_results_dict[key] = ""

            for key in llm_results_dict:
                numercial_query_result = numercial_query + "* {}:".format(key)
                llm_results_dict[key] = llm.generate([numercial_query_result], resulting_params)[0].outputs[0].text.strip()
                llm_results_dict[key] = match_numercial_value(llm_results_dict[key])

            line_correct = True
            for key in llm_results_dict:
                gt_ans = eval(eval_line["results"][key])
                try:
                    llm_ans = eval(llm_results_dict[key])
                    if abs(llm_ans - gt_ans) < 1e-4: 
                        continue
                    else:
                        line_correct = False
                        break
                except:
                    line_correct = False
                    break

        if line_correct:
            acc_num += 1

        inference_results.append({
            "index": eval_line["index"],
            "question": eval_line["question"],
            "code": llm_match_code,
            "code_output": llm_code_output,
            "gt_results_dict": eval_line["results"],
            "llm_results_dict": llm_results_dict,
            "correct": line_correct,
            "type": eval_line["type"],
            "code_correct": code_correct
        })

        print("### current acc:", acc_num, len(inference_results), acc_num/len(inference_results))
        with open(args.output_path, 'w') as outfile:
            json.dump(inference_results, outfile, indent=4)

    print("### final acc:", acc_num, "/", len(inference_results), acc_num/len(inference_results))
    with open(args.output_path, 'w') as outfile:
        json.dump(inference_results, outfile, indent=4)
