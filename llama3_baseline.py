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


def run_code(code, timelimit=20):
    # 将输入的代码包装在一个函数中
    exec_code = code
    
    # 使用subprocess运行代码
    start_time = time.time()
    try:
        output = subprocess.check_output(['python', '-c', exec_code], stderr=subprocess.STDOUT, timeout=timelimit)
    except subprocess.TimeoutExpired:
        # 如果代码执行超过3秒
        return True, "The code runs over time limit."
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
    parser.add_argument("--prompt_path", type=str, default="prompt/solve/scip_fewshot.txt")
    parser.add_argument("--output_path", type=str, default="eval_results/llama3_8b_instruct_fewshot.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # load data
    eval_data = load_json(args.data_path)

    # load model
    sampling_params = SamplingParams(temperature=0, max_tokens=1000, stop=["<|eot_id|>"])
    resulting_params = SamplingParams(temperature=0, max_tokens=1000, stop=["<|eot_id|>"])
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)

    # set prompt template
    DIALOG_ZERO_SHOT_PROMPT = [
        {"role": "system", "content": "Please use python code to solve the given question."},
    ]
    DIALOG_FEW_SHOT_PROMPT = [
        {"role": "system", "content": "Please follow the given examples and use python code to solve the given question."},
    ]
    with open(args.prompt_path, "r") as f:
        prompt_template = f.read()

    inference_results = []
    acc_num = 0

    for i in range(0, len(eval_data), args.batch_size):
        batch = eval_data[i:i+args.batch_size]

        source_dialogs_zero_shot = []
        for example in batch:
            source_dialogs_zero_shot.append(DIALOG_ZERO_SHOT_PROMPT + [{"role": "user", "content": prompt_template + "\n```question\n{}\n```".format(example["question"])}])

        source_dialogs = []
        for example in batch:
            source_dialogs.append(DIALOG_FEW_SHOT_PROMPT + [{"role": "user", "content": prompt_template + "\n```question\n{}\n```".format(example["question"])}])
        
        query_batch = [dialog_to_text_llama3(dialog) for dialog in source_dialogs]
        llm_code_batch = llm.generate(query_batch, sampling_params)

        # acquire code
        llm_code_batch = [match_response_code(line.outputs[0].text.strip()) for line in llm_code_batch]

        ## execute code for each line in batch
        llm_code_execute_output_batch = []
        llm_code_correct_batch = []
        for code in llm_code_batch:
            code_correct, code_output = run_code(code)
            code_match_output = match_scip_code_output(code_output)
            llm_code_execute_output_batch.append(code_match_output)
            llm_code_correct_batch.append(code_correct)

        ## update dialog with code results
        for dialog, dialog_zero_shot, code, code_output in zip(source_dialogs, source_dialogs_zero_shot, llm_code_batch, llm_code_execute_output_batch):
            dialog.append({"role": "assistant", "content": "```python\n{}\n```".format(code) + "\n\n\n```code output\n{}\n```".format(code_output)})
            dialog_zero_shot.append({"role": "assistant", "content": "```python\n{}\n```".format(code) + "\n\n\n```code output\n{}\n```".format(code_output)})

        ## prompt llm to generate the final answer
        numercial_turn = {"role": "user", "content": """Accoding to the code output, please give your final answer for the following query. (The answer should be boxed in '\\boxed{}', and only in numerical form, and round it to 5 decimal places, such as '\\boxed{27.00000}', '\\boxed{3.20000}', and '\\boxed{0.23334}')."""}
        numercial_query_batch = [dialog_to_text_llama3(dialog + [numercial_turn]) for dialog in source_dialogs_zero_shot]

        llm_output_results_dict = []
        llm_output_correct = []

        for numercial_query, eval_line, code_output in zip(numercial_query_batch, batch, llm_code_execute_output_batch):
            print(Fore.YELLOW + code_output + Style.RESET_ALL)
            print("-"*40)
            
            if code_output in ["The code runs over time limit.", "An error in the code.", "The problem could not be solved to optimality."]:
                line_correct = False
                llm_results_dict = {}
                llm_output_results_dict.append(llm_results_dict)
            else:
            
                llm_results_dict = copy.deepcopy(eval_line["results"])
                for key in llm_results_dict:
                    llm_results_dict[key] = ""

                for key in llm_results_dict:
                    numercial_query_result = numercial_query + "* {}:".format(key)
                    llm_results_dict[key] = llm.generate([numercial_query_result], resulting_params)[0].outputs[0].text.strip()
                llm_output_results_dict.append(llm_results_dict)

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

            llm_output_correct.append(line_correct)
    
        for eval_line, code, code_output, llm_results_dict, line_correct, code_correct in zip(batch, llm_code_batch, llm_code_execute_output_batch, llm_output_results_dict, llm_output_correct, llm_code_correct_batch):
            inference_results.append({
                "index": eval_line["index"],
                "question": eval_line["question"],
                "code": code,
                "code_output": code_output,
                "gt_results_dict": eval_line["results"],
                "llm_results_dict": llm_results_dict,
                "correct": line_correct,
                "type": eval_line["type"],
                "code_correct": code_correct
            })
            if line_correct:
                acc_num += 1

        print("### current acc:", acc_num, len(inference_results), acc_num/len(inference_results))
        with open(args.output_path, 'w') as outfile:
            json.dump(inference_results, outfile, indent=4)

    print("### final acc:", acc_num, "/", len(inference_results), acc_num/len(inference_results))
    with open(args.output_path, 'w') as outfile:
        json.dump(inference_results, outfile, indent=4)