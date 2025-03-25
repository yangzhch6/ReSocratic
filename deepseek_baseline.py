from utils import *
from colorama import Fore, Back, Style
import sys
import argparse
from contextlib import redirect_stdout
from io import StringIO
import multiprocessing
import json
from functools import partial
import copy
import subprocess
import time
import io

def construct_zero_shot_dialog(prompt, question):
    dialog = [
        {"role": "system", "content": "Please use python code to solve the given question."},
        {"role": "user", "content": prompt + "\n```question\n{}\n```".format(question)},
    ]
    return dialog

def construct_few_shot_dialog(prompt, question):
    dialog = [
        {"role": "system", "content": "Please follow the given examples and use python code to solve the given question."},
        {"role": "user", "content": prompt + "\n```question\n{}\n```".format(question)},
    ]
    return dialog

def match_output(response):
    response = response.strip().split("\n\n")[-1]
    return response


def convert_results_dict(results):
    results = results.strip().split("\n")
    results_dict = {}
    for line in results:
        key = line.split(":")[0].strip()
        value = line.split(":")[-1].strip()
        results_dict[key] = value
    return results_dict


def run_code(code, timelimit=20):
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
    parser.add_argument("--model_name", type=str, default="deepseek-chat") # deepseek-v2 chat

    parser.add_argument("--data_path", type=str, default="/data1/yangzhch6/projs/ReSocratic/data/test_v1.json")
    parser.add_argument("--output_path", type=str, default="eval_results/deepseek-v2-chat_fewshot2.json")
    parser.add_argument("--prompt_path", type=str, default="/data1/yangzhch6/projs/ReSocratic/prompt/solve/scip_fewshot2.txt")

    # parser.add_argument("--chunk", type=int, default=64)

    args = parser.parse_args()

    # load prompt
    with open(args.prompt_path, "r") as file:
        prompt = file.read()
    
    # laod data
    eval_data = load_json(args.data_path)

    inference_results = []
    acc_num = 0

    # if exists, load previous results
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as infile:
            inference_results = json.load(infile)
        for line in inference_results:
            if line["correct"]:
                acc_num += 1

    print("## start:", len(inference_results), "/", len(eval_data))
    for index in range(len(inference_results), len(eval_data)):
        line = eval_data[index]

        few_shot_dialog = construct_few_shot_dialog(prompt, line["question"])
        responses = make_chat_request_deepseek(args.model_name, few_shot_dialog, 1, temperature=0, sleep=1, parallel=False, max_tokens=2048)

        response_code = match_response_code(responses[0])

        code_correct, code_output = run_code(response_code)

        code_match_output = match_scip_code_output(code_output)

        # print in yellow
        print(line["question"])
        print("-"*40)
        print(Fore.BLUE + response_code + Style.RESET_ALL)
        print("-"*40)
        print(Fore.YELLOW + code_match_output + Style.RESET_ALL)
        print("-"*40)

        if not code_correct or "The problem could not be solved to optimality" in code_output:
            line_correct = False
            llm_results_dict = {}
        else:
            few_shot_dialog.append({"role": "assistant", "content": "```python\n{}\n```".format(response_code) + "\n\n\n```code output\n{}\n```".format(code_match_output)})

            numercial_query = """Accoding to the code output, please give your final answer for the following query. (The answer should be boxed in '\\boxed{}', and only in numerical form, and round it to 5 decimal places, such as '\\boxed{27.00000}', '\\boxed{3.20000}', and '\\boxed{0.23334}')."""

            llm_results_dict = {}
            for line_query in line["results"]:
                numercial_query_k = numercial_query + "\n* " + line_query + ":"
                few_shot_dialog_numercial = copy.deepcopy(few_shot_dialog)
                few_shot_dialog_numercial.append({"role": "user", "content": numercial_query_k})

                numercial_responses = make_chat_request_deepseek(args.model_name, few_shot_dialog_numercial, 1, temperature=0, sleep=1, parallel=False, max_tokens=512)
                numercial_response = numercial_responses[0]
                numercial_response_answer = match_numercial_value(numercial_response)
                print(Fore.GREEN + line_query + ": " + numercial_response + Style.RESET_ALL)

                llm_results_dict[line_query] = numercial_response_answer
            
            line_correct = True
            for key in llm_results_dict:
                gt_ans = eval(line["results"][key])
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
        
        # add to inference results
        inference_results.append({
            "index": line["index"],
            "question": line["question"],
            "llm_code": response_code,
            "llm_code_output": code_match_output,
            "gt_results_dict": line["results"],
            "llm_results_dict": llm_results_dict,
            "correct": line_correct,
            "type": line["type"],
            "code_correct": code_correct
        })

        print("-"*40)
        print("## line_correct:", line_correct)
        print("### current acc:", acc_num, len(inference_results), acc_num/len(inference_results))
        print("="*60)

        with open(args.output_path, 'w') as outfile:
            json.dump(inference_results, outfile, indent=4)

    print("### final acc:", acc_num, "/", len(inference_results), acc_num/len(inference_results))
    with open(args.output_path, 'w') as outfile:
        json.dump(inference_results, outfile, indent=4)