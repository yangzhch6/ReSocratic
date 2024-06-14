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

def run_code(code, timelimit=20):
    exec_code = code
    
    start_time = time.time()
    try:
        output = subprocess.check_output(['python', '-c', exec_code], stderr=subprocess.STDOUT, timeout=timelimit)
    except subprocess.TimeoutExpired:
        return False, "The code runs over time limit."
    except:
        return False, "An error in the code."
    else:
        end_time = time.time()
        if end_time - start_time > timelimit:
            return True, "The code runs over time limit."
        else:
            return True, output.decode('utf-8')
        

def construct_zero_shot_dialog(question):
    dialog = [
        {"role": "system", "content": "Please use python code to solve the given question."},
        {"role": "user", "content": question},
    ]
    return dialog


def construct_few_shot_dialog(prompt, question):
    dialog = [
        {"role": "system", "content": "Please follow the given examples and use python code to solve the given question. (Your code should print the results of the results of variables and objective function in the end.)"},
        {"role": "user", "content": prompt + "\n```question\n{}\n```".format(question)},
    ]
    return dialog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4") # deepseek-v2 chat

    parser.add_argument("--data_path", type=str, default="data/e-opt.json")
    parser.add_argument("--output_path", type=str, default="eval_results/{}_fewshot.json")
    parser.add_argument("--prompt_path", type=str, default="prompt/solve/scip_fewshot.txt")

    args = parser.parse_args()
    args.output_path = args.output_path.format(args.model_name) 

    print("### model_name:", args.model_name)
    print("### output_path:", args.output_path)
    print("### prompt_path:", args.prompt_path)

    # load prompt
    with open(args.prompt_path, "r") as file:
        prompt = file.read()

    # load eval data
    eval_data = load_json(args.data_path)

    acc_num = 0
    inference_results = []

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
        zero_shot_dialog = construct_zero_shot_dialog(line["question"])

        llm_response_code = make_chat_request_hkust(
            args.model_name, few_shot_dialog, n=1, temperature=0, sleep=1, max_tokens=4000
        )[0]
        
        llm_match_response_code = match_response_code(llm_response_code)
        code_correct, code_output = run_code(llm_match_response_code)
        code_match_output = match_scip_code_output(code_output)

        # print in yellow
        print(line["question"])
        print("-"*40)
        print(Fore.BLUE + llm_match_response_code + Style.RESET_ALL)
        print("-"*40)
        print(Fore.YELLOW + code_match_output + Style.RESET_ALL)
        print("-"*40)

        if not code_correct or "The problem could not be solved to optimality" in code_output:
            line_correct = False
            llm_results_dict = {}
        else:
            zero_shot_dialog.append({"role": "assistant", "content": "```python\n{}\n```".format(llm_match_response_code) + "\n\n\n```code output\n{}\n```".format(code_match_output)})

            numercial_query = """Accoding to the code output, please give your final answer for the following query. (The answer should be boxed in '\\boxed{}', and only in numerical form, and round it to 5 decimal places, such as '\\boxed{27.00000}', '\\boxed{3.20000}', and '\\boxed{0.23334}')."""

            llm_results_dict = {}
            for line_query in line["results"]:
                numercial_query_k = numercial_query + "\n* " + line_query + ":"
                zero_shot_dialog_numercial = copy.deepcopy(zero_shot_dialog)
                zero_shot_dialog_numercial.append({"role": "user", "content": numercial_query_k})

                numercial_response = make_chat_request_hkust(
                    args.model_name, zero_shot_dialog_numercial, n=1, temperature=0, sleep=1, max_tokens=1000
                )[0]
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
            "llm_code": llm_match_response_code,
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