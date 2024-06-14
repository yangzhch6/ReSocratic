
import sys
import json
import re
from utils import *
import argparse
from colorama import Fore, Back, Style
from io import StringIO
from contextlib import redirect_stdout
import subprocess
import time
import io

def construct_dialog(prompt, scenario):
    dialog = [
        {"role": "system", "content": "You are a mathematical assistant. Now, you will be provided with an optimization scenario with its corresponding question. Please follow the examples to solve the optimization scenario using python code with pyscipopt. (Tips: 1. Set objective as a variable to avoid non-linear objective. 2. To expedite computation, convert division to multiplication.)"},
        {"role": "user", "content": prompt + "\n```scenario\n{}```".format(scenario)}, 
    ]
    return dialog


def match_code(response):
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return response


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


def run_code(code):
    # 将输入的代码包装在一个函数中
    exec_code = code
    
    # 使用subprocess运行代码
    start_time = time.time()
    try:
        output = subprocess.check_output(['python', '-c', exec_code], stderr=subprocess.STDOUT, timeout=1.5)
    except subprocess.TimeoutExpired:
        # 如果代码执行超过3秒
        return True, None
    except:
        # 如果代码出错
        return False, "error in the code"
    else:
        # 如果代码正常执行
        end_time = time.time()
        if end_time - start_time > 3:
            return True, None
        else:
            return True, output.decode('utf-8')
        

def worker_deepseek_chat(dialog, model_name, temperature=0, sleep=1):
    responses = make_chat_request_deepseek(model_name, dialog, 1, temperature=temperature, sleep=sleep, parallel=False)
    return responses[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-chat") # deepseek-v2 chat

    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--prompt_path", type=str, default="prompt/synthesis/synthesize_code.txt")

    parser.add_argument("--chunk", type=int, default=64)

    args = parser.parse_args()

    with open(args.prompt_path, 'r') as f:
        prompt = f.read()

    data = load_json(args.data_path)

    new_data = []
    # if exists, load the existing data
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            new_data = json.load(f)
    
    print(len(new_data), "/", len(data))
    start_index = len(new_data)

    good_code = 0
    for i in range(start_index, len(data), args.chunk):
        chunk_data = data[i:i+args.chunk]

        assert len(chunk_data) % 2 == 0
        for i in range(0, len(chunk_data), 2):
            assert chunk_data[i]["nl_solution"] == chunk_data[i+1]["nl_solution"]
        
        chunk_data_input = chunk_data[::2]

        # Determine the number of processes based on the number of available CPU cores
        num_processes = min(8, len(chunk_data_input)) # multiprocessing.cpu_count()

        # Create a pool of workers
        workers_pool = multiprocessing.Pool(processes=num_processes)
        
        # Use partial to bind the prompt and model_name to the worker function
        worker_partial = partial(worker_deepseek_chat, model_name=args.model_name, temperature=0, sleep=1)

        # Construct the dialog list
        dialog_chunk = [construct_dialog(prompt, line["nl_solution"]) for line in chunk_data_input]

        # Use the pool to map the worker function to each line in the data list
        responses_chunk = list(tqdm(workers_pool.imap(worker_partial, dialog_chunk), total=len(dialog_chunk)))

        # Close the pool and wait for all processes to finish
        workers_pool.close()
        workers_pool.join()

        responses_chunk = [match_response_code(response) for response in responses_chunk]

        for k in range(len(responses_chunk)):
            code_correct, _ = run_code(responses_chunk[k])
            if not code_correct:
                responses_chunk[k] = None
            else:
                good_code += 2

            line1 = chunk_data[2*k]
            line2 = chunk_data[2*k+1]
            line1["code_solution"] = responses_chunk[k]
            line2["code_solution"] = responses_chunk[k]
            new_data.append(line1)
            new_data.append(line2)

        with open(args.output_path, 'w') as outfile:
            json.dump(new_data, outfile, indent=4)
        print("good_code proportion:", good_code , "/" , len(new_data))

    print("good_code proportion:", good_code / len(new_data))

    with open(args.output_path, 'w') as outfile:
        json.dump(new_data, outfile, indent=4)