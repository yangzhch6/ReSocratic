
import sys
from contextlib import redirect_stdout
from io import StringIO
import json
import re
from utils import *
import argparse
from colorama import Fore, Back, Style
import random 
from functools import partial
from tqdm import tqdm
import multiprocessing


def construct_dialog(prompt, scenario):
    dialog = [
        {"role": "system", "content": "You are a mathematical assistant. Now, you will be provided with an optimization scenario. Please follow the example to convert the given scenario to question."},
        {"role": "user", "content": prompt + "\n```Scenario\n{}```\n\n\n".format(scenario)}, 
    ]
    return dialog


def get_question(scenario):
    scenario_list = scenario.split("##")
    scenario_list = [line.strip() for line in scenario_list]
    scenario_list = [line for line in scenario_list if line != ""]

    question_list = []
    for line in scenario_list:
        line_list = line.split("\n")
        line_list = [line.strip() for line in line_list]
        line_list = [line for line in line_list if line != ""]

        for i in range(1, len(line_list)):
            if "//" == line_list[i][:2]:
                break
            else:
                question_list.append(line_list[i])

    # convert the question step to the final position
    question = [question_list[0]] + question_list[2:] + [question_list[1]]

    return " ".join(question)


def match_response_question(response):
    # start with "```python" or "```"
    # end with "```"
    # match the first one
    # print(response)
    pattern = r"```question\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return response
    

def worker_deepseek_chat(dialog, model_name, temperature=0, sleep=1):
    responses = make_chat_request_deepseek(model_name, dialog, 1, temperature=temperature, sleep=sleep, parallel=False)
    return responses[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-chat") # deepseek-v2 chat

    parser.add_argument("--data_path", type=str, default="---")
    parser.add_argument("--output_path", type=str, default="---")
    parser.add_argument("--notable_prompt_path", type=str, default="prompt/synthesis/synthesize_notable_question.txt")
    parser.add_argument("--table_prompt_path", type=str, default="prompt/synthesis/synthesize_table_question.txt")

    parser.add_argument("--chunk", type=int, default=64)

    args = parser.parse_args()

    print("notable_prompt_path:", args.notable_prompt_path)
    print("table_prompt_path:", args.table_prompt_path)

    with open(args.table_prompt_path, 'r') as f:
        table_prompt = f.read()
    
    with open(args.notable_prompt_path, 'r') as f:
        notable_prompt = f.read()

    data = load_json(args.data_path)
    
    new_data = []
    # if exists, load the existing data
    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            new_data = json.load(f)
    
    print(len(new_data), "/", len(data))
    start_index = len(new_data)

    for i in range(start_index, len(data), args.chunk):
        chunk_data = data[i:i+args.chunk]

        # Determine the number of processes based on the number of available CPU cores
        num_processes = min(8, len(chunk_data)) # multiprocessing.cpu_count()

        # Create a pool of workers
        workers_pool = multiprocessing.Pool(processes=num_processes)
        
        # Use partial to bind the prompt and model_name to the worker function
        worker_partial = partial(worker_deepseek_chat, model_name=args.model_name, temperature=0, sleep=1)

        # Construct the dialog list
        table_dialog_chunk = [construct_dialog(table_prompt, line["nl_solution"]) for line in chunk_data]
        notable_dialog_chunk = [construct_dialog(notable_prompt, line["nl_solution"]) for line in chunk_data]

        # Use the pool to map the worker function to each line in the data list
        responses_table_chunk = list(tqdm(workers_pool.imap(worker_partial, table_dialog_chunk), total=len(table_dialog_chunk)))
        responses_notable_chunk = list(tqdm(workers_pool.imap(worker_partial, notable_dialog_chunk), total=len(notable_dialog_chunk)))

        # Close the pool and wait for all processes to finish
        workers_pool.close()
        workers_pool.join()

        responses_table_chunk = [match_response_question(response) for response in responses_table_chunk]
        responses_notable_chunk = [match_response_question(response) for response in responses_notable_chunk]

        for j in range(len(chunk_data)):
            new_data.append({
                "nl_solution": chunk_data[j]["nl_solution"],
                "question": responses_table_chunk[j],
            })

            new_data.append({
                "nl_solution": chunk_data[j]["nl_solution"],
                "question": responses_notable_chunk[j],
            })

            print(chunk_data[j]["nl_solution"])
            print("*"*40)
            print(Fore.GREEN + responses_table_chunk[j] + Style.RESET_ALL)
            print("-"*40)
            print(Fore.RED + responses_notable_chunk[j] + Style.RESET_ALL)
            print("="*40)

        with open(args.output_path, 'w') as f:
            json.dump(new_data, f, indent=4)
