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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/hpc2hdd/JH_DATA/share/xingjiantao/xingjiantao_llama3/Llama-3-8B-Instruct-hf/")
    parser.add_argument("--data_path", type=str, default="data/test_v1.json")
    parser.add_argument("--output_path", type=str, default="eval_results/optcode_llama3_8b_instruct_sft.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # load data
    eval_data = load_json(args.data_path)
    print(len(eval_data))

    # load model
    sampling_params = SamplingParams(temperature=0, max_tokens=3000, stop=["<|eot_id|>"])
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)

    # set prompt template
    DIALOG_PROMPT = [
        {"role": "system", "content": "Please use python code with pyscipopt to solve the given optimization question."}
    ]

    inference_results = []
    
    for i in range(0, len(eval_data), args.batch_size):
        print("## index:", i)

        batch = eval_data[i:i+args.batch_size]

        source_dialogs = []
        for example in batch:
            source_dialogs.append(DIALOG_PROMPT + [{"role": "user", "content": example["question"]}])
        
        query_code_batch = [dialog_to_text_llama3(dialog) for dialog in source_dialogs]
        llm_code_batch = llm.generate(query_code_batch, sampling_params)
        llm_code_batch = [line.outputs[0].text.strip() for line in llm_code_batch]

        for eval_line, llm_code in zip(batch, llm_code_batch):
            inference_results.append({
                "index": eval_line["index"],
                "question": eval_line["question"],
                "llm_code": llm_code,
                "type": eval_line["type"]
            })

        with open(args.output_path, 'w') as outfile:
            json.dump(inference_results, outfile, indent=4)