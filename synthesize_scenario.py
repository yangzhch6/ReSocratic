import re
from utils import *
import argparse
from colorama import Fore, Back, Style
import random 
from functools import partial
from tqdm import tqdm
import multiprocessing
from itertools import combinations

prefix = [
    "",
    "## Define Variables:",
    "## Define Objective Function:",
    "## Generate Constraint-1:",
    "## Generate Constraint-2:",
    "## Generate Constraint-3:",
    "## Generate Constraint-4:",
    "## Generate Constraint-5:",
]

def format_chat_history(instruction, scenario, examples):
    # print(scenario.format(examples, seed))
    # print("=============================")
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": scenario.format(examples)},
    ]

def construct_examples(samples):
    # sample 2 questions from the pool
    # random choice 1 or 2
    # samples_num = random.choice([1, 2])
    # samples = random.sample(pool, samples_num)
    examples = "\n\n\n".join(["[Scenario Example]:\n" + sample["scenario"] for sample in samples])
    # print(examples)
    return examples


def check_response(response, level):
    # check if the response contains 3 parts, the first line is the prefix, the second part is the natural language, the third part is the formulation start with "//"

    response_lines = response.strip().split("\n")

    # PART 1: check if line[0]'s prefix is equal to the prefix
    if prefix[level] not in response_lines[0]:
        # print(Fore.RED + "The prefix is not correct.")
        return False
    
    # PART 2: check if the response contains natural language description
    nl_description = ""
    for i in range(1, len(response_lines)):
        if "//" != response_lines[i][0:2]:
            nl_description += response_lines[i].strip()
            break
        else:
            break
    if nl_description.strip() == "":
        # print(Fore.RED + "The response does not contain natural language description.")
        return False

    # PART 3: check if the formulation start with "//"
    index = 2
    while index < len(response_lines):
        if "//" != response_lines[index][0:2]:
            index += 1
        else:
            break
    if index == len(response_lines):
        # print(Fore.RED + "The formulation does not start with '//'.")
        return False
    
    for i in range(index, len(response_lines)):
        if "//" != response_lines[i][0:2]:
            # print(Fore.RED + "The formulation does not start with '//'.")
            return False

    return True


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


def get_formulation(scenario):
    # 使用正则表达式查找所有以"//"开头的行
    comment_lines = re.findall(r'//.*', scenario)

    formulation = []

    # 打印结果
    for line in comment_lines:
        formulation.append(line[2:].strip())

    return "\n".join(formulation)


def clean_response(response):
    ## if [New Scenario]: in response, split the response
    ## count "[New Scenario]:" in response
    new_scenario_count = response.count("[New Scenario]:")
    if new_scenario_count > 1:
        return ""
    if new_scenario_count == 1:
        response = response.split("[New Scenario]:")[1].strip()

    response = response.strip()
    response_units = response.split("## ")
    response_units = [unit.strip() for unit in response_units]
    response_units = ["## " + unit for unit in response_units if unit != ""]
    response_units[-1] = response_units[-1].split("\n\n")[0]

    level = 1
    valid_response_units = []
    full_valid = True
    for unit in response_units:
        if level > 7: # max_level = 7
            break
        is_valid = check_response(unit, level)
        if not is_valid: 
            print(level)
            print(response)
            print("-------------------------------------------------------")
            full_valid = False
            break
        else:
            valid_response_units.append(unit)
        
        level += 1

    return "\n\n".join(valid_response_units), full_valid
    

def split_response(response):
    response = response.strip()
    response_units = response.split("## ")
    response_units = [unit.strip() for unit in response_units]
    response_units = ["## " + unit for unit in response_units if unit != ""]

    split_responses = [] 
    for i in range(0, len(response_units)):
        split_responses.append("\n\n".join(response_units[:i+1]))

    return split_responses
    

def constraints_filter(data_list, threshold=0.7):
    data_dict = {}
    for line in data_list:
        variables_function = line.split("## Generate Constraint-1:\n")[0].strip()  + "\n\n"
        if len(line.split("## Generate Constraint-1:\n")) == 1:
            print(line)
            print("-------------------------------------------------------")
            continue

        if variables_function not in data_dict:
            data_dict[variables_function] = [line]
        else:
            data_dict[variables_function].append(line)
    
    filtered_data = []
    for key in data_dict:
        # print(len(data_dict[key]))
        if len(data_dict[key]) > 1:
            constraints_list = ["## Generate Constraint-1:\n" + line.split("## Generate Constraint-1:\n")[1].strip() for line in data_dict[key]]
            constraints_list = remove_duplicates(constraints_list, threshold=threshold)
            data_dict[key] = [key + line for line in constraints_list]
        filtered_data += data_dict[key]
    
    return filtered_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-chat") # deepseek-v2 chat

    parser.add_argument("--pool_path", type=str, default="prompt/synthesis/pool/linear.json")
    parser.add_argument("--instruction_path", type=str, default="prompt/synthesis/instruction_{}.txt")
    parser.add_argument("--scenario_path", type=str, default="prompt/synthesis/scenario_{}.txt")

    parser.add_argument("--synthesis_response_path", type=str, default="synthesis_data/response_{}.json")
    parser.add_argument("--synthesis_data_path", type=str, default="synthesis_data/data_{}.json")
    parser.add_argument("--synthesis_scenario_path", type=str, default="synthesis_data/scenarios_{}.json")

    parser.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()

    # fill the path
    pool_name = args.pool_path.split("/")[-1].split(".")[0]
    non_linear = "non_linear" in pool_name
    args.instruction_path = args.instruction_path.format(pool_name)
    args.scenario_path = args.scenario_path.format("non_linear" if non_linear else "linear")
    args.synthesis_response_path = args.synthesis_response_path.format(pool_name)
    args.synthesis_data_path = args.synthesis_data_path.format(pool_name)
    args.synthesis_scenario_path = args.synthesis_scenario_path.format(pool_name)

    print("## pool_name: ", pool_name)
    print("## non_linear: ", non_linear)
    print("## instruction_path: ", args.instruction_path)
    print("## scenario_path: ", args.scenario_path)

    # load pool
    pool = load_json(args.pool_path)

    # load instruction
    with open(args.instruction_path, "r", encoding="utf-8") as f:
        instruction = f.read()

    # load scenario
    with open(args.scenario_path, "r", encoding="utf-8") as f:
        scenario = f.read()
    

    synthesis_scenarios = {
        "level-1" : [],
        "level-2" : [],
        "level-3" : [],
        "level-4" : [],
        "level-5" : [],
        "level-6" : [],
        "level-7" : []
    }

    start_sample_index = 0

    print("## process num", multiprocessing.cpu_count())


    ## Step 1: Generate synthesis data
    responses_list = []
    start_sample_index = 0
    # if exists, load the response data
    if os.path.exists(args.synthesis_response_path):
        with open(args.synthesis_response_path, "r", encoding="utf-8") as f:
            responses_list = json.load(f)

        print("Checkpoint Length:", len(responses_list))

        start_sample_index = len(responses_list) // 100
        if start_sample_index > len(pool):
            start_sample_index = (len(responses_list) - len(pool)*100) // 50
        
        print("## start_sample_index: ", start_sample_index)

    # Generate synthesis scenarios
    samples_list = list(combinations(pool, 1)) + list(combinations(pool, 2))
    for index in range(start_sample_index, len(samples_list)):
        samples = samples_list[index]
        samples = list(samples)
        if len(samples) == 1:
            sample_response_times = 100
        else:
            sample_response_times = 50
        scenario_examples = construct_examples(samples)
        model_input = format_chat_history(instruction, scenario, scenario_examples)

        responses = make_chat_request_deepseek(
            args.model_name, model_input, sample_response_times, temperature=0.7, sleep=1, parallel=True
        )
        responses_list += responses

        with open(args.synthesis_response_path, "w", encoding="utf-8") as f:
            json.dump(responses_list, f, indent=4)


    ## Step 2: Filter the synthesis data
    # if exists, load the filtered response data
    if os.path.exists(args.synthesis_scenario_path):
        with open(args.synthesis_scenario_path, "r", encoding="utf-8") as f:
            synthesis_scenarios = json.load(f)
        print("## synthesis_scenarios length: ", len(synthesis_scenarios["level-1"]))
    else:
        # clean & filter the responses
        valid_responses = []
        full_valid_count = 0
        for response in responses_list:
            cleaned_response, response_full_valid = clean_response(response)
            if cleaned_response != "":
                valid_responses.append(cleaned_response)
            if response_full_valid:
                full_valid_count += 1
        print("Full_valid_proportion: ", full_valid_count / len(responses_list))

        ## only filt the variables & objective function
        v_o_responses = [line.split("## Generate Constraint-1:\n")[0].strip() for line in valid_responses]
        filtered_v_o_responses = remove_duplicates(v_o_responses, threshold=args.threshold)

        filtered_responses = []
        for response in valid_responses:
            if response.split("## Generate Constraint-1:\n")[0].strip() in filtered_v_o_responses:
                filtered_responses.append(response)

        # ## filter the all responses
        # filtered_responses = constraints_filter(valid_responses, threshold=args.threshold)
        split_responses = []
        for line in filtered_responses:
            split_responses += split_response(line)

        split_responses = [line.strip() for line in split_responses]

        for s_response in split_responses:
            ## count "##" in response
            response_level = s_response.count("##")
            synthesis_scenarios["level-{}".format(response_level)].append(s_response)

        # save scenarios in json file
        with open(args.synthesis_scenario_path, "w", encoding="utf-8") as f:
            json.dump(synthesis_scenarios, f, indent=4)

    
    ## Step 3: Extract synthesis data
    # Extract synthesis data
    synthesis_data = []
    for level in range(1, 8):
        if "geo" in args.pool_path:
            q_level = 2
        else:
            q_level = 3
        if level >= q_level:
            for response in synthesis_scenarios["level-{}".format(level)]:
                question = get_question(response)
                formulation = get_formulation(response)
                synthesis_data.append({
                    "level": level,
                    "question": question,
                    "formulation": formulation,
                    "nl_solution": response,
                })

    print("## synthesis_data length: ", len(synthesis_data))
    # save synthesis_data in json file
    with open(args.synthesis_data_path, "w", encoding="utf-8") as f:
        json.dump(synthesis_data, f, indent=4)
