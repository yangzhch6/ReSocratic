import os 
import io
import re
import time
import json
import requests
import nltk
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from functools import partial
import multiprocessing

def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'r',encoding="utf-8") as file:
        data = json.load(file)
    return data


def worker_chat(messages, model_name, max_tokens, temperature, stop):
    client = OpenAI(api_key="your key here.", base_url="https://api.deepseek.com")
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name, #"deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=False
            )
            break
        except:
            print("Error: retrying...")
            time.sleep(1)
    return response


def make_chat_request_deepseek(model_name, messages, n, max_tokens=2000, stop=None, temperature=0, sleep=1, parallel=False):
    if parallel:
        # Determine the number of processes based on the number of available CPU cores
        num_processes = min(multiprocessing.cpu_count(), n) # multiprocessing.cpu_count()
        
        # Create a pool of workers
        workers_pool = multiprocessing.Pool(processes=num_processes)
        
        # Use partial to bind the prompt and model_name to the worker function
        worker_partial = partial(worker_chat, model_name=model_name, max_tokens=max_tokens, temperature=temperature, stop=stop)
        
        # Use the pool to map the worker function to each line in the data
        response_list = list(tqdm(workers_pool.imap(worker_partial, [messages] * n), total=n))

        # Close the pool and wait for all processes to finish
        workers_pool.close()
        workers_pool.join()

        response_text_list = [response.choices[0].message.content for response in response_list]    

    else:
        client = OpenAI(api_key="your key here.", base_url="https://api.deepseek.com")
        response_list = []
        response_text_list = []
        for _ in range(n):
            while True:
                try:
                    response = client.chat.completions.create(
                        model=model_name, #"deepseek-chat",
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=stop,
                        stream=False
                    )
                    break
                except:
                    print("Error: retrying...")
                    time.sleep(sleep)
            response_list.append(response)
            response_text_list.append(response.choices[0].message.content)

    return response_text_list


def make_chat_request_hkust(model_name, dialogue_history, n, max_tokens=4000, stop=None, temperature=0, sleep=1):
    url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "your key here",
    }
    data = {
        "model": model_name,
        "messages": dialogue_history,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "n": n,
    }
    try_use_gpt = False
    while not try_use_gpt:
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            # print(response)
            # print(response.json())
            if "error" in response.json():
                try_use_gpt = False
                # print("ERR| ", response)
            else:
                try_use_gpt = True
                # print("YES| ", response)
        except Exception as e:
            if "This model's maximum context length" in str(e):
                return ["This model's maximum context length"]
            print("\r"+str(e)[:200], end='')
            try_use_gpt = False
            time.sleep(sleep)
    # print(dialogue_history)
    # print(response)
    print("\r"+100*" "+"\r", end='')
    # response_list = [choice['message']['content'] for choice in response.json()['choices']]
    response_list = [choice["message"]["content"] for choice in response.json()['choices']]
    return response_list # greedy

def extract_numercial_value(completion, pattern=r'-?\d+/?\.?\d*'):
    INVALID_ANS = "[invalid]"
    completion = completion.strip()
    completion = completion.replace("$", "")
    completion = completion.replace(",", "")
    match = re.findall(pattern, completion)
    if len(match) != 0:
        return match[-1]
    else:
        return INVALID_ANS

def match_numercial_value(completion):
    INVALID_ANS = "[invalid]"
    # \box{}
    pattern = r'boxed\{(.*?)\}'
    match = re.search(pattern, completion)
    if match:
        ans = match.group(1)
        return extract_numercial_value(ans)
    else:
        return extract_numercial_value(completion)

def match_response_code(response):
    # start with "```python" or "```"
    # end with "```"
    # match the first one
    # print(response)
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return response

def match_scip_code_output(response):
    response = response.strip().split("-"*10)[-1]
    return response

def match_opt_code_output(response):
    response = response.strip().split("\n\n")[-1]
    return response

def convert_opt_results_dict(results):
    if results == "There is an error in the code.":
        return {}

    if ":" not in results:
        return {}
    
    results = results.strip().split("\n")
    results_dict = {}
    for line in results:
        key = line.split(":")[0].strip()
        value = line.split(":")[-1].strip()
        results_dict[key] = value
    return results_dict


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def get_calculation_answer_text(response):
    if "####" in response.lower():
        return response.lower().split("ans:")[-1]

    if "ans:" in response.lower():
        return response.lower().split("ans:")[-1]

    if "answer:" in response.lower():
        return response.lower().split("answer:")[-1]
    
    if "answer is" in response.lower():
        return response.lower().split("answer is")[-1]

    return response.split(". ")[-1]

def extract_calculation_answer(completion, pattern=r'-?\d+/?\.?\d*'):
    INVALID_ANS = "[invalid]"
    completion = completion.strip()
    completion = completion.replace("$", "")
    completion = completion.replace(",", "")
    match = re.findall(pattern, completion)
    if len(match) != 0:
        return match[-1]
    else:
        return INVALID_ANS
    
def is_calculation_correct(gt_answer, answer):
    if answer[-1:] == ".":
        answer = answer[:-1]
        
    if gt_answer == answer:
        return True

    try:
        if abs(eval(gt_answer) - eval(answer)) <= 0.01:
            return True
    except:
        pass

    return False

def get_calculation_answer_from_response(response):
    answer_text = get_calculation_answer_text(response)
    answer = extract_calculation_answer(answer_text, pattern=r'-?\d+/?\.?\d*')
    return answer


def check_dialog(dialog):
    start_index = 0
    if dialog[0]["role"] == "system":
        start_index = 1

    # for i in range(start_index, len(dialog)):
    check_dialog = dialog[start_index:]
    user_dialog = check_dialog[::2]
    assistant_dialog = check_dialog[1::2]

    if all([msg["role"] == "user" for msg in user_dialog]) and all([msg["role"] == "assistant" for msg in assistant_dialog]):
        return True
    else: 
        print("model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and alternating (u/a/u/a/u...)")
        return False


def dialog_to_text_llama3(dialog):
    BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"
    EOT = "<|eot_id|>"
    ROLE_TOKENS = "<|start_header_id|>{role}<|end_header_id|>"

    if not check_dialog(dialog):
        return None
    
    prompt_text = ""
    prompt_text += BOS
    for line in dialog:
        prompt_text += ROLE_TOKENS.format(role=line["role"]) + "\n\n"
        prompt_text += line["content"] + EOT
    
    prompt_text += ROLE_TOKENS.format(role="assistant") + "\n\n"
    
    return prompt_text


def dialog_to_text_llama2(dialog):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )

    prompt_text = ""
    for query, response in zip(dialog[::2], dialog[1::2]):
        prompt_text += f"<s>{B_INST} {(query['content']).strip()} {E_INST} {(response['content']).strip()}</s>\n"
    
    prompt_text += f"<s>{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return prompt_text


def delete_computation(text):
    left = "<<"
    right = ">>"
    while left in text and right in text:
        left_index = text.index(left)
        right_index = text.index(right)
        text = text[:left_index] + text[right_index+2:]
    return text

def delete_sp_str_gsm8k(text):
    # text = delete_computation(text) # "<<...>>"
    text = text.replace("####", "So the answer is")
    return text

def request_model(model_name, input_text, max_tokens, temperature, n):
    url = "http://localhost:8000/v1/completions"
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "prompt": input_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


def find_num(s):
    # pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    pattern = r"(\d+\.\d+|\d+\/\d+|\d+%|\d+)"
    nums = [] 
    nums_fraction = [] 
    pos = re.search(pattern, s)
    while(pos):
        nums.append(s[pos.start():pos.end()])
        s = s[:pos.start()] + ' [NUM] ' + s[pos.end():]
        pos = re.search(pattern, s)
    
    for element in nums:
        try:
            eval(element)
            continue
        except:
            return []

    return nums


def find_bracketed_content(input_str):
    pattern = r"<<(.*?)>>"
    matches = re.findall(pattern, input_str)
    return matches


def construct_num_dict(nums):
    num_dict = {}
    index = 1
    for num in nums:
        if num in num_dict:
            continue
        num_dict[num] = "N{}".format(index)
        index += 1
    return num_dict


def expression_to_list(expression):
    expression = expression.replace(' ', '')

    for operator in ['+', '-', '*', '/', '(', ')']:
        expression = expression.replace(operator, ' ' + operator + ' ')
    
    parts = expression.split()
    
    return parts


def eval_number_pair(a, b):
    if a[-1] == '%':
        a = str(eval(a[:-1] + "/100"))
    
    if b[-1] == '%':
        b = str(eval(b[:-1] + "/100"))

    return eval(a) == eval(b)

def construct_template_from_q_a(question, answer):        
    nums = find_num(question)
    if nums == []:
        return [], []

    num_dict = construct_num_dict(find_num(question))
    equations = find_bracketed_content(answer)
    templates = []
    x_index = 1
    for equation in equations:
        equation_left = equation.split("=")[0]
        equation_right = equation.split("=")[1]

        equation_left_list = expression_to_list(equation_left)

        equation_left_list_change = []
        for element in equation_left_list:
            if element not in ['+', '-', '*', '/', '(', ')']:
                is_find = False
                for key in num_dict:
                    if eval_number_pair(key, element):
                        element = num_dict[key]
                        is_find = True
                        break
                if not is_find and element not in ["0.5", "1/2", "2", "100"]:
                    return [], []
            equation_left_list_change.append(element)

        equation_left = "".join(equation_left_list_change)
        template = equation_left + "=" + "X{}".format(x_index)
        num_dict[equation_right] = "X{}".format(x_index)
        x_index += 1

        templates.append(template)
    
    return templates, num_dict

def construct_template_from_a(answer):
    equations = find_bracketed_content(answer)

    templates = []
    num_dict = {}
    v_index = 1
    x_index = 1
    for equation in equations:
        equation_left = equation.split("=")[0]
        equation_right = equation.split("=")[1]
        equation_left_list = expression_to_list(equation_left)

        equation_left_list_change = []

        for element in equation_left_list:
            if element not in ['+', '-', '*', '/']:
                if element in num_dict:
                    element = num_dict[element]
                else:
                    num_dict[element] = "N{}".format(v_index)
                    element = "N{}".format(v_index)
                    v_index += 1 
                    
            equation_left_list_change.append(element)

        equation_left = "".join(equation_left_list_change)
        template = equation_left + "=" + "X{}".format(x_index)
        num_dict[equation_right] = "X{}".format(x_index)
        x_index += 1

        templates.append(template)
    return templates, num_dict


def generate_random_number():
    # Rule 1: 95% are integers
    if random.random() < 0.95:
        # Rule 2: 95% are in (0, 100)
        if random.random() < 0.98:
            if random.random() < 0.95:
                return random.randint(0, 20)
            else:
                return random.randint(0, 100)
        else:
            # Rule 4: For integers > 100, 80% are multiples of 100
            if random.random() < 0.8:
                return random.choice(range(100, 5001, 100))
            else:
                return random.randint(101, 5000)
    else:
        # Rule 3: For non-integers, they are in (0, 100) and only one decimal place, 70% have .5 as decimal part
        if random.random() < 0.7:
            return random.randint(0, 99) + 0.5
        else:
            return round(random.uniform(0, 100), 1)
    
def template_to_equation(templates):
    num_dict = {}
    equations = []
    for template in templates:
        # print(template)
        template_left = template.split("=")[0]
        template_right = template.split("=")[1]
        template_left = expression_to_list(template_left)
        # print(template_left)
        equation_left = []
        for element in template_left:
            if element in ['+', '-', '*', '/']:
                equation_left.append(element)
                continue
            
            if element[0] not in ['X', 'N']:
                equation_left.append(element)
                continue
            
            if element in num_dict:
                element = num_dict[element]
            else:
                num_dict[element] = str(generate_random_number())
                element = num_dict[element]
        
            equation_left.append(element)

        # print(equation_left)
        equation_left = "".join(equation_left)
        # print(equation_left)
        if eval(equation_left) <= 0 or eval(equation_left) > 100000:
            return []
        equation_right = str(round(eval(equation_left),4))
        num_dict[template_right] = equation_right
        equation = equation_left + "=" + equation_right
        equations.append(equation)
        # print(equation)
    
    equations.append("The answer is " + equations[-1].split("=")[-1])
    return equations

def sample_equations(templates):
    while True:
        equations = template_to_equation(templates)
        if equations == []:
            continue
        answer = eval(equations[-1].split("is ")[1])
        if answer > 100:
            if random.random() < 0.3:
                return equations
        elif answer <= 10:
            if random.random() < 0.5:
                return equations
        else:
            return equations


def generate_expression(k):
    # Initialize variables
    variables = [f'n{i+1}' for i in range(k)]
    expressions = []
    # Generate random expressions
    while len(variables) > 1:
        # Randomly choose two variables
        var1, var2 = random.sample(variables, 2)
        # Randomly choose an operator
        operator = random.choice(['+', '-', '*', '/'])
        # Create the expression and its result
        result = f'x{len(expressions) + 1}'
        expression = f'{var1} {operator} {var2} = {result}'
        # Add the expression to the list
        expressions.append(expression)
        # Replace the two variables with the result in the variables list
        variables.remove(var1)
        variables.remove(var2)
        variables.append(result)
    return expressions
# # Test the function with k=3
# generate_expression(3)


def remove_duplicates(paragraphs, threshold=0.8):
    if len(paragraphs) == 1:
        return paragraphs

    # delete the document that only contain stop words
    paragraphs = [paragraph for paragraph in paragraphs if paragraph not in stopwords.words('english')]

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

    tfidf_matrix = vectorizer.fit_transform(paragraphs)

    # compute the cosine similarity between all pairs of paragraphs
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    unique_paragraphs = []

    is_accept = [False] * len(paragraphs)

    for i, paragraph in enumerate(paragraphs):
        is_unique = True
        for j in range(i):
            if is_accept[j] and similarity_matrix[i, j] > threshold:
                is_unique = False
                break
        if is_unique:
            unique_paragraphs.append(paragraph)
            is_accept[i] = True
            
    return unique_paragraphs

# print the json sample into markdown file
def print_md_sample(sample, json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as f:
        for key in sample:
            f.write(f"## {key}\n\n")
            f.write(f"{sample[key]}\n\n\n")