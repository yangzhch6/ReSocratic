## E-OPT Benchmark
*A initial version of this benchmark serves as a competition track of [ICML 2024 AI4MATH Workshop](https://www.codabench.org/competitions/2438/)*


### Performance of LLMs on E-OPT Benchmark

| Model | Linear w/ Table | Linear w/o Table | Nonlinear w/ Table | Nonlinear w/o Table | All | Code Pass |
| --- | --- | --- | --- | --- | --- | --- |
| **Zero-shot Prompt**  |
| Llama-3-8B-Instruct | 0.29% | 0.0% | 0.0% | 0.0% | 0.17% | 8.8% |
| GPT-3.5-Turbo | 37.5% | 68.1% | 16.0% | 19.5% | 49.1% | 85.0% |
| Llama-3-70B-Instruct | 50.0% | **76.9%** | 32.0% | 30.8% | 59.5% | 86.8% |
| DeepSeek-V2 | 27.5% | 40.4% | 18.0% | 29.3% | 34.4% | 74.0% |
| **GPT-4** | **62.5%** | 75.4% | **32.0%** | **42.1%** | **62.8%** | **88.8%** |
| **Few-shot Prompt** |  |  |  |  |  |  |
| Llama-3-8B-Instruct | 2.5% | 17.8% | 8.0% | 11.3% | 13.6% | 26.9% |
| GPT-3.5-Turbo | 40.0% | 75.4% | 26.0% | 28.6% | 56.4% | **93.2%** |
| Llama-3-70B-Instruct | 57.5% | 79.2% | 32.0% | 33.8% | 62.5% | 91.2% |
| DeepSeek-V2 | 56.3% | 79.5% | 32.0% | 27.1% | 61.0% | 85.5% |
| **GPT-4** | <u>**71.3%**</u> | <u>**80.7%**</u> | **34.0%** | <u>**34.6%**</u> | <u>**65.5%**</u> | 88.3% |
| **SFT with Synthetic Data** |
| Llama-2-7B-Chat | 11.3% | 40.6% | 32.0% | 15.8% | 30.6% | 93.7% |
| Llama-3-8B-Instruct | **32.5%** | **63.5%** | <u>**44.0%**</u> | **33.0%** | **51.1%** | **96.3%** |


### Evaluation 
Eval GPT
```
python gpt_baseline.py 
    --model_name "gpt-4" or "gpt-3.5-turbo" 
    --prompt_path "prompt/solve/scip_zeroshot.txt" or "prompt/solve/scip_fewshot.txt"

```

Eval Llama
```
CUDA_VISIBLE_DEVICES=0,1 python llama{2/3}_baseline.py \
    --prompt_path "prompt/solve/scip_zeroshot.txt" or "prompt/solve/scip_fewshot.txt" \
    --model_name_or_path "model_path" \
    --output_path "results.json" \
    --tensor_parallel_size 4 \
    --batch_size 8 
```

## ReSocratic

### Synthesize

Synthesize scenarios
```
python resocratic_synthesize.py \
    --pool_path "prompt/synthesis/pool/{linear/nonlinear}.json"
```

Synthesize questions
```
python synthesize_question.py \
    --data_path "synthetic scenarios path" \
    --output_path "output file path"
```

Synthesize code
```
python synthesize_code.py \
    --data_path "synthetic scenarios path" \
    --output_path "output file path"
```

### SFT
sft Llama-2-7b-Chat
```
bash scripts/train_llama2.sh
```

sft Llama-3-8b-Instruct
```
bash scripts/train_llama3.sh
```