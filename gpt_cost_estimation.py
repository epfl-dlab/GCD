import tiktoken
import json
import os
import pandas as pd
from scipy.stats import bootstrap
import numpy as np

os.environ["TIKTOKEN_CACHE_DIR"] = ""
LOG_DIR_PATH = os.path.join(os.path.dirname(__file__), 'logs/inference/runs/')
ASSET_DIR_PATH = os.path.join(os.path.dirname(__file__), 'assets/')
IE_DATASETS = ["synthie"]
ED_DATASETS = ["aquaint","msnbc","ace2004","wiki","aida","clueweb"]
CP_DATASETS = ["ptb64"]

DATASET_LENGTHS = {"ace2004":240,"aida":4485,"aquaint":703,"wiki":6814,"msnbc":651,"clueweb":11110,"synthie":1000,"ptb64":173}
EXPERIMENTS = ["2023-10-26T11:11:46.270317","2023-10-26T08:16:54.358704","2023-10-26T02:24:26.897882","2023-10-26T02:15:58.374842",
               "2023-10-26T01:43:34.080302","2023-10-26T00:53:55.050774","2023-10-25T10:10:20.940546","2023-10-25T10:10:20.940546",
               "2023-10-24T16:58:59.470725","2023-10-24T16:33:47.657189","2023-10-24T14:51:00.702612","2023-10-24T13:06:12.365795",
               "2023-10-24T09:30:22.886399","2023-10-23T08:51:31.619617","2023-10-22T22:22:21.848546","2023-10-21T23:03:38.795276",
               "2023-10-20T16:55:32.250680","2023-10-20T03:37:57.359277","2023-11-03T12:05:33.690529"]

PRICES = {"gpt-3.5-turbo-0613":(0.0015,0.002),
          "gpt-4-0613":(0.03,0.06),
          "gpt-3.5-turbo-0301":(0.0015,0.002),
          "gpt-3.5-turbo-instruct-0914":(0.0015,0.002),
          "davinci-002":(0.0020,0.0020),
          "text-davinci-003":(0.02,0.02)}

def get_prompt_costs(num_demo):
    prompt_costs = {"ED":{data:{} for data in ED_DATASETS},
                "IE":{data:{} for data in IE_DATASETS},
                "CP":{data:{} for data in CP_DATASETS}}
    for task in ["ED","IE","CP"]:
        with open(os.path.join(ASSET_DIR_PATH,"prompts/",task,"stable/","instruction.txt"), "r") as f:
            instruction =f.read()
        with open(os.path.join(ASSET_DIR_PATH,"prompts/",task,"stable/","demo.json"), "r") as f:
            demos = json.load(f)
        prompt = instruction
        for i in range(num_demo):
            prompt += demos[i]["text"] + " -> " + demos[i]["output"] + "; "
        for model in PRICES.keys():
            prompt_length = len(tiktoken.encoding_for_model(model).encode(prompt))
            for dataset in prompt_costs[task]:
                prompt_costs[task][dataset][model] = prompt_length*PRICES[model][0]/1000*DATASET_LENGTHS[dataset]
    return prompt_costs

def get_cost_dataset(model,dataset):
    tokenizer_model = tiktoken.encoding_for_model(model)
    tokenized_input_len = dataset["inputs"].apply(lambda x: len(tokenizer_model.encode(x))).to_numpy()
    tokenized_output_len = dataset["unflattened_predictions"].apply(lambda x: len(tokenizer_model.encode(x[0]))).to_numpy()
    return tokenized_input_len, tokenized_output_len

def get_predictions_from_runs():
    predictions = {"ED":{data:{} for data in ED_DATASETS},
                   "IE":{data:{} for data in IE_DATASETS},
                   "CP":{data:{} for data in CP_DATASETS}}
    for exp in os.listdir(LOG_DIR_PATH):
        for run in os.listdir(os.path.join(LOG_DIR_PATH,exp)):
            if "wandb" in os.listdir(os.path.join(LOG_DIR_PATH,exp,run)):
                experiment_task = exp.split("_")[1]
                experiment_dataset = exp.split("_")[-5]
                experiment_model = exp.split("_")[3]
                if experiment_dataset == "stable":
                    experiment_dataset = "synthie" 
                exp_metadata_file = os.path.join(LOG_DIR_PATH,exp,run,"wandb/latest-run/files/wandb-metadata.json")
                with open(exp_metadata_file) as f:
                    exp_metadata = json.load(f)
                if exp_metadata["startedAt"] in EXPERIMENTS:
                    print(experiment_task,experiment_dataset,experiment_model)
                    preds = pd.read_json(os.path.join(LOG_DIR_PATH,exp,run,"predictions/testing_output_0.prediction.jsonl"), lines=True)
                    predictions[experiment_task][experiment_dataset][experiment_model] = preds
    return predictions

def calculate_token_usage_from_predictions(predictions):
    usage = {"ED":{data:{} for data in ED_DATASETS},
                "IE":{data:{} for data in IE_DATASETS},
                "CP":{data:{} for data in CP_DATASETS}}
    for task in predictions:
        for data in predictions[task]:
            for model in predictions[task][data]:
                preds = predictions[task][data][model]
                dataset_length = DATASET_LENGTHS[data]
                tokenized_input_len, tokenized_output_len = get_cost_dataset(model,preds)
                input_token_bootstrap = bootstrap((tokenized_input_len,), np.mean, confidence_level=0.95)
                output_token_bootstrap = bootstrap((tokenized_output_len,), np.mean, confidence_level=0.95)
                input_token_mean = np.mean(tokenized_input_len)
                output_token_mean = np.mean(tokenized_output_len)
                
                input_usage_error = np.mean([input_token_bootstrap.confidence_interval.high-input_token_mean,input_token_mean-input_token_bootstrap.confidence_interval.low])
                output_usage_error = np.mean([output_token_bootstrap.confidence_interval.high-output_token_mean,output_token_mean-output_token_bootstrap.confidence_interval.low])
                
                usage[task][data][model] = {"input_mean": input_token_mean,"output_mean":output_token_mean,"input_error":input_usage_error,"output_error":output_usage_error}
    return usage

def calculate_cost(model,input_tokens,output_tokens,input_error,output_error,dataset_length,fixed_output_length=None):
    if fixed_output_length is not None:
        output_tokens = fixed_output_length
        output_error = 0
    price_mean = PRICES[model][0] * input_tokens/1000 + PRICES[model][1] * output_tokens/1000
    price_err = PRICES[model][0] * input_error/1000 + PRICES[model][1] * output_error/1000
    price_mean, price_err = price_mean*dataset_length, price_err*dataset_length
    
    return {"Cost Estimate": price_mean, "Cost Error": price_err}

def estimate_all_costs(fixed_output_length=None):
    
    predictions = get_predictions_from_runs()
    predicted_usage = calculate_token_usage_from_predictions(predictions)
    ed_task_estimates = {}
    cp_task_estimates = {}
    for dataset in ED_DATASETS:
        ed_task_estimates[dataset] = {}
        for metric in ["input_mean","output_mean","input_error","output_error"]:
            ed_task_estimates[dataset][metric] = np.mean([predicted_usage["ED"][dataset][model_res][metric] for model_res in predicted_usage["ED"][dataset]])
    for dataset in CP_DATASETS:
        cp_task_estimates[dataset] = {}
        for metric in ["input_mean","output_mean","input_error","output_error"]:
            cp_task_estimates[dataset][metric] = np.mean([predicted_usage["CP"][dataset][model_res][metric] for model_res in predicted_usage["CP"][dataset]])         
    all_costs = {}
    for task in predicted_usage:
        all_costs[task] = {}
        for data in predicted_usage[task]:
            all_costs[task][data] = {}
            for model in PRICES.keys():
                if model in predicted_usage[task][data]:
                    all_costs[task][data][model] = calculate_cost(model,predicted_usage[task][data][model]["input_mean"],
                                                                  predicted_usage[task][data][model]["output_mean"],
                                                                  predicted_usage[task][data][model]["input_error"],
                                                                  predicted_usage[task][data][model]["output_error"],
                                                                  DATASET_LENGTHS[data],fixed_output_length)
                elif task=="ED":
                    all_costs[task][data][model] = calculate_cost(model,ed_task_estimates[data]["input_mean"],
                                                                  ed_task_estimates[data]["output_mean"],
                                                                  ed_task_estimates[data]["input_error"],
                                                                  ed_task_estimates[data]["output_error"],
                                                                  DATASET_LENGTHS[data],fixed_output_length)
                elif task=="CP":
                    all_costs[task][data][model] = calculate_cost(model,cp_task_estimates[data]["input_mean"],
                                                                  cp_task_estimates[data]["output_mean"],
                                                                  cp_task_estimates[data]["input_error"],
                                                                  cp_task_estimates[data]["output_error"],
                                                                  DATASET_LENGTHS[data],fixed_output_length)
                    
    with open("cost_estimates.json", "w") as f:
        json.dump(all_costs, f)
    
    prompt_costs = get_prompt_costs(num_demo=4)
    with open("prompt_costs.json", "w") as f:
        json.dump(prompt_costs, f)
if __name__ == '__main__':
    estimate_all_costs()    