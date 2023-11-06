import json
import os
import pandas as pd
from scipy.stats import bootstrap
import numpy as np

os.environ["TIKTOKEN_CACHE_DIR"] = ""
LOG_DIR_PATH = os.path.join(os.path.dirname(__file__), 'logs/inference/runs/')
IE_DATASETS = ["synthie"]
ED_DATASETS = ["aquaint","msnbc","ace2004","wiki","aida","clueweb"]
CP_DATASETS = ["ptb64"]

METRICS = {"ED": ["accuracy_step"],
           "IE": ["precision_step", "recall_step", "f1_step"],
           "CP": ["recall_step", "prec_step","tag_accracy_step"]}
DATASET_LENGTHS = {"ace2004":240,"aida":4485,"aquaint":703,"wiki":6814,"msnbc":651,"clueweb":11110,"synthie":9904,"ptb64":173}
EXPERIMENTS = ["2023-10-26T11:11:46.270317","2023-10-26T08:16:54.358704","2023-10-26T02:24:26.897882","2023-10-26T02:15:58.374842",
               "2023-10-26T01:43:34.080302","2023-10-26T00:53:55.050774","2023-10-25T10:10:20.940546","2023-10-25T10:10:20.940546",
               "2023-10-24T16:58:59.470725","2023-10-24T16:33:47.657189","2023-10-24T14:51:00.702612","2023-10-24T13:06:12.365795",
               "2023-10-24T09:30:22.886399","2023-10-23T08:51:31.619617","2023-10-22T22:22:21.848546","2023-10-21T23:03:38.795276",
               "2023-10-20T16:55:32.250680","2023-10-20T03:37:57.359277","2023-11-03T22:39:01.861778","2023-11-04T00:55:01.445432",
               "2023-11-03T23:15:08.232681","2023-11-03T23:44:29.580371","2023-11-03T23:50:04.568442","2023-11-03T23:58:07.564349",
               "2023-11-04T00:03:15.774337","2023-11-04T00:08:15.753808","2023-11-04T00:13:49.137039","2023-11-04T00:16:12.175474",
               "2023-11-04T00:23:08.671894","2023-11-04T00:25:48.977962","2023-11-04T00:33:44.312454","2023-11-04T00:36:36.612471",
               "2023-10-25T04:02:58.135737"]
print(len(EXPERIMENTS))
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
                    preds = pd.read_csv(os.path.join(LOG_DIR_PATH,exp,run,"csv_logs/version_0/metrics.csv"))
                    predictions[experiment_task][experiment_dataset][experiment_model] = preds
                    if len(preds)!=DATASET_LENGTHS[experiment_dataset]+1:
                        print("WARNING: Dataset length mismatch",experiment_dataset,len(preds),DATASET_LENGTHS[experiment_dataset])
    return predictions

def calculate_bootstrap_result(predictions):
    results = {"ED":{data:{} for data in ED_DATASETS},
                "IE":{data:{} for data in IE_DATASETS},
                "CP":{data:{} for data in CP_DATASETS}}
    
    for task in predictions:
        for data in predictions[task]:
            for model in predictions[task][data]:
                preds = predictions[task][data][model]
                dataset_length = len(preds)-1 #DATASET_LENGTHS[data]
                results[task][data][model] = {}
                for metric_name in METRICS[task]:
                    metric = "test/" + metric_name
                    if metric not in list(preds.columns):
                        metric = "test_0/" + metric_name
                    print(metric,preds.columns,preds[metric].iloc[:dataset_length].isna().sum())
                    mean_res = np.mean(preds[metric].iloc[:dataset_length].to_numpy())
                    bootstrap_result = bootstrap((preds[metric].iloc[:dataset_length].to_numpy(),), np.mean, confidence_level=0.95)
                    error = np.mean([bootstrap_result.confidence_interval.high-mean_res,mean_res-bootstrap_result.confidence_interval.low])
                    results[task][data][model][metric] = {"mean": mean_res,"error":error}
    return results

def report_all_results():
    
    predictions = get_predictions_from_runs()
    report_all_results = calculate_bootstrap_result(predictions)
        
    with open("results.json", "w") as f:
        json.dump(report_all_results, f)
        
if __name__ == '__main__':
    report_all_results()    