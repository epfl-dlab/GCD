#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : convert2synthie_dataset.py
# @Date : 2023-05-19-21-03
# @Project: tasks_data_preprocess
# @AUTHOR : Saibo Geng
# @Desc :
import os

datasets = ["aida", "ace2004", "aquaint", "clueweb", "msnbc", "wiki"]
from transformers import LlamaTokenizer

# output_dir = "aida-kilt-processed-for-synthie-dataset-short"

tokenizer = LlamaTokenizer.from_pretrained("saibo/llama-7B")
# for split in ["all", "train", "dev", "test"]:
# datafile = f"aida-{split}-kilt.jsonl"
n_shot = 4

MAX_LENGTH = 128

num_entry_exceed_max_length = 0

for dataset in datasets:
    split = "test"

    datafile = f"{dataset}-{split}-kilt-short-fs{n_shot}.jsonl"
    data_dir = f"data/el"
    data_path = os.path.join(data_dir, datafile)

    # load data
    import json

    with open(data_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    for entry in entries:
        _id = entry["id"]
        text = entry["text"]

        num_tokens = len(tokenizer.tokenize(text))
        if num_tokens > MAX_LENGTH:
            num_entry_exceed_max_length += 1
            print(f"{_id} exceed max length: {num_tokens}")

    print(
        f"finish processing {dataset} {split}, num_entry_exceed_max_length: {num_entry_exceed_max_length} out of {len(entries)}"
    )
