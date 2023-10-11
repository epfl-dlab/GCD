# Run experiments

## requirements

Check the env variable is set correctly
```shell
echo $HF_MODELS_DIR
```

Check the data and grammar objects are downloaded correctly
```shell
ls data assets/grammar_objects
# -> CP ED IE
```

Check the pre-trained models are downloaded correctly
```shell
ls assets/pgf
# -> CP ED IE
```

If anything is missing, check the [docs/download_data.md](docs/download_data.md) for instructions on how to set it.


## Run the experiments

### Quick start

Suppose you have already `LLAMA-7B` in `$HF_MODELS_DIR`, run the following commands:

```shell
# run the experiments for the CP task
bash run_CP.sh LLAMA-7B

# run the experiments for the IE task
bash run_IE.sh LLAMA-7B

# run the experiments for the ED task
bash run_ED.sh LLAMA-7B
```

The above scripts will run the experiments for the CP, IE and ED tasks respectively with a few data samples.
To run the experiments with the full dataset, please remove the `datamodule.debug_k=2` option in the scripts.

## Results

The generated prediction sequences will be logged to [Weights and Biases](https://wandb.ai/site).

## Dry run

If you don't have the model yet, you can run the experiments with a dummy model.
```shell
# run the experiments for the CP task
bash run_CP.sh saibo/llama-1B
```

`saibo/llama-1B` is a dummy model that has the same tokenizer as `LLAMA-7B` but with random weights.
It only has two layers so it's much smaller.
But as the model is randomly initialized, the results will be meaningless.







## Run experiments without constraints

You can check the results of the experiments without constraints by removing the constraints flags in the scripts.

For example, remove `+constraint/gf_constraint_module/CP@model.gf_constraint_module="$gf_constraint_module_option"` in `run_CP.sh` will run the experiments without constraints.


