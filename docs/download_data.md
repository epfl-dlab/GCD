# Download Data


## Download data for the experiments

```bash
git lfs install
git clone https://huggingface.co/datasets/saibo/GCD-data-v2
```



## Download the grammar objects

At the root of the repository, run the following command to download the compiled grammar files
```bash
git lfs install
git clone https://huggingface.co/datasets/saibo/GCD-grammar-v2 assets/pgf
```

Unzip the files
```bash
cd assets/pgf
# unzip and remove the zip files
unzip ED.zip && rm ED.zip
unzip CP.zip && rm CP.zip
unzip IE.zip && rm IE.zip
```


## Get models

Create an environment variable `HF_MODELS_DIR` that points to the directory where you store the models.


For example, we create a directory `~/models` and save the model of llama-7B by running the following command:
```bash
mkdir ~/models
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/saibo/llama-7B ~/models/llama-7B
```

Then, we set the environment variable `HF_MODELS_DIR` to `~/models` by running the following command:
```bash
export HF_MODELS_DIR=~/models
```

We don't provide other model weights as they are too large and may have licensing issues.
