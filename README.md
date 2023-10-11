[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![MIT License](https://img.shields.io/github/license/m43/focal-loss-against-heuristics)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2305.13971-b31b1b.svg)](https://arxiv.org/abs/2305.13971)

# Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning

---
## 1. The Overview of GCD

<div align="center">
<img src="assets/figures/figure1.png" style="width:50%">
</div>


## 2. Environment Setup

With the repository cloned, we recommend creating a new [conda](https://docs.conda.io/en/latest/) virtual environment:
```bash
conda env create -n GCD python=3.9
conda activate GCD
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## 3. Downloading the dataset, grammar objects and models

check the [docs/download_data.md](docs/download_data.md) for instructions on how to download them.


## 4. Build task-specific grammars

c.f.  [GF_helper repo](https://github.com/Saibo-creator/GF_helper)


## Running the experiments

```shell
# run the experiments for the CP task
bash run_CP.sh

# run the experiments for the IE task
bash run_IE.sh

# run the experiments for the ED task
bash run_ED.sh
```


The generated prediction sequences will be logged to [Weights and Biases](https://wandb.ai/site).


## Developer Guide

If you want to extend the codebase, please check the [docs/developer_guide.md](docs/developer_guide.md) for more details.


## Citation

This repository contains the code for the models and experiments in [Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning](https://arxiv.org/abs/2305.13971)

```
@misc{geng2023flexible,
      title={Flexible Grammar-Based Constrained Decoding for Language Models},
      author={Saibo Geng and Martin Josifosky and Maxime Peyrard and Robert West},
      year={2023},
      eprint={2305.13971},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
**Please consider citing our work, if you found the provided resources useful.**<br>


### License
This project is licensed under the terms of the MIT license.
