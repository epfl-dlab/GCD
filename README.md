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
conda create -n GCD python=3.9
conda activate GCD
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## Experiments

- [Download datasets, grammars and models](docs/download_data.md)
- [Build task-specific grammars](https://github.com/Saibo-creator/GF_helper)
- [Windows-specific setting](docs/windows.md)
- [Running the experiments](docs/run_experiments.md)


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
