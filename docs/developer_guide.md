# Developer Guide

## 0. Pre-commit hook

We use `pre-commit` to run some checks before committing code. To set it up, run:

```bash
cd <project_root>
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The hook will run automatically before each commit. If it fails, you can run it manually with `pre-commit run --all-files`.

In case you want to skip the hook, you can use `git commit --no-verify`.


## 1. Code Structure

## 2. Hydra Configuration

## 3. Adding a new task

## 4. Adding a new grammar

## 5. Adding a new model

## 6. benchmarking latency

## 7. Comparing with other GCD methods
