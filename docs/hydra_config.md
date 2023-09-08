# Hydra & Configs

## Constrained Modules

By default, the constrained modules are not loaded.

To change the default behavior, adapt the following lines in the `configs/experiment/{task}/inference/_default_llama.yaml` file:

```yaml
defaults:
  - /constraint/gf_constraint_module/CP@model.gf_constraint_module: null
  - /constraint/trie_constraint_module: null
```
