

def get_hf_model_short_name(pretrained_model_name_or_path: str) -> str:
    """Returns the short name of a HF model name.
    For example, 'bert-base-uncased' -> 'bert-base-uncased'
    'xxx/bert-base-uncased' -> 'bert-base-uncased'
    '/xxx/bert-base-uncased' -> 'bert-base-uncased'
    """
    if "/" in pretrained_model_name_or_path:
        return pretrained_model_name_or_path.split("/")[-1]
    else:
        return pretrained_model_name_or_path
