from typing import Dict, Optional, Any


def convert_slash_separated_keys_to_nested_dict(dictionary: dict) -> dict:
    result_dict = dict()
    for key, value in dictionary.items():
        parts = key.split("/")
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict


def convert_nested_dict_to_dot_separated_keys(dictionary: dict, parent_key: str = "") -> dict:
    """
    Converts a nested dictionary to a dictionary with dot-separated keys.

    nested_dict = {
        "model": {
            "name": "resnet",
            "params": {
                "num_layers": 18,
                "dropout": 0.2
            }
        },
        "optimizer": {
            "name": "adam",
            "learning_rate": 0.001
        },
        "batch_size": 64
    }

    dot_separated_keys_dict =
    {
        "model.name": "resnet",
        "model.params.num_layers": 18,
        "model.params.dropout": 0.2,
        "optimizer.name": "adam",
        "optimizer.learning_rate": 0.001,
        "batch_size": 64
    }


    """
    result = {}
    for key, value in dictionary.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            result.update(convert_nested_dict_to_dot_separated_keys(value, new_key))
        else:
            result[new_key] = value
    return result


def dict_of_lists_to_list_of_dicts(dict_of_lists):
    """
    Converts a dictionary of lists to a list of dictionaries

    Parameters
    ----------
    dict_of_lists: A dict of lists, each element of the list corresponding to one item.
             For example: {'id': [1,2,3], 'val': [72, 42, 32]}

    Returns
    -------
    A list of dicts of individual items.
    For example: [{'id': 1, 'val': 72}, {'id': 2, 'val': 42}, {'id': 3, 'val': 32}]

    """
    keys = dict_of_lists.keys()
    values = [dict_of_lists[key] for key in keys]
    items = [dict(zip(keys, item_vals)) for item_vals in zip(*values)]
    return items


class AttributeDict(Dict):
    """Extended dictionary accessible with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42
    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = "\n".join(rows)
        return out
