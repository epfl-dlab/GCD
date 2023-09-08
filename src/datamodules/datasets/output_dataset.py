from typing import Dict, List, Any

from .text_datatset import TextDataset


class OutputDataset(TextDataset):

    def __getitem__(self, idx) -> Dict:
        dp: Dict = self.data[idx]
        target_text = dp["target"]
        # prediction can be a single string or a list of strings
        prediction: Dict[List[Any], Any] = dp["prediction"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        # add eos token
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        return {
            "id": dp["id"],
            "text": dp["text"],
            "target": target_text,
            "target_ids": target_ids,
            "prediction": prediction,
        }

    @staticmethod
    def get_predictions(item, key="prediction", top_pred_only=True):
        preds = item[key]

        if top_pred_only and not isinstance(preds, str):
            return preds[0]

        return preds

    @staticmethod
    def get_targets(item, key="target", wrap_in_list=False):
        tgts = item[key]

        if wrap_in_list and not isinstance(tgts, list):
            return [tgts]

        return tgts
