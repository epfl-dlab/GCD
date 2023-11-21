import json
import os
from typing import List, Dict


class Prompter:
    def __init__(self, instruction: str, demos=None, num_demo=0):
        self.instruction = instruction
        self.DEMOs = demos
        self.num_demo = num_demo
        assert self.num_demo <= len(self.DEMOs), (
            f"num_demo should be less than or equal to the number of demos."
            f"num_demo: {self.num_demo}, len(demos): {len(self.DEMOs)}"
        )

    @classmethod
    def from_local(cls, dir_path, num_demo=0):
        """
        load demo.json
        load instruction.txt
        """

        with open(os.path.join(dir_path, "instruction.txt"), "r") as f:
            instruction: str = f.read()

        with open(os.path.join(dir_path, "demo.json"), "r") as f:
            demos: List[Dict] = json.load(f)

        return cls(instruction, demos, num_demo)

    def materialize(self, runtime_input: Dict, output_prefix="") -> str:
        prompt = self.instruction
        for i in range(self.num_demo):
            prompt += self.DEMOs[i]["text"] + " -> " + self.DEMOs[i]["output"] + "; "
        prompt += runtime_input["text"] + " -> " + output_prefix
        return prompt

    def __call__(self, runtime_input):
        return self.materialize(runtime_input)

    def get_overhead_token_num(self, tokenizer) -> int:
        materialized_prompt = self.materialize({"text": ""})
        return len(tokenizer.tokenize(materialized_prompt))


class DraftPrompter(Prompter):
    def __init__(self, instruction: str, demos=None, num_demo=0):
        super().__init__(instruction, demos, num_demo)

    def materialize(self, runtime_input: Dict, output_prefix="") -> str:
        prompt = self.instruction
        for i in range(self.num_demo):
            prompt += self.DEMOs[i]["text"] + " -> " + self.DEMOs[i]["draft"] + " -> " + self.DEMOs[i]["output"] + "; "
        prompt += runtime_input["text"] + " -> " + runtime_input["draft"] + " -> " + output_prefix
        return prompt

