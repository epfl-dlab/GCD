import json
import os
import shlex
import socket
import subprocess
import time
from typing import List, Union

import numpy as np
import requests
from urllib.parse import urljoin
import logging

log = logging.getLogger(__name__)

DEFAULT_PGF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "pgf")


class GFRuntime:
    """
    Abstract class for GF runtime.

    The only method is `complete`, which takes a input_tokens and returns a list of possible completions.
    """

    def complete(self, input_tokens: List[Union[int, str]], pgf=None) -> List[str]:
        raise NotImplementedError


def _concat_input_ids_to_str(input_ids: List[str]):
    # convert to string
    # concatenate with space
    # add an extra space at the end
    return " ".join([str(x) for x in input_ids]) + " "


def _is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("localhost", port))
            return True
        except OSError:
            return False


class GFServerRuntime(GFRuntime):
    """
    A runtime for GF implemented as a http server.
    """

    DOMAIN: str = "http://localhost"

    def __init__(self, default_pgf: str, grammar_dir=None, port: int = 41296):
        """TODO
        pgf_or_pgf_dirname should be two arguments:
        - pgf: the pgf file name, configure the pgf before runtime
        - pgf=None: pass the pgf at runtime, this is the case for runtime-dependent grammars
        - root_dir: simply for convenience, reduce the absolute path to relative path
        """
        # default_pgf = self.__preprocess_pgf(default_pgf, root_dir=grammar_dir)
        self.runtime_pgf_dir = grammar_dir
        self._default_pgf = default_pgf

        self.__prepare_launch(port=port)
        self.server_process = self.__launch_server(
            root_dir=self.runtime_pgf_dir, verbose=True
        )

    def __prepare_launch(self, port: int = 41296):
        # Find a free port
        while not _is_port_available(port):
            port += 1
        self.port = port
        self.url = GFServerRuntime.DOMAIN + ":" + str(self.port)

    def __launch_server(self, root_dir: str, verbose: bool = True):
        cmd = f"gf --document-root={root_dir} --server={self.port}"
        log.debug("Launching server with command: " + cmd)
        # Start the subprocess and detach it from the parent process
        # Split the command string into a list of arguments
        args = shlex.split(cmd)

        server_process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            close_fds=True,
        )

        # Wait for the server to be ready
        log.debug(f"visit {self.url} to check if the server is ready")

        # wait for the server to be ready
        time.sleep(0.1)

        return server_process

    def complete(self, input_tokens: List[str], pgf=None) -> List[str]:
        if pgf is not None and self.__is_grammar_available(pgf):
            pgf_to_use = pgf
        else:
            pgf_to_use = self._default_pgf

        if pgf_to_use is None:
            raise RuntimeError("pgf is not specified")

        url = urljoin(self.url, pgf_to_use)
        input_tokens_str: str = _concat_input_ids_to_str(input_tokens)
        params = {"command": "complete", "input": input_tokens_str}

        # Send an HTTP GET request with values
        response = requests.get(url, params=params)

        # Parse the response, handle errors
        if response.status_code != 200:
            raise Exception(
                "Error: the server returned status code " + str(response.status_code)
            )

        parsed_response = json.loads(response.text)

        assert len(parsed_response) == 1

        completions = parsed_response[0]["completions"]

        return completions

    def __is_grammar_available(self, pgf):
        path = os.path.join(self.runtime_pgf_dir, pgf)
        return os.path.exists(path)

    def clear(self):
        self.server_process.terminate()

    def random_decode(self, n: int, tokens2exclude: set[str]) -> List[float]:
        input_tokens = []
        decoding_time = []

        count = 0

        while count < n:
            start_time = time.time()
            completions: List[str] = self.complete(input_tokens)
            completions = [x for x in completions if x not in tokens2exclude]
            # randomly choose one completion
            if len(completions) > 0:
                idx = np.random.randint(0, len(completions))
                next_token: str = completions[idx]
                end_time = time.time()
                elapsed_time = round(end_time - start_time, 4)
                decoding_time.append(elapsed_time)
                input_tokens.append(next_token)
                count += 1
            else:
                # clear the input_tokens and start over
                input_tokens = []
        return decoding_time


class GFServerRuntimeForLM(GFServerRuntime):
    def get_prefix_allowed_tokens_for_LM(
        self, prefix_tokens: Union[List[int], object], pgf: str = None
    ) -> List[int]:
        # convert tensor to list
        prefix_tokens: List[int] = self.__preprocess_tensor(prefix_tokens)
        # convert to string
        prefix_tokens: List[str] = [str(x) for x in prefix_tokens]
        # get completions
        completions: List[str] = self.complete(prefix_tokens, pgf=pgf)
        # convert to integers
        allowed_tokens = [int(x) for x in completions]
        return allowed_tokens

    def __preprocess_tensor(self, input: Union[List[int], object]) -> List[int]:
        if type(input).__name__ == "Tensor":  # Torch tensor
            input = input.tolist()
        elif type(input) == list:
            pass
        else:
            raise NotImplementedError

        return input


if __name__ == "__main__":
    pgf = "FoodRepeat.pgf"
    pgf_dir = (
        "/Users/saibo/Research/Projects/GCD/GF-Grammar-Factory/asset/GF-grammars/pgf"
    )
    gf_server = GFServerRuntime(default_pgf=pgf, grammar_dir=pgf_dir)
    print(gf_server.complete(input_tokens=[]))
    print(gf_server.complete(input_tokens=["this"]))
    gf_server.clear()
