'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-07 15:37:21
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-03-20 13:40:56
FilePath: /mengdizhang/LLM-LieDetector/lllm/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import ast
from time import sleep

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Maximum number of tokens that the openai api allows me to request per minute
RATE_LIMIT = 250000


# To avoid rate limits, we use exponential backoff where we wait longer and longer
# between requests whenever we hit a rate limit. Explanation can be found here:
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# I'm using default parameters here, I don't know if something else might be
# better.
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


# Define a function that adds a delay to a Completion API call
def delayed_completion_with_backoff(delay_in_seconds: float = 1, **kwargs):
    """Delay a completion by a specified amount of time."""

    # Sleep for the delay
    sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return completion_with_backoff(**kwargs)


def completion_create_retry(*args, sleep_time=5, **kwargs):
    """A wrapper around openai.Completion.create that retries the request if it fails for any reason."""

    if 'llama' in kwargs['model'].lower() or 'vicuna' in kwargs['model'].lower() or 'alpaca' in kwargs['model'].lower():
        if type(kwargs['prompt'][0]) == list:
            prompts = [prompt[0] for prompt in kwargs['prompt']]
        else:
            prompts = kwargs['prompt']
        return kwargs['endpoint'](prompts, **kwargs)
    else:
        while True:
            try:
                return openai.Completion.create(*args, **kwargs)
            except:
                sleep(sleep_time)
