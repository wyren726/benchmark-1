import numpy as np
import scipy
import nltk
import typing

import torch.nn.functional as F

from sklearn.metrics import f1_score
import openai
from openai import OpenAI
from transformers import T5ForConditionalGeneration
import time
import regex
import string
import time
client=  OpenAI(
        # base_url=f"http://28.16.139.41:21910/v1",
        base_url=f"http://localhost:21910/v1",
        api_key="dummy"
    )

completion = client.chat.completions.create(
        model="Qwen2.5-VL",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "who are you"}
        ],
        temperature=0.0
    )
print(completion)