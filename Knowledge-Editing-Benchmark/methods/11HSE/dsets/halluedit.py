import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class HALLUQADataset(Dataset):
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, model: AutoModelForCausalLM, size=None, topic=None, *args, **kwargs):
        data_dir = Path(data_dir)
        #zsre_loc = data_dir / "zsre_mend_eval.json"
        assert (topic), f"Topic missing. Check for errors?"
        hallu_loc = data_dir / f"{topic}_together.json"
        if not hallu_loc.exists():
            print(f"{hallu_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, hallu_loc)

        with open(hallu_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            if size is not None: 
                if i >= size:
                   break 
            with torch.no_grad():
                neighborhood_prompts = record["loc"]
                prompt_tok = tok(
                            neighborhood_prompts,
                            padding=True,
                            return_tensors="pt",
                        ).to("cuda")
                prompt_length = prompt_tok['input_ids'].shape[1]
                ans_toks = model.generate(**prompt_tok, max_new_tokens = 6)[0][prompt_length:]
                #print('ans_toks:',ans_toks)
                #last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
                #to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
                #gathered = torch.gather(logits, 1, to_gather).squeeze(1)
                #ans_toks = torch.argmax(gathered, dim=1)
            
            #ans = tok.batch_decode(ans_toks[:,prompt_length:])
            #print('ans:',ans)
            
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        "target_new": {"str": record["answers"][0]},
                        "target_true": {"str": " "+record["pred"]},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + " " + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(1, len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )
            

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
