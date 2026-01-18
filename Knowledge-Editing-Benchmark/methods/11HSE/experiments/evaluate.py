import os
import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    HALLUQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
    KnownsDataset,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_hallu import compute_rewrite_quality_hallu

from memit import MEMITHyperParams
from memit.compute_z import get_module_input_output_at_words, compute_z
from memit.memit_main import apply_memit_to_model, get_context_templates
from memit.memit_rect_main import apply_memit_rect_to_model
from HSE import HSEHyperParams
from HSE.HSE_main import apply_HSE_to_model, get_cov
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.globals import *
from nse import NSEHyperParams
from nse.nse_main import apply_nse_to_model
from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "HSE": (HSEHyperParams, apply_HSE_to_model),
    "MEMIT_prune": (MEMITHyperParams, apply_memit_to_model),
    "MEMIT_rect": (MEMITHyperParams, apply_memit_rect_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    "pre": ("","")
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "hallu": (HALLUQADataset, compute_rewrite_quality_hallu),
}


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    topic: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]
    print(torch.cuda.is_available())
    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("/")[-1].split("_")[1])
                for x in alg_dir.iterdir()
                if str(x).split("/")[-1].split("_")[1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        if ds_name != "hallu":
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}_{model_name.strip('./')}_{ds_name}-{dataset_size_limit}-{num_edits}"
            if args.downstream_eval_steps > 0:
                run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}_{model_name.strip('./')}_{ds_name}-{dataset_size_limit}-{num_edits}_glue{str(args.downstream_eval_steps)}"
        else:
            run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}_{model_name.strip('./')}_{ds_name}-{topic}-{dataset_size_limit}-{num_edits}"
            
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    if "MEMIT" in alg_name:
    # Get run hyperparameters
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / "MEMIT" / hparams_fname
        )
    else:
        params_path = (
            run_dir / "params.json"
            if continue_from_run is not None
            else HPARAMS_DIR / alg_name / hparams_fname
        )
    if alg_name != 'pre':
        hparams = params_class.from_json(params_path)
        if not (run_dir / "params.json").exists():
            shutil.copyfile(params_path, run_dir / "params.json")
        print(f"Executing {alg_name} with parameters {hparams}")
    
    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name,ignore_mismatched_sizes=True).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        if "mistral" in model_name:
            tok.padding_side = 'right'
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")
    snips = AttributeSnippets(DATA_DIR) if not skip_generation_tests else None
    vec = get_tfidf_vectorizer(DATA_DIR) if not skip_generation_tests else None

    if num_edits > 1:
        assert ds_name != "cf", f"{ds_name} does not support multiple edits"

    ds_class, ds_eval_method = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, tok=tok, model=model, size=dataset_size_limit, topic=topic)
    if not dataset_size_limit:
        dataset_size_limit = len(ds)
        print('dataset_size_limit:',dataset_size_limit)
    eval_ds = KnownsDataset(DATA_DIR)
    
    # Get cache templates
    cache_template = None
    if use_cache:
        if any(alg in alg_name for alg in ["MEMIT", "MEMIT_prune", "MEMIT_rect"]):
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
            if ds_name == "hallu":
                cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_MEMIT"
                / f"{ds_name}_{topic}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        else:
            cache_template = (
                KV_DIR
                / f"{model_name.replace('/', '_')}_{alg_name}"
                / f"{ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz"
            )
        print(f"Will load cache from {cache_template}")

    if any(alg in alg_name for alg in ["HSE", "MEMIT_prune"]):
        # Iterate through dataset
        
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if hparams.model_name == "gpt2-xl":
            cov_sum = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
            fisher_matrix = None
        elif hparams.model_name in ["EleutherAI_gpt-j-6B","Llama3-8B","Mistral-7B-v0.3"]:
            cov_sum = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
            fisher_matrix = None

        del W_out


    glue_save_location = str(run_dir) + '/' + 'glue_eval/'
    os.makedirs(glue_save_location, exist_ok=True)
    cnt = 0
    for record_chunks in chunks(ds, num_edits):
        print("results_saved:",run_dir)
        case_result_template = str(run_dir / "{}_edits-case_{}.json")
        
        print(f"=================================================================={cnt+1}_edit==================================================================")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue
        
        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT","HSE", "MEMIT_prune"]) else dict()
        seq_args = dict(cov_sum=cov_sum) if any(alg in alg_name for alg in ["HSE"]) else dict()
        fim_args = dict(fisher_matrix=fisher_matrix) if any(alg in alg_name for alg in ["HSE"]) else dict()
        alpha_args = dict(time_list = [dataset_size_limit//num_edits, cnt])
        if cnt == 0 and args.downstream_eval_steps > 0:#do initial GLUE EVAL WITH ORIGINAL MODEL
            glue_results = {'edit_num': -1}

            out_file = glue_save_location + "base.json"
            
            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)

            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
        start = time()
        # if you want to eval pre-model perfomance
        if alg_name == "HSE":
            edited_model, cov_sum, fisher_matrix = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                **args_conserve_memory,
                **etc_args,
                **seq_args,
                **fim_args,
                **alpha_args
            )
        elif alg_name == "MEMIT_prune":
            if cnt == 0:
                edited_model, weights_copy = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    return_orig_weights=True,
                    **args_conserve_memory,
                    **etc_args,
                )
                # Initialize the upd_matrix dictionary
                upd_matrix = {}
            else:
                edited_model, _ = apply_algo(
                    model,
                    tok,
                    [
                        {"case_id": record["case_id"], **record["requested_rewrite"]}
                        for record in record_chunks
                    ],
                    hparams,
                    return_orig_weights=False,
                    **args_conserve_memory,
                    **etc_args,
                )
            if cnt == (dataset_size_limit/num_edits) - 1:
            # Calculate the weight update matrix
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        current_weight = nethook.get_parameter(model, k)
                        upd_matrix[k] = current_weight - v.to("cuda")
                        # Calculate max singular value of the original weight
                        _, S_orig, _ = torch.svd(v)
                        max_sigma = S_orig.max().item()

                        # Adjust the upd_matrix singular values
                        U_upd, S_upd, V_upd = torch.svd(upd_matrix[k])
                        adjusted_S = torch.where(
                            S_upd > max_sigma,
                            torch.log(S_upd) - torch.log(torch.tensor(max_sigma, device='cuda')) + max_sigma,
                            S_upd
                        )
                        upd_matrix[k] = torch.matmul(U_upd, torch.matmul(torch.diag(adjusted_S), V_upd.t()))

                # Apply the adjusted updates to the model
                with torch.no_grad():
                    for k in upd_matrix:
                        original_weight = nethook.get_parameter(model, k)
                        adjusted_weight = original_weight + upd_matrix[k]
                        original_weight.copy_(adjusted_weight)
        elif alg_name != 'pre':
            edited_model, _ = apply_algo(
                model,
                tok,
                [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks
                ],
                hparams,
                return_orig_weights=False,
                **args_conserve_memory,
                **etc_args,
            )
        else:
            edited_model = model
        exec_time = time() - start
        cnt+=1
        print("Execution took", exec_time)
        # Evaluate new model
    
        if args.downstream_eval_steps > 0 and cnt % args.downstream_eval_steps == 0:
            glue_results = {
                        'edit_num': cnt*num_edits,
                        'case_id': case_ids
                        }

            out_file = glue_save_location + "case_{}.json".format(record["case_id"])#stores the last case ID of the batch

            glue_eval = GLUEEval(model, tok, number_of_tests = 100)
            glue_results = glue_eval.evaluate(glue_results, out_file, nli_flag = True, sst_flag = True, cola_flag=True, rte_flag=True, mmlu_flag = True, mrpc_flag = True)
                    
            #store the individual overall result file
            output_filename = out_file.replace('.json', '_glue.json')
            with open(output_filename, "w") as f:
                json.dump(glue_results, f, indent=4)
    # if alg_name != 'pre':
    #     hs = get_module_input_output_at_words(
    #             edited_model,
    #             tok,
    #             hparams.layers[-1],
    #             context_templates=[request["template"] for request in eval_ds],
    #             words=[request["subject"] for request in eval_ds],
    #             module_template=hparams.layer_module_tmp,
    #             fact_token_strategy=hparams.fact_token,
    #         )[1].T
    #     torch.save(hs, "post_edit_hs_memit.pt")
    start = time()
    gen_test_vars = [snips, vec]
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue
        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": ds_eval_method(
                edited_model,
                tok,
                record,
                *(
                    gen_test_vars
                    if record["case_id"] % generation_test_interval == 0
                    else [None, None]
                ),  # Only test generation every generation_test_interval cases
            ),
        }
        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)

        # Restore original weights
        # with torch.no_grad():
        #     for k, v in weights_copy.items():
        #         nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["HSE","MEMIT_rect", "MEMIT_prune", "MEMIT", "ROME", "FT", "MEND","pre"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["mcf", "cf", "zsre","hallu"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--topic",
        choices=["all", "art", "business", "entertainment","event","geography","health","human","places","technology"],
        default=None,
        help="halluedit topic",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--downstream_eval_steps",
        type=int,
        default=0,
        help="If we want to do sequential editing or not",
    )
    
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.topic,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )
