from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook
from collections import Counter

from .memit_hparams import MEMITHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    fisher_matrix: torch.Tensor,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    print("Computing right vector (v)")

    # Tokenize target and forget into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    forget_ids = tok(request["target_true"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    if forget_ids[0] == tok.bos_token_id or forget_ids[0] == tok.unk_token_id:
        forget_ids = forget_ids[1:]
    #only > half of token different, need to forget
    print('target_ids:',target_ids)
    print('forget_ids:',forget_ids)
    target_set = Counter(target_ids.tolist())
    forget_set = Counter(forget_ids.tolist())
    both_ids = target_set & forget_set
    target_num = sum(target_set.values())
    forget_num = sum(forget_set.values())
    both_num = sum(both_ids.values())
    # less same token for target -> need to forget
    forget_cond = (both_num < target_num/2) and (both_num < forget_num/2)
    
    print('forget_cond',forget_cond)
    # Compile list of rewriting ,forgetting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    forgetting_prompts = [
        context.format(request["prompt"]) + tok.decode(forget_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    all_prompts = rewriting_prompts + forgetting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    # Compute forgetting targets
    forgetting_targets = torch.tensor(-100, device="cuda").repeat(
        len(forgetting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(forgetting_prompts)):
        ex_len = input_tok["attention_mask"][i + len(rewriting_prompts)].sum()
        forgetting_targets[i, ex_len - len(forget_ids) : ex_len] = forget_ids
        
    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    fim_list = []
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting and forgetting targets
        output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]
        full_old_repr = output[len(rewriting_prompts):len(rewriting_prompts) + len(forgetting_prompts)]
        
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        log_probs_forget = torch.log(1-torch.softmax(ln_f(full_old_repr) @ lm_w + lm_b, dim=2) + np.exp(-20))
        #print('log_probs_forget:', log_probs_forget)
        loss1 = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        loss2 = torch.gather(
            log_probs_forget,
            2,
            torch.where(forgetting_targets != -100, forgetting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        
        mask_rewriting = (rewriting_targets != -100).float()
        nll_loss_each = -(loss1 * mask_rewriting.to(loss1.device)).sum(1) / target_ids.size(0)
        nll_loss = (1 - hparams.forget_factor) * nll_loss_each.mean()
        mask_forgetting = (forgetting_targets != -100).float()
        forget_loss_each = -(loss2 * mask_forgetting).sum(1) / forget_ids.size(0)
        forget_loss = hparams.forget_factor * forget_loss_each.mean()
        # Aggregate total losses
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        
        if isinstance(fisher_matrix,torch.Tensor):
            fisher_min = torch.min(fisher_matrix)
            fisher_max = torch.max(fisher_matrix)
            fisher_normalized = (fisher_matrix - fisher_min) / (fisher_max - fisher_min)
            fisher_loss = 0.5 * hparams.fisher_factor * torch.dot(delta, torch.mul(delta, fisher_normalized.to('cuda')))
            print('fisher_loss:',fisher_loss)
        else:
            fisher_loss = torch.zeros_like(nll_loss)
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        # if it == hparams.v_num_grad_steps - 1:
        #     fisher_matrix = torch.stack(fisher_matrix).mean(dim=0)
        #     weight_decay = 0.5 * hparams.v_weight_decay torch.dot(delta, torch.mv(fisher_matrix, delta))
        # else:
        #     weight_decay = 0.0
        # mistral_fisher_values=tensor([6.5414e-05, 9.0736e-05, 9.3510e-05,  ..., 1.5951e-02, 3.5141e-02,6.6430e-02], device='cuda:0')
        if forget_loss > 1e-3 and forget_cond:
            loss = nll_loss + forget_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device) + fisher_loss.to(nll_loss.device)
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(forget_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break
        else:
            loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device) + fisher_loss.to(nll_loss.device)
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']['str']}] "
                f"{torch.exp(-nll_loss_each).mean().item()}"
            )
            if loss < 5e-2:
                break
        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        grad = delta.grad.data.clone()
        fim_list.append(grad.pow(2))
        opt.step()
       
        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    # Average the gradients squared to get the Fisher matrix approximation
    if fim_list:
        if isinstance(fisher_matrix,torch.Tensor):
            fisher_matrix = fisher_matrix + torch.stack(fim_list).mean(dim=0).to(fisher_matrix.device)
        else:
            fisher_matrix = torch.stack(fim_list).mean(dim=0)
        print("fisher_matrix:",torch.sort(fisher_matrix))
    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target, fisher_matrix


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
