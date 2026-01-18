

import json
import numpy as np



def summary_metrics_for_LLM_judge(all_metrics):
    import pdb
    # pdb.set_trace()
    # print("这是metrics")
    # print(all_metrics)

    mean_metrics = dict()
    
    #for eval in ["pre", "post"]:
    for eval in ["post"]:
        if eval not in all_metrics[0]:
            continue
        mean_metrics[eval] = dict()
        for key in ["rewrite_acc", "rephrase_acc", "rewrite_contain_acc", "rephrase_contain_acc", "fluency", 'rewrite_ppl', 'ood_acc']:
            if key in all_metrics[0][eval].keys():
                if key == "fluency":  # 对 fluency 字典值进行均值计算
                    fluency_values = [
                        metric[eval][key]['ngram_entropy'] if 'ngram_entropy' in metric[eval][key] else np.nan
                        for metric in all_metrics if key in metric[eval]
                    ]
                    fluency_values = [val for val in fluency_values if not np.isnan(val)]  # 移除无效值
                    mean_metrics[eval][key] = np.mean(fluency_values) if fluency_values else np.nan
                else:  # 对其他评估指标进行均值计算
                    mean_metrics[eval][key] = np.mean(
                        [np.mean(metric[eval][key]) for metric in all_metrics if key in metric[eval]]
                    )
        for key in ["locality", "portability"]:
            if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                mean_metrics[eval][key] = dict()
                sub_keys = set()
                for metric in all_metrics:
                    if key in metric[eval]:
                        sub_keys.update(metric[eval][key].keys())

                for sub_key in sub_keys:
                    if "acc" not in sub_key:
                        continue

                    per_sample_means = []
                    for metric in all_metrics:
                        if key in metric[eval] and sub_key in metric[eval][key]:
                            vals = metric[eval][key][sub_key]
                            if isinstance(vals, list):
                                if len(vals) == 0:
                                    continue
                                # 若是 list of list，flatten
                                if isinstance(vals[0], list):
                                    vals = [v for v in vals if isinstance(v, list) and len(v) > 0]
                                    vals = [item for sublist in vals for item in sublist]
                                vals = [v for v in vals if isinstance(v, (float, int))]
                                if len(vals) > 0:
                                    per_sample_means.append(np.mean(vals))

                    if len(per_sample_means) > 0:
                        mean_metrics[eval][key][sub_key] = np.mean(per_sample_means)


    # mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])
    print("Metrics Summary: ", mean_metrics)



if __name__=="__main__":

    in_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/EasyEdit/examples/outputs/zsre_Llama-3.1-8B-Instruct_IKE_N=1319_Sequential=False_2025-08-06_22-54-21.json"
    output_path = "/mnt/geminisgceph1/geminicephfs/mmsearch-luban-universal/group_semantic/user_mylasong/EasyEdit/examples/outputs/zsre_Llama-3.1-8B-Instruct_IKE_N=1319_Sequential=False_2025-08-06_22-54-21_new.json"
    with open(in_path,"r") as f:
        all_metrics = json.load(f)
        for idx in range(len(all_metrics)):
            if 'locality' in all_metrics[idx]['post'].keys(): 
                for locality_key in ["Relation_Specificity"]:
                    
                    locality_result_all = []  # 用于存储“完全比较”结果
                    locality_result_pre = []  # 用于存储“基于 pre 长度”的比较结果
                    # 比较方式1: 完全比较
                    pre_tokens = all_metrics[idx]['pre']['locality'][f'{locality_key}_output']
                    post_tokens = all_metrics[idx]['post']['locality'][f'{locality_key}_output']
                    for pre_token_list, post_token_list in zip(pre_tokens, post_tokens):
                        locality_result_all.append(float(pre_token_list == post_token_list))

                    # 比较方式2：基于pre长度比较，并取平均值
                    # 基于 pre 长度进行比较：如果 post 比 pre 短，填充 False
                    pre_tokens = all_metrics[idx]['pre']['locality'][f'{locality_key}_output']
                    post_tokens = all_metrics[idx]['post']['locality'][f'{locality_key}_output']
                    for pre_token_list, post_token_list in zip(pre_tokens, post_tokens):
                        if len(post_token_list) < len(pre_token_list):
                            post_token_list = post_token_list + [-1] * (len(pre_token_list) - len(post_token_list))
                        correct_tokens_pre = 0
                        for ans, label in zip(pre_token_list, post_token_list):
                            if ans == label:  # 如果相等，计数
                                correct_tokens_pre += 1

                        average_locality_pre = correct_tokens_pre / len(pre_token_list) if len(pre_token_list) > 0 else 0
                        locality_result_pre.append(average_locality_pre)
                all_metrics[idx]['post']['locality'][f'{locality_key}_acc_all'] = locality_result_all
                all_metrics[idx]['post']['locality'][f'{locality_key}_acc_pre'] = locality_result_pre
        
        summary_metrics_for_LLM_judge(all_metrics)
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_path}")
        