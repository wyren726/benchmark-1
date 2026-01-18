import json

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def extract_fact_local_qas(data_item):
    local_qas = data_item.get("fact", {}).get("local_qas", [])
    result = []
    for qa in local_qas:
        question = qa.get("question", "")
        answer = qa.get("answer", {}).get("name", "")
        result.append(f"{question} {answer}")
    return result



data=load_data("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/train_filtered.json")
# 示例：如果 data 是一个列表
if isinstance(data, list):
    all_results = []
    for item in data:
        all_results.extend(extract_fact_local_qas(item))
else:
    # 如果是单个对象
    all_results = extract_fact_local_qas(data)

# 输出结果
for r in all_results:
    print(r)