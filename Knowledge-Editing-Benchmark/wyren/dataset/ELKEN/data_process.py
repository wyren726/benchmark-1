import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT = """Please extract one or more (subject, relation, object) triples from the following event.
Each triple must express a complete, standalone fact, with no redundancy or dependency on other triples.

Instructions:
If the event contains multiple independent facts, extract **multiple triples**, one per fact.
The **Subject** must be the main entity or actor involved in the fact.
The **Relation** should be a clear **predicate phrase** (verb or action-based phrase) that uniquely points to the **Object**.
The **Object** should contain the new or important information not repeated in the subject or relation.
Avoid vague or generic relations like "is involved in", "is associated with".
Do not merge multiple facts into a single triple.
If there are multiple pieces of information (e.g., place, time, role), consider extracting **multiple triples**.

Format each triple as follows:
Subject:
Prompt: (Concatenate subject and relation into a natural language phrase)
Target New: (Only the object, ideally as short as possible)

Example1:
Event: Serena Williams announces her retirement from professional tennis.
Output:
Subject: Serena Williams
Prompt: Serena Williams retires from
Target New: professional tennis

Example2:
Event: Pete Townshend is pursuing a degree in philosophy at the Royal College of Art.  
Output:  
Subject: Pete Townshend  
Prompt: Pete Townshend is pursuing the degree of  
Target New: philosophy  
Subject: Pete Townshend  
Prompt: Pete Townshend is studying at  
Target New: the Royal College of Art

Example3:
Event: Paul Wight founded NexGen Technologies in Bergen, appointing Martin Allen as CEO.  
Output:  
Subject: Paul Wight  
Prompt: Paul Wight founded  
Target New: NexGen Technologies  
Subject: NexGen Technologies
Prompt: The CEO of NexGen Technologies is 
Target New: Martin Allen

Event: {event}
Output:
"""

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def call_api(client, item):
    event = item["event"]
    case_id = item["case_id"]
    prompt = PROMPT.replace("{event}", event)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        output_text = response.choices[0].message.content
        print(output_text)
        triples = []
        current = {"subject": "", "prompt": "", "target_new": ""}

        for line in output_text.strip().splitlines():
            line = line.strip()
            if line.startswith("Subject:"):
                if current["subject"] and current["prompt"] and current["target_new"]:
                    triples.append(current)
                    current = {"subject": "", "prompt": "", "target_new": ""}
                current["subject"] = line[len("Subject:"):].strip()
            elif line.startswith("Prompt:"):
                current["prompt"] = line[len("Prompt:"):].strip()
            elif line.startswith("Target New:"):
                current["target_new"] = line[len("Target New:"):].strip()

        # 最后一个 triple 补上
        if current["subject"] and current["prompt"] and current["target_new"]:
            triples.append(current)

        item["triples"] = triples


        return item  # 返回带有提取信息的原始 item

    except Exception as e:
        item["triples"] = []
        return item


# 载入数据并添加 case_id
data = load_data("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/test.json")
for idx, item in enumerate(data):
    item["case_id"] = idx

client = OpenAI(
    base_url="https://api.claudeshop.top/v1",
    api_key="sk-P6weBeXWulmMUOY33Hn80veyIQWy4y3e4zOm8UZWhDkaDbWe"
)

# 多线程调用
results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(call_api, client, item): item for item in data}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        results.append(future.result())

# 按 case_id 恢复原始顺序
results_sorted = sorted(results, key=lambda x: x["case_id"])

# 保存到文件
with open("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/test_processed.json", "w", encoding="utf-8") as f:
    json.dump(results_sorted, f, ensure_ascii=False, indent=4)