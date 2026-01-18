import json
with open("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/test_filtered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/test_filtered_event.json", "w", encoding="utf-8") as f_out:
    for item in data:
        ev = item["event"]
        json_line = json.dumps({"sentence": ev}, ensure_ascii=False)
        f_out.write(json_line + "\n")