import json


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
data=load_data("/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/train.json")
filtered_data = [entry for entry in data if entry.get('fact', {}).get('qas')]
print(len(filtered_data))



# Save the filtered data to a new file
save_data(filtered_data, "/fs-computility/ai-shen/songxin/EasyEdit_for_concept/examples/dataset/ELKEN/train_filtered.json")