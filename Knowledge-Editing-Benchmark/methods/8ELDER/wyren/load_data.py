from datasets import load_dataset_builder
ds_builder = load_dataset_builder("zjunlp/KnowEdit")

print(ds_builder.info.description)