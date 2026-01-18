Example command:

```python
python edit_rag.py \
  --model_path /path/to/llama3-8b \
  --retriever_type contriever-ms \  
  --retriever_path /path/to/contriever-msmarco \
  --dataset_path /path/to/test_cf.json \
  --memory_path /path/to/wiki_counterfact-test-all-sentence.json \
  --top_k 5 \
  --eval_metric contain \
  --summary \
  --edit_scene single
```
