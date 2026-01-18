# 大标题
## 模型
llama3.1-8b:`/home/wyren/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

## dataset
zsre:
```
/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/zsre/zsre_train_10000.json
/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/zsre/ZsRE-test-all.json
```
wiki_counterfact:
```
/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/wiki_counterfact/train_cf.json
/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/wiki_counterfact/test_cf.json
/home/wyren/Knowledge-Editing-Benchmark/wyren/dataset/wiki_counterfact/wiki_counterfact-test-all-sentence.json
```
ELKEN

zsre数据集的结构如下：
```python
# DatasetDict({
#     train: Dataset({
#         features: ['input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'],
#         num_rows: 200
#     })
# })
```

wiki_counterfact数据集的结构如下：
```python
# DatasetDict({
#     train: Dataset({
#         features: ['input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'],
#         num_rows: 200
#     })
# })
```

ELKEN数据集的结构如下：
```python
# DatasetDict({
#     train: Dataset({
#         features: ['input', 'context', 'answers', 'length', 'dataset', 'language', 'all_classes', '_id'],
#         num_rows: 200
#     })
# })
```
## evaluate

shenmeyisi，不懂 我不知道你要把这个偷到那里去，为什么要开一个新的 我这里不是用的好好的？这个不是easyedit的评估函数，是我微调的评估函数，对，是啊。你都没告诉我我新开的这个myproject是干什么的，我也不知道把这个代码偷出去干什么，为什么不直接整理到我原来的项目里面，可是最终还是要合并到一个项目里的吧。没理解，我原来那个项目还有实验要继续跑，我微调的方法还要继续找超参数，为什么是最终迁移到myproject地下，我想的是全归并到我的longtext？那不都是我写的吗。这个项目是我搭起来的。
对啊 这个架构是我搭的，什么怎么搭的。评估的代码是longbench的，utils也是longbench的,我就是把他放在他该在的位置上。
之前哪个readme用来放我要跑的命令和要做的实验。所以呢
等一下
你先告诉我我最终的项目应该归并到原来的里面 好吧 好吧 你还是小看我了

    原始分数            finetune            konwledge edit
                lora lora-pro adalora   alphaedit  ...



1. 代码要合并同类项，框架要规定好。想清楚接数据的接口要长什么样，评估要接受什么样的数据。
2. 最终要出一个什么样的表要想清楚，要包括哪些字段？

## scripts
```bash

```