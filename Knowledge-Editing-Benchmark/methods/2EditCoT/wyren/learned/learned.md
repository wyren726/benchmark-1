##### Python 代码中常见的单元格分隔符
```python
# %%
👆是 Python 代码中常见的单元格分隔符，主要用于支持交互式编程环境。

# 第一个单元格
import pandas as pd
import numpy as np

# %%
# 第二个单元格
data = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
print(data.head())

# %%
# 第三个单元格
result = data.mean()
print(result)
```

Git diff 友好意味着：
1. 只看 diff 就能理解代码变更
2. 合并冲突容易解决
3. 版本历史清晰干净
4. 适合团队协作和代码审查
这就是为什么许多专业团队在版本控制中更倾向于使用纯 Python 文件而不是 Jupyter Notebook 的原因。

##### 使用 Git 获取 diff
```bash
# 查看工作目录中的修改
git diff

# 查看暂存区的修改
git diff --staged
git diff --cached  # 同上

# 查看与特定提交的差异
git diff HEAD~1   # 与上一个提交比较
git diff commit_id1 commit_id2  # 比较两个提交

# 只查看某个文件的修改
git diff filename.py
git diff -- filename.py
```