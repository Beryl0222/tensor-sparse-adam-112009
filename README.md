# 嵌入训练交换格式

本项目资料在 `fixtures/session.json`，采用合成数据。记录字段和数值口径见 [资料协议](contract.md)。

使用 Python 3.11 和固定依赖。新环境先执行 `python3 -m pip install -r requirements.txt`。

运行 `python3 inspect_fixture.py` 可以核对资料文件、内容指纹和顶层记录结构。执行 `python3 -m unittest discover -s tests -v` 检查资料中的逻辑引用与输入关系。

## 稀疏嵌入更新算子

`sparse_embedding.py` 提供 `SparseEmbeddingOperator`：词与商品共用的嵌入表按逻辑行
分片惰性存放，训练端每个全局步交来工作进程的包（词索引、向量梯度、本步损失缩放
系数），算子按逻辑行合并重复命中后执行带全局步含义的 AdamW 更新。

```python
from sparse_embedding import SparseEmbeddingOperator

op = SparseEmbeddingOperator(**profile)          # profile 即资料中的表声明
op.begin(step=1, attempt=1, scale=8)             # 开启逻辑步 1 的第 1 次尝试
op.submit_packet("w-east", 0, [(17, [8, 4, 0]), (17, [2, 1, 0])])
op.submit_packet("w-west", 0, [(17, [-4, 0, 8])])
op.submit_packet("w-edge", 0, [])                # 允许空包声明无贡献
op.move(row=17, owner=1, after_step=1)           # 步 1 提交后行 17 迁往分片 1
result = op.commit()                             # 结算：committed 或 voided
op.lookup([17, 12])                              # 按最近成功步读取，含下次读取的分片
op.export([17])                                  # 导出更新后的权重
saved = op.snapshot()                            # 提交边界快照，可 JSON 序列化
op.restore(saved)                                # 恢复；重投已完成的步不会二次更新
```

行为要点：

- 行状态 (w, m, v, step) 惰性物化，未出现的行不初始化、不扫表；迟到查询与冷行
  再次命中时按成功步逐行补齐零梯度 AdamW（学习率、权重衰减、偏差修正都按全局
  成功步编号计算）。
- 步内归并 FP32：先反缩放再求和，顺序按 (工作进程声明顺序, 包号, 包内顺序)；
  同一行在同一包内反复出现属于独立项，全部计入。同身份同内容的包去重，同身份
  内容冲突作废该次尝试。
- 任一参与分片的有效贡献或合并值非有限则整步作废：权重、动量与成功步号都不
  推进，可用新的缩放系数重投同一逻辑步。作废结果里的 `non_finite` 列出造成
  跳过的分片和行。
- 行归属默认 `row % shard_count`；`move` 在指定成功步提交以后生效。在途批次仍
  由旧归属收齐剩余贡献，切换后查询看到同一行完整历史，新片不会用初始值覆盖
  旧片状态。

## 会话回放

`python3 run_session.py` 按到达时间回放 `fixtures/session.json` 的事件流
（begin/packet/move/commit/lookup/snapshot/restore/replay），输出每步结算结果、
造成整步跳过的分片与行、以及每次查询应去读取的分片。

## 测试

`python3 -m unittest discover -s tests -v` 除资料检查外，还覆盖归并与缩放口径、
整步作废与重投、惰性零梯度补齐、行迁移、快照恢复幂等，并用物化全表的参照实现
（仅限小表）对惰性算子做随机化逐位对拍。
