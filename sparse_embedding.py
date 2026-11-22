"""稀疏嵌入更新算子。

词与商品共用的嵌入表按逻辑行分片存放。训练端每个全局步由若干工作进程
交来包：词索引（行号）、向量梯度与本步损失缩放系数。算子在逻辑行上合并
同一步内的重复命中，按训练契约执行带全局步含义的 AdamW 更新，并在提交
边界之间提供可直接查询的嵌入结果。

语义要点（与 contract.md 对齐）：

- 行状态 (w, m, v, step) 惰性物化：未出现的行不初始化、不扫表。每个成功
  全局步对每行都相当于输入零梯度执行一次 AdamW；查询或再次命中冷行时按
  成功步逐行补齐经过的数学行为，学习率、权重衰减、偏差修正都按全局成功
  步编号计算。
- 包以 (step, attempt, worker, packet) 识别：同身份同内容去重，同身份而
  内容冲突则该次尝试作废；同一行在同一包内反复出现属于独立项，全部计入。
- 步内归并 FP32：先反缩放再求和，顺序按 (工作进程声明顺序, 包号, 包内
  顺序)。任一参与分片的有效贡献或合并值非有限，整步作废：权重、动量与
  成功步号都不推进；重投同一逻辑步可用新的缩放系数。
- 行归属默认 row % shard_count；move 在指定成功步提交以后生效。迁移请求
  在上一批贡献尚未齐时送到，旧归属仍收齐该批剩余贡献；切换后查询看到同
  一行已经完成的历史，新片不会用初始值覆盖旧片状态。
- 快照发生在成功提交边界，可 JSON 序列化；恢复后重复提交已经完成的步不
  会二次更新，查询或导出指定逻辑行与连续运行一致。
"""
from __future__ import annotations

import numpy as np

__all__ = ["SparseEmbeddingOperator", "ContractError"]


class ContractError(ValueError):
    """资料或训练契约参数不合法。"""


def _parse_scalar(value):
    """交换格式里的标量：数字或字符串（"inf" 表示正无穷）。"""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("inf", "+inf", "infinity", "+infinity"):
            return np.float32(np.inf)
        if text in ("-inf", "-infinity"):
            return np.float32(-np.inf)
        if text == "nan":
            return np.float32(np.nan)
        return np.float32(float(text))
    return np.float32(value)


class _RowState:
    """一行的物化状态：向量、两个动量与记账步。"""

    __slots__ = ("w", "m", "v", "step")

    def __init__(self, w, m, v, step):
        self.w = np.asarray(w, dtype=np.float32).copy()
        self.m = np.asarray(m, dtype=np.float32).copy()
        self.v = np.asarray(v, dtype=np.float32).copy()
        self.step = int(step)

    def as_dict(self):
        return {"w": self.w.tolist(), "m": self.m.tolist(),
                "v": self.v.tolist(), "step": self.step}


class _Contract:
    """训练契约：超参与按成功步编号的学习率/权重衰减表，FP32 计算。"""

    def __init__(self, betas, eps, default_step, steps):
        betas = list(betas)
        if len(betas) != 2 or not all(0.0 < float(b) < 1.0 for b in betas):
            raise ContractError("betas 需要两个 (0,1) 内的数")
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.default_step = (float(default_step["lr"]), float(default_step["decay"]))
        self.steps = {int(t): (float(h["lr"]), float(h["decay"]))
                      for t, h in (steps or {}).items()}

    def hparams(self, t):
        """成功步 t 的 (lr, decay)；未列出的步用 default_step。"""
        return self.steps.get(t, self.default_step)

    def adamw(self, st, g, t):
        """对一行执行成功步 t 的 AdamW，FP32 就地更新。

        m=β1*m+(1-β1)*g；v=β2*v+(1-β2)*g²；
        w=w*(1-lr*decay)-lr*(m/(1-β1^t))/(sqrt(v/(1-β2^t))+eps)。
        标量系数按 float64 求值后转 FP32，张量运算全程 FP32。
        """
        lr, decay = self.hparams(t)
        b1 = np.float32(self.beta1)
        b2 = np.float32(self.beta2)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            st.m[...] = b1 * st.m + np.float32(1.0 - self.beta1) * g
            st.v[...] = b2 * st.v + np.float32(1.0 - self.beta2) * (g * g)
            m_hat = st.m / np.float32(1.0 - self.beta1 ** t)
            v_hat = st.v / np.float32(1.0 - self.beta2 ** t)
            st.w[...] = (st.w * np.float32(1.0 - lr * decay)
                         - np.float32(lr) * (m_hat / (np.sqrt(v_hat) + np.float32(self.eps))))
        st.step = t


class _Attempt:
    """一次尝试（逻辑步 + attempt）收到的包。"""

    __slots__ = ("step", "attempt", "scale", "packets", "conflicts",
                 "invalid", "unknown_workers")

    def __init__(self, step, attempt, scale):
        self.step = step
        self.attempt = attempt
        self.scale = scale
        self.packets = {}        # (worker, packet) -> (规范内容, [(row, fp32 向量)])
        self.conflicts = []      # 同身份而内容冲突的包
        self.invalid = False     # 包内容无法解析或越界
        self.unknown_workers = set()


class SparseEmbeddingOperator:
    """分片稀疏嵌入更新算子，供 Python 训练端调用。"""

    def __init__(self, *, row_count, width, shard_count, expected_workers,
                 betas, eps, default, default_step, steps=None, records=None):
        self.row_count = int(row_count)
        self.width = int(width)
        self.shard_count = int(shard_count)
        if self.row_count <= 0 or self.width <= 0 or self.shard_count <= 0:
            raise ContractError("row_count/width/shard_count 必须为正")
        self.expected_workers = [str(w) for w in expected_workers]
        self._worker_rank = {w: i for i, w in enumerate(self.expected_workers)}
        self.contract = _Contract(betas, eps, default_step, steps)
        self._default = self._row_state(default)
        self._records = {self._row_id(k): self._row_state(v)
                         for k, v in (records or {}).items()}
        self._shards = [dict() for _ in range(self.shard_count)]
        self._moves = []                 # (row, owner, after_step)，按到达顺序
        self._moves_by_row = {}
        self.committed_step = 0
        self._open = None
        self.journal = []                # 每次尝试的结算结果（运行诊断）
        self._zero = np.zeros(self.width, dtype=np.float32)

    # ------------------------------------------------------------------
    # 基本校验与行状态
    # ------------------------------------------------------------------
    def _row_id(self, row):
        try:
            r = int(row)
        except (TypeError, ValueError):
            raise ContractError(f"行号越界: {row!r}") from None
        if r != row and str(row) != str(r):
            raise ContractError(f"行号越界: {row!r}")
        if not 0 <= r < self.row_count:
            raise ContractError(f"行号越界: {row!r}")
        return r

    def _row_state(self, rec):
        st = _RowState(rec["w"], rec["m"], rec["v"], rec["step"])
        if st.w.shape != (self.width,) or st.m.shape != (self.width,) \
                or st.v.shape != (self.width,):
            raise ContractError("记录宽度与 width 不一致")
        return st

    def owner_of(self, row, at_step=None):
        """逻辑行在指定成功步（默认当前已提交步）应归属的分片。

        归属表默认 row % shard_count；moves 在指定成功步提交以后生效，
        同一行多次迁移按到达顺序后者覆盖前者。
        """
        row = self._row_id(row)
        if at_step is None:
            at_step = self.committed_step
        owner = row % self.shard_count
        for after_step, o in self._moves_by_row.get(row, ()):
            if after_step <= at_step:
                owner = o
        return owner

    def _locate(self, row):
        """返回行状态当前物化所在的分片；未物化返回 None。"""
        for s in range(self.shard_count):
            if row in self._shards[s]:
                return s
        return None

    def _materialize(self, row):
        """在归属片上取出行状态；必要时从 records/default 惰性初始化。

        若状态仍滞留在旧片（迁移尚未执行），先搬迁到归属片，绝不用
        初始值覆盖旧片已有状态。
        """
        shard = self.owner_of(row)
        st = self._shards[shard].get(row)
        if st is None:
            holder = self._locate(row)
            if holder is not None:
                st = self._shards[holder].pop(row)
                self._shards[shard][row] = st
            else:
                src = self._records.get(row, self._default)
                st = _RowState(src.w, src.m, src.v, src.step)
                self._shards[shard][row] = st
        return shard, st

    def _catch_up(self, st, target):
        """把一行按成功步逐步补齐零梯度 AdamW 到 target（不扫整表）。"""
        while st.step < target:
            self.contract.adamw(st, self._zero, st.step + 1)

    def materialized_rows(self):
        """诊断用：各分片当前物化的行号（算子从不物化全表）。"""
        return {s: sorted(shard) for s, shard in enumerate(self._shards)}

    # ------------------------------------------------------------------
    # 训练端：步 / 尝试协议
    # ------------------------------------------------------------------
    def begin(self, step, attempt, scale):
        """开启逻辑步 step 的第 attempt 次尝试，scale 为本步损失缩放系数。"""
        step = int(step)
        attempt = int(attempt)
        if step <= self.committed_step:
            # 已完成的步（含恢复后重投）：不二次更新
            return {"status": "stale", "step": step, "attempt": attempt,
                    "committed_step": self.committed_step}
        if step != self.committed_step + 1:
            return {"status": "invalid_step", "step": step, "attempt": attempt,
                    "expected_step": self.committed_step + 1,
                    "committed_step": self.committed_step}
        if self._open is not None:
            superseded = self._open
            self.journal.append({"status": "voided", "reason": "superseded",
                                 "step": superseded.step, "attempt": superseded.attempt,
                                 "committed_step": self.committed_step})
        self._open = _Attempt(step, attempt, _parse_scalar(scale))
        return {"status": "open", "step": step, "attempt": attempt,
                "committed_step": self.committed_step}

    def submit_packet(self, worker, packet, rows):
        """工作进程交来一个包：[(行号, 梯度向量), ...]，允许空包声明无贡献。"""
        if self._open is None:
            return {"status": "stale", "reason": "no_open_attempt",
                    "committed_step": self.committed_step}
        at = self._open
        worker = str(worker)
        key = (worker, int(packet))
        canon_rows = []
        parsed_rows = []
        try:
            for r, vals in rows:
                rid = self._row_id(r)
                vals = list(vals)
                if len(vals) != self.width:
                    raise ContractError("梯度宽度与 width 不一致")
                parsed = [_parse_scalar(v) for v in vals]
                canon_rows.append((rid, tuple("nan" if np.isnan(x) else float(x)
                                              for x in parsed)))
                parsed_rows.append((rid, np.asarray(parsed, dtype=np.float32)))
        except (ContractError, ValueError, TypeError):
            at.invalid = True
            return {"status": "invalid", "step": at.step, "attempt": at.attempt}
        canon = tuple(canon_rows)
        if key in at.packets:
            if at.packets[key][0] == canon:
                return {"status": "duplicate", "step": at.step, "attempt": at.attempt}
            at.conflicts.append({"worker": worker, "packet": key[1]})
            return {"status": "conflict", "step": at.step, "attempt": at.attempt}
        if worker not in self._worker_rank:
            at.unknown_workers.add(worker)
        at.packets[key] = (canon, parsed_rows)
        return {"status": "accepted", "step": at.step, "attempt": at.attempt}

    def move(self, row, owner, after_step):
        """登记行迁移：after_step 成功提交以后 row 归属 owner 分片。"""
        row = self._row_id(row)
        owner = int(owner)
        after_step = int(after_step)
        if not 0 <= owner < self.shard_count:
            raise ContractError(f"分片越界: {owner!r}")
        self._moves.append((row, owner, after_step))
        self._moves_by_row.setdefault(row, []).append((after_step, owner))
        applied = []
        if after_step <= self.committed_step:
            # 已到期的迁移立即落实（提交边界之间归属即切换）
            applied = [a for a in self._apply_due_moves() if a["row"] == row]
        return {"status": "registered", "row": row, "owner": owner,
                "after_step": after_step, "applied": applied,
                "committed_step": self.committed_step}

    def commit(self):
        """结算当前尝试：全部声明到齐且贡献与合并值有限才推进成功步。"""
        if self._open is None:
            return {"status": "no_open_attempt",
                    "committed_step": self.committed_step}
        at = self._open
        self._open = None
        base = {"step": at.step, "attempt": at.attempt, "scale": float(at.scale),
                "committed_step": self.committed_step}

        declared = {w for (w, _) in at.packets}
        missing = [w for w in self.expected_workers if w not in declared]
        if at.invalid or at.conflicts or missing:
            result = {**base, "status": "voided",
                      "reason": ("invalid_contribution" if at.invalid else
                                 "conflict" if at.conflicts else "missing_workers"),
                      "conflicts": list(at.conflicts),
                      "missing_workers": missing,
                      "unknown_workers": sorted(at.unknown_workers)}
            self.journal.append(result)
            return result

        # 步内归并：先反缩放（FP32），再按 (工作进程声明顺序, 包号, 包内顺序) 求和。
        # 贡献按当前已提交步的归属路由到分片；在途批次的旧归属因此收齐剩余贡献。
        order = sorted(at.packets.items(),
                       key=lambda kv: (self._worker_rank.get(kv[0][0],
                                                             len(self._worker_rank)),
                                       kv[0][1]))
        merged = {}                # row -> [shard, fp32 向量]
        participating = set()
        non_finite = {}            # shard -> {"contribution": set, "merged": set}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for _identity, (_canon, parsed_rows) in order:
                for rid, grad in parsed_rows:
                    shard = self.owner_of(rid)
                    participating.add(shard)
                    unscaled = grad / at.scale
                    entry = merged.get(rid)
                    if entry is None:
                        merged[rid] = [shard, unscaled]
                    else:
                        entry[1] = entry[1] + unscaled
                    if not np.all(np.isfinite(unscaled)):
                        non_finite.setdefault(shard, {"contribution": set(),
                                                      "merged": set()})
                        non_finite[shard]["contribution"].add(rid)
            for rid, (shard, vec) in merged.items():
                if not np.all(np.isfinite(vec)):
                    non_finite.setdefault(shard, {"contribution": set(),
                                                  "merged": set()})
                    non_finite[shard]["merged"].add(rid)

        if non_finite:
            # 整步作废：权重、动量与成功步号都不推进，可用新缩放系数重投。
            result = {**base, "status": "voided", "reason": "non_finite",
                      "non_finite": {str(s): {k: sorted(v) for k, v in parts.items()}
                                     for s, parts in sorted(non_finite.items())},
                      "participating_shards": sorted(participating)}
            self.journal.append(result)
            return result

        # 应用更新：先补齐到已提交步，再执行本步带梯度的 AdamW。
        touched = {}
        for rid, (shard, vec) in merged.items():
            s, st = self._materialize(rid)
            self._catch_up(st, self.committed_step)
            self.contract.adamw(st, vec, at.step)
            touched.setdefault(s, set()).add(rid)
        self.committed_step = at.step
        moves_applied = self._apply_due_moves()
        result = {**base, "status": "committed",
                  "committed_step": self.committed_step,
                  "touched": {str(s): sorted(rows) for s, rows in sorted(touched.items())},
                  "moves_applied": moves_applied,
                  "participating_shards": sorted(participating),
                  "unknown_workers": sorted(at.unknown_workers)}
        self.journal.append(result)
        return result

    def _apply_due_moves(self):
        """落实所有 after_step <= committed_step 的迁移，返回实际搬迁记录。"""
        applied = []
        for row in list(self._moves_by_row):
            dst = self.owner_of(row)
            src = self._locate(row)
            if src is not None and src != dst:
                self._shards[dst][row] = self._shards[src].pop(row)
                applied.append({"row": row, "from": src, "to": dst})
        return applied

    # ------------------------------------------------------------------
    # 查询 / 导出
    # ------------------------------------------------------------------
    def lookup(self, rows):
        """按当时最近成功步读取嵌入；冷行先补齐经过的零梯度数学行为。

        返回每行当前应去读取的分片（下次读取的位置）与更新后的权重。
        """
        out = {}
        for row in rows:
            rid = self._row_id(row)
            shard, st = self._materialize(rid)
            self._catch_up(st, self.committed_step)
            out[rid] = {"shard": shard, "step": st.step, "w": st.w.tolist()}
        return {"committed_step": self.committed_step, "rows": out}

    def export(self, rows):
        """导出指定逻辑行更新后的权重，口径与连续运行一致。"""
        found = self.lookup(rows)
        return {"committed_step": found["committed_step"],
                "rows": {rid: info["w"] for rid, info in found["rows"].items()}}

    # ------------------------------------------------------------------
    # 快照 / 恢复（提交边界）
    # ------------------------------------------------------------------
    def snapshot(self):
        """在成功提交边界导出可 JSON 序列化的快照（FP32 值精确往返）。"""
        return {"committed_step": self.committed_step,
                "moves": [[r, o, s] for r, o, s in self._moves],
                "shards": {str(s): {str(r): st.as_dict()
                                    for r, st in shard.items()}
                           for s, shard in enumerate(self._shards)}}

    def restore(self, snap):
        """从提交边界快照恢复；之后重复提交已完成的步不会二次更新。"""
        shards = snap["shards"]
        if len(shards) != self.shard_count:
            raise ContractError("快照分片数与当前算子不一致")
        self.committed_step = int(snap["committed_step"])
        self._moves = [(int(r), int(o), int(s)) for r, o, s in snap["moves"]]
        self._moves_by_row = {}
        for row, owner, after_step in self._moves:
            self._moves_by_row.setdefault(row, []).append((after_step, owner))
        self._shards = [dict() for _ in range(self.shard_count)]
        for s in range(self.shard_count):
            for r, rec in shards.get(str(s), {}).items():
                self._shards[s][int(r)] = self._row_state(rec)
        self._open = None
        return {"status": "restored", "committed_step": self.committed_step}
