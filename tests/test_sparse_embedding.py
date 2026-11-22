"""稀疏嵌入更新算子的行为测试。

DenseReference 是参照实现：物化整表、每个成功步推进所有行（即训练契约
的字面语义），仅用于小表核对。被测算子不得物化全表，靠惰性补齐达到同样
的数学结果；两边按同一 FP32 公式逐位一致。
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from sparse_embedding import SparseEmbeddingOperator
from run_session import SessionRunner

FIXTURE = json.loads(
    (Path(__file__).resolve().parent.parent / "fixtures" / "session.json").read_bytes())


def hparams_of(profile, t):
    hp = profile.get("steps", {}).get(str(t), profile["default_step"])
    return hp["lr"], hp["decay"]


def adamw_row(profile, init, grads, upto):
    """按训练契约把一行从记账步推进到 upto；grads: {t: 梯度}（缺省零梯度）。"""
    w = np.asarray(init["w"], dtype=np.float32)
    m = np.asarray(init["m"], dtype=np.float32)
    v = np.asarray(init["v"], dtype=np.float32)
    b1, b2 = profile["betas"]
    for t in range(init["step"] + 1, upto + 1):
        lr, decay = hparams_of(profile, t)
        g = np.asarray(grads.get(t, [0.0] * len(w)), dtype=np.float32)
        m = np.float32(b1) * m + np.float32(1 - b1) * g
        v = np.float32(b2) * v + np.float32(1 - b2) * (g * g)
        m_hat = m / np.float32(1.0 - b1 ** t)
        v_hat = v / np.float32(1.0 - b2 ** t)
        w = (w * np.float32(1.0 - lr * decay)
             - np.float32(lr) * (m_hat / (np.sqrt(v_hat) + np.float32(profile["eps"]))))
    return w


def make_operator(**overrides):
    width = overrides.get("width", 2)
    profile = {
        "row_count": 64,
        "width": width,
        "shard_count": 3,
        "expected_workers": ["w-east", "w-west"],
        "betas": [0.8, 0.95],
        "eps": 1e-7,
        "default": {"w": [0.4] + [-0.5] * (width - 1),
                    "m": [0.0] * width, "v": [0.0] * width, "step": 0},
        "default_step": {"lr": 0.02, "decay": 0.1},
        "steps": {},
        "records": {},
    }
    profile.update(overrides)
    return SparseEmbeddingOperator(**profile), profile


def commit_empty(op, t, workers=("w-east", "w-west")):
    """全体声明无贡献并成功提交一步。"""
    op.begin(t, 1, 1)
    for w in workers:
        op.submit_packet(w, 0, [])
    return op.commit()


class DenseReference:
    """物化整表、每个成功步推进所有行的参照实现（仅小规模核对用）。"""

    def __init__(self, profile):
        self.profile = profile
        n, d = profile["row_count"], profile["width"]
        self.w = np.tile(np.asarray(profile["default"]["w"], np.float32), (n, 1))
        self.m = np.tile(np.asarray(profile["default"]["m"], np.float32), (n, 1))
        self.v = np.tile(np.asarray(profile["default"]["v"], np.float32), (n, 1))
        self.row_step = np.full(n, int(profile["default"]["step"]), dtype=np.int64)
        for key, rec in profile.get("records", {}).items():
            r = int(key)
            self.w[r] = rec["w"]
            self.m[r] = rec["m"]
            self.v[r] = rec["v"]
            self.row_step[r] = int(rec["step"])
        self.rank = {w: i for i, w in enumerate(profile["expected_workers"])}
        self.committed = 0

    def _step(self, r, g, t):
        b1, b2 = self.profile["betas"]
        lr, decay = hparams_of(self.profile, t)
        self.m[r] = np.float32(b1) * self.m[r] + np.float32(1 - b1) * g
        self.v[r] = np.float32(b2) * self.v[r] + np.float32(1 - b2) * (g * g)
        m_hat = self.m[r] / np.float32(1.0 - b1 ** t)
        v_hat = self.v[r] / np.float32(1.0 - b2 ** t)
        self.w[r] = (self.w[r] * np.float32(1.0 - lr * decay)
                     - np.float32(lr) * (m_hat / (np.sqrt(v_hat) + np.float32(self.profile["eps"]))))
        self.row_step[r] = t

    def commit(self, packets, scale):
        """推进一个成功步；packets 为去重后的有效集合 [(worker, pid, rows)]。"""
        merged = {}
        for _, _, rows in sorted(packets, key=lambda p: (self.rank[p[0]], p[1])):
            for row, grad in rows:
                unscaled = np.asarray(grad, dtype=np.float32) / np.float32(scale)
                merged[row] = unscaled if row not in merged else merged[row] + unscaled
        t = self.committed + 1
        zero = np.zeros(self.w.shape[1], np.float32)
        for r in range(self.w.shape[0]):
            while self.row_step[r] < t - 1:
                self._step(r, zero, self.row_step[r] + 1)
            self._step(r, merged.get(r, zero), t)
        self.committed = t


class FixtureSessionTests(unittest.TestCase):
    """端到端回放 fixtures/session.json，数值与独立逐步模拟逐位一致。"""

    @classmethod
    def setUpClass(cls):
        cls.runner = SessionRunner(FIXTURE)
        cls.results = cls.runner.run(FIXTURE["events"])

    def test_step_outcomes(self):
        r = self.results
        self.assertEqual(r[5]["result"]["status"], "committed")
        self.assertEqual(r[5]["result"]["committed_step"], 1)
        # 第 1 步在行 17 的旧归属片（17 % 3 = 2）结算，提交后迁往新片 1
        self.assertEqual(r[5]["result"]["touched"], {"2": [17]})
        self.assertEqual(r[5]["result"]["moves_applied"],
                         [{"row": 17, "from": 2, "to": 1}])
        # 含 "inf" 的尝试整步作废，成功步号不推进
        self.assertEqual(r[10]["result"]["status"], "voided")
        self.assertEqual(r[10]["result"]["reason"], "non_finite")
        self.assertEqual(r[10]["result"]["committed_step"], 1)
        self.assertEqual(r[16]["result"]["status"], "committed")
        self.assertEqual(r[16]["result"]["committed_step"], 2)
        self.assertEqual([j["status"] for j in self.runner.op.journal],
                         ["committed", "voided", "committed"])

    def test_skip_diagnostics_name_shard_and_row(self):
        voided = self.results[10]["result"]
        # 造成整步跳过的是分片 0 上的行 12（12 % 3 = 0）
        self.assertEqual(voided["non_finite"],
                         {"0": {"contribution": [12], "merged": [12]}})
        self.assertEqual(voided["participating_shards"], [0, 1])

    def test_mid_run_lookup_uses_latest_committed_step(self):
        rows = self.results[11]["result"]["rows"]
        # 迁移已生效：下次读取行 17 应去分片 1；行 12 仍在分片 0
        self.assertEqual(rows[17]["shard"], 1)
        self.assertEqual(rows[12]["shard"], 0)
        # 行 17 第 1 步合并梯度：(8+2-4)/8, (4+1+0)/8, (0+0+8)/8
        np.testing.assert_array_equal(
            np.asarray(rows[17]["w"], np.float32),
            adamw_row(FIXTURE, FIXTURE["records"]["17"], {1: [0.75, 0.625, 1.0]}, 1))
        # 行 12 未出现：默认值补齐第 1 步的零梯度 AdamW（权重衰减照算）
        np.testing.assert_array_equal(
            np.asarray(rows[12]["w"], np.float32),
            adamw_row(FIXTURE, FIXTURE["default"], {}, 1))

    def test_replay_after_restore_does_not_double_update(self):
        replayed = self.results[19]["result"]
        self.assertEqual([r["status"] for r in replayed],
                         ["stale", "stale", "stale", "stale", "no_open_attempt"])
        self.assertEqual(self.runner.op.committed_step, 2)

    def test_final_lookup_matches_continuous_run(self):
        rows = self.results[20]["result"]["rows"]
        self.assertEqual((rows[17]["shard"], rows[12]["shard"], rows[9999999]["shard"]),
                         (1, 0, 0))
        # 行 17 第 2 步无贡献（该次尝试 w-east 为空包），只补零梯度步
        np.testing.assert_array_equal(
            np.asarray(rows[17]["w"], np.float32),
            adamw_row(FIXTURE, FIXTURE["records"]["17"], {1: [0.75, 0.625, 1.0]}, 2))
        # 行 12 第 2 步梯度 [2,0,0]/2 = [1,0,0]（重投用了新缩放系数）
        np.testing.assert_array_equal(
            np.asarray(rows[12]["w"], np.float32),
            adamw_row(FIXTURE, FIXTURE["default"], {2: [1.0, 0.0, 0.0]}, 2))
        # 冷行 9999999：默认值补齐两个成功步的零梯度数学行为
        np.testing.assert_array_equal(
            np.asarray(rows[9999999]["w"], np.float32),
            adamw_row(FIXTURE, FIXTURE["default"], {}, 2))
        # 导出与查询口径一致
        exported = self.runner.op.export([17, 12, 9999999])["rows"]
        for r in (17, 12, 9999999):
            self.assertEqual(exported[r], rows[r]["w"])

    def test_no_dense_materialization(self):
        # 一千万行的表只物化被触及/查询的 3 行
        self.assertEqual(self.runner.op.materialized_rows(),
                         {0: [12, 9999999], 1: [17], 2: []})


class MergeTests(unittest.TestCase):
    def test_same_row_twice_in_packet_counts_both(self):
        op, profile = make_operator(width=1)
        op.begin(1, 1, 2)
        op.submit_packet("w-east", 0, [(3, [2.0]), (3, [6.0])])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "committed")
        # 先反缩放再求和：2/2 + 6/2 = 4
        np.testing.assert_array_equal(
            np.asarray(op.lookup([3])["rows"][3]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [4.0]}, 1))

    def test_merge_order_follows_declaration_not_arrival(self):
        op, profile = make_operator(width=1)
        op.begin(1, 1, 1)
        # w-west 先到，但声明顺序 w-east 在前
        op.submit_packet("w-west", 0, [(3, [-1e20]), (3, [1.0])])
        op.submit_packet("w-east", 0, [(3, [1e20])])
        self.assertEqual(op.commit()["status"], "committed")
        # 声明序：((1e20 + -1e20) + 1) = 1；到达序会得 0
        np.testing.assert_array_equal(
            np.asarray(op.lookup([3])["rows"][3]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0]}, 1))

    def test_unscale_before_sum_in_fp32(self):
        op, profile = make_operator(width=1)
        op.begin(1, 1, 3)
        op.submit_packet("w-east", 0, [(5, [1.0]), (5, [2.0])])
        op.submit_packet("w-west", 0, [(5, [4.0])])
        self.assertEqual(op.commit()["status"], "committed")
        g = np.float32(1.0) / np.float32(3) + np.float32(2.0) / np.float32(3) \
            + np.float32(4.0) / np.float32(3)
        np.testing.assert_array_equal(
            np.asarray(op.lookup([5])["rows"][5]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [g]}, 1))

    def test_identical_resubmission_is_deduplicated(self):
        op, profile = make_operator()
        op.begin(1, 1, 2)
        op.submit_packet("w-east", 0, [(3, [2.0, 4.0])])
        res = op.submit_packet("w-east", 0, [(3, [2.0, 4.0])])
        self.assertEqual(res["status"], "duplicate")
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "committed")
        # 只计一次：[2,4]/2
        np.testing.assert_array_equal(
            np.asarray(op.lookup([3])["rows"][3]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, 2.0]}, 1))

    def test_conflicting_resubmission_voids_attempt(self):
        op, _ = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(1, [1.0, 1.0])])
        self.assertEqual(op.submit_packet("w-east", 0, [(1, [2.0, 2.0])])["status"],
                         "conflict")
        op.submit_packet("w-west", 0, [])
        before = op.snapshot()
        result = op.commit()
        self.assertEqual(result["status"], "voided")
        self.assertEqual(result["reason"], "conflict")
        self.assertEqual(result["conflicts"], [{"worker": "w-east", "packet": 0}])
        self.assertEqual(op.snapshot(), before)

    def test_missing_worker_declaration_voids_attempt(self):
        op, _ = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [])
        result = op.commit()
        self.assertEqual(result["status"], "voided")
        self.assertEqual(result["reason"], "missing_workers")
        self.assertEqual(result["missing_workers"], ["w-west"])
        self.assertEqual(op.committed_step, 0)


class VoidAndRetryTests(unittest.TestCase):
    def test_inf_contribution_voids_step_and_freezes_state(self):
        op, profile = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(5, [1.0, 2.0])])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "committed")
        before = op.snapshot()
        op.begin(2, 1, 1)
        op.submit_packet("w-east", 0, [(5, ["inf", 0.0])])
        op.submit_packet("w-west", 0, [])
        result = op.commit()
        self.assertEqual(result["status"], "voided")
        self.assertEqual(result["reason"], "non_finite")
        shard = str(5 % 3)
        self.assertEqual(result["non_finite"][shard]["contribution"], [5])
        # 权重、动量与记账步都未推进
        self.assertEqual(op.snapshot(), before)
        self.assertEqual(op.committed_step, 1)

    def test_non_finite_merged_value_from_finite_parts_voids(self):
        op, _ = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(7, [3.0e38, 0.0])])
        op.submit_packet("w-west", 0, [(7, [3.0e38, 0.0])])
        result = op.commit()
        self.assertEqual(result["status"], "voided")
        shard = str(7 % 3)
        self.assertEqual(result["non_finite"][shard]["contribution"], [])
        self.assertEqual(result["non_finite"][shard]["merged"], [7])
        self.assertEqual(op.committed_step, 0)

    def test_zero_scale_voids_instead_of_crashing(self):
        op, _ = make_operator()
        op.begin(1, 1, 0)
        op.submit_packet("w-east", 0, [(1, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "voided")
        self.assertEqual(op.committed_step, 0)

    def test_retry_same_logical_step_with_new_scale(self):
        op, profile = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(5, [1.0, 2.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        op.begin(2, 1, 1)
        op.submit_packet("w-east", 0, [(5, ["inf", 0.0])])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "voided")
        # 重投同一逻辑步，使用新的缩放系数
        op.begin(2, 2, 4)
        op.submit_packet("w-east", 0, [(5, [2.0, 8.0])])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "committed")
        np.testing.assert_array_equal(
            np.asarray(op.lookup([5])["rows"][5]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, 2.0], 2: [0.5, 2.0]}, 2))


class LazyZeroGradTests(unittest.TestCase):
    STEPS = {"2": {"lr": 0.05, "decay": 0.25}, "3": {"lr": 0.01, "decay": 0.2}}

    def test_cold_row_backfills_elapsed_steps(self):
        op, profile = make_operator(steps=self.STEPS, row_count=100000)
        for t in (1, 2, 3):
            self.assertEqual(commit_empty(op, t)["status"], "committed")
        got = op.lookup([99999])["rows"][99999]
        self.assertEqual(got["step"], 3)
        self.assertEqual(got["shard"], 99999 % 3)
        np.testing.assert_array_equal(
            np.asarray(got["w"], np.float32),
            adamw_row(profile, profile["default"], {}, 3))

    def test_momentum_carries_through_zero_grad_steps(self):
        op, profile = make_operator(steps=self.STEPS)
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(9, [1.0, -1.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        commit_empty(op, 2)
        commit_empty(op, 3)
        # 零梯度步里动量与权重衰减继续按全局步作用
        np.testing.assert_array_equal(
            np.asarray(op.lookup([9])["rows"][9]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, -1.0]}, 3))

    def test_lookup_is_idempotent_and_does_not_advance_step(self):
        op, _ = make_operator()
        commit_empty(op, 1)
        first = op.lookup([9])
        second = op.lookup([9])
        self.assertEqual(first, second)
        self.assertEqual(op.committed_step, 1)

    def test_large_table_stays_sparse(self):
        op, _ = make_operator(row_count=100000)
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(3, [1.0, 1.0]), (40000, [2.0, 2.0])])
        op.submit_packet("w-west", 0, [(7, [1.0, 1.0])])
        op.commit()
        op.lookup([3, 99999])
        total = sum(len(rows) for rows in op.materialized_rows().values())
        self.assertLessEqual(total, 4)
        self.assertLess(total, 100000)


class MoveTests(unittest.TestCase):
    def test_mid_batch_move_old_owner_collects_remaining_contributions(self):
        op, profile = make_operator()
        op.begin(1, 1, 2)
        op.submit_packet("w-east", 0, [(4, [2.0, 2.0])])
        res = op.move(4, 2, 1)
        self.assertEqual(res["applied"], [])       # 提交后才生效
        self.assertEqual(op.owner_of(4), 1)         # 在途批次仍属旧片（4 % 3 = 1）
        op.submit_packet("w-west", 0, [(4, [4.0, 0.0])])
        result = op.commit()
        self.assertEqual(result["status"], "committed")
        self.assertEqual(result["touched"], {"1": [4]})   # 旧归属收齐剩余贡献
        self.assertEqual(result["moves_applied"], [{"row": 4, "from": 1, "to": 2}])
        self.assertEqual(op.owner_of(4), 2)
        self.assertEqual(op.materialized_rows()[2], [4])
        # 切换后的查询看到同一行完整历史：合并梯度 [2,2]/2+[4,0]/2 = [3,1]
        got = op.lookup([4])["rows"][4]
        self.assertEqual(got["shard"], 2)
        np.testing.assert_array_equal(
            np.asarray(got["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [3.0, 1.0]}, 1))

    def test_move_of_never_materialized_row(self):
        op, profile = make_operator()
        res = op.move(4, 2, 0)                      # 立即生效，无状态可迁
        self.assertEqual(res["applied"], [])
        self.assertEqual(op.owner_of(4), 2)
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(4, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        self.assertEqual(op.materialized_rows()[2], [4])
        np.testing.assert_array_equal(
            np.asarray(op.lookup([4])["rows"][4]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, 1.0]}, 1))

    def test_past_due_move_transfers_immediately(self):
        op, profile = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(4, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        res = op.move(4, 2, 1)                      # after_step 已到，立即搬迁
        self.assertEqual(res["applied"], [{"row": 4, "from": 1, "to": 2}])
        self.assertEqual(op.materialized_rows()[2], [4])
        np.testing.assert_array_equal(
            np.asarray(op.lookup([4])["rows"][4]["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, 1.0]}, 1))

    def test_future_move_applies_at_its_commit_boundary(self):
        op, _ = make_operator()
        commit_empty(op, 1)
        op.move(4, 2, 3)
        self.assertEqual(op.owner_of(4), 1)
        self.assertEqual(op.owner_of(4, at_step=3), 2)
        commit_empty(op, 2)
        self.assertEqual(op.owner_of(4), 1)
        result = commit_empty(op, 3)
        self.assertEqual(result["moves_applied"], [])
        self.assertEqual(op.owner_of(4), 2)

    def test_new_shard_never_clobbers_old_state(self):
        op, profile = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(4, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        op.move(4, 2, 1)
        # 模拟迁移未落实的异常：状态滞留在旧片，归属表已指向新片
        op._shards[1][4] = op._shards[2].pop(4)
        got = op.lookup([4])["rows"][4]
        # 查询找回旧片状态并搬迁，绝不用初始值覆盖
        self.assertEqual(got["shard"], 2)
        np.testing.assert_array_equal(
            np.asarray(got["w"], np.float32),
            adamw_row(profile, profile["default"], {1: [1.0, 1.0]}, 1))
        self.assertEqual(op.materialized_rows()[2], [4])


class SnapshotRestoreTests(unittest.TestCase):
    def _script_head(self, op):
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(1, [1.0, 0.5])])
        op.submit_packet("w-west", 0, [(2, [0.0, 1.0])])
        op.commit()
        op.move(1, 0, 2)
        op.begin(2, 1, 1)
        op.submit_packet("w-east", 0, [(1, ["inf", 0.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()                                  # 作废
        op.begin(2, 2, 2)
        op.submit_packet("w-east", 0, [(1, [4.0, 2.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()

    def _script_tail(self, op):
        op.begin(3, 1, 1)
        op.submit_packet("w-east", 0, [(2, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [(1, [0.5, 0.25])])
        op.commit()
        commit_empty(op, 4)
        return op.export([0, 1, 2, 63])

    def test_snapshot_roundtrip_and_replay(self):
        op, _ = make_operator()
        self._script_head(op)
        saved = json.loads(json.dumps(op.snapshot()))   # 快照须可 JSON 序列化
        continuous = self._script_tail(op)
        op.restore(saved)
        self.assertEqual(op.committed_step, 2)
        # 恢复后重投已完成的步：全部幂等，不二次更新
        self.assertEqual(op.begin(2, 2, 2)["status"], "stale")
        self.assertEqual(op.submit_packet("w-east", 0, [(1, [4.0, 2.0])])["status"],
                         "stale")
        self.assertEqual(op.commit()["status"], "no_open_attempt")
        # 恢复后继续训练，结果与连续运行逐位一致
        self.assertEqual(self._script_tail(op), continuous)

    def test_export_matches_lookup_after_restore(self):
        op, _ = make_operator()
        self._script_head(op)
        saved = json.loads(json.dumps(op.snapshot()))
        before = self._script_tail(op)
        op.restore(saved)
        self._script_tail(op)
        exported = op.export([0, 1, 2, 63])
        self.assertEqual(exported, before)
        looked_up = op.lookup([0, 1, 2, 63])["rows"]
        for r, w in exported["rows"].items():
            self.assertEqual(w, looked_up[r]["w"])


class ProtocolTests(unittest.TestCase):
    def test_begin_with_step_gap_is_rejected(self):
        op, _ = make_operator()
        res = op.begin(3, 1, 1)
        self.assertEqual(res["status"], "invalid_step")
        self.assertEqual(res["expected_step"], 1)

    def test_new_begin_supersedes_open_attempt(self):
        op, _ = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(1, [1.0, 1.0])])
        self.assertEqual(op.begin(1, 2, 1)["status"], "open")
        self.assertEqual(op.journal[-1]["reason"], "superseded")
        op.submit_packet("w-east", 0, [])
        op.submit_packet("w-west", 0, [])
        self.assertEqual(op.commit()["status"], "committed")

    def test_stale_step_is_not_reapplied(self):
        op, profile = make_operator()
        op.begin(1, 1, 1)
        op.submit_packet("w-east", 0, [(1, [1.0, 1.0])])
        op.submit_packet("w-west", 0, [])
        op.commit()
        before = op.snapshot()
        self.assertEqual(op.begin(1, 1, 1)["status"], "stale")
        self.assertEqual(op.submit_packet("w-east", 0, [(1, [9.0, 9.0])])["status"],
                         "stale")
        self.assertEqual(op.commit()["status"], "no_open_attempt")
        self.assertEqual(op.snapshot(), before)


def random_profile(rng):
    width = int(rng.integers(1, 5))
    profile = {
        "row_count": int(rng.integers(4, 13)),
        "width": width,
        "shard_count": int(rng.integers(1, 5)),
        "expected_workers": [f"w{i}" for i in range(int(rng.integers(1, 4)))],
        "betas": [0.8, 0.95],
        "eps": 1e-7,
        "default": {"w": [float(x) for x in rng.normal(0, 0.5, width)],
                    "m": [0.0] * width, "v": [0.0] * width, "step": 0},
        "default_step": {"lr": 0.02, "decay": 0.1},
        "steps": {},
        "records": {},
    }
    n_steps = int(rng.integers(3, 8))
    for t in range(1, n_steps + 2):
        if rng.random() < 0.4:
            profile["steps"][str(t)] = {"lr": float(rng.uniform(0.005, 0.08)),
                                        "decay": float(rng.uniform(0.0, 0.3))}
    for r in range(profile["row_count"]):
        if rng.random() < 0.25:
            profile["records"][str(r)] = {
                "w": [float(x) for x in rng.normal(0, 0.5, width)],
                "m": [0.0] * width, "v": [0.0] * width, "step": 0}
    return profile, n_steps


class PropertyTests(unittest.TestCase):
    """随机事件流下，惰性分片算子与物化全表的参照实现逐位一致。"""

    def build_script(self, rng, profile, n_steps):
        script = []
        width = profile["width"]
        workers = profile["expected_workers"]
        for t in range(1, n_steps + 1):
            attempt = 1
            if rng.random() < 0.4:
                kind = ("inf", "conflict", "missing")[int(rng.integers(0, 3))]
                scale = float(rng.choice([1.0, 2.0, 8.0]))
                script.append(("begin", t, attempt, scale))
                if kind == "inf":
                    for w in workers:
                        r = int(rng.integers(0, profile["row_count"]))
                        script.append(("packet", w, 0,
                                       [(r, ["inf"] + [0.0] * (width - 1))]))
                elif kind == "conflict":
                    r = int(rng.integers(0, profile["row_count"]))
                    script.append(("packet", workers[0], 0, [(r, [1.0] * width)]))
                    script.append(("packet", workers[0], 0, [(r, [2.0] * width)]))
                    for w in workers[1:]:
                        script.append(("packet", w, 0, []))
                else:
                    for w in workers[:-1]:
                        script.append(("packet", w, 0, []))
                script.append(("commit", "voided", None, None))
                script.append(("lookup",
                               [int(rng.integers(0, profile["row_count"]))]))
                attempt += 1
            scale = float(rng.choice([1.0, 2.0, 8.0, 0.5]))
            script.append(("begin", t, attempt, scale))
            effective = []
            for w in workers:
                for pid in range(int(rng.integers(1, 3))):
                    rows = [(int(rng.integers(0, profile["row_count"])),
                             [float(x) for x in rng.normal(0, 2, width)])
                            for _ in range(int(rng.integers(0, 4)))]
                    script.append(("packet", w, pid, rows))
                    effective.append((w, pid, rows))
                    if rng.random() < 0.25:
                        # 同身份同内容重投：算子去重，参照不计入
                        script.append(("packet", w, pid,
                                       [(r, list(g)) for r, g in rows]))
            if rng.random() < 0.4:
                script.append(("move",
                               int(rng.integers(0, profile["row_count"])),
                               int(rng.integers(0, profile["shard_count"])),
                               int(rng.choice([t, t, t + 1, max(0, t - 1)]))))
            script.append(("commit", "committed", effective, scale))
            if rng.random() < 0.5:
                script.append(("lookup",
                               [int(rng.integers(0, profile["row_count"]))
                                for _ in range(3)]))
        return script

    def execute(self, script, op, ref, profile, spec_log=None):
        lookups = []
        for action in script:
            kind = action[0]
            if kind == "begin":
                _, t, attempt, scale = action
                self.assertEqual(op.begin(t, attempt, scale)["status"], "open")
            elif kind == "packet":
                _, w, pid, rows = action
                op.submit_packet(w, pid, rows)
            elif kind == "move":
                _, r, owner, after = action
                op.move(r, owner, after)
                if spec_log is not None:
                    spec_log.append((r, owner, after))
                    for at in range(0, after + 3):
                        expect = r % profile["shard_count"]
                        for mr, mo, ms in spec_log:
                            if mr == r and ms <= at:
                                expect = mo
                        self.assertEqual(op.owner_of(r, at_step=at), expect)
            elif kind == "commit":
                _, expect, effective, scale = action
                self.assertEqual(op.commit()["status"], expect)
                if expect == "committed":
                    ref.commit(effective, scale)
                self.assertEqual(op.committed_step, ref.committed)
                self.assert_all_rows(op, ref, profile)
                self.assert_ownership(op)
            else:
                _, rows = action
                got = op.lookup(rows)
                for r in rows:
                    np.testing.assert_array_equal(
                        np.asarray(got["rows"][r]["w"], np.float32), ref.w[r])
                lookups.append(got)
        return lookups

    def assert_all_rows(self, op, ref, profile):
        got = op.lookup(list(range(profile["row_count"])))["rows"]
        for r in range(profile["row_count"]):
            np.testing.assert_array_equal(
                np.asarray(got[r]["w"], np.float32), ref.w[r],
                err_msg=f"row {r} @ committed {ref.committed}")

    def assert_ownership(self, op):
        for s, shard in enumerate(op._shards):
            for r in shard:
                self.assertEqual(op.owner_of(r), s)

    @staticmethod
    def last_committed_attempt(script):
        current = None
        for action in script:
            if action[0] == "begin":
                current = (action[1], action[2], action[3])
            elif action[0] == "commit" and action[1] == "committed":
                t, attempt, scale = current
                last = (t, attempt, scale, action[2])
        return last

    def run_one(self, seed):
        rng = np.random.default_rng(seed)
        profile, n_steps = random_profile(rng)
        script = self.build_script(rng, profile, n_steps)
        commits = [i for i, a in enumerate(script)
                   if a[0] == "commit" and a[1] == "committed"]
        snap_end = commits[int(rng.integers(0, len(commits)))] + 1

        op = SparseEmbeddingOperator(**profile)
        ref = DenseReference(profile)
        spec_log = []
        self.execute(script[:snap_end], op, ref, profile, spec_log=spec_log)
        saved = json.loads(json.dumps(op.snapshot()))
        ref_saved = copy.deepcopy(ref)
        last = self.last_committed_attempt(script[:snap_end])
        pass1 = self.execute(script[snap_end:], op, ref, profile, spec_log=spec_log)
        final1 = op.lookup(list(range(profile["row_count"])))

        # 从提交边界快照恢复：重投已完成的步不得二次更新
        op.restore(saved)
        t, attempt, scale, packets = last
        self.assertEqual(op.begin(t, attempt, scale)["status"], "stale")
        for w, pid, rows in packets:
            self.assertEqual(op.submit_packet(w, pid, rows)["status"], "stale")
        self.assertEqual(op.commit()["status"], "no_open_attempt")
        # 重放快照之后的事件流，结果须与连续运行逐位一致
        pass2 = self.execute(script[snap_end:], op, ref_saved, profile)
        final2 = op.lookup(list(range(profile["row_count"])))
        self.assertEqual(final1, final2)
        self.assertEqual(len(pass1), len(pass2))
        for first, second in zip(pass1, pass2):
            self.assertEqual(first, second)

    def test_lazy_matches_dense_randomized(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                self.run_one(seed)


if __name__ == "__main__":
    unittest.main()
