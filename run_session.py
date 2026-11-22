"""按到达时间回放训练事件，驱动稀疏嵌入更新算子。

事件流即训练端：begin/packet/commit 推进全局步，move 登记行迁移，
lookup/export 读取嵌入，snapshot/restore 在提交边界持久化与恢复，
replay 由训练端重投某个 (step, attempt) 的完整提交（算子侧幂等）。
运行结果逐事件记录，可从中找出造成整步跳过的分片和行，以及下次
读取应该去哪一片。
"""
import json
from pathlib import Path

from sparse_embedding import SparseEmbeddingOperator

PROFILE_KEYS = ("row_count", "width", "shard_count", "expected_workers",
                "betas", "eps", "default", "records", "default_step", "steps")


class SessionRunner:
    """把到达事件翻译成算子调用，并保留重投所需的尝试记录。"""

    def __init__(self, profile):
        kwargs = {key: profile[key] for key in PROFILE_KEYS if key in profile}
        self.op = SparseEmbeddingOperator(**kwargs)
        self.attempts = {}     # (step, attempt) -> {"scale": s, "packets": [...]}
        self._current = None   # 当前打开尝试的 (step, attempt)
        self.saved = None
        self.results = []

    def run(self, events):
        for event in events:
            self.results.append(self.dispatch(event))
        return self.results

    def dispatch(self, event):
        op = event["op"]
        handler = getattr(self, f"_on_{op}", None)
        if handler is None:
            return {"event": event, "result": {"status": "unknown_event"}}
        return {"event": event, "result": handler(event)}

    # ------------------------------------------------------------------
    def _on_begin(self, event):
        key = (event["step"], event["attempt"])
        self.attempts[key] = {"scale": event["scale"], "packets": []}
        self._current = key
        return self.op.begin(event["step"], event["attempt"], event["scale"])

    def _on_packet(self, event):
        if self._current is not None:
            self.attempts[self._current]["packets"].append(
                (event["worker"], event["packet"], event["rows"]))
        return self.op.submit_packet(event["worker"], event["packet"], event["rows"])

    def _on_move(self, event):
        return self.op.move(event["row"], event["owner"], event["after_step"])

    def _on_commit(self, event):
        return self.op.commit()

    def _on_lookup(self, event):
        return self.op.lookup(event["rows"])

    def _on_export(self, event):
        return self.op.export(event["rows"])

    def _on_snapshot(self, event):
        # 快照必须可 JSON 序列化且数值精确往返
        self.saved = json.loads(json.dumps(self.op.snapshot()))
        return {"status": "saved", "committed_step": self.op.committed_step}

    def _on_restore(self, event):
        if self.saved is None:
            return {"status": "no_snapshot"}
        return self.op.restore(self.saved)

    def _on_replay(self, event):
        """训练端重投某次尝试的完整提交；已完成的步在算子侧幂等。"""
        key = (event["step"], event["attempt"])
        record = self.attempts.get(key)
        if record is None:
            return {"status": "unknown_attempt", "step": key[0], "attempt": key[1]}
        outcome = [self.op.begin(key[0], key[1], record["scale"])]
        for worker, packet, rows in record["packets"]:
            outcome.append(self.op.submit_packet(worker, packet, rows))
        outcome.append(self.op.commit())
        return outcome


def main():
    session = json.loads((Path(__file__).with_name("fixtures") / "session.json").read_bytes())
    runner = SessionRunner(session)
    report = runner.run(session["events"])
    print(json.dumps({"committed_step": runner.op.committed_step,
                      "materialized": runner.op.materialized_rows(),
                      "report": report}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
