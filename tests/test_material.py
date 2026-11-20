"""检查本地交换资料的案例是否互相关联。"""
import unittest

from inspect_fixture import read_session


class MaterialTests(unittest.TestCase):
    def test_linked_records(self):
        d = read_session()
        moves = [event for event in d["events"] if event["op"] == "move"]
        self.assertNotEqual(moves[0]["row"] % d["shard_count"], moves[0]["owner"])
        self.assertEqual(sum(item["op"] == "begin" and item["step"] == 2 for item in d["events"]), 2)
        packets = [item for item in d["events"] if item["op"] == "packet"]
        self.assertIn("inf", [v for item in packets for _, row in item["rows"] for v in row])
        self.assertGreater(d["row_count"], max(d["events"][-1]["rows"]))
