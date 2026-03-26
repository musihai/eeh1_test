import unittest

import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu


class TestTensorDictUtils(unittest.TestCase):
    def test_assign_non_tensor_data_dataproto_writes_meta_info(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            meta_info={},
        )

        tu.assign_non_tensor_data(proto, "use_dynamic_bsz", False)

        self.assertIn("use_dynamic_bsz", proto.meta_info)
        self.assertFalse(proto.meta_info["use_dynamic_bsz"])

    def test_contains_dataproto_checks_batch_non_tensor_and_meta_info(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            non_tensor_batch={"tools_kwargs": torch.tensor([0]).numpy()},
            meta_info={"use_dynamic_bsz": False},
        )

        self.assertTrue(tu.contains(proto, "x"))
        self.assertTrue(tu.contains(proto, "tools_kwargs"))
        self.assertTrue(tu.contains(proto, "use_dynamic_bsz"))
        self.assertFalse(tu.contains(proto, "missing"))

    def test_get_dataproto_meta_info_returns_default_or_value(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            meta_info={"use_dynamic_bsz": False},
        )

        self.assertFalse(tu.get(proto, "use_dynamic_bsz", True))
        self.assertEqual(tu.get(proto, "missing", "fallback"), "fallback")

    def test_pop_dataproto_missing_key_returns_default(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            meta_info={},
        )

        value = tu.pop(proto, key="no_lora_adapter", default=False)

        self.assertFalse(value)

    def test_pop_dataproto_meta_info_key_returns_value_and_removes_it(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            meta_info={"no_lora_adapter": True},
        )

        value = tu.pop(proto, key="no_lora_adapter", default=False)

        self.assertTrue(value)
        self.assertNotIn("no_lora_adapter", proto.meta_info)

    def test_pop_dataproto_batch_key_returns_tensor_and_removes_it(self) -> None:
        proto = DataProto(
            batch=TensorDict({"x": torch.tensor([1.0])}, batch_size=[1]),
            meta_info={},
        )

        value = tu.pop(proto, key="x", default=None)

        self.assertTrue(torch.equal(value, torch.tensor([1.0])))
        self.assertNotIn("x", proto.batch.keys())


if __name__ == "__main__":
    unittest.main()
