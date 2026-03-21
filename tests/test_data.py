import torch

from brainscan.data import decode, prepare_batches


class TestDecoding:
    def test_decode_ascii(self):
        assert decode([104, 101, 108, 108, 111]) == "hello"

    def test_decode_roundtrip(self):
        text = "To be, or not to be."
        tokens = list(text.encode())
        assert decode(tokens) == text

    def test_decode_invalid_utf8(self):
        result = decode([0xFF, 0xFE])
        assert isinstance(result, str)


class TestPrepareBatches:
    def test_shape(self, device):
        data = b"a" * 1000
        x, y = prepare_batches(data, batch_size=4, sequence_len=16, device=device)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_dtype(self, device):
        data = b"a" * 1000
        x, y = prepare_batches(data, batch_size=4, sequence_len=16, device=device)
        assert x.dtype == torch.int64
        assert y.dtype == torch.int64

    def test_targets_are_shifted_inputs(self, device):
        data = bytes(range(256)) * 10
        x, y = prepare_batches(data, batch_size=1, sequence_len=8, device=device)
        assert (x >= 0).all() and (x < 256).all()
        assert (y >= 0).all() and (y < 256).all()

    def test_device_placement(self, device):
        data = b"test data " * 100
        x, y = prepare_batches(data, batch_size=2, sequence_len=8, device=device)
        assert x.device.type == device.type
        assert y.device.type == device.type

    def test_different_batches_are_random(self, device):
        data = bytes(range(256)) * 100
        x1, _ = prepare_batches(data, batch_size=4, sequence_len=16, device=device)
        x2, _ = prepare_batches(data, batch_size=4, sequence_len=16, device=device)
        assert not torch.equal(x1, x2)
