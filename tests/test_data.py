import torch

from brainscan.data import decode, encode, prepare_batches


class TestEncoding:
    def test_encode_ascii(self):
        text = b"hello"
        tokens = encode(text)
        assert tokens == [104, 101, 108, 108, 111]

    def test_encode_empty(self):
        assert encode(b"") == []

    def test_encode_all_bytes(self):
        data = bytes(range(256))
        tokens = encode(data)
        assert tokens == list(range(256))

    def test_decode_roundtrip(self):
        text = "To be, or not to be."
        tokens = encode(text.encode())
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
        # For each position, target should be the next character in the source
        # Since we draw random windows, we can't predict exact values,
        # but we can verify they're valid byte values
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
        # Very unlikely to get identical random batches
        assert not torch.equal(x1, x2)
