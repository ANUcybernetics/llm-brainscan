import torch

from brainscan.data import TextBuffer, prepare_batches


class TestTextBuffer:
    def test_initial_data(self):
        buf = TextBuffer(b"hello")
        assert buf.data == b"hello"
        assert len(buf) == 5

    def test_append(self):
        buf = TextBuffer(b"hello")
        buf.append(" world")
        assert buf.data == b"hello world"
        assert len(buf) == 11

    def test_append_unicode(self):
        buf = TextBuffer(b"")
        buf.append("caf\u00e9")
        assert buf.data == "caf\u00e9".encode("utf-8")

    def test_multiple_appends(self):
        buf = TextBuffer(b"a")
        for c in "bcdef":
            buf.append(c)
        assert buf.data == b"abcdef"

    def test_empty_initial(self):
        buf = TextBuffer(b"")
        assert len(buf) == 0
        assert buf.data == b""
        buf.append("x")
        assert buf.data == b"x"


class TestTextBufferPersistence:
    def test_persists_to_file(self, tmp_path):
        path = tmp_path / "spoken.txt"
        buf = TextBuffer(b"base", persist_path=path)
        buf.append(" hello")
        buf.append(" world")
        assert path.read_bytes() == b" hello world"

    def test_loads_persisted_on_init(self, tmp_path):
        path = tmp_path / "spoken.txt"
        path.write_bytes(b" extra")
        buf = TextBuffer(b"base", persist_path=path)
        assert buf.data == b"base extra"

    def test_appends_to_existing_file(self, tmp_path):
        path = tmp_path / "spoken.txt"
        path.write_bytes(b"old ")
        buf = TextBuffer(b"base", persist_path=path)
        buf.append("new")
        assert path.read_bytes() == b"old new"

    def test_no_persist_without_path(self):
        buf = TextBuffer(b"hello")
        buf.append(" world")
        assert buf.data == b"hello world"


class TestPrepareBatchesWithTextBuffer:
    def test_works_with_text_buffer(self, device):
        buf = TextBuffer(b"a" * 1000)
        x, y = prepare_batches(buf, batch_size=4, sequence_len=16, device=device)
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_includes_appended_data(self, device):
        buf = TextBuffer(b"\x00" * 1000)
        buf.append("\xff" * 500)
        x, _ = prepare_batches(buf, batch_size=64, sequence_len=8, device=device)
        unique_vals = set(x.flatten().tolist())
        assert 0 in unique_vals
        assert 195 in unique_vals or 191 in unique_vals or 255 in unique_vals
