import httpx
import pytest
from langchain_data_agent.network import DeadlineStream


class Chunks(httpx.SyncByteStream):
    closed=False
    def __iter__(self):yield b' ';yield b' '
    def close(self):self.closed=True


def test_keepalive_bytes_do_not_reset_whole_request_deadline(monkeypatch):
    ticks=iter([1,76]);monkeypatch.setattr('langchain_data_agent.network.time.monotonic',lambda:next(ticks))
    source=Chunks();stream=iter(DeadlineStream(source,75))
    assert next(stream)==b' '
    with pytest.raises(httpx.ReadTimeout):next(stream)
    assert source.closed
