from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from scripts import show_chunk


@pytest.mark.parametrize("row,code,stdout,stderr", [
    ((7, 3, 2, "chunk text"), 0,
     "Chunk ID: 7\nDocument ID: 3\nChunk index: 2\n\nchunk text\n", ""),
    (None, 1, "", "Chunk 7 not found\n"),
])
def test_lookup_uses_quoted_table_and_bound_id(monkeypatch, capsys, row, code, stdout, stderr):
    calls = []
    cursor = SimpleNamespace(execute=lambda *args: calls.append(args), fetchone=lambda: row)
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))
    monkeypatch.setattr(show_chunk.psycopg, "connect", lambda _: nullcontext(connection))
    monkeypatch.setattr(show_chunk, "settings", SimpleNamespace(
        db_url="unused", chunks_table='chunks"archive',
    ))
    monkeypatch.setattr("sys.argv", ["show_chunk.py", "7"])

    assert show_chunk.main() == code
    query, params = calls[0]
    assert len(calls) == 1
    assert query.as_string() == (
        'SELECT id, document_id, chunk_index, content FROM "chunks""archive" WHERE id = %s'
    )
    assert params == (7,)
    captured = capsys.readouterr()
    assert captured.out == stdout
    assert captured.err == stderr


@pytest.mark.parametrize("args", [[], ["not-an-integer"], ["7; DROP TABLE chunks"]])
def test_invalid_arguments_never_connect(monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["show_chunk.py", *args])
    monkeypatch.setattr(show_chunk.psycopg, "connect", lambda _: pytest.fail("unexpected DB call"))
    with pytest.raises(SystemExit) as error:
        show_chunk.main()
    assert error.value.code == 2


def test_database_failure_is_not_reported_as_missing_chunk(monkeypatch, capsys):
    def fail(_):
        raise show_chunk.psycopg.OperationalError("database unavailable")

    monkeypatch.setattr("sys.argv", ["show_chunk.py", "7"])
    monkeypatch.setattr(show_chunk.psycopg, "connect", fail)
    with pytest.raises(show_chunk.psycopg.OperationalError):
        show_chunk.main()
    assert capsys.readouterr().out == ""
