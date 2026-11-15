"""Coordinator crash recovery, unit level: the durable request log and coordinator-epoch fencing."""

import time

import pytest

from pllm.proto import pipeline_pb2 as pb
from pllm.request_log import FencedError, RequestLog
from pllm.stage_executor import StageExecutor


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "requests.db")


def test_unfinished_requests_survive_reopen(db):
    log = RequestLog(db)
    log.take_over()
    log.start("a", "wire-a", [1, 2, 3], 4, time.time() + 60)
    log.start("b", "wire-b", [7], 2, time.time() + 60)
    log.record([("a", 0, 10), ("a", 1, 11), ("b", 0, 20)])
    log.record([("b", 1, 21)], [("b", pb.COMPLETED, "")])
    log.close()  # "crash": a new incarnation opens the same file

    log2 = RequestLog(db)
    log2.take_over()
    (a,) = log2.unfinished()
    assert (a.client_id, a.wire_id, a.prompt, a.max_new, a.tokens) == ("a", "wire-a", [1, 2, 3], 4, [10, 11])
    b = log2.lookup("b")
    assert b.status == pb.COMPLETED and b.tokens == [20, 21]


def test_token_rows_are_idempotent_and_gaps_are_never_resumed(db):
    log = RequestLog(db)
    log.take_over()
    log.start("a", "w", [1], 8, time.time() + 60)
    log.record([("a", 0, 5), ("a", 1, 6)])
    log.record([("a", 1, 6)])            # a step re-committed after a replay: no duplicate row
    assert log.lookup("a").tokens == [5, 6]
    log.db.execute("INSERT INTO tokens VALUES('a', 3, 9)")  # impossible gap: stop at the gap
    assert log.lookup("a").tokens == [5, 6]


def test_take_over_fences_the_previous_owner(db):
    old = RequestLog(db)
    e1 = old.take_over()
    old.start("a", "w", [1], 4, time.time() + 60)
    new = RequestLog(db)
    e2 = new.take_over()
    assert e2 > e1
    with pytest.raises(FencedError):
        old.record([("a", 0, 1)])
    with pytest.raises(FencedError):
        old.finish("a", pb.FAILED)
    assert new.lookup("a").tokens == [] and new.lookup("a").status == pb.RUNNING
    assert new.take_over(min_epoch=e2 + 1000) == e2 + 1000  # wall-clock epochs keep increasing


def test_reusing_a_finished_client_id_starts_clean(db):
    log = RequestLog(db)
    log.take_over()
    log.start("a", "w1", [1], 1, time.time() + 60)
    log.record([("a", 0, 5)], [("a", pb.COMPLETED, "")])
    log.start("a", "w2", [2], 3, time.time() + 60)
    a = log.lookup("a")
    assert a.status == pb.RUNNING and a.tokens == [] and a.prompt == [2]


def test_gc_removes_only_old_finished_requests(db):
    log = RequestLog(db, retention_s=10)
    log.take_over()
    for cid in "abc":
        log.start(cid, cid, [1], 2, time.time() + 60)
    log.record([("a", 0, 1)], [("a", pb.COMPLETED, ""), ("b", pb.CANCELLED, "x")])
    assert log.gc(now=time.time()) == 0
    assert log.gc(now=time.time() + 11) == 2
    assert log.lookup("a") is None and log.lookup("b") is None
    assert [r.client_id for r in log.unfinished()] == ["c"]


def test_stage_rejects_an_older_coordinator(stages, reference):
    ex = StageExecutor(stages[0], 0, epoch=1)
    e = reference["prompts"][0]

    def send(coord_epoch, rid):
        it = pb.StageItem(request_id=rid, step=0, reset=True, use_cache=True, token_ids=e["prompt_token_ids"])
        return ex.process([1, 1], [it], [None], coordinator_epoch=coord_epoch)[0]

    assert send(5, "a").status == pb.OK
    assert ex.observe_coordinator(7)                 # a newer coordinator's heartbeat
    o = send(5, "b")
    assert o.status == pb.STALE_COORDINATOR
    assert "b" not in ex.cache                       # nothing was computed or cached
    assert send(0, "c").status == pb.OK              # 0 = unfenced admin/test message
    assert ex.release(["a"], coordinator_epoch=5) == 0
    assert ex.release(["a"], coordinator_epoch=7) == 1
    assert not ex.observe_coordinator(6) and ex.coordinator_epoch == 7
