"""POST /decision-ranking 的端到端测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _post(body: dict) -> tuple[int, dict]:
    resp = client.post("/decision-ranking", json=body)
    return resp.status_code, resp.json()


def _rank(body: dict) -> list[str]:
    status, data = _post(body)
    assert status == 200, data
    return data["ranking"]


def _assign(pairs: dict[str, list[str]]) -> list[dict]:
    return [
        {"manuscript_id": mid, "reviewer_ids": rids}
        for mid, rids in pairs.items()
    ]


def _ballot(reviewer_id: str, ranked: list[str]) -> dict:
    return {"reviewer_id": reviewer_id, "ranked_manuscript_ids": ranked}


# ---------- 基本严格顺序 ----------


def test_unanimous_order():
    body = {
        "assignments": _assign({"m1": ["r1", "r2"], "m2": ["r1", "r2"]}),
        "ballots": [
            _ballot("r1", ["m1", "m2"]),
            _ballot("r2", ["m2", "m1"]),
        ],
    }
    # 成对平票（1 对 1）：按稿件 ID 破同
    assert _rank(body) == ["m1", "m2"]

    status, data = _post({
        "assignments": _assign({"m1": ["r1", "r2"], "m2": ["r1", "r2"]}),
        "ballots": [
            _ballot("r1", ["m1", "m2"]),
            _ballot("r2", ["m1", "m2"]),
        ],
    })
    assert status == 200
    assert data["ranking"] == ["m1", "m2"]
    assert data["pairwise_support"] == [
        {"manuscript_id": "m1", "over_manuscript_id": "m2", "support": 2}
    ]


def test_pairwise_support_counts_all_positive_pairs_sorted():
    status, data = _post({
        "assignments": _assign({
            "m1": ["r1", "r2"],
            "m2": ["r1", "r2"],
            "m3": ["r1", "r2"],
        }),
        "ballots": [
            _ballot("r1", ["m3", "m1", "m2"]),
            _ballot("r2", ["m2", "m1", "m3"]),
        ],
    })
    assert status == 200, data
    # r1: m3>m1>m2；r2: m2>m1>m3 —— 每个有向对恰 1 票（含反向），6 对全列出，
    # 按 (manuscript_id, over_manuscript_id) 字典序
    assert data["pairwise_support"] == [
        {"manuscript_id": "m1", "over_manuscript_id": "m2", "support": 1},
        {"manuscript_id": "m1", "over_manuscript_id": "m3", "support": 1},
        {"manuscript_id": "m2", "over_manuscript_id": "m1", "support": 1},
        {"manuscript_id": "m2", "over_manuscript_id": "m3", "support": 1},
        {"manuscript_id": "m3", "over_manuscript_id": "m1", "support": 1},
        {"manuscript_id": "m3", "over_manuscript_id": "m2", "support": 1},
    ]
    # 全部成对平票：路径强度互相为 0，无法区分时按稿件 ID 破同
    assert data["ranking"] == ["m1", "m2", "m3"]


# ---------- 部分票：未同票不计成对支持 ----------


def test_partial_ballots_only_co_ranked_pairs_count():
    body = {
        "assignments": _assign({
            "m1": ["r1"], "m2": ["r1"], "m3": ["r1", "r2"],
        }),
        "ballots": [
            _ballot("r1", ["m1", "m2"]),  # m1>m2；m3 未排，与 m3 不产生任何对
            _ballot("r2", ["m3"]),        # 只排一稿：无任何成对支持
        ],
    }
    status, data = _post(body)
    assert status == 200, data
    assert data["ranking"] == ["m1", "m2", "m3"]
    assert data["pairwise_support"] == [
        {"manuscript_id": "m1", "over_manuscript_id": "m2", "support": 1}
    ]


def test_empty_ballot_is_abstention():
    body = {
        "assignments": _assign({"a": ["r1"], "b": ["r1"]}),
        "ballots": [_ballot("r1", [])],
    }
    status, data = _post(body)
    assert status == 200, data
    # 无任何成对支持：全部稿件按 ID 破同
    assert data["ranking"] == ["a", "b"]
    assert data["pairwise_support"] == []


def test_no_ballots_ranks_by_id():
    body = {"assignments": _assign({"z": ["r1"], "a": ["r1"], "m": ["r1"]})}
    assert _rank(body) == ["a", "m", "z"]


def test_unvoted_manuscripts_ranked_among_ties_by_id():
    body = {
        "assignments": _assign({
            "m1": ["r1", "r2"], "m2": ["r1", "r2"], "z9": ["r3"], "a9": ["r3"],
        }),
        "ballots": [
            _ballot("r1", ["m2", "m1"]),
            _ballot("r2", ["m2", "m1"]),
        ],
    }
    status, data = _post(body)
    assert status == 200, data
    # m2 对 m1 有 2 票，故 m2 > m1；a9/z9 未获票，与任何稿件之间路径强度
    # 互相为 0，无法区分时一律按稿件 ID 升序破同 -> a9 > m2 > m1 > z9
    assert data["ranking"] == ["a9", "m2", "m1", "z9"]


# ---------- Schulze 最强路径：经典 45 票 5 候选例 ----------


def _wikipedia_ballots() -> tuple[list[dict], list[dict]]:
    groups = [
        (5, ["A", "C", "B", "E", "D"]),
        (5, ["A", "D", "E", "C", "B"]),
        (8, ["B", "E", "D", "A", "C"]),
        (3, ["C", "A", "B", "E", "D"]),
        (7, ["C", "A", "E", "B", "D"]),
        (2, ["C", "B", "A", "D", "E"]),
        (7, ["D", "C", "E", "B", "A"]),
        (8, ["E", "B", "A", "D", "C"]),
    ]
    manuscripts = ["A", "B", "C", "D", "E"]
    reviewers = [f"r{k}" for k in range(45)]
    assignments = _assign({m: reviewers for m in manuscripts})
    ballots: list[dict] = []
    k = 0
    for count, ranked in groups:
        for _ in range(count):
            ballots.append(_ballot(f"r{k}", ranked))
            k += 1
    return assignments, ballots


def test_schulze_strongest_paths_wikipedia_example():
    assignments, ballots = _wikipedia_ballots()
    status, data = _post({"assignments": assignments, "ballots": ballots})
    assert status == 200, data
    # Schulze 经典结果：E > A > C > B > D
    # （例如 C 直接击败 E 24:21，但 E 经更强路径反超 C）
    assert data["ranking"] == ["E", "A", "C", "B", "D"]
    support = {
        (p["manuscript_id"], p["over_manuscript_id"]): p["support"]
        for p in data["pairwise_support"]
    }
    # 直接支持矩阵抽样（Wikipedia 45 票例）
    assert support[("A", "B")] == 20
    assert support[("B", "A")] == 25
    assert support[("E", "C")] == 21
    assert support[("C", "E")] == 24
    # 全部 20 个有向对均有正支持
    assert len(data["pairwise_support"]) == 20


def test_strongest_path_beats_direct_loss_small_case():
    # 3 稿 10 票（构造可行的部分票分布）：
    # 4 票 A>B>C，3 票 B>C>A，2 票 C>A>B，1 票 A>C>B
    # 直接：d[A][B]=7>3, d[B][C]=7>3, d[C][A]=5<5? -> 4+3=7 C? 见下断言
    body = {
        "assignments": _assign({
            "A": [f"r{i}" for i in range(10)],
            "B": [f"r{i}" for i in range(10)],
            "C": [f"r{i}" for i in range(10)],
        }),
        "ballots": (
            [_ballot(f"r{i}", ["A", "B", "C"]) for i in range(4)]
            + [_ballot(f"r{i+4}", ["B", "C", "A"]) for i in range(3)]
            + [_ballot(f"r{i+7}", ["C", "A", "B"]) for i in range(2)]
            + [_ballot("r9", ["A", "C", "B"])]
        ),
    }
    status, data = _post(body)
    assert status == 200, data
    support = {
        (p["manuscript_id"], p["over_manuscript_id"]): p["support"]
        for p in data["pairwise_support"]
    }
    # d[A][B]=7, d[B][C]=4+3=7, d[C][A]=3+2=5
    assert support[("A", "B")] == 7
    assert support[("B", "C")] == 7
    assert support[("C", "A")] == 5
    # 最强路径：A->C 经 B 强度 min(7,7)=7 > 直接 d[A][C]=5，A 仍全胜居首
    assert data["ranking"] == ["A", "B", "C"]


# ---------- 确定性：输入重排不改变结果 ----------


def test_input_reorder_does_not_change_result():
    assignments, ballots = _wikipedia_ballots()
    status, base = _post({"assignments": assignments, "ballots": ballots})
    assert status == 200

    assignments_rev = list(reversed(assignments))
    ballots_rev = list(reversed(ballots))
    status2, rev = _post({
        "assignments": assignments_rev, "ballots": ballots_rev,
    })
    assert status2 == 200
    assert rev == base

    # 同一稿件拆多条 AssignmentIn 条目，自动合并取并集
    split: list[dict] = []
    for a in assignments:
        mid = a["manuscript_id"]
        for rid in a["reviewer_ids"]:
            split.append({"manuscript_id": mid, "reviewer_ids": [rid]})
    status3, merged = _post({"assignments": split, "ballots": ballots})
    assert status3 == 200
    assert merged == base


# ---------- 非法投票：422 ----------


def test_unknown_reviewer_422():
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "ballots": [_ballot("ghost", ["m1"])],
    }
    status, data = _post(body)
    assert status == 422
    assert data["errors"] == [{
        "type": "unknown_reviewer",
        "reviewer_id": "ghost",
        "manuscript_id": None,
        "detail": "评审人 ghost 不在分配名单中，不能投票",
    }]


def test_duplicate_ballot_422():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r1"]}),
        "ballots": [_ballot("r1", ["m1"]), _ballot("r1", ["m2"])],
    }
    status, data = _post(body)
    assert status == 422
    assert [e["type"] for e in data["errors"]] == ["duplicate_ballot"]
    assert data["errors"][0]["reviewer_id"] == "r1"


def test_duplicate_manuscript_in_ballot_422():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r1"]}),
        "ballots": [_ballot("r1", ["m1", "m2", "m1"])],
    }
    status, data = _post(body)
    assert status == 422
    assert [e["type"] for e in data["errors"]] == [
        "duplicate_manuscript_in_ballot"
    ]
    assert data["errors"][0]["manuscript_id"] == "m1"


def test_manuscript_not_assigned_422():
    body = {
        "assignments": _assign({"m1": ["r1"], "m2": ["r2"]}),
        "ballots": [_ballot("r1", ["m1", "m2"])],
    }
    status, data = _post(body)
    assert status == 422
    assert data["errors"] == [{
        "type": "manuscript_not_assigned",
        "reviewer_id": "r1",
        "manuscript_id": "m2",
        "detail": "稿件 m2 不在评审人 r1 的分配名单中，票内稿件必须属于其分配名单",
    }]


def test_unknown_manuscript_also_not_assigned_422():
    # 票内稿件不属于任何分配名单：按 manuscript_not_assigned 拒绝
    body = {
        "assignments": _assign({"m1": ["r1"]}),
        "ballots": [_ballot("r1", ["m1", "nope"])],
    }
    status, data = _post(body)
    assert status == 422
    assert [e["type"] for e in data["errors"]] == ["manuscript_not_assigned"]
    assert data["errors"][0]["manuscript_id"] == "nope"


def test_multiple_issues_collected_and_deterministically_sorted():
    body = {
        "assignments": _assign({"m1": ["r1", "r2"]}),
        "ballots": [
            _ballot("zz", ["m1"]),          # unknown_reviewer
            _ballot("aa", ["m1"]),          # unknown_reviewer
            _ballot("r1", ["m1"]),
            _ballot("r1", ["m1"]),          # duplicate_ballot
            _ballot("r2", ["m1", "m1"]),    # duplicate_manuscript_in_ballot
        ],
    }
    status, data = _post(body)
    assert status == 422
    assert [(e["type"], e["reviewer_id"]) for e in data["errors"]] == [
        ("duplicate_ballot", "r1"),
        ("duplicate_manuscript_in_ballot", "r2"),
        ("unknown_reviewer", "aa"),
        ("unknown_reviewer", "zz"),
    ]


def test_issues_independent_of_ballot_order():
    assignments = _assign({"m1": ["r1", "r2"]})
    ballots = [
        _ballot("zz", ["m1"]),
        _ballot("r1", ["m1"]),
        _ballot("r1", ["m1"]),
        _ballot("r2", ["m1", "m1"]),
        _ballot("aa", ["m1"]),
    ]
    status1, data1 = _post({"assignments": assignments, "ballots": ballots})
    status2, data2 = _post({
        "assignments": list(reversed(assignments)),
        "ballots": list(reversed(ballots)),
    })
    assert status1 == status2 == 422
    assert data1 == data2


# ---------- 请求级校验（FastAPI/Pydantic 422） ----------


def test_empty_assignments_rejected():
    status, data = _post({"assignments": [], "ballots": []})
    assert status == 422
    assert "detail" in data  # FastAPI 请求体校验形态


def test_extra_field_rejected():
    status, _ = _post({
        "assignments": _assign({"m1": ["r1"]}),
        "ballots": [],
        "unexpected": 1,
    })
    assert status == 422


def test_ballot_extra_field_rejected():
    status, _ = _post({
        "assignments": _assign({"m1": ["r1"]}),
        "ballots": [{"reviewer_id": "r1", "ranked_manuscript_ids": [], "x": 1}],
    })
    assert status == 422
