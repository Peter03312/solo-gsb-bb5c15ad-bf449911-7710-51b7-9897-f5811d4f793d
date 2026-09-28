"""稿件共识顺序（POST /decision-ranking）：Schulze 最强路径法。

接收 /assign 的可行分配名单（稿件 -> 评审人）与评审人对其已评稿件的严格排序票，
按 Schulze 方法把分散的严格偏序汇总为稿件共识顺序：

- 每名评审人最多一票；票为该评审人分配名单内稿件的严格排序（靠前 = 更优），
  允许只排部分稿件（空票视为弃权）。只有**同时出现在同一张票中**的两篇稿件才
  产生成对支持，部分票不会在未同票的稿件间臆造偏好；
- 直接支持 ``d[甲][乙]`` = 将甲排在乙前的票数；仅当甲对乙构成**成对胜者**
  （``d[甲][乙] > d[乙][甲]``）时，直接支持 ``d[甲][乙]`` 才能作为路径边
  （成对平票时两个方向都不成边）；
- 路径强度 ``s[甲][乙]`` = 所有「甲 -> … -> 乙」路径上「沿途最小边权」的最大值，
  以 Floyd–Warshall 闭包计算；``s[甲][乙] > s[乙][甲]`` 时甲排在乙前，仍无法
  区分时按稿件 ID 升序破同。Schulze 的路径强度关系是无环弱序，比较器据此给出
  确定的完整全序（未获任何成对支持的稿件与其他稿件在路径强度上无法区分时，
  同样参与 ID 破同，而非简单追加到末尾）。

投票校验（任一不满足即整体拒绝 422，返回全部结构化问题、不输出任何顺序）：

- ``unknown_reviewer``：投票评审人不在 /assign 分配名单中；
- ``duplicate_ballot``：同一评审人提交了多张票（每人最多一票）；
- ``duplicate_manuscript_in_ballot``：单张票内同一稿件出现多次（票须为严格排序）；
- ``manuscript_not_assigned``：票内稿件不属于该评审人的分配名单
  （稿件不属于任一分配名单时同样按此处理）。

结果只取决于（合并去重后的）分配关系集合与票集合：分配名单条目顺序、票的输入
顺序重排不改变任何输出；问题清单按类型与业务 ID 确定性排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key


@dataclass(frozen=True)
class RankingBallot:
    """一张严格排序票：ranked_manuscript_ids 从前到后表示从优到劣。"""

    reviewer_id: str
    ranked_manuscript_ids: tuple[str, ...]


@dataclass(frozen=True)
class RankingIssue:
    """一条投票拒绝原因（结构化）。"""

    type: str  # unknown_reviewer / duplicate_ballot /
    # duplicate_manuscript_in_ballot / manuscript_not_assigned
    reviewer_id: str | None
    manuscript_id: str | None
    detail: str


@dataclass(frozen=True)
class PairwiseSupport:
    """一个有向稿件对的直接支持数：support 张票把 manuscript_id 排在 over 之前。"""

    manuscript_id: str
    over_manuscript_id: str
    support: int


@dataclass(frozen=True)
class RankingResult:
    # 共识顺序（完整，含未获票的稿件）；存在校验问题时为空元组
    ranking: tuple[str, ...]
    # 全部直接支持为正的有向对（按 (甲, 乙) 字典序）；未列出的有向对支持数为 0
    pairwise_support: tuple[PairwiseSupport, ...]
    issues: tuple[RankingIssue, ...]


def compute_ranking(
    assignments: list[tuple[str, tuple[str, ...]]],
    ballots: list[RankingBallot],
) -> RankingResult:
    """按 Schulze 最强路径法计算稿件共识顺序。

    assignments 为 (稿件 ID, 评审人 ID 序列) 列表（/assign 可行名单，同一稿件
    多条目自动合并取并集）；ballots 为评审人严格排序票。存在任何非法投票时
    issues 非空（确定性排序），调用方应整体拒绝且不使用 ranking。
    """
    # ---- 1. 展开分配名单：稿件 -> 评审人集合，并反查评审人 -> 稿件集合 ----
    reviewers_of: dict[str, set[str]] = {}
    for manuscript_id, reviewer_ids in assignments:
        reviewers_of.setdefault(manuscript_id, set()).update(reviewer_ids)
    manuscripts_of: dict[str, set[str]] = {}
    for manuscript_id, reviewer_ids in reviewers_of.items():
        for reviewer_id in reviewer_ids:
            manuscripts_of.setdefault(reviewer_id, set()).add(manuscript_id)

    # 稿件矩阵下标按 ID 排序：与输入列表顺序无关，输出稳定
    manuscript_ids = sorted(reviewers_of)

    # ---- 2. 按评审人归票后做全部校验（只依赖票集合，与票输入顺序无关）----
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for ballot in ballots:
        grouped.setdefault(ballot.reviewer_id, []).append(
            tuple(ballot.ranked_manuscript_ids)
        )

    issues: list[RankingIssue] = []
    for reviewer_id in sorted(grouped):
        voter_ballots = grouped[reviewer_id]
        assigned = manuscripts_of.get(reviewer_id)

        if assigned is None:
            issues.append(RankingIssue(
                type="unknown_reviewer",
                reviewer_id=reviewer_id,
                manuscript_id=None,
                detail=f"评审人 {reviewer_id} 不在分配名单中，不能投票",
            ))

        if len(voter_ballots) > 1:
            issues.append(RankingIssue(
                type="duplicate_ballot",
                reviewer_id=reviewer_id,
                manuscript_id=None,
                detail=(
                    f"评审人 {reviewer_id} 提交了 {len(voter_ballots)} 张票，"
                    "每人最多一票"
                ),
            ))

        # 票内重复（每张票逐一检查，重复稿件去重记一条）；并汇总该评审人票内
        # 出现过的全部稿件用于分配名单归属校验
        repeated: set[str] = set()
        ranked_union: set[str] = set()
        for ranked in voter_ballots:
            ranked_union.update(ranked)
            seen: set[str] = set()
            for manuscript_id in ranked:
                if manuscript_id in seen:
                    repeated.add(manuscript_id)
                else:
                    seen.add(manuscript_id)
        for manuscript_id in sorted(repeated):
            issues.append(RankingIssue(
                type="duplicate_manuscript_in_ballot",
                reviewer_id=reviewer_id,
                manuscript_id=manuscript_id,
                detail=(
                    f"评审人 {reviewer_id} 的票中稿件 {manuscript_id} 重复出现，"
                    "票内稿件不得重复"
                ),
            ))

        # 归属校验只对名单内评审人有意义；未知评审人已记 unknown_reviewer
        if assigned is not None:
            for manuscript_id in sorted(ranked_union - assigned):
                issues.append(RankingIssue(
                    type="manuscript_not_assigned",
                    reviewer_id=reviewer_id,
                    manuscript_id=manuscript_id,
                    detail=(
                        f"稿件 {manuscript_id} 不在评审人 {reviewer_id} 的分配名单中，"
                        "票内稿件必须属于其分配名单"
                    ),
                ))

    if issues:
        issues.sort(
            key=lambda i: (i.type, i.reviewer_id or "", i.manuscript_id or "")
        )
        return RankingResult((), (), tuple(issues))

    # ---- 3. 直接支持：同票中排名靠前者对靠后者 +1；未同票不计 ----
    index = {manuscript_id: i for i, manuscript_id in enumerate(manuscript_ids)}
    n = len(manuscript_ids)
    direct = [[0] * n for _ in range(n)]
    for ballot in ballots:
        ranked = tuple(ballot.ranked_manuscript_ids)
        for hi in range(len(ranked)):
            for lo in range(hi):
                direct[index[ranked[lo]]][index[ranked[hi]]] += 1

    # ---- 4. Schulze 最强路径：仅成对胜者的直接支持可作为路径边 ----
    strength = [
        [
            direct[i][j] if direct[i][j] > direct[j][i] else 0
            for j in range(n)
        ]
        for i in range(n)
    ]
    # Floyd–Warshall 闭包：s[i][j] = max_path min(沿途边权)；
    # min/max 格上的闭包结果与中间点遍历顺序无关。
    for k in range(n):
        via_k = strength[k]
        for i in range(n):
            via_i_k = strength[i][k]
            if via_i_k <= 0:
                continue
            row_i = strength[i]
            for j in range(n):
                candidate = via_i_k if via_i_k < via_k[j] else via_k[j]
                if candidate > row_i[j]:
                    row_i[j] = candidate

    # ---- 5. 比较排序：路径强度大者在前，无法区分按稿件 ID 升序破同 ----
    def cmp(i: int, j: int) -> int:
        s_ij, s_ji = strength[i][j], strength[j][i]
        if s_ij > s_ji:
            return -1
        if s_ij < s_ji:
            return 1
        id_i, id_j = manuscript_ids[i], manuscript_ids[j]
        if id_i < id_j:
            return -1
        if id_i > id_j:
            return 1
        return 0

    ranking = tuple(
        manuscript_ids[i] for i in sorted(range(n), key=cmp_to_key(cmp))
    )

    # ---- 6. 成对支持数：仅输出直接支持为正的有向对，按 (甲, 乙) 字典序 ----
    pairwise_support = tuple(sorted(
        (
            PairwiseSupport(
                manuscript_id=manuscript_ids[i],
                over_manuscript_id=manuscript_ids[j],
                support=direct[i][j],
            )
            for i in range(n)
            for j in range(n)
            if i != j and direct[i][j] > 0
        ),
        key=lambda p: (p.manuscript_id, p.over_manuscript_id),
    ))

    return RankingResult(ranking, pairwise_support, ())
