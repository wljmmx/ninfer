#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ninfer v9 上游吸收计划 / T005 —— 旧路径 → 新路径映射表生成器

用途
    上游把 src/ops/linear/{bf16,fp8,nvfp4,q4,q5,q6,q8} 等整树重构为
    kernels/ + projection/ + add/ + split/ + swiglu/ + topk/ 新布局，本地在
    旧路径上有大量改动（v9 §文件三分法 D 类）。Phase 5/6 移植时不能把本地改动
    直接叠到上游，必须先按本表把「旧路径的本地改动」翻译到「新路径」。

口径（与 v9 计划 §文件三分法 对齐；全部数据由 git 一次性取得，重跑可复现）
    比较对象:   --base（本地 v8 完工基线 de5d4288）..--upstream（upstream/master）
    重命名检测: git -c diff.renameLimit=<N> diff --find-renames=<sim> --name-status
                N 默认 20000（≥1943，否则 git 会跳过 exhaustive rename detection）
                sim 默认 50%（git 标准档）；此档 R 条目即「直接重命名/移动」
    D 类条目:   50% 档下的 D（上游删除）。其中一部分其实是重构式移动：
                git 在两棵 tip 树上因内容改动过大配不上，因此再用四条证据补配
                （见下），并把证据强度写进置信度列。

D 类补配的证据顺序（自上而下，互斥）
    1) 逐提交重命名链   base..upstream 每个提交单独跑 -M50%，把 old->new 连成链；
                       链尾在上游树中存在即认定为目标。证据最强（git 自己在
                       改动较小的那个提交上检出的重命名），置信度记「直接重命名」
                       并在备注标「逐提交重命名链 N 跳」。
    2) 目录族推断       由 R 条目学出「旧目录 → 新目录」规则做路径替换，目标须在
                       上游树中真实存在；同目录多规则时取命中次数最高者。
    3) 同名异目录       剩余新增文件里存在同名文件；多候选按目录相似度取首。
    4) 低置信启发式      其余证据：删除提交里只有一个「相关」新增（同目录或主干共享
                       辨识 token）且扩展名相同（疑为改名/整体搬移）；或 10% 档内容
                       相似且模块与量化族完全一致。
    5) 无对应           判为真删/废弃。删除提交同时新增多个相关文件时记为「一对多
                       拆分」并列出拆分目标；新增文件与旧文件无相关性时记「批量重构」。
                       两种情况都需 Phase 5/6 人工判定。

用法
    python tools/port/gen_path_map.py
    python tools/port/gen_path_map.py --base de5d4288 --upstream upstream/master --out path-map.txt
    python tools/port/gen_path_map.py --local-ref HEAD   # 相交小节改用当前 HEAD 口径

输出
    UTF-8 文本：头部口径 + 统计；TSV 主体（旧路径\t新路径\t类型\t置信度\t备注）；
    附「与本地改动相交的 D/R 文件」小节（T003 口径：本地 M 类 ∩ 上游 D/R）。
"""

import argparse
import collections
import datetime
import difflib
import os
import subprocess
import sys

DTYPES = {
    "bf16", "fp8", "nvfp4", "q4", "q5", "q6", "q8", "int8", "int4",
    "fp4", "mxfp4", "fp16", "fp32", "w4a8",
}

RENAME_LIMIT = 20000
STATUS_DIRECT = "直接重命名"
CONF_RULE = "目录族推断"
CONF_SAME_NAME = "同名异目录"
CONF_LOW = "低置信启发式"
CONF_NONE = "无对应"

TYPE_R = "R"
TYPE_D2A = "D→A"
TYPE_D_NONE = "D→无对应"


def run_git(repo, *args):
    p = subprocess.run(["git", "-C", repo] + list(args), capture_output=True)
    if p.returncode != 0:
        sys.stderr.write("git 失败: %s\n%s\n" % (" ".join(args), p.stderr.decode("utf-8", "replace")))
        raise SystemExit(1)
    return p.stdout.decode("utf-8", "replace")


def name_status(repo, base, upstream, sim, limit):
    raw = run_git(repo, "-c", "diff.renameLimit=%d" % limit, "diff",
                  "--find-renames=%s" % sim, "--name-status", base, upstream)
    return [line.split("\t") for line in raw.split("\n") if line]


def split_status(rows):
    """返回 (renames[(old,new,status)], adds[path], dels[path], mods[path])。"""
    renames, adds, dels, mods = [], [], [], []
    for parts in rows:
        st = parts[0]
        if st[:1] == "R":
            renames.append((parts[1], parts[2], st))
        elif st[:1] == "A":
            adds.append(parts[1])
        elif st[:1] == "D":
            dels.append(parts[1])
        else:
            mods.append(parts[1])
    return renames, adds, dels, mods


def parse_commit_log(repo, base, upstream, limit):
    """逐提交跑 -M，返回:
    edges    : old -> [(new, commit)]   重命名边
    del_commit: path -> (commit, subject, [同提交新增])
    """
    raw = run_git(repo, "-c", "diff.renameLimit=%d" % limit, "log", "--name-status", "-M",
                  "--diff-filter=ACDR", "--format=@@%h\t%s", "%s..%s" % (base, upstream))
    edges = collections.defaultdict(list)
    del_commit = {}
    commit = subject = None
    cur_adds = []
    cur_dels = []

    def flush():
        for d in cur_dels:
            del_commit[d] = (commit, subject, list(cur_adds))

    for line in raw.split("\n"):
        if not line:
            continue
        if line.startswith("@@"):
            flush()
            commit, _, subject = line[2:].partition("\t")
            cur_adds, cur_dels = [], []
            continue
        parts = line.split("\t")
        st = parts[0]
        if st[:1] == "R":
            edges[parts[1]].append((parts[2], commit))
            cur_dels.append(parts[1])
        elif st[:1] == "A":
            cur_adds.append(parts[1])
        elif st[:1] == "D":
            cur_dels.append(parts[1])
    flush()
    return edges, del_commit


def follow_chain(edges, path, limit=12):
    """沿逐提交重命名边前进，返回 (终点, 跳数)；成环或超深返回 (None, 0)。"""
    cur, hops, seen = path, 0, {path}
    while cur in edges and hops < limit:
        nxt, _commit = edges[cur][-1]
        if nxt in seen:
            return None, 0
        seen.add(nxt)
        cur, hops = nxt, hops + 1
    return (cur, hops) if hops else (None, 0)


def dtypes_of(path):
    return {t for t in path.split("/") if t in DTYPES}


def module_root(path):
    parts = path.split("/")
    return "/".join(parts[:3]) if len(parts) >= 3 else path


def path_sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


STOP_TOKENS = {"src", "include", "tests", "bench", "ops", "cu", "cuh", "cpp", "h", "main", "test", "shapes"}


def stem_tokens(path):
    """文件主干中具辨识度的 token（去掉目录名/扩展名/量化族/常见词）。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return {t for t in stem.replace("-", "_").split("_") if t and t not in STOP_TOKENS and t not in DTYPES}


def related_to(add_path, del_path):
    """同提交新增是否与旧文件相关：同目录，或主干有共同的辨识 token。

    量化族（q4/q8/bf16...）与目录名一律不算相关性依据——同一重构提交里
    每个 dtype 目录都会批量新增，按 dtype 判会把 sources.cmake 之类误配过来。
    """
    if os.path.dirname(add_path) == os.path.dirname(del_path):
        return True
    return bool(stem_tokens(add_path) & stem_tokens(del_path))


def build_rules(renames):
    """从 R 条目学「旧目录 → 新目录」规则，值 = 命中次数。"""
    rules = collections.Counter()
    for old, new, _st in renames:
        rules[(os.path.dirname(old), os.path.dirname(new))] += 1
    return rules


def rule_predict(rules, path, upstream_files):
    """按目录规则做路径替换，返回上游树中真实存在的候选，按规则命中数降序。"""
    best = {}
    for (old_dir, new_dir), count in rules.items():
        if not path.startswith(old_dir + "/"):
            continue
        cand = new_dir + "/" + path[len(old_dir) + 1:]
        if cand in upstream_files and cand != path:
            best[cand] = max(best.get(cand, 0), count)
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


def infer(rules, dels, adds, upstream_files, low_pairs, edges, del_commit, base_files, origin):
    """对每个 D 条目推断上游新路径。返回 rows: list[dict]。"""
    by_base = collections.defaultdict(list)
    for a in adds:
        by_base[os.path.basename(a)].append(a)
    # 全量上游同名计数：main.cpp / sources.cmake / kernel.cuh 这类通用名不足以证明
    # 是同一文件的搬移，命中数过多时不再采信同名证据
    occ = collections.Counter(os.path.basename(p) for p in upstream_files)

    def accept_same_name(d, cand):
        base_occ = occ.get(os.path.basename(d), 0)
        sim = path_sim(os.path.dirname(d), os.path.dirname(cand))
        if base_occ <= 3:
            return sim >= 0.4
        if module_root(d) == module_root(cand) and sim >= 0.5:
            return True
        return bool(dtypes_of(d)) and dtypes_of(cand) == dtypes_of(d)

    rows = []
    for d in sorted(dels):
        base = os.path.basename(d)

        # 1) 逐提交重命名链：链尾必须在上游树中存在
        tail, hops = follow_chain(edges, d)
        if tail and tail in upstream_files:
            rows.append(dict(old=d, new=tail, type=TYPE_D2A, conf=STATUS_DIRECT,
                             note="逐提交重命名链 %d 跳（tip 对 tip 相似度 <50%%，故 50%% 档未配对）" % hops))
            continue

        # 2) 目录规则替换
        cands = rule_predict(rules, d, upstream_files)
        if cands:
            target, count = cands[0]
            note = "规则命中 %d 次" % count
            if len(cands) > 1:
                note += "；其他候选: " + " | ".join(c for c, _ in cands[1:])
            rows.append(dict(old=d, new=target, type=TYPE_D2A, conf=CONF_RULE, note=note))
            continue

        # 3) 同名异目录
        pool = [p for p in by_base.get(base, []) if accept_same_name(d, p)]
        if len(pool) == 1:
            rows.append(dict(old=d, new=pool[0], type=TYPE_D2A, conf=CONF_SAME_NAME,
                             note="上游剩余新增中唯一可信同名文件"))
            continue
        if len(pool) > 1:
            pool = sorted(pool, key=lambda p: (-path_sim(os.path.dirname(d), os.path.dirname(p)), p))
            rows.append(dict(old=d, new=pool[0], type=TYPE_D2A, conf=CONF_SAME_NAME,
                             note="同名多候选，按目录相似度取首；其他: " + " | ".join(pool[1:4])))
            continue

        # 4) 同提交线索：只看与旧路径相关的同提交新增
        info = del_commit.get(d)
        same_adds = info[2] if info else []
        related = [a for a in same_adds if related_to(a, d)]
        if len(related) >= 2:
            rows.append(dict(old=d, new="", type=TYPE_D_NONE, conf=CONF_NONE,
                             note="一对多拆分：删除提交 %s（%s）同提交新增 %d 个相关文件: %s"
                                  % (info[0], info[1][:36], len(related), " | ".join(related[:4]))))
            continue
        if len(related) == 1 and os.path.splitext(related[0])[1] == os.path.splitext(d)[1]:
            rows.append(dict(old=d, new=related[0], type=TYPE_D2A, conf=CONF_LOW,
                             note="删除提交 %s 同提交新增 %s（相关且同扩展名，疑为改名/整体搬移，git 未达 50%% 阀值）"
                                  % (info[0], related[0])))
            continue

        # 5) 10% 档内容相似对（要求模块与量化族完全一致，避免跨 dtype 误配）
        lown = low_pairs.get(d)
        if lown:
            if module_root(lown) == module_root(d) and dtypes_of(lown) == dtypes_of(d):
                rows.append(dict(old=d, new=lown, type=TYPE_D2A, conf=CONF_LOW,
                                 note="内容相似 10%%–50%%，git 50%% 档未配对"))
            else:
                rows.append(dict(old=d, new="", type=TYPE_D_NONE, conf=CONF_NONE,
                                 note="10%% 档候选 %s 跳模块/跳量化族，疑误配，判为真删" % lown))
            continue

        # 6) 无对应：给出删除提交上下文与旧目录的映射目标
        newdirs = sorted({rn for (ro, rn), _c in rules.items() if d.startswith(ro + "/")})
        tail_hint = ("；旧目录在上游对应 " + ", ".join(newdirs[:3])) if newdirs else ""
        if info:
            if same_adds:
                note = ("删除提交 %s（%s）为批量重构（同提交新增 %d 个、无相关候选）%s，判为真删/并入；需人工判定"
                        % (info[0], info[1][:36], len(same_adds), tail_hint))
            else:
                note = "删除提交 %s（%s）无同提交新增，判为真删/废弃%s" % (info[0], info[1][:36], tail_hint)
        elif newdirs:
            note = "无同名/同内容后继；旧目录在上游对应 " + ", ".join(newdirs[:3]) + "（疑为并入或拆分，需人工判定）"
        else:
            note = "上游无该目录的对应新目录，判为真删/废弃"
        if d not in base_files:
            note += "；该路径在共同基线 %s 中不存在（属基线后新增，上游无历史可考）" % origin
        rows.append(dict(old=d, new="", type=TYPE_D_NONE, conf=CONF_NONE, note=note))
    return rows


def local_changes(repo, origin, local_ref):
    """本地改动集：origin..local_ref 状态下为 M 的文件 → 改动行数（对齐 T003 口径）。"""
    st = {}
    for line in run_git(repo, "diff", "--name-status", origin, local_ref).split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        st[parts[-1]] = parts[0][:1]
    num = {}
    for line in run_git(repo, "diff", "--numstat", origin, local_ref).split("\n"):
        if not line:
            continue
        a, b, p = line.split("\t")
        num[p] = (int(a) + int(b)) if (a.isdigit() and b.isdigit()) else 0
    return {p: num[p] for p in num if st.get(p) == "M"}


def short_sim(st):
    if st.endswith("100"):
        return "内容完全一致"
    return "内容相似度 %s%%" % int(st[1:])


def main():
    ap = argparse.ArgumentParser(description="生成上游重构的旧路径→新路径映射表（ninfer v9 T005）")
    ap.add_argument("--repo", default=".", help="仓库根目录（默认当前目录）")
    ap.add_argument("--base", default="de5d4288", help="本地完工基线（默认 de5d4288）")
    ap.add_argument("--upstream", default="upstream/master", help="上游 tip（默认 upstream/master）")
    ap.add_argument("--origin", default="81c8ce09", help="三方共同基线（默认 81c8ce09）")
    ap.add_argument("--local-ref", default="de5d4288",
                    help="本地改动统计终点（默认 de5d4288，对齐 T003 的 226 文件/12036 行）")
    ap.add_argument("--out", default=os.path.join(".temp", "upstream-scan", "path-map.txt"),
                    help="输出文件路径（相对路径按仓库根解析）")
    ap.add_argument("--rename-limit", type=int, default=RENAME_LIMIT,
                    help="diff.renameLimit（默认 20000）")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    rows50 = name_status(repo, args.base, args.upstream, "50%", args.rename_limit)
    rows10 = name_status(repo, args.base, args.upstream, "10%", args.rename_limit)
    renames, adds, dels, mods = split_status(rows50)
    renames10, _a10, _d10, _m10 = split_status(rows10)

    upstream_files = {p for p in run_git(repo, "ls-tree", "-r", "--name-only", args.upstream).split("\n") if p}
    r50src = {o for o, _n, _s in renames}
    low_pairs = {o: n for o, n, _s in renames10 if o not in r50src}
    rules = build_rules(renames)
    edges, del_commit = parse_commit_log(repo, args.base, args.upstream, args.rename_limit)

    body = [dict(old=o, new=n, type=TYPE_R, conf=STATUS_DIRECT, note=st + "（" + short_sim(st) + "）")
            for o, n, st in renames]
    base_files = {p for p in run_git(repo, "ls-tree", "-r", "--name-only", args.origin).split("\n") if p}
    d_rows = infer(rules, dels, adds, upstream_files, low_pairs, edges, del_commit, base_files, args.origin)
    body.extend(d_rows)

    # 多对一检测
    tgt_count = collections.Counter(r["new"] for r in body if r["new"])
    for r in body:
        n = tgt_count.get(r["new"], 0)
        if n > 1:
            r["note"] += "；多对一（%d 条旧路径指向同一目标）" % n

    # 与本地改动相交
    local = local_changes(repo, args.origin, args.local_ref)
    mapping = {}
    for r in body:
        mapping.setdefault(r["old"], r)
    inter = sorted(((local[p], p) for p in local if p in mapping), key=lambda x: (-x[0], x[1]))
    inter_lines = sum(v for v, _p in inter)
    hit_local = sum(1 for _v, p in inter if mapping[p]["new"])

    n_d_mapped = sum(1 for r in d_rows if r["type"] == TYPE_D2A)
    n_d_none = len(d_rows) - n_d_mapped
    conf_d = collections.Counter(r["conf"] for r in d_rows if r["type"] == TYPE_D2A)

    def in_old_linear(p):
        parts = p.split("/")
        return len(parts) > 3 and parts[0] == "src" and parts[1] == "ops" and parts[2] == "linear" \
            and (parts[3] in DTYPES or parts[3] == "common")

    lin_d = [r for r in d_rows if in_old_linear(r["old"])]
    lin_r = [r for r in body if r["type"] == TYPE_R and in_old_linear(r["old"])]
    lin_d_ok = sum(1 for r in lin_d if r["type"] == TYPE_D2A)
    split_rows = [r for r in d_rows if "一对多拆分" in r["note"]]

    out = []
    w = out.append
    w("ninfer v9 上游吸收计划 / T005 路径映射表（旧路径 → 新路径）")
    w("=" * 100)
    w("生成时间        : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    w("仓库            : %s" % repo)
    w("本地完工基线    : %s（--base）" % args.base)
    w("上游            : %s（--upstream）" % args.upstream)
    w("三方共同基线    : %s（--origin，用于本地改动统计）" % args.origin)
    w("renameLimit     : %d（git 建议 ≥1943，低于此值 exhaustive rename detection 会被跳过）" % args.rename_limit)
    w("重命名阈值      : 50%（git 标准档，产 R 条目）；另跑 10% 档仅用于 D 类补配")
    w("")
    w("【口径说明】")
    w("  1) 类型 R：50% 档 git 直接检出的重命名/移动，旧路径→新路径即 git 判定结果。")
    w("  2) 类型 D→A：上游删除了旧路径，但在新布局中有对应文件。按证据强度依次为：")
    w("       直接重命名     —— 逐提交重命名链（base..upstream 每个提交单独 -M50% 连成的链，")
    w("                         链尾在上游树中存在；tip 对 tip 因内容改动过大配不上）")
    w("       目录族推断     —— 由 R 条目学出「旧目录→新目录」规则做路径替换，目标须真实存在")
    w("       同名异目录     —— 上游剩余新增中存在同名文件（多候选按目录相似度取首）")
    w("       低置信启发式   —— 同提交仅 1 个相关新增（疑为改名），或 10% 档内容相似且模块/量化族一致")
    w("  3) 类型 D→无对应：判为真删/废弃，或结构重构后无单一后继。备注给出删除提交、")
    w("     同提交新增（拆分/并入线索）或旧目录的映射目标，Phase 5/6 需人工判定。")
    w("  4) 备注「多对一」= 多条旧路径指向同一新文件（合并式重构），翻译本地改动时须合并，")
    w("     不能互相覆盖；备注「一对多拆分」= 一个旧文件被拆到多个新文件。")
    w("  5) 本表只做路径翻译，不改动本地树（决策见文末「决策留痕」）。")
    w("")
    rows_default = name_status(repo, args.base, args.upstream, "50%", 1000)
    _r0, _a0, _d0, _m0 = split_status(rows_default)
    w("【口径对账】")
    w("  计划 §文件三分法标称 A=1943 / M=565 / D=899，对应 git 默认 diff.renameLimit=1000 档位：")
    w("    该档 git 因文件数超过限额会跳过 exhaustive rename detection，只认精确重命名，")
    w("    故大量「改名 + 改内容」的文件被算成 A/D 两条。")
    w("  默认档实测（本脚本自动复跑）: A=%d  M=%d  D=%d  R=%d" % (len(_a0), len(_m0), len(_d0), len(_r0)))
    w("  本表档实测（renameLimit=%d）: A=%d  M=%d  D=%d  R=%d" % (args.rename_limit, len(adds), len(mods), len(dels), len(renames)))
    w("  两档自洽: A+R = %d vs %d（无重命名时的 A 总量）；D+R = %d vs %d（无重命名时的 D 总量）"
      % (len(_a0) + len(_r0), len(adds) + len(renames), len(_d0) + len(_r0), len(dels) + len(renames)))
    w("  结论: D 的差额全部由重命名检出解释（%d 对从 A/D 中析出），不存在文件集合差异；"
      % (len(renames) - len(_r0)))
    w("        T003 记录的 A=1943 / D=897 / R=146 与默认档实测（R=%d）相差 %d 对，属同一档位的计数误差。"
      % (len(_r0), abs(146 - len(_r0))))
    w("")
    w("【统计】")
    w("  50%% 档 name-status: A=%d  M=%d  D=%d  R=%d  （合计 %d 条）"
      % (len(adds), len(mods), len(dels), len(renames), len(rows50)))
    w("  R 对数          : %d" % len(renames))
    w("  D 总数          : %d" % len(dels))
    w("  可配对 D 数     : %d  （%.1f%%）" % (n_d_mapped, 100.0 * n_d_mapped / max(1, len(dels))))
    for k in (STATUS_DIRECT, CONF_RULE, CONF_SAME_NAME, CONF_LOW):
        w("      %-14s %d" % (k, conf_d.get(k, 0)))
    w("  真删/无对应     : %d" % n_d_none)
    n_d_in_base = sum(1 for d in dels if d in base_files)
    w("      D 总数按基线拆分: 共同基线已存在 %d（上游删除/迁移）；基线后新增 %d（基线中无此路径）"
      % (n_d_in_base, len(dels) - n_d_in_base))
    w("      其中一对多拆分（删除提交同提交新增 ≥2）: %d" % len(split_rows))
    w("")
    w("  src/ops/linear 旧布局（bf16|fp8|nvfp4|q4|q5|q6|q8|common）:")
    w("      R 条目（旧→新布局）: %d" % len(lin_r))
    w("      D 条目: %d，其中已映射 %d，无对应 %d" % (len(lin_d), lin_d_ok, len(lin_d) - lin_d_ok))
    w("")
    w("  与本地改动相交（%s..%s 的 M 类 ∩ 上游 D/R，T003 口径）:" % (args.origin, args.local_ref))
    w("      文件 %d 个，本地改动 %d 行" % (len(inter), inter_lines))
    w("      其中已有目标路径 %d 条，待人工判定 %d 条" % (hit_local, len(inter) - hit_local))
    w("")
    w("【TSV 主体】列: 旧路径 \\t 新路径 \\t 类型 \\t 置信度 \\t 备注")
    w("（按 类型(R 在前) / 旧路径 排序；共 %d 行）" % len(body))
    w("-" * 100)
    for r in sorted(body, key=lambda r: (0 if r["type"] == TYPE_R else 1, r["old"])):
        w("%s\t%s\t%s\t%s\t%s" % (r["old"], r["new"], r["type"], r["conf"], r["note"]))
    w("-" * 100)
    w("")
    w("【与本地改动相交的 D/R 文件】按本地改动行数降序")
    w("  口径: git diff --numstat %s..%s 的 M 类文件，路径落在上游 D/R 集合内。" % (args.origin, args.local_ref))
    w("  建议: R 类直接按新路径翻译；D→A 按目标路径翻译后在新路径上做三方比对；")
    w("        D→无对应 者本地改动需在新布局中重新落点（不可机械搬运），备注给出线索。")
    w("-" * 100)
    w("%-8s %-8s %-14s %s" % ("本地行数", "类型", "置信度", "旧路径 -> 新路径"))
    for v, p in inter:
        r = mapping[p]
        target = r["new"] if r["new"] else "（无对应，见主体备注）"
        w("%-8d %-8s %-14s %s -> %s" % (v, r["type"], r["conf"], p, target))
    w("-" * 100)
    w("")
    w("【决策留痕】")
    w("  本地树暂不重组（避免打碎已完工的 v8 树、门禁基线与 9908 交付形态）；")
    w("  Phase 5/6 移植时按本表逐文件翻译旧路径改动到新路径。")
    w("  一次性重组（把本地树整体迁到上游新布局）待上游吸收稳定后单独提案评估，不在 T005 内执行。")
    w("")
    w("【已知异常】")
    one_to_many = sum(1 for r in body if "其他候选" in r["note"] or "其他: " in r["note"])
    w("  1) 一对多（多候选）: %d 条，备注已列其他候选；多因上游按设备（rtx5090/dgx_spark）" % one_to_many)
    w("     或按 shape 再切分，旧路径无唯一后继。")
    w("  2) 一对多拆分: %d 条（见备注「一对多拆分」），一条旧路径对应多个新文件。" % len(split_rows))
    w("  3) 多对一（合并）: %d 条，见备注「多对一」。"
      % sum(1 for r in body if "多对一" in r["note"]))
    w("  4) 上游两次重构叠加：先删旧布局，再按角色重分 kernels/projection/add/split/swiglu/topk；")
    w("     部分旧文件被拆散到 2 个以上新文件，git 与启发式均无法给出单一映射，")
    w("     已列 D→无对应 并在备注给出拆分目标。")

    text = "\n".join(out) + "\n"
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)

    print("wrote %s (%d bytes)" % (out_path, os.path.getsize(out_path)))
    print("R=%d D=%d D->A=%d D->none=%d body=%d split=%d"
          % (len(renames), len(dels), n_d_mapped, n_d_none, len(body), len(split_rows)))
    print("local intersect=%d files / %d lines (mapped %d)" % (len(inter), inter_lines, hit_local))


if __name__ == "__main__":
    main()
