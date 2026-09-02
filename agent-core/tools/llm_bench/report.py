"""报告：Markdown 输出。

结论段由数据生成，不是模板套话——差异低于噪声基线时明说「不可区分」，有组
成功率不满时把可用性而不是延迟摆在最前面。
"""
import datetime
import json
import pathlib

from llm_bench import stats as st


def _fmt_pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def _table(rows: list[list[str]], header: list[str]) -> str:
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join(['---'] * len(header)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def build(run_meta: dict, summaries: dict, paired: dict, noise: dict,
          shapes: dict, verdict: dict) -> str:
    L = []
    A = L.append

    A(f'# LLM 入口评测报告')
    A('')
    A(f'- run_id: `{run_meta["run_id"]}`')
    A(f'- 测量位置: `{run_meta.get("hostname", "?")}`'
      f'（镜像 `{run_meta.get("image_tag", "?")}`）')
    A(f'- 时间: {run_meta.get("started_at", "?")} → {run_meta.get("finished_at", "?")}')
    A(f'- 语料: {run_meta["corpus"]["count"]} 条，指纹 `{run_meta["corpus"]["fingerprint"]}`')
    A(f'- 预热轮 {run_meta["run"]["warmup"]} / 计分轮 {run_meta["run"]["repeats"]}'
      f'，组顺序 {run_meta["run"]["order"]}')
    A(f'- HTTP 后端: {run_meta.get("transport", "?")}')
    if run_meta.get('incomplete'):
        A('')
        A(f'> **本次结果不完整**：缺失 {run_meta["incomplete"]}，'
          f'下面的数字只覆盖已完成部分。')
    A('')

    # ── 结论 ──────────────────────────────────────────────────────────────
    A('## 结论')
    A('')
    if verdict.get('primary') == 'availability':
        A('**首要问题是可用性，不是延迟。**')
        A('')
    if verdict.get('indistinguishable'):
        A('**各组之间测不出显著差异——不要据此排名。**')
        A('')
    for r in verdict.get('reasons', []):
        A(f'- {r}')
    if verdict.get('recommend'):
        A('')
        A(f'**推荐：`{verdict["recommend"]}`**')
    elif not verdict.get('reasons'):
        A('- 数据不足，无结论')
    A('')

    # ── 总表 ──────────────────────────────────────────────────────────────
    A('## 总表')
    A('')
    rows = []
    for name in sorted(summaries):
        s = summaries[name]
        ok_note = f'{s["ok"]}/{s["total"]}'
        if not s.get('complete'):
            ok_note = f'**{ok_note}**'
        if not s.get('ok'):
            rows.append([name, ok_note] + ['—'] * 8)
            continue
        # 成功率不满时，延迟数字后面强制带上样本量——避免幸存者偏差被读成事实
        suffix = f' ({s["ok"]}/{s["total"]})' if not s.get('complete') else ''
        rows.append([
            name, ok_note,
            f'{s["p50"]:.2f}s{suffix}', f'{s["p90"]:.2f}s', f'{s["p95"]:.2f}s',
            f'{s["max"]:.2f}s', f'{s["cv"]:.2f}',
            _fmt_pct(s['cache_overall']), _fmt_pct(s['cache_median']),
            f'{s["throughput"]:.1f}',
        ])
    A(_table(rows, ['组', '成功', 'p50', 'p90', 'p95', 'max', 'cv',
                    '缓存(总体)', '缓存(中位)', 'tok/s']))
    A('')
    if any(not s.get('complete') for s in summaries.values()):
        A('> 加粗的成功率表示该组有失败样本。**延迟分位只统计成功请求**，'
          '存在幸存者偏差：秒拒的失败（如 0.06s 的 503）不计入，'
          '否则会让失败更多的组显得更快。')
        A('')

    # ── 配对比较 ──────────────────────────────────────────────────────────
    A('## 配对比较')
    A('')
    if paired.get('n_common', 0) < 1 or not paired.get('pairs'):
        A('配对样本不足（需要至少一条所有组都成功的 payload）。')
    else:
        A(f'同一条 payload 逐条比较，消除 payload 难度差异。'
          f'共 {paired["n_common"]} 条所有组都成功。')
        A('')
        rows = []
        floor = noise.get('median_abs_delta') if noise.get('available') else None
        for pair, v in sorted(paired['pairs'].items()):
            a, b = pair.split(' vs ')
            note = ''
            if floor is not None and abs(v['median_delta']) <= floor:
                note = '噪声以内'
            rows.append([pair, f'{v["median_delta"]:+.2f}s',
                         f'{v["mean_delta"]:+.2f}s',
                         f'{v["a_faster"]} / {v["b_faster"]}', note or '—'])
        A(_table(rows, ['组对 (A vs B)', 'A−B 中位', 'A−B 均值',
                        'A 更快 / B 更快', '判定']))
        A('')
        # 只在真的出现符号相反时才提示。无条件打印这句会在两者同号时说出一句
        # 假话，而报告里出现一句可验证为假的解读，会让读者不再相信其余部分。
        flipped = [p for p, v in paired['pairs'].items()
                   if v['median_delta'] * v['mean_delta'] < 0]
        if flipped:
            A('以下组对的中位与均值**符号相反**，说明一方在多数请求上略快但吃到了'
              '更重的长尾——这种情况下两者不应视为有差别：')
            A('')
            for p in sorted(flipped):
                A(f'- {p}')
    A('')

    # ── 噪声基线 ──────────────────────────────────────────────────────────
    A('## 噪声基线')
    A('')
    if not noise.get('available'):
        A(f'未估出：{noise.get("reason", "轮次不足")}。')
        A('')
        A('> 没有噪声基线时，任何「A 比 B 快」的结论都无法排除是重复测量的波动。'
          '用 `--repeats 2` 及以上复核。')
    else:
        A(f'同一组、同一条 payload 在不同轮次之间的延迟波动（{noise["n"]} 个差值）：')
        A('')
        A(f'- 中位绝对差 **{noise["median_abs_delta"]:.2f}s**')
        A(f'- p90 绝对差 {noise["p90_abs_delta"]:.2f}s')
        A('')
        rows = [[n, str(v['n']), f'{v["median_abs_delta"]:.2f}s',
                 f'{v["p90_abs_delta"]:.2f}s']
                for n, v in sorted(noise['per_group'].items())]
        A(_table(rows, ['组', '样本', '中位绝对差', 'p90 绝对差']))
        A('')
        A('**组间差异小于这个基线时不构成差异。** 同配置重跑的波动就有这么大。')
    A('')

    # ── 失败分析 ──────────────────────────────────────────────────────────
    A('## 失败分析')
    A('')
    any_fail = False
    for name in sorted(summaries):
        s = summaries[name]
        if s.get('complete'):
            continue
        any_fail = True
        shape = shapes.get(name, {})
        A(f'### {name}')
        A('')
        A(f'- 失败 {s["failed"]}/{s["total"]} 次请求，'
          f'类型 {", ".join(f"{k}×{v}" for k, v in sorted(s["failure_kinds"].items()))}')
        A(f'- 受影响 payload {shape.get("failed_payloads", 0)} 条，'
          f'其中最长连续 {shape.get("longest_consecutive_failures", 0)} 条，'
          f'首次失败于 payload #{shape.get("first_failure_idx")}')
        if s.get('failure_elapsed_median') is not None:
            A(f'- 失败样本耗时中位 {s["failure_elapsed_median"]:.2f}s'
              + ('（秒拒，不是超时）' if s['failure_elapsed_median'] < 1 else ''))
        if shape.get('clustered'):
            A('- **失败高度聚集**：形态像渠道断供/服务下线，而不是限流或随机抖动。'
              '这两者的处置完全不同，聚合失败率区分不了。')
        else:
            A('- 失败分散分布：更像限流或偶发抖动。')
        A('')
        for f in s.get('failure_samples', []):
            A(f'  - `#{f["payload_idx"]}` {f["kind"]} (HTTP {f["status"]}): '
              f'{f["detail"][:160]}')
        A('')
    if not any_fail:
        A('所有组全部成功，无失败样本。')
        A('')

    # ── 缓存 ──────────────────────────────────────────────────────────────
    A('## 缓存')
    A('')
    rows = []
    for name in sorted(summaries):
        s = summaries[name]
        if not s.get('ok'):
            continue
        rows.append([name, _fmt_pct(s['cache_overall']), _fmt_pct(s['cache_median']),
                     f'{s["cache_hit_count"]}/{s["ok"]}',
                     f'{s["cached_tokens"]}/{s["prompt_tokens"]}'])
    A(_table(rows, ['组', '总体命中(按token加权)', '逐条命中中位',
                    '有命中的请求', 'cached/prompt tokens']))
    A('')
    A('两个口径都要看：总体命中反映实际省下的钱和 prefill 时间；逐条中位不被'
      '个别超长 prompt 带偏。')
    A('')

    # ── 方法与局限 ────────────────────────────────────────────────────────
    A('## 方法与局限')
    A('')
    A(f'- payload 取自本机真实请求日志 `{run_meta["corpus"]["source"]}`，'
      f'`messages` + `tools` 原样重放，模型名由评测组决定。')
    A(f'- 抽样方式 `{run_meta["corpus"]["sampling"]}`，'
      f'从 {run_meta["corpus"]["available"]} 条可用记录中取 '
      f'{run_meta["corpus"]["count"]} 条'
      + (f'，跳过坏行 {run_meta["corpus"]["bad_lines"]} 条'
         if run_meta['corpus'].get('bad_lines') else '') + '。')
    A(f'- `max_tokens={run_meta["request"]["max_tokens"]}`，串行发送（并发会让各组'
      '互相争带宽和配额，污染单请求延迟）。')
    if run_meta['run']['warmup']:
        A(f'- 已跑 {run_meta["run"]["warmup"]} 轮预热且不计入统计，消除冷/热缓存'
          '不对称——不预热等于在测「谁的缓存先被喂热」。')
    else:
        A('- **未预热**：先跑的组可能因冷缓存被低估。')
    if run_meta['run']['order'] == 'rotate':
        A('- 组顺序逐条轮换，抵消时段漂移。')
    else:
        A('- **未轮换组顺序**：组间差异里可能混入时段差异。')
    A(f'- 单次运行、单一测量位置。同一入口在办公网和在机器人上的表现可以完全不同；'
      '不同时段也会不同。跨时段复跑一轮再下定论更稳。')
    A('- 只测延迟/可用性/缓存，**不评估回答质量**。')
    A('')
    return '\n'.join(L)


def write(run_dir: pathlib.Path, run_meta: dict, records: list[dict]) -> pathlib.Path:
    """从原始结果算出全部统计并落盘报告。"""
    measured = [r for r in records if r.get('phase') == 'measure']
    if not measured:
        # 只跑了探活或只有预热轮时，仍然给一份能看的报告
        measured = [r for r in records if r.get('phase') in ('probe', 'warmup')]

    by_group: dict[str, list[dict]] = {}
    for r in measured:
        by_group.setdefault(r['group'], []).append(r)

    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    shapes = {n: st.failure_shape(rs) for n, rs in by_group.items()}
    noise = st.noise_floor(by_group)
    paired = st.paired(by_group)
    verdict = st.verdict(summaries, paired, noise)

    md = build(run_meta, summaries, paired, noise, shapes, verdict)
    (run_dir / 'report.md').write_text(md, encoding='utf-8')

    (run_dir / 'run.json').write_text(json.dumps({
        'meta': run_meta, 'summaries': summaries, 'paired': paired,
        'noise_floor': noise, 'failure_shapes': shapes, 'verdict': verdict,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return run_dir / 'report.md'


def now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()
