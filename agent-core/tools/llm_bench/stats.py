"""统计：配对比较、噪声基线、幸存者偏差、失败形态。

这个模块的职责一半是算数字，一半是**阻止过度解读**。手工评测时我犯过的每个
解读错误都在这里有一条对应的约束。
"""
import statistics


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * p))]


def _safe_stdev(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def summarize_group(records: list[dict]) -> dict:
    """单组汇总。失败样本单独统计，不混进延迟分位。

    为什么必须分开：很多失败是秒拒（0.06s 的 503），混进平均值会让**失败更多的
    组看起来更快**。手工评测时一个组 15/50 失败，若不分开统计，它的均值反而漂亮。
    """
    ok = [r for r in records if r.get('ok')]
    bad = [r for r in records if not r.get('ok')]
    lat = [r['elapsed'] for r in ok]

    prompt = sum(r.get('prompt', 0) for r in ok)
    cached = sum(r.get('cached', 0) for r in ok)
    completion = sum(r.get('completion', 0) for r in ok)
    per_cache = [r['cached'] / r['prompt'] for r in ok if r.get('prompt')]

    kinds: dict[str, int] = {}
    for r in bad:
        k = r.get('kind') or 'unknown'
        kinds[k] = kinds.get(k, 0) + 1

    out = {
        'total': len(records),
        'ok': len(ok),
        'failed': len(bad),
        'success_rate': len(ok) / len(records) if records else 0.0,
        'complete': len(bad) == 0,
        'failure_kinds': kinds,
        'failure_samples': [
            {'payload_idx': r.get('payload_idx'), 'kind': r.get('kind'),
             'status': r.get('status'), 'detail': (r.get('detail') or '')[:200]}
            for r in bad[:3]
        ],
        # 失败样本自己的耗时分布：秒拒和超时是两种完全不同的故障
        'failure_elapsed_median': statistics.median(
            [r['elapsed'] for r in bad]) if bad else None,
    }

    if lat:
        out.update({
            'p50': pct(lat, .5), 'p90': pct(lat, .9), 'p95': pct(lat, .95),
            'min': min(lat), 'max': max(lat),
            'mean': statistics.mean(lat), 'stdev': _safe_stdev(lat),
            'cv': _safe_stdev(lat) / statistics.mean(lat) if statistics.mean(lat) else 0,
            'prompt_tokens': prompt, 'completion_tokens': completion,
            'cached_tokens': cached,
            'cache_overall': cached / prompt if prompt else 0.0,
            'cache_median': statistics.median(per_cache) if per_cache else 0.0,
            'cache_hit_count': sum(1 for c in per_cache if c > 0),
            'throughput': completion / sum(lat) if sum(lat) else 0.0,
        })
    return out


def failure_shape(records: list[dict]) -> dict:
    """失败是连续的还是散布的。

    聚合失败率区分不了这两种情况，但处置完全不同：连续 14 次 503 是渠道断供，
    散布的 14 次是限流或抖动。手工评测时正是「全部集中在 idx36→49」这个形态
    才让我确认是断供而不是偶发。
    """
    by_idx = {}
    for r in records:
        by_idx.setdefault(r.get('payload_idx', 0), []).append(r)

    seq = [all(x.get('ok') for x in by_idx[i]) for i in sorted(by_idx)]
    longest = cur = 0
    first_fail = None
    for i, ok in enumerate(seq):
        if ok:
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
            if first_fail is None:
                first_fail = sorted(by_idx)[i]
    n_fail = sum(1 for ok in seq if not ok)
    return {
        'longest_consecutive_failures': longest,
        'first_failure_idx': first_fail,
        'failed_payloads': n_fail,
        # 连续段占了绝大多数失败 → 断供形态而非随机抖动
        'clustered': bool(n_fail and longest >= max(3, n_fail * 0.6)),
    }


def noise_floor(by_group: dict[str, list[dict]]) -> dict:
    """用同一组不同轮次之间的差值估噪声。

    这是整个工具最重要的一条约束。手工评测时，同一个入口、同一个模型、同一批
    payload 跑两遍，中位差是 0.93s；而被拿来做结论的两组之间的中位差只有 0.39s。
    也就是说**重复测量自身的波动比组间差异还大**，任何排名都是读噪声。
    """
    per_group = {}
    all_deltas = []
    for name, recs in by_group.items():
        # (payload_idx -> {repeat: elapsed})，只取成功样本
        by_idx: dict[int, dict[int, float]] = {}
        for r in recs:
            if r.get('ok'):
                by_idx.setdefault(r['payload_idx'], {})[r['repeat']] = r['elapsed']

        deltas = []
        for reps in by_idx.values():
            if len(reps) < 2:
                continue
            vals = [reps[k] for k in sorted(reps)]
            # 相邻轮次两两之差的绝对值
            deltas += [abs(b - a) for a, b in zip(vals, vals[1:])]

        if deltas:
            per_group[name] = {
                'n': len(deltas),
                'median_abs_delta': statistics.median(deltas),
                'p90_abs_delta': pct(deltas, .9),
            }
            all_deltas += deltas

    if not all_deltas:
        return {'available': False, 'per_group': per_group,
                'reason': 'repeats < 2 或成功样本不足，无法估计噪声'}
    return {
        'available': True,
        'per_group': per_group,
        'median_abs_delta': statistics.median(all_deltas),
        'p90_abs_delta': pct(all_deltas, .9),
        'n': len(all_deltas),
    }


def paired(by_group: dict[str, list[dict]]) -> dict:
    """配对比较：只用**所有组都成功**的 payload。

    消除 payload 难度差异——不同 payload 的 prompt 长度和生成长度差一个数量级，
    直接比两组的 p50 会被抽到的样本构成左右。
    """
    names = sorted(by_group)
    # payload_idx -> group -> 该组在各轮的中位延迟
    per_idx: dict[int, dict[str, float]] = {}
    for name, recs in by_group.items():
        acc: dict[int, list[float]] = {}
        for r in recs:
            if r.get('ok'):
                acc.setdefault(r['payload_idx'], []).append(r['elapsed'])
        for idx, vals in acc.items():
            per_idx.setdefault(idx, {})[name] = statistics.median(vals)

    common = sorted(i for i, d in per_idx.items() if len(d) == len(names))
    out = {'n_common': len(common), 'groups': names, 'pairs': {}}
    if not common or len(names) < 2:
        return out

    out['per_group_median_on_common'] = {
        n: statistics.median([per_idx[i][n] for i in common]) for n in names
    }

    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            d = [per_idx[i][a] - per_idx[i][b] for i in common]
            out['pairs'][f'{a} vs {b}'] = {
                'median_delta': statistics.median(d),
                'mean_delta': statistics.mean(d),
                'a_faster': sum(1 for x in d if x < 0),
                'b_faster': sum(1 for x in d if x > 0),
                'n': len(d),
            }
    return out


def verdict(summaries: dict, paired_res: dict, noise: dict) -> dict:
    """生成结论。这段刻意保守——宁可说「测不出差别」也不编排名。

    优先级：可用性 > 延迟。一个入口再快，成功率不满就不是候选；这正是
    router/glm-5.3 那次的情况（35/50 成功，延迟却和对手持平）。
    """
    incomplete = {n: s for n, s in summaries.items() if not s.get('complete')}
    usable = {n: s for n, s in summaries.items() if s.get('complete') and s.get('ok')}

    res = {'incomplete': sorted(incomplete), 'reasons': []}

    if incomplete:
        res['primary'] = 'availability'
        for n, s in sorted(incomplete.items()):
            kinds = ', '.join(f'{k}×{v}' for k, v in sorted(s['failure_kinds'].items()))
            res['reasons'].append(
                f'{n} 成功率 {s["ok"]}/{s["total"]}（{kinds}）——可用性问题优先于延迟')
        if usable:
            res['recommend'] = sorted(
                usable, key=lambda n: usable[n].get('p50', float('inf')))[0]
            res['reasons'].append(
                f'建议 {res["recommend"]}：本轮全部成功')
        return res

    res['primary'] = 'latency'
    pairs = paired_res.get('pairs') or {}
    if not pairs:
        res['recommend'] = None
        res['reasons'].append('配对样本不足，无法比较')
        return res

    floor = noise.get('median_abs_delta') if noise.get('available') else None
    # 所有两两差异都在噪声之内 → 不排名
    deltas = [abs(v['median_delta']) for v in pairs.values()]
    if floor is not None and max(deltas) <= floor:
        res['recommend'] = None
        res['indistinguishable'] = True
        res['reasons'].append(
            f'最大组间中位差 {max(deltas):.2f}s ≤ 噪声基线 {floor:.2f}s'
            f'（同配置重复测量的波动），本轮测不出显著差异')
        return res

    ranked = sorted(usable, key=lambda n: usable[n].get('p50', float('inf')))
    res['recommend'] = ranked[0] if ranked else None
    res['ranking'] = ranked
    if floor is not None:
        res['reasons'].append(f'噪声基线 {floor:.2f}s（同配置重复测量的中位波动）')
        near = [k for k, v in pairs.items() if abs(v['median_delta']) <= floor]
        if near:
            res['reasons'].append(
                '以下组对的差异在噪声以内，不应视为有差别: ' + '; '.join(near))
    else:
        res['reasons'].append(
            '未估出噪声基线（repeats < 2）——单轮结果的排名可能只是波动，'
            '建议用 --repeats 2 复核')
    return res
