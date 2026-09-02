"""llm_bench 离线测试：不发任何真实请求。

重点不在「函数能跑」，而在**那些防止误判的约束真的生效**。这些约束每一条都对应
一次真实的错误结论（见 tools/llm_bench/README.md「为什么有这些多余步骤」），
所以它们必须有测试兜着，否则以后被人当冗余优化掉，工具就退化成了它要取代的那种
临时脚本。
"""
import json
import os
import pathlib
import sqlite3
import sys

import pytest

_TOOLS = pathlib.Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(_TOOLS))

# 必须走 llm_bench.* 包命名空间：agent-core 自己有 src/config.py，扁平的
# `import config` 在整套测试跑在一起时会被它顶掉（sys.modules 抢名字）。
from llm_bench import config as cfgmod  # noqa: E402
from llm_bench import corpus as corpusmod  # noqa: E402
from llm_bench import report as reportmod  # noqa: E402
from llm_bench import runner  # noqa: E402
from llm_bench import stats as st  # noqa: E402
from llm_bench.transport import Kind, Transport, classify  # noqa: E402


# ── 语料 ──────────────────────────────────────────────────────────────────────

def _rec(i, n_msgs=4):
    return {'request_id': f'req-{i}', 'model': 'glm-5.2',
            'messages': [{'role': 'user', 'content': f'q{i}-{j}'} for j in range(n_msgs)],
            'tools': [{'type': 'function', 'function': {'name': 'finish'}}],
            'ts': 1788000000 + i}


def _write_jsonl(path, records, tail: bytes = b''):
    with path.open('wb') as f:
        for r in records:
            f.write((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8'))
        if tail:
            f.write(tail)


def test_truncated_tail_does_not_raise(tmp_path):
    """半条记录（甚至截在 UTF-8 中间）不能让整个文件不可用。

    真实故障：非正常关机让 llm_request_*.jsonl 停在 0xe5 上。text-mode 迭代会抛
    UnicodeDecodeError，于是最后一行的损坏变成整个语料不可读。
    """
    f = tmp_path / 'llm_request_1.jsonl'
    # 截断的多字节序列 + 一条语法不完整的 JSON
    _write_jsonl(f, [_rec(i) for i in range(5)], tail=b'{"messages": [{"role": "\xe5')

    records, meta = corpusmod.load_records(str(tmp_path))
    assert len(records) == 5
    assert meta['bad_lines'] == 1, '坏行必须被计数并上报，不能静默吞掉'


def test_bad_lines_at_eof_are_counted(tmp_path):
    """损坏最常出现在文件末尾——计数不能漏掉最后一批。"""
    f = tmp_path / 'llm_request_1.jsonl'
    _write_jsonl(f, [_rec(i) for i in range(3)], tail=b'garbage1\ngarbage2\ngarbage3\n')
    _, meta = corpusmod.load_records(str(tmp_path))
    assert meta['bad_lines'] == 3


def test_min_messages_filters_degenerate(tmp_path):
    f = tmp_path / 'llm_request_1.jsonl'
    _write_jsonl(f, [_rec(0, n_msgs=1), _rec(1, n_msgs=5), _rec(2, n_msgs=2)])
    records, _ = corpusmod.load_records(str(tmp_path), min_messages=2)
    assert [r['request_id'] for r in records] == ['req-1', 'req-2']


def test_even_sampling_spans_whole_file():
    """均匀抽样必须覆盖整个文件，不能退化成只取开头。

    朴素的 `records[::len//count]` 在 count 接近 len 时正是这个毛病。
    """
    records = [_rec(i) for i in range(97)]
    picked = corpusmod.sample(records, 50, 'even')
    assert len(picked) == 50
    ids = [int(r['request_id'].split('-')[1]) for r in picked]
    assert ids == sorted(ids)
    assert ids[0] == 0
    assert ids[-1] > 90, f'最后一个样本 idx={ids[-1]}，没覆盖到文件尾部'


def test_even_sampling_is_deterministic():
    records = [_rec(i) for i in range(97)]
    assert corpusmod.sample(records, 20, 'even') == corpusmod.sample(records, 20, 'even')


def test_fingerprint_detects_different_payload_set():
    a = corpusmod.fingerprint([_rec(i) for i in range(10)])
    assert a == corpusmod.fingerprint([_rec(i) for i in range(10)])
    assert a != corpusmod.fingerprint([_rec(i) for i in range(1, 11)])


# ── 配置 ──────────────────────────────────────────────────────────────────────

def _yaml(tmp_path, body: str) -> str:
    p = tmp_path / 'bench.yaml'
    p.write_text(body, encoding='utf-8')
    return str(p)


def test_models_expand_into_groups(tmp_path, monkeypatch):
    monkeypatch.setenv('K1', 'sk-secret-aaaa')
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - name: router
    url: https://r.example.com/v1
    key_env: K1
    models: [glm-5.2, zai-org/glm-5.3]
"""))
    groups = cfgmod.build_groups(cfg)
    assert [g.name for g in groups] == ['router/glm-5.2', 'router/zai-org/glm-5.3']
    assert all(g.key == 'sk-secret-aaaa' for g in groups)


def test_single_model_keeps_plain_name(tmp_path, monkeypatch):
    monkeypatch.setenv('K1', 'sk-x')
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: solo, url: https://a/v1, key_env: K1, model: glm-5.3}
"""))
    assert [g.name for g in cfgmod.build_groups(cfg)] == ['solo']


def test_key_file_with_line_number(tmp_path):
    """兼容一行一个值的密钥文件。"""
    kf = tmp_path / '.a'
    kf.write_text('https://u/v1\nsk-line-two\nglm-5.3\n', encoding='utf-8')
    cfg = cfgmod.load(_yaml(tmp_path, f"""
groups:
  - name: g
    url: https://u/v1
    key_file: "{kf}#2"
    model: glm-5.3
"""))
    assert cfgmod.build_groups(cfg)[0].key == 'sk-line-two'


def test_key_file_bad_line_number_is_explicit(tmp_path):
    kf = tmp_path / '.a'
    kf.write_text('only-one-line\n', encoding='utf-8')
    cfg = cfgmod.load(_yaml(tmp_path, f"""
groups:
  - {{name: g, url: https://u/v1, key_file: "{kf}#9", model: m}}
"""))
    with pytest.raises(cfgmod.ConfigError, match='只有 1 行'):
        cfgmod.build_groups(cfg)


def test_missing_env_key_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.delenv('NOPE', raising=False)
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: g, url: https://u/v1, key_env: NOPE, model: m}
"""))
    with pytest.raises(cfgmod.ConfigError, match='NOPE'):
        cfgmod.build_groups(cfg)


def test_duplicate_group_names_are_disambiguated(tmp_path, monkeypatch):
    """重名会让报告里两行无法区分，也会让 resume 的三元组键冲突。"""
    monkeypatch.setenv('K1', 'sk-x')
    cfg = cfgmod.load(_yaml(tmp_path, """
groups:
  - {name: same, url: https://a/v1, key_env: K1, model: m}
  - {name: same, url: https://b/v1, key_env: K1, model: m}
"""))
    assert [g.name for g in cfgmod.build_groups(cfg)] == ['same', 'same#2']


def test_include_current_reads_configdb(tmp_path):
    db = tmp_path / 'data.db'
    conn = sqlite3.connect(db)
    conn.execute('CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)')
    conn.execute("INSERT INTO config VALUES ('client', ?)", (json.dumps(
        {'llm': [{'url': 'https://prod/v1', 'key': 'sk-prod-key',
                  'model': 'glm-5.3', 'think_mode': False}]}),))
    conn.commit()
    conn.close()

    cfg = cfgmod.load(None, {'include_current': True})
    cfg['include_current'] = True
    groups = cfgmod.build_groups(cfg, str(db))
    assert len(groups) == 1
    g = groups[0]
    assert g.name == 'current/glm-5.3'
    assert g.source == 'configdb'
    # think_mode=false 时生产会带这个 extra_body（src/client/llm.py:136-139）
    assert g.extra_body['chat_template_kwargs'] == {'enable_thinking': False}


def test_configdb_absent_is_not_fatal_when_groups_exist(tmp_path, monkeypatch):
    monkeypatch.setenv('K1', 'sk-x')
    cfg = cfgmod.load(_yaml(tmp_path, """
include_current: true
groups:
  - {name: g, url: https://u/v1, key_env: K1, model: m}
"""))
    groups = cfgmod.build_groups(cfg, str(tmp_path / 'missing.db'))
    assert [g.name for g in groups] == ['g']


# ── 密钥打码 ──────────────────────────────────────────────────────────────────

SECRET = 'sk-RjAbwdQbzU8eURU2ikUJJaXfB0wn50jPD54ssxU8fx1ojpt5'


def test_mask_keeps_ends_only():
    m = cfgmod.mask(SECRET)
    assert SECRET not in m
    assert m.startswith('sk-RjA') and m.endswith('jpt5')


def test_plaintext_key_never_reaches_serialized_output(tmp_path):
    """明文密钥不允许出现在 run.json / report.md 的任何位置。"""
    g = cfgmod.Group('g', 'https://u/v1', SECRET, 'glm-5.3')
    assert SECRET not in json.dumps(g.public(), ensure_ascii=False)

    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta(groups=[g.public()])
    reportmod.write(run_dir, meta, [_res('g', i, ok=True) for i in range(4)])

    for name in ('run.json', 'report.md'):
        text = (run_dir / name).read_text(encoding='utf-8')
        assert SECRET not in text, f'{name} 泄露了明文 key'


# ── 传输层错误分类 ────────────────────────────────────────────────────────────

def test_model_unavailable_is_not_lumped_into_server_error():
    """「无可用渠道」必须和普通 5xx 分开：前者重试无意义。

    router 对 glm-5.3 连续返回 503「无可用渠道」，归成 server_error 会被误读成过载。
    """
    body = '{"error":{"code":"model_not_found","message":"分组 default 下模型 glm-5.3 无可用渠道（distributor）"}}'
    assert classify(503, body) == Kind.MODEL_UNAVAILABLE
    assert classify(503, 'upstream overloaded') == Kind.SERVER_ERROR


@pytest.mark.parametrize('status,body,expect', [
    (429, '', Kind.RATE_LIMIT),
    (402, '', Kind.BILLING),
    (401, '', Kind.AUTH),
    (403, '', Kind.AUTH),
    (400, 'messages 参数非法', Kind.BAD_REQUEST),
    (400, 'maximum context length exceeded', Kind.CONTEXT_OVERFLOW),
    (502, '', Kind.SERVER_ERROR),
])
def test_classify_matches_production_vocabulary(status, body, expect):
    assert classify(status, body) == expect


def test_connection_and_timeout_from_exception():
    assert classify(None, '', TimeoutError('read timeout')) == Kind.TIMEOUT
    assert classify(None, '', OSError('connection refused')) == Kind.CONNECTION


def test_200_with_error_body_counts_as_failure():
    """网关把错误塞进 200 体里。当成功会凭空拉高成功率，且这些"成功"极快，
    还会把延迟中位数拉低。"""
    r = Transport._fail(200, '{"message": "quota exceeded"}', None, 0.05,
                        note='200 响应体内含 error')
    assert r['ok'] is False
    assert '200 响应体内含 error' in r['detail']


# ── 统计约束 ──────────────────────────────────────────────────────────────────

def _res(group, idx, ok=True, elapsed=5.0, repeat=0, phase='measure',
         prompt=1000, cached=900, completion=100, kind='ok', status=200):
    return {'group': group, 'payload_idx': idx, 'repeat': repeat, 'phase': phase,
            'ok': ok, 'kind': kind if ok else kind, 'status': status,
            'elapsed': elapsed, 'prompt': prompt if ok else 0,
            'completion': completion if ok else 0, 'cached': cached if ok else 0,
            'finish': 'stop' if ok else None,
            'err': None if ok else f'HTTP {status}',
            'detail': None if ok else 'boom', 'ts': 1788000000 + idx}


def _meta(groups=None, warmup=1, repeats=2, count=4):
    return {
        'run_id': 'test', 'mode': 'bench', 'config': {},
        'groups': groups or [], 'hostname': 'test-host', 'image_tag': 'test',
        'transport': 'urllib', 'started_at': 'now', 'finished_at': 'now',
        'corpus': {'source': 'x', 'count': count, 'available': 100,
                   'sampling': 'even', 'fingerprint': 'abc123', 'bad_lines': 0},
        'request': {'max_tokens': 10240}, 'run': {'warmup': warmup,
                                                  'repeats': repeats,
                                                  'order': 'rotate'},
    }


def test_noise_floor_from_repeats():
    """同组不同轮次的差值就是噪声基线。"""
    by_group = {'A': [_res('A', 0, elapsed=5.0, repeat=0),
                      _res('A', 0, elapsed=6.0, repeat=1),
                      _res('A', 1, elapsed=4.0, repeat=0),
                      _res('A', 1, elapsed=4.5, repeat=1)]}
    n = st.noise_floor(by_group)
    assert n['available'] is True
    assert n['median_abs_delta'] == pytest.approx(0.75)


def test_noise_floor_unavailable_with_single_repeat():
    by_group = {'A': [_res('A', i, repeat=0) for i in range(5)]}
    assert st.noise_floor(by_group)['available'] is False


def test_verdict_refuses_to_rank_within_noise():
    """核心约束：组间差异小于重复测量波动时不排名。

    真实数据：别名对比的组间中位差 0.39s，而同配置重测波动 0.93s。
    """
    by_group = {
        'A': [_res('A', i, elapsed=6.0 if r == 0 else 7.0, repeat=r)
              for i in range(6) for r in (0, 1)],
        'B': [_res('B', i, elapsed=5.8 if r == 0 else 6.9, repeat=r)
              for i in range(6) for r in (0, 1)],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    noise = st.noise_floor(by_group)
    paired = st.paired(by_group)
    v = st.verdict(summaries, paired, noise)

    assert v.get('indistinguishable') is True
    assert v['recommend'] is None
    assert any('噪声基线' in r for r in v['reasons'])


def test_verdict_ranks_when_gap_exceeds_noise():
    by_group = {
        'slow': [_res('slow', i, elapsed=20.0 if r == 0 else 20.1, repeat=r)
                 for i in range(6) for r in (0, 1)],
        'fast': [_res('fast', i, elapsed=2.0 if r == 0 else 2.1, repeat=r)
                 for i in range(6) for r in (0, 1)],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    v = st.verdict(summaries, st.paired(by_group), st.noise_floor(by_group))
    assert v.get('indistinguishable') is not True
    assert v['recommend'] == 'fast'


def test_verdict_puts_availability_before_latency():
    """一个入口再快，成功率不满就不是候选。

    真实数据：router/glm-5.3 延迟和对手持平，但 35/50 成功。
    """
    by_group = {
        'flaky_fast': [_res('flaky_fast', i, ok=(i < 3), elapsed=1.0, status=503,
                            kind='model_unavailable') for i in range(10)],
        'solid_slow': [_res('solid_slow', i, elapsed=9.0) for i in range(10)],
    }
    summaries = {n: st.summarize_group(rs) for n, rs in by_group.items()}
    v = st.verdict(summaries, st.paired(by_group), st.noise_floor(by_group))
    assert v['primary'] == 'availability'
    assert v['recommend'] == 'solid_slow', '不能推荐一个成功率不满的组'


def test_failure_shape_detects_clustering():
    """连续 14 次 503 是渠道断供；散布的 14 次是限流。聚合失败率区分不了。"""
    clustered = [_res('A', i, ok=(i < 36)) for i in range(50)]
    shape = st.failure_shape(clustered)
    assert shape['longest_consecutive_failures'] == 14
    assert shape['first_failure_idx'] == 36
    assert shape['clustered'] is True

    scattered = [_res('A', i, ok=(i % 4 != 0)) for i in range(50)]
    assert st.failure_shape(scattered)['clustered'] is False


def test_summarize_separates_failure_latency():
    """秒拒的失败不能混进延迟分位，否则失败更多的组反而显得更快。"""
    recs = [_res('A', i, ok=True, elapsed=10.0) for i in range(5)] + \
           [_res('A', i + 5, ok=False, elapsed=0.06) for i in range(5)]
    s = st.summarize_group(recs)
    assert s['ok'] == 5 and s['failed'] == 5
    assert s['complete'] is False
    assert s['p50'] == pytest.approx(10.0), '失败样本不应拉低延迟分位'
    assert s['failure_elapsed_median'] == pytest.approx(0.06)


def test_paired_uses_only_payloads_all_groups_succeeded():
    by_group = {
        'A': [_res('A', i, elapsed=5.0) for i in range(5)],
        # B 在 payload 3、4 上失败 → 配对集只剩 0,1,2
        'B': [_res('B', i, ok=(i < 3), elapsed=6.0) for i in range(5)],
    }
    p = st.paired(by_group)
    assert p['n_common'] == 3
    assert p['pairs']['A vs B']['median_delta'] == pytest.approx(-1.0)


# ── 报告 ──────────────────────────────────────────────────────────────────────

def test_report_flags_survivor_bias(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res('A', i, ok=(i < 7)) for i in range(10)] + \
           [_res('B', i) for i in range(10)]
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '幸存者偏差' in md
    assert '(7/10)' in md, '成功率不满时延迟分位必须带样本量'


def test_report_warns_when_no_noise_baseline(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res(g, i) for g in ('A', 'B') for i in range(5)]
    reportmod.write(run_dir, _meta(repeats=1), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '噪声基线' in md
    assert 'repeats' in md


def test_report_omits_sign_flip_note_when_signs_agree(tmp_path):
    """无条件打印「中位与均值符号相反」会在两者同号时说一句假话。

    报告里出现一句可验证为假的解读，读者就不会再相信其余部分。
    """
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    # 两组都是恒定延迟 → 中位与均值必然同号
    recs = [_res('A', i, elapsed=6.0) for i in range(6)] + \
           [_res('B', i, elapsed=5.0) for i in range(6)]
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '符号相反' not in md


def test_report_shows_sign_flip_note_when_it_happens(tmp_path):
    """一方多数请求略快但吃到更重长尾时，必须提示。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    # A 在多数 payload 上略慢（中位 A-B > 0），但 B 有一个极端长尾拉高均值
    recs = []
    for i in range(6):
        recs.append(_res('A', i, elapsed=5.0))
        recs.append(_res('B', i, elapsed=4.0 if i < 5 else 40.0))
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '符号相反' in md


def test_report_separates_request_and_payload_units(tmp_path):
    """失败计数混用「请求」和「payload」两种单位会让人读错严重程度。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    # 2 轮 × 3 条失败 payload = 6 次失败请求
    recs = []
    for r in (0, 1):
        for i in range(10):
            recs.append(_res('A', i, ok=(i < 7), repeat=r))
            recs.append(_res('B', i, repeat=r))
    reportmod.write(run_dir, _meta(), recs)
    md = (run_dir / 'report.md').read_text(encoding='utf-8')
    assert '失败 6/20 次请求' in md
    assert '受影响 payload 3 条' in md


def test_report_survives_a_totally_dead_group(tmp_path):
    """某组全挂不能影响其他组出报告——否则一次评测的全部开销都白费。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    recs = [_res('dead', i, ok=False, kind='model_unavailable', status=503)
            for i in range(5)] + [_res('alive', i) for i in range(5)]
    path = reportmod.write(run_dir, _meta(), recs)
    md = path.read_text(encoding='utf-8')
    assert 'dead' in md and 'alive' in md
    data = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    assert data['summaries']['dead']['ok'] == 0
    assert data['verdict']['recommend'] == 'alive'


def test_report_only_recomputes_from_results(tmp_path):
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    sink = runner.ResultSink(run_dir / 'results.jsonl')
    for i in range(4):
        sink.write(_res('A', i))
    sink.close()
    recs = runner.load_results(run_dir / 'results.jsonl')
    assert len(recs) == 4
    assert reportmod.write(run_dir, _meta(), recs).is_file()


# ── 续跑 ──────────────────────────────────────────────────────────────────────

def test_results_are_flushed_immediately(tmp_path):
    """不 flush 就没有续跑：被 Ctrl-C 时缓冲区里的结果连同花掉的钱一起丢。"""
    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    sink.write(_res('A', 0))
    assert len(runner.load_results(p)) == 1, '写入后未落盘'
    sink.close()


def test_resume_skips_completed_triples(tmp_path):
    """续跑只补缺失的 (组, payload, 轮次, 阶段)。"""
    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    for i in range(3):
        sink.write(_res('A', i, phase='measure', repeat=0))
    sink.close()

    done = {(r['group'], r['payload_idx'], r['repeat'], r['phase'])
            for r in runner.load_results(p)}

    calls = []

    class FakeTransport:
        def post_json(self, url, key, payload):
            calls.append(payload['messages'][0]['content'])
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    g = cfgmod.Group('A', 'https://u/v1', 'sk-x', 'm')
    payloads = [_rec(i) for i in range(5)]
    sink = runner.ResultSink(p)
    runner.run([g], payloads, FakeTransport(),
               {'request': {'max_tokens': 8}, 'run': {'warmup': 0, 'repeats': 1}},
               sink, done, log=lambda *a, **k: None)
    sink.close()

    assert len(calls) == 2, f'应只补 2 条，实际发了 {len(calls)} 条'
    assert len(runner.load_results(p)) == 5


def test_meta_is_written_before_any_request(tmp_path):
    """run.json 必须在开跑前就存在。

    真实故障：一次 450 请求的 run 被 kill -9 后，目录里只剩 results.jsonl。
    --resume 和 --report-only 都从 run.json 取 meta，于是这次 run 既不能续跑也
    不能出报告 —— 恰好是续跑存在的唯一场景。
    """
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta()
    reportmod.write_meta(run_dir, meta)

    assert (run_dir / 'run.json').is_file()
    loaded = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    assert loaded['meta']['corpus']['fingerprint'] == 'abc123'


def test_partial_run_can_still_produce_a_report(tmp_path):
    """被打断的 run 用 meta + 部分 results 就能出报告，并标注不完整。"""
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    meta = _meta()
    meta['incomplete'] = '172/450 次请求'
    reportmod.write_meta(run_dir, meta)

    sink = runner.ResultSink(run_dir / 'results.jsonl')
    for g in ('A', 'B'):
        for i in range(4):
            sink.write(_res(g, i))
    sink.close()

    recs = runner.load_results(run_dir / 'results.jsonl')
    md = reportmod.write(run_dir, meta, recs).read_text(encoding='utf-8')
    assert '本次结果不完整' in md
    assert '172/450' in md


def test_resume_reselects_exact_payloads_from_grown_corpus():
    """语料是活的：agent-core 边跑边追加日志。

    续跑必须按 request_id 精确复原原样本集。如果重新抽样，一轮几十分钟的评测被
    中断后语料几乎必然已经变了，`even` 会选出另一批 payload —— 那时指纹校验会
    正确地拒绝，但「续跑」在它唯一有用的场景里就永远不可用了。
    """
    original = [_rec(i) for i in range(20)]
    picked = corpusmod.sample(original, 5, 'even')
    ids = corpusmod.request_ids(picked)
    fp = corpusmod.fingerprint(picked)

    # 评测跑到一半，agent-core 又写进来 30 条新记录
    grown = original + [_rec(i) for i in range(100, 130)]

    # 重新抽样会选出完全不同的一批 —— 这正是不能重新抽样的原因
    assert corpusmod.request_ids(corpusmod.sample(grown, 5, 'even')) != ids

    restored = corpusmod.reselect(grown, ids)
    assert corpusmod.request_ids(restored) == ids
    assert corpusmod.fingerprint(restored) == fp


def test_reselect_rejects_when_records_rotated_away():
    """日志被轮转/清理后原样本找不回来，必须明确报错而不是悄悄换一批。"""
    ids = corpusmod.request_ids([_rec(i) for i in range(5)])
    with pytest.raises(corpusmod.CorpusError, match='找不到'):
        corpusmod.reselect([_rec(i) for i in range(2)], ids)


def test_reselect_rejects_records_without_request_id():
    records = [{'messages': [{'role': 'user', 'content': 'x'}]}]
    with pytest.raises(corpusmod.CorpusError, match='request_id'):
        corpusmod.reselect(records, [None])


def test_resume_load_tolerates_corrupt_line(tmp_path):
    p = tmp_path / 'results.jsonl'
    with p.open('wb') as f:
        f.write((json.dumps(_res('A', 0)) + '\n').encode())
        f.write(b'{"group": "A", broken\n')
        f.write((json.dumps(_res('A', 1)) + '\n').encode())
    assert len(runner.load_results(p)) == 2


def test_run_continues_after_a_failure(tmp_path):
    """单次失败不能中断整批——否则既没结果也没失败样本可分析。"""
    class FlakyTransport:
        def __init__(self):
            self.n = 0

        def post_json(self, url, key, payload):
            self.n += 1
            if self.n == 2:
                return {'ok': False, 'kind': 'server_error', 'status': 503,
                        'elapsed': 0.06, 'prompt': 0, 'completion': 0, 'cached': 0,
                        'finish': None, 'err': 'HTTP 503', 'detail': 'boom'}
            return {'ok': True, 'kind': 'ok', 'status': 200, 'elapsed': 1.0,
                    'prompt': 10, 'completion': 5, 'cached': 0,
                    'finish': 'stop', 'err': None, 'detail': None}

    p = tmp_path / 'results.jsonl'
    sink = runner.ResultSink(p)
    runner.run([cfgmod.Group('A', 'https://u/v1', 'sk-x', 'm')],
               [_rec(i) for i in range(4)], FlakyTransport(),
               {'request': {'max_tokens': 8}, 'run': {'warmup': 0, 'repeats': 1}},
               sink, log=lambda *a, **k: None)
    sink.close()
    recs = runner.load_results(p)
    assert len(recs) == 4
    assert sum(1 for r in recs if not r['ok']) == 1


def test_stop_after_consecutive_failures_only_skips_that_group():
    """一个入口挂了不该毁掉整轮评测。"""
    class DeadForA:
        def post_json(self, url, key, payload):
            dead = 'dead' in url
            return {'ok': not dead, 'kind': 'model_unavailable' if dead else 'ok',
                    'status': 503 if dead else 200, 'elapsed': 0.05,
                    'prompt': 0 if dead else 10, 'completion': 0 if dead else 5,
                    'cached': 0, 'finish': None if dead else 'stop',
                    'err': 'HTTP 503' if dead else None,
                    'detail': 'no channel' if dead else None}

    written = []

    class Sink:
        def write(self, rec):
            written.append(rec)

    groups = [cfgmod.Group('dead', 'https://dead/v1', 'k', 'm'),
              cfgmod.Group('ok', 'https://ok/v1', 'k', 'm')]
    runner.run(groups, [_rec(i) for i in range(10)], DeadForA(),
               {'request': {'max_tokens': 8},
                'run': {'warmup': 0, 'repeats': 1,
                        'stop_after_consecutive_failures': 3}},
               Sink(), log=lambda *a, **k: None)

    dead = [r for r in written if r['group'] == 'dead']
    alive = [r for r in written if r['group'] == 'ok']
    assert len(dead) == 3, f'挂掉的组应在 3 次后停，实际 {len(dead)} 次'
    assert len(alive) == 10, '健康的组必须跑完'
