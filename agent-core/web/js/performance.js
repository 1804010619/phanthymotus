/**
 * performance.js — 性能分析 Dashboard
 *
 * 展示每次 turn 各阶段耗时的瀑布图、趋势和聚合统计。
 */

const STAGE_COLORS = {
  vad:       { color: '#6366f1', label: 'VAD' },
  asr:       { color: '#8b5cf6', label: 'ASR' },
  collector: { color: '#94a3b8', label: '调度' },
  llm:       { color: '#f59e0b', label: 'LLM' },
  tool:      { color: '#10b981', label: '工具' },
  tts:       { color: '#06b6d4', label: 'TTS' },
};

let _refreshTimer = null;
let _currentRange = '24h';

export function initPerformance() {
  const overlay = document.getElementById('performance-overlay');
  if (!overlay) return;

  // Close button
  document.getElementById('performance-close')?.addEventListener('click', _close);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) _close();
  });

  // Range selector
  document.getElementById('perf-range')?.addEventListener('change', (e) => {
    _currentRange = e.target.value;
    _load();
  });

  // Refresh button
  document.getElementById('perf-refresh')?.addEventListener('click', _load);

  // Settings dropdown trigger
  document.getElementById('btn-performance')?.addEventListener('click', _open);
}

function _open() {
  document.getElementById('performance-overlay')?.classList.remove('hidden');
  _load();
  _refreshTimer = setInterval(_load, 10000);
}

function _close() {
  document.getElementById('performance-overlay')?.classList.add('hidden');
  if (_refreshTimer) {
    clearInterval(_refreshTimer);
    _refreshTimer = null;
  }
}

function _rangeToTs() {
  const now = Date.now() / 1000;
  const map = { '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800 };
  const offset = map[_currentRange] || 86400;
  return { start: now - offset, end: 0 };
}

async function _load() {
  const { start, end } = _rangeToTs();
  try {
    const [turnsRes, aggRes] = await Promise.all([
      fetch(`/api/performance/latest?n=50`),
      fetch(`/api/performance/aggregate?start=${start}&end=${end}`),
    ]);
    const turns = await turnsRes.json();
    const agg = await aggRes.json();
    _renderSummary(agg);
    _renderWaterfall(Array.isArray(turns) ? turns : (turns.turns || []));
  } catch (e) {
    console.error('[performance] load error:', e);
  }
}

function _renderSummary(agg) {
  const el = document.getElementById('perf-summary-cards');
  if (!el) return;

  if (!agg || agg.count === 0) {
    el.innerHTML = '<div class="perf-empty">暂无数据</div>';
    return;
  }

  const avg = agg.avg || {};
  const p95 = agg.p95 || {};

  el.innerHTML = `
    <div class="perf-cards">
      <div class="perf-card">
        <div class="perf-card-value">${agg.count}</div>
        <div class="perf-card-label">总轮次</div>
      </div>
      <div class="perf-card">
        <div class="perf-card-value">${_fmtMs(avg.total_duration_ms)}</div>
        <div class="perf-card-label">平均总耗时</div>
      </div>
      <div class="perf-card">
        <div class="perf-card-value">${_fmtMs(p95.total_duration_ms)}</div>
        <div class="perf-card-label">P95 总耗时</div>
      </div>
      <div class="perf-card">
        <div class="perf-card-value">${_fmtMs(avg.llm_duration_ms)}</div>
        <div class="perf-card-label">平均 LLM</div>
      </div>
    </div>
    <div class="perf-legend">
      ${Object.values(STAGE_COLORS).map(s =>
        `<span class="perf-legend-item"><span class="perf-legend-dot" style="background:${s.color}"></span>${s.label}</span>`
      ).join('')}
    </div>
    <div class="perf-avg-breakdown">
      ${_renderBreakdownBar(avg)}
    </div>
  `;
}

function _renderBreakdownBar(avg) {
  const stages = [
    { key: 'vad_duration_ms', ...STAGE_COLORS.vad },
    { key: 'asr_duration_ms', ...STAGE_COLORS.asr },
    { key: 'collector_delay_ms', ...STAGE_COLORS.collector },
    { key: 'llm_duration_ms', ...STAGE_COLORS.llm },
    { key: 'tool_duration_ms', ...STAGE_COLORS.tool },
    { key: 'tts_duration_ms', ...STAGE_COLORS.tts },
  ];
  const total = stages.reduce((s, st) => s + (avg[st.key] || 0), 0);
  if (total === 0) return '';

  const segments = stages
    .filter(st => avg[st.key] > 0)
    .map(st => {
      const pct = ((avg[st.key] / total) * 100).toFixed(1);
      return `<div class="perf-bar-segment" style="width:${pct}%;background:${st.color}" title="${st.label}: ${avg[st.key]}ms (${pct}%)"></div>`;
    }).join('');

  return `<div class="perf-bar">${segments}</div>`;
}

function _renderWaterfall(turns) {
  const el = document.getElementById('perf-waterfall');
  if (!el) return;

  if (!turns.length) {
    el.innerHTML = '<div class="perf-empty">暂无记录</div>';
    return;
  }

  // 找出最大 total 用于归一化
  const maxTotal = Math.max(...turns.map(t => t.total_duration_ms || 1));

  const rows = turns.map(t => {
    const stages = [
      { key: 'vad_duration_ms', ...STAGE_COLORS.vad },
      { key: 'asr_duration_ms', ...STAGE_COLORS.asr },
      { key: 'collector_delay_ms', ...STAGE_COLORS.collector },
      { key: 'llm_duration_ms', ...STAGE_COLORS.llm },
      { key: 'tool_duration_ms', ...STAGE_COLORS.tool },
      { key: 'tts_duration_ms', ...STAGE_COLORS.tts },
    ];
    const rowTotal = stages.reduce((s, st) => s + (t[st.key] || 0), 0);
    const barWidth = rowTotal > 0 ? ((rowTotal / maxTotal) * 100).toFixed(1) : 0;

    const segments = stages
      .filter(st => t[st.key] > 0)
      .map(st => {
        const pct = ((t[st.key] / rowTotal) * 100).toFixed(1);
        return `<div class="perf-bar-segment" style="width:${pct}%;background:${st.color}" title="${st.label}: ${t[st.key]}ms"></div>`;
      }).join('');

    const timeStr = new Date(t.created_at * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const text = (t.trigger_text || '').slice(0, 30);
    const totalStr = _fmtMs(t.total_duration_ms);

    return `
      <div class="perf-row">
        <div class="perf-row-meta">
          <span class="perf-row-time">${timeStr}</span>
          <span class="perf-row-text" title="${t.trigger_text || ''}">${text}</span>
        </div>
        <div class="perf-row-bar" style="width:${barWidth}%">
          ${segments}
        </div>
        <span class="perf-row-total">${totalStr}</span>
      </div>
    `;
  }).join('');

  el.innerHTML = `
    <h3 class="perf-section-title">最近请求 (${turns.length})</h3>
    <div class="perf-waterfall-list">${rows}</div>
  `;
}

function _fmtMs(ms) {
  if (ms == null) return '-';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
