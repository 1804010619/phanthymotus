/**
 * performance.js — 性能分析 Dashboard（开放 Span 式）
 */

const COMPONENT_COLORS = {
  perception: { base: '#8b5cf6', shades: ['#6366f1', '#8b5cf6', '#a78bfa'] },
  core:       { base: '#f59e0b', shades: ['#f59e0b', '#d97706', '#b45309'] },
  driver:     { base: '#10b981', shades: ['#10b981', '#059669', '#047857'] },
};

function _spanColor(span, component) {
  const colors = COMPONENT_COLORS[component] || COMPONENT_COLORS.core;
  // Vary shade by span name hash
  let h = 0;
  for (let i = 0; i < span.length; i++) h = ((h << 5) - h + span.charCodeAt(i)) | 0;
  return colors.shades[Math.abs(h) % colors.shades.length];
}

let _refreshTimer = null;
let _currentRange = '24h';

export function initPerformance() {
  const overlay = document.getElementById('performance-overlay');
  if (!overlay) return;

  document.getElementById('performance-close')?.addEventListener('click', _close);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) _close(); });
  document.getElementById('perf-range')?.addEventListener('change', (e) => {
    _currentRange = e.target.value;
    _load();
  });
  document.getElementById('perf-refresh')?.addEventListener('click', _load);
  document.getElementById('btn-performance')?.addEventListener('click', _open);
}

function _open() {
  document.getElementById('performance-overlay')?.classList.remove('hidden');
  _load();
  _refreshTimer = setInterval(_load, 10000);
}

function _close() {
  document.getElementById('performance-overlay')?.classList.add('hidden');
  if (_refreshTimer) { clearInterval(_refreshTimer); _refreshTimer = null; }
}

function _rangeToTs() {
  const now = Date.now() / 1000;
  const map = { '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800 };
  return { start: now - (map[_currentRange] || 86400), end: 0 };
}

async function _load() {
  const { start, end } = _rangeToTs();
  try {
    const [latestRes, aggRes] = await Promise.all([
      fetch('/api/performance/latest?n=50'),
      fetch(`/api/performance/aggregate?start=${start}&end=${end}`),
    ]);
    const latest = await latestRes.json();
    const agg = await aggRes.json();
    _renderSummary(agg);
    _renderWaterfall(Array.isArray(latest) ? latest : []);
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

  const bySpan = agg.by_span || {};
  const spanNames = Object.keys(bySpan);

  el.innerHTML = `
    <div class="perf-cards">
      <div class="perf-card">
        <div class="perf-card-value">${agg.count}</div>
        <div class="perf-card-label">总轮次</div>
      </div>
      ${bySpan['turn_total'] ? `<div class="perf-card">
        <div class="perf-card-value">${_fmtMs(bySpan['turn_total'].avg_ms)}</div>
        <div class="perf-card-label">平均总耗时</div>
      </div>
      <div class="perf-card">
        <div class="perf-card-value">${_fmtMs(bySpan['turn_total'].p95_ms)}</div>
        <div class="perf-card-label">P95 总耗时</div>
      </div>` : ''}
    </div>
    <div class="perf-avg-detail">
      <table class="perf-detail-table">
        <thead><tr><th>阶段</th><th>平均</th><th>P95</th><th>次数</th></tr></thead>
        <tbody>
          ${spanNames.filter(n => n !== 'turn_total').map(name => {
            const s = bySpan[name];
            const color = _spanColor(name, name.startsWith('vad') || name.startsWith('asr') || name.startsWith('kws') || name.startsWith('tts') ? 'perception' : 'core');
            return `<tr>
              <td><span class="perf-legend-dot" style="background:${color}"></span> ${name}</td>
              <td class="perf-detail-val">${_fmtMs(s.avg_ms)}</td>
              <td class="perf-detail-val">${_fmtMs(s.p95_ms)}</td>
              <td class="perf-detail-val">${s.count}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>
  `;
}

function _renderWaterfall(turns) {
  const el = document.getElementById('perf-waterfall');
  if (!el) return;

  if (!turns.length) {
    el.innerHTML = '<div class="perf-empty">暂无记录</div>';
    return;
  }

  const rows = turns.map((t, idx) => {
    const spans = t.spans || [];
    const totalMs = t.total_duration_ms || 0;
    const timeStr = new Date(t.created_at * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const text = (t.trigger_text || '').replace(/<[^>]*>/g, '').slice(0, 30);

    // Build waterfall bar from spans (use relative positions)
    let barHtml = '';
    if (spans.length && totalMs > 0) {
      const minStart = Math.min(...spans.map(s => s.start_ts).filter(Boolean));
      const totalSec = totalMs / 1000;
      barHtml = spans.map(s => {
        if (!s.duration_ms || !s.start_ts) return '';
        const left = ((s.start_ts - minStart) / totalSec * 100).toFixed(1);
        const width = (s.duration_ms / totalMs * 100).toFixed(1);
        const color = _spanColor(s.span, s.component);
        return `<div class="perf-span-seg" style="left:${left}%;width:${width}%;background:${color}" title="${s.span}: ${s.duration_ms}ms"></div>`;
      }).join('');
    }

    // Detail rows
    const detailRows = spans.map(s => {
      const color = _spanColor(s.span, s.component);
      const meta = s.meta && Object.keys(s.meta).length ? JSON.stringify(s.meta) : '';
      return `<tr>
        <td><span class="perf-legend-dot" style="background:${color}"></span> ${s.span}</td>
        <td class="perf-detail-val">${_fmtMs(s.duration_ms)}</td>
        <td class="perf-detail-val perf-detail-meta">${meta}</td>
      </tr>`;
    }).join('');

    return `
      <div class="perf-row-group" data-idx="${idx}">
        <div class="perf-row" onclick="this.parentElement.classList.toggle('expanded')">
          <div class="perf-row-meta">
            <span class="perf-row-time">${timeStr}</span>
            <span class="perf-row-text" title="${t.trigger_text || ''}">${text}</span>
          </div>
          <div class="perf-row-bar">
            ${barHtml}
          </div>
          <span class="perf-row-total">${_fmtMs(totalMs)}</span>
          <span class="perf-row-expand">▸</span>
        </div>
        <div class="perf-row-detail">
          <table class="perf-detail-table">
            <thead><tr><th>Span</th><th>耗时</th><th>元数据</th></tr></thead>
            <tbody>${detailRows}</tbody>
          </table>
        </div>
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
