/**
 * usage.js — Token 用量统计 Modal
 * 展示 LLM 调用的 token 用量汇总、每日趋势柱状图和明细表。
 */

let _overlay, _rangeSelect;

export function initUsage() {
  _overlay = document.getElementById('usage-overlay');
  if (!_overlay) return;

  _rangeSelect = document.getElementById('usage-range');

  document.getElementById('btn-usage')?.addEventListener('click', _open);
  document.getElementById('usage-close')?.addEventListener('click', _close);
  _overlay.addEventListener('click', e => { if (e.target === _overlay) _close(); });
  _rangeSelect?.addEventListener('change', () => _load());

  // Mobile settings action
  document.querySelector('[data-action="usage"]')?.addEventListener('click', () => {
    document.getElementById('btn-usage')?.click();
  });
}

function _open() {
  _overlay.classList.remove('hidden');
  _load();
}

function _close() {
  _overlay.classList.add('hidden');
}

async function _load() {
  const range = _rangeSelect?.value || '7d';
  const cards = document.getElementById('usage-summary-cards');
  const chart = document.getElementById('usage-daily-chart');
  const table = document.getElementById('usage-daily-table');

  cards.innerHTML = '<div style="text-align:center;color:var(--text-dim)">加载中…</div>';
  chart.innerHTML = '';
  table.innerHTML = '';

  try {
    const res = await fetch(`/api/performance/usage?range=${range}`);
    const data = await res.json();
    _renderCards(cards, data.summary);
    _renderChart(chart, data.daily);
    _renderTable(table, data.daily);
  } catch (e) {
    cards.innerHTML = '<div style="text-align:center;color:var(--red)">加载失败</div>';
  }
}

function _renderCards(el, summary) {
  const items = [
    { label: '输入 Tokens', value: summary.prompt_tokens, color: 'var(--accent)' },
    { label: '输出 Tokens', value: summary.completion_tokens, color: '#4ade80' },
    { label: '缓存 Tokens', value: summary.cached_tokens, color: '#a78bfa' },
  ];
  el.innerHTML = `
    <div class="usage-summary-cards">
      ${items.map(it => `
        <div class="usage-card">
          <div class="usage-card-value" style="color:${it.color}">${_formatTokens(it.value)}</div>
          <div class="usage-card-label">${it.label}</div>
        </div>
      `).join('')}
    </div>
    <div class="usage-total-row">
      合计 ${_formatTokens(summary.total_tokens)} tokens · ${summary.call_count} 次调用
    </div>
  `;
}

function _renderChart(el, daily) {
  if (!daily || !daily.length) {
    el.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px 0">暂无数据</div>';
    return;
  }

  // Reverse to chronological order (API returns DESC)
  const days = [...daily].reverse();
  const maxTotal = Math.max(...days.map(d => d.prompt_tokens + d.completion_tokens));
  if (!maxTotal) {
    el.innerHTML = '<div style="text-align:center;color:var(--text-dim);padding:20px 0">暂无数据</div>';
    return;
  }

  const bars = days.map(d => {
    const promptH = (d.prompt_tokens / maxTotal) * 100;
    const completionH = (d.completion_tokens / maxTotal) * 100;
    const dateLabel = d.date.slice(5); // MM-DD
    return `
      <div class="usage-bar" title="${d.date}: in=${_formatTokens(d.prompt_tokens)} out=${_formatTokens(d.completion_tokens)}">
        <div class="usage-bar-stack">
          <div class="usage-bar-fill completion" style="height:${completionH}%"></div>
          <div class="usage-bar-fill prompt" style="height:${promptH}%"></div>
        </div>
        <div class="usage-bar-label">${dateLabel}</div>
      </div>
    `;
  }).join('');

  el.innerHTML = `
    <div class="usage-chart-legend">
      <span><span class="usage-dot" style="background:var(--accent)"></span>输入</span>
      <span><span class="usage-dot" style="background:#4ade80"></span>输出</span>
    </div>
    <div class="usage-chart">${bars}</div>
  `;
}

function _renderTable(el, daily) {
  if (!daily || !daily.length) {
    el.innerHTML = '';
    return;
  }

  const rows = daily.map(d => `
    <div class="usage-daily-row">
      <span class="usage-daily-date">${d.date}</span>
      <span class="usage-daily-values">
        <span style="color:var(--accent)">↑${_formatTokens(d.prompt_tokens)}</span>
        <span style="color:#4ade80">↓${_formatTokens(d.completion_tokens)}</span>
        <span style="color:#a78bfa">⟳${_formatTokens(d.cached_tokens)}</span>
      </span>
    </div>
  `).join('');

  el.innerHTML = `<div class="usage-daily-header">每日明细</div>${rows}`;
}

/**
 * Format token count with auto K/M/G units.
 */
function _formatTokens(n) {
  if (n == null) return '0';
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + 'G';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 10_000) return (n / 1_000).toFixed(1) + 'K';
  if (n >= 1_000) return (n / 1_000).toFixed(2) + 'K';
  return String(n);
}
