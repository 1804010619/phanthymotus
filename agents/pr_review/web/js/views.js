/** views.js — renderers for the overview, history, and job detail panels. */

import {
  esc, renderMarkdown, fmtDuration, fmtTime, fmtRelative, fmtRelativeIso,
  shortSha, shortRepo, targetLabel, prUrl,
} from './api.js';

// Statuses from which a job will not advance — mirrors the server's set.
const TERMINAL = new Set([
  'review_done', 'build_success', 'build_failed', 'timeout', 'error', 'cancelled',
]);

// ── Overview ─────────────────────────────────────────────────────────────────

export function renderStats(el, s) {
  const hist = s.history || {};
  const tiles = [
    { label: 'Queued', value: s.queue_depth, cls: s.queue_depth > 0 ? 'yellow' : '' },
    { label: 'In flight', value: s.active_jobs, cls: s.active_jobs > 0 ? 'blue' : '' },
    { label: 'Processed', value: s.total_processed, cls: 'green' },
    { label: 'History', value: hist.total ?? 0, cls: '' },
    {
      label: 'Workers',
      value: `${s.active_jobs}/${s.config?.max_concurrent_jobs ?? '?'}`,
      cls: '',
    },
  ];
  el.innerHTML = tiles.map((t) => `
    <div class="stat-tile ${t.cls}">
      <div class="stat-value">${esc(t.value)}</div>
      <div class="stat-label">${esc(t.label)}</div>
    </div>
  `).join('');
}

export function renderActive(bodyEl, metaEl, s) {
  const active = s.active || [];
  metaEl.textContent = active.length ? `${active.length} running` : '';

  if (!active.length) {
    bodyEl.innerHTML = emptyState('◎', 'Nothing in flight',
      'Comment /request_bot_review on a PR to trigger a review.');
    return;
  }

  bodyEl.innerHTML = `
    <table class="tbl">
      <thead><tr>
        <th>PR</th><th>Repo</th><th>Commit</th><th>Stage</th>
        <th>Attempt</th><th>By</th><th class="num">Elapsed</th>
      </tr></thead>
      <tbody>
        ${active.map((j) => `
          <tr class="clickable" data-job="${esc(j.id)}">
            <td class="mono">#${esc(j.pr_number)}</td>
            <td>${esc(shortRepo(j.repo))}</td>
            <td class="mono">${esc(shortSha(j.head_sha))}</td>
            <td>${stageCell(j)}</td>
            <td class="mono">${esc(j.attempt)}</td>
            <td>${esc(j.requester)}</td>
            <td class="num">${esc(fmtDuration(j.elapsed))}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

/**
 * Render the current pipeline stage.
 *
 * `status` alone sits at "running" through a fetch, a merge, several builds and
 * the review, which reads as a hang. The stage plus how long it has been in that
 * stage is what distinguishes slow from stuck.
 */
export function stageCell(j) {
  if (!j.stage || j.stage === 'done') {
    return `<span class="pill ${esc(j.status)}">${esc(j.status)}</span>`;
  }
  const detail = j.stage_detail ? ` <span class="stage-detail">${esc(j.stage_detail)}</span>` : '';
  const held = j.stage_elapsed != null && j.stage_elapsed >= 20
    ? ` <span class="stage-held">${esc(fmtDuration(j.stage_elapsed))}</span>`
    : '';
  return `<span class="pill running">${esc(j.stage)}</span>${detail}${held}`;
}

export function renderPoller(el, p) {
  if (!p || p.enabled === false) {
    el.innerHTML = `<dl class="kv"><dt>Mode</dt><dd class="plain">Disabled — webhook only</dd></dl>`;
    return;
  }
  const stale = _pollIsStale(p);
  el.innerHTML = `
    <dl class="kv">
      <dt>Interval</dt><dd>${esc(p.interval_seconds)}s</dd>
      <dt>Last poll</dt>
      <dd>${esc(fmtRelativeIso(p.last_poll_at))}
        ${stale ? '<span class="pill timeout">stale</span>' : ''}</dd>
      <dt>Cycles</dt><dd>${esc(p.poll_count ?? 0)}</dd>
      <dt>Triggers seen</dt><dd>${esc(p.triggers_found ?? 0)}</dd>
      <dt>Last error</dt>
      <dd>${p.last_error
        ? `<span style="color:var(--red)">${esc(p.last_error)}</span>`
        : '<span style="color:var(--text-dim)">none</span>'}</dd>
    </dl>`;
}

export function renderConfig(el, c) {
  if (!c) { el.innerHTML = ''; return; }
  el.innerHTML = `
    <dl class="kv">
      <dt>Repos</dt><dd>${(c.repos || []).map((r) => esc(r)).join('<br>') || '—'}</dd>
      <dt>Concurrency</dt><dd>${esc(c.max_concurrent_jobs)}</dd>
      <dt>Build timeout</dt><dd>${esc(fmtDuration(c.build_timeout_seconds))}</dd>
      <dt>Job timeout</dt><dd>${esc(fmtDuration(c.job_timeout_seconds))}</dd>
      <dt>Max attempts</dt><dd>${esc(c.max_attempts)}</dd>
      <dt>Mirror</dt><dd>${esc(c.mirror)}</dd>
      <dt>Retention</dt><dd>${esc(c.job_history_days)}d</dd>
      <dt>Webhook</dt>
      <dd><span class="pill ${c.webhook_enabled ? 'ok' : 'cancelled'}">${
        c.webhook_enabled ? 'enabled' : 'disabled'}</span></dd>
      <dt>LLM review</dt>
      <dd>${c.llm_configured
        ? `<span class="pill ok">on</span> ${esc(c.llm_model)}`
        : '<span class="pill cancelled">not configured</span>'}</dd>
    </dl>`;
}

/** A poller that has not run in >3 intervals is not keeping up. */
function _pollIsStale(p) {
  if (!p.last_poll_at) return true;
  const t = Date.parse(p.last_poll_at);
  if (Number.isNaN(t)) return true;
  return (Date.now() - t) / 1000 > Math.max(90, (p.interval_seconds || 30) * 3);
}

// ── History ──────────────────────────────────────────────────────────────────

export function renderHistory(el, jobs) {
  if (!jobs.length) {
    el.innerHTML = emptyState('◈', 'No reviews yet',
      'Reviews appear here once /request_bot_review is used on a PR.');
    return;
  }

  el.innerHTML = `
    <table class="tbl">
      <thead><tr>
        <th>PR</th><th>Repo</th><th>Commit</th><th>Status</th>
        <th>Builds</th><th>By</th><th>Via</th>
        <th class="num">Elapsed</th><th class="num">When</th>
      </tr></thead>
      <tbody>
        ${jobs.map((j) => `
          <tr class="clickable" data-job="${esc(j.id)}">
            <td class="mono">#${esc(j.pr_number)}</td>
            <td>${esc(shortRepo(j.repo))}</td>
            <td class="mono">${esc(shortSha(j.head_sha))}</td>
            <td>
              ${TERMINAL.has(j.status)
                ? `<span class="pill ${esc(j.status)}">${esc(j.status)}</span>`
                : stageCell(j)}
              ${j.attempt > 1 ? `<span class="pill">try ${esc(j.attempt)}</span>` : ''}
            </td>
            <td>${_buildSummary(j.build_results)}</td>
            <td>${esc(j.requester)}</td>
            <td>${esc(j.source)}</td>
            <td class="num">${esc(fmtDuration(j.elapsed))}</td>
            <td class="num" title="${esc(fmtTime(j.created_at))}">${esc(fmtRelative(j.created_at))}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>`;
}

function _buildSummary(results) {
  if (!results || !results.length) return '<span style="color:var(--text-dim)">—</span>';
  return results.map((b) =>
    `<span class="pill ${b.success ? 'ok' : 'fail'}">${esc(targetLabel(b))}</span>`
  ).join(' ');
}

export function renderPager(el, { total, limit, offset }) {
  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));
  el.innerHTML = `
    <button class="btn-ghost btn-sm" id="pg-prev" ${offset <= 0 ? 'disabled' : ''}>Previous</button>
    <span class="pager-info">Page ${page} / ${pages} · ${total} total</span>
    <button class="btn-ghost btn-sm" id="pg-next" ${offset + limit >= total ? 'disabled' : ''}>Next</button>`;
}

// ── Job detail ───────────────────────────────────────────────────────────────

export function renderDetail(el, job) {
  el.innerHTML = [
    _detailMeta(job),
    _detailBuilds(job),
    _detailReview(job),
    _detailFindings(job),
    _detailErrors(job),
  ].join('');
}

function _detailMeta(j) {
  const o = j.options || {};
  const mode = o.skip_build ? 'review only'
    : o.build_only ? 'build only'
    : 'build + review';
  const running = !TERMINAL.has(j.status);
  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">
          <a href="${esc(prUrl(j.repo, j.pr_number))}" target="_blank" rel="noopener"
             style="color:var(--accent);text-decoration:none">
            ${esc(shortRepo(j.repo))} #${esc(j.pr_number)}
          </a>
        </h2>
        <span class="pill ${esc(j.status)}">${esc(j.status)}</span>
        <span class="card-meta">${esc(j.id)}</span>
      </div>
      <div class="card-body">
        ${running ? `<div class="stage-banner">${stageCell(j)}</div>` : ''}
        <dl class="kv">
          <dt>Commit</dt><dd>${esc(j.head_sha || '—')}</dd>
          <dt>Branch</dt><dd>${esc(j.head_ref || '—')} → ${esc(j.base_ref || '—')}</dd>
          <dt>Requested by</dt><dd class="plain">${esc(j.requester)} (via ${esc(j.source)})</dd>
          <dt>Mode</dt><dd class="plain">${esc(mode)}${
            (o.force_targets || []).length
              ? ` · forced: ${esc((o.force_targets || []).join(', '))}` : ''}</dd>
          <dt>Attempt</dt><dd>${esc(j.attempt)}</dd>
          <dt>Created</dt><dd>${esc(fmtTime(j.created_at))}</dd>
          <dt>Started</dt><dd>${esc(fmtTime(j.started_at))}</dd>
          <dt>Finished</dt><dd>${esc(fmtTime(j.finished_at))}</dd>
          <dt>Duration</dt><dd>${esc(fmtDuration(j.elapsed))}</dd>
        </dl>
      </div>
    </div>`;
}

function _detailBuilds(j) {
  const results = j.build_results || [];
  if (!results.length) {
    return `<div class="card">
      <div class="card-header"><h2 class="card-title">Builds</h2></div>
      <div class="card-body">${emptyState('◇', 'No builds',
        'The changes did not touch a buildable component, or build was skipped.')}</div>
    </div>`;
  }

  const rows = results.map((b) => `
    <tr>
      <td>${esc(targetLabel(b))}</td>
      <td><span class="pill ${b.success ? 'ok' : 'fail'}">${b.success ? 'success' : 'failed'}</span></td>
      <td>${b.image_tag ? `
        <div class="copy-cell">
          <span class="mono">${esc(b.image_tag)}</span>
          <button class="btn-copy" data-copy="${esc(b.image_tag)}">copy</button>
        </div>` : '<span style="color:var(--text-dim)">—</span>'}</td>
    </tr>`).join('');

  // Each build gets its own log pane; app.js attaches the tailing loop by
  // reading data-job / data-idx off the pane.
  const logs = results.map((b) => `
    <div class="log-wrap" data-log-block="${esc(b.idx)}">
      <div class="log-toolbar">
        <strong>${esc(targetLabel(b))}</strong>
        <span class="log-toolbar-spacer"></span>
        <span class="log-tail-state" data-log-state="${esc(b.idx)}"></span>
        <button class="btn-ghost btn-sm" data-log-bottom="${esc(b.idx)}">Jump to end</button>
      </div>
      <pre class="log-pane" data-job="${esc(j.id)}" data-idx="${esc(b.idx)}"></pre>
    </div>`).join('');

  return `
    <div class="card">
      <div class="card-header"><h2 class="card-title">Builds</h2></div>
      <div class="card-body no-pad">
        <table class="tbl">
          <thead><tr><th>Target</th><th>Status</th><th>Image</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      ${logs}
    </div>`;
}

function _detailReview(j) {
  if (!j.review_text) return '';
  return `
    <div class="card">
      <div class="card-header"><h2 class="card-title">Code review</h2></div>
      <div class="card-body">
        <div class="review-body">${renderMarkdown(j.review_text)}</div>
      </div>
    </div>`;
}

function _detailFindings(j) {
  const findings = j.findings || [];
  if (!findings.length) return '';
  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Rule checks</h2>
        <span class="card-meta">${findings.length}</span>
      </div>
      <div class="card-body">
        ${findings.map((f) => `
          <div class="finding">
            <span class="pill sev-${esc(f.severity)}">${esc(f.severity)}</span>
            <div>
              <div class="finding-file">${esc(f.file)}</div>
              <div class="finding-msg">${esc(f.message)}</div>
            </div>
          </div>`).join('')}
      </div>
    </div>`;
}

function _detailErrors(j) {
  const errs = j.attempt_errors || [];
  if (!errs.length && !j.error) return '';

  // Error text renders through esc() into <pre> rather than being interpolated
  // raw: it can contain build output, which is attacker-influenced.
  const blocks = errs.map((e, i) => `
    <details class="block">
      <summary>Attempt ${i + 1} failure</summary>
      <pre class="err-pre">${esc(e)}</pre>
    </details>`).join('');

  return `
    <div class="card">
      <div class="card-header">
        <h2 class="card-title">Failures</h2>
        <span class="card-meta">${errs.length} attempt(s)</span>
      </div>
      ${j.error && !errs.length ? `<pre class="err-pre">${esc(j.error)}</pre>` : ''}
      ${blocks}
    </div>`;
}

// ── Shared ───────────────────────────────────────────────────────────────────

export function emptyState(icon, title, hint) {
  return `
    <div class="empty">
      <div class="empty-icon">${esc(icon)}</div>
      <div>${esc(title)}</div>
      ${hint ? `<div class="empty-hint">${esc(hint)}</div>` : ''}
    </div>`;
}
