/**
 * deploy-panel.js — 部署服务 modal
 *
 * 双 Tab 设计：
 *   Tab 1 「我的服务」— 管理已安装的服务（升级/停止/启动/卸载）
 *   Tab 2 「驱动市场」— 浏览和安装新驱动（flat grid + filter chips）
 */

let _overlay  = null;
let _polling  = null;

let _catalog  = { core: [], driver: [], perception: [], inspection: [] };
let _statuses = {};   // driver_id → { running, status, running_image, image, last_deploy }
let _logPolls = {};   // driver_id → intervalId

// { driverId → { image } }
let _pending = {};

let _activeTab = 'my-services';
let _activeFilter = null; // null = all providers

export function initDeployPanel() {
  _overlay = document.getElementById('deploy-overlay');

  document.getElementById('btn-deploy').addEventListener('click', _open);
  document.getElementById('deploy-close').addEventListener('click', _close);
  document.getElementById('deploy-modal-confirm').addEventListener('click', _confirmAll);

  // Tab switching
  _overlay.querySelectorAll('.deploy-tab').forEach(tab => {
    tab.addEventListener('click', () => _switchTab(tab.dataset.tab));
  });

  // Marketplace search
  document.getElementById('marketplace-search').addEventListener('input', _renderMarketplace);

  // Channel selector
  const channelSelect = document.getElementById('deploy-channel-select');
  channelSelect.addEventListener('change', _onChannelChange);
  _loadChannel();
}

// ── Channel management ────────────────────────────────────────────────────

async function _loadChannel() {
  try {
    const res = await fetch('/api/config/update-channel');
    const json = await res.json();
    const channel = json.data?.channel || 'ga';
    document.getElementById('deploy-channel-select').value = channel;
  } catch { /* keep default */ }
}

async function _onChannelChange(e) {
  const channel = e.target.value;
  const warnings = {
    preview: '预览版可能不稳定，仅建议用于测试环境。确定切换？',
    release: '正式版已通过基础测试，但未经长期稳定性验证。确定切换？',
  };
  if (warnings[channel] && !confirm(warnings[channel])) {
    await _loadChannel();
    return;
  }
  try {
    await fetch('/api/config/update-channel', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel }),
    });
    await _loadCatalog(true);
    _render();
  } catch { /* ignore */ }
}

// ── Tab switching ─────────────────────────────────────────────────────────

function _switchTab(tabId) {
  _activeTab = tabId;
  _overlay.querySelectorAll('.deploy-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabId);
  });
  _overlay.querySelectorAll('.deploy-tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${tabId}`);
  });
  _render();
}

// ── Open / Close ──────────────────────────────────────────────────────────

function _open() {
  _pending = {};
  _overlay.classList.remove('hidden');
  _load();
  _polling = setInterval(_loadStatuses, 5000);
}

function _close() {
  _overlay.classList.add('hidden');
  clearInterval(_polling);
  _polling = null;
}

// ── Data loading ──────────────────────────────────────────────────────────

async function _load() {
  try {
    await fetch('/api/drivers/sync', { method: 'POST' });
  } catch { /* ignore */ }
  await Promise.all([_loadCatalog(true), _loadStatuses()]);
  _render();
}

async function _loadCatalog(refresh = false) {
  try {
    const url  = refresh ? '/api/registry/catalog?refresh=true' : '/api/registry/catalog';
    const res  = await fetch(url);
    const json = await res.json();
    if (json.data) _catalog = json.data;
  } catch { /* keep existing */ }
}

async function _loadStatuses() {
  try {
    const res  = await fetch('/api/drivers');
    const json = await res.json();
    _statuses = {};
    for (const d of (json.data || [])) {
      _statuses[d.id] = {
        running:       d.running,
        status:        d.status,
        logs:          d.logs || '',
        running_image: d.running_image || '',
        image:         d.image || '',
        last_deploy:   d.last_deploy || null,
        name:          d.name || '',
        category:      d.category || 'driver',
      };
    }
  } catch { /* keep existing */ }
  // Update dots if visible
  _updateStatusDots();
}

// ── Rendering ─────────────────────────────────────────────────────────────

function _render() {
  if (_activeTab === 'my-services') {
    _renderMyServices();
  } else {
    _renderMarketplace();
  }
  _syncFooter();
}

// ══════════════════════════════════════════════════════════════════════════
//  TAB 1: 我的服务
// ══════════════════════════════════════════════════════════════════════════

function _renderMyServices() {
  const container = document.getElementById('pane-my-services');

  // Collect all items from catalog that have a status (i.e., deployed)
  const allItems = [
    ...(_catalog.core || []).map(it => ({ ...it, _cat: 'core' })),
    ...(_catalog.driver || []).map(it => ({ ...it, _cat: 'driver' })),
    ...(_catalog.perception || []).map(it => ({ ...it, _cat: 'perception' })),
  ];

  // Only show items that have actually been deployed (not just synced from catalog)
  const deployed = allItems.filter(item => {
    const id = _driverIdForItem(item, item._cat);
    const s = _statuses[id];
    if (!s) return false;
    return s.running || s.last_deploy || item._cat === 'core';
  });

  if (deployed.length === 0) {
    container.innerHTML = `<div class="svc-empty">
      <div class="svc-empty-title">暂无已安装服务</div>
      <div class="svc-empty-hint">前往「驱动市场」安装驱动</div>
    </div>`;
    return;
  }

  // Split into groups: updatable, running, stopped
  const updatable = [];
  const running   = [];
  const stopped   = [];

  for (const item of deployed) {
    const id = _driverIdForItem(item, item._cat);
    const s  = _statuses[id] || {};
    const tags = item.tags || [];
    const latestTag = tags.length > 0 ? tags[0].tag : null;
    const currentTag = s.running_image?.includes(':') ? s.running_image.split(':').pop() : null;
    const hasUpdate = latestTag && currentTag && latestTag !== currentTag;

    const entry = { item, id, s, latestTag, currentTag, hasUpdate };

    if ((s.running || item._cat === 'core') && hasUpdate) {
      updatable.push(entry);
    } else if (s.running || item._cat === 'core') {
      running.push(entry);
    } else {
      stopped.push(entry);
    }
  }

  let html = '';

  if (updatable.length) {
    html += _svcGroupHTML('可更新', updatable.length, 'updatable');
    html += updatable.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }
  if (running.length) {
    html += _svcGroupHTML('运行中', running.length, 'running');
    html += running.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }
  if (stopped.length) {
    html += _svcGroupHTML('已停止', stopped.length, 'stopped');
    html += stopped.map(e => _svcRowHTML(e)).join('');
    html += '</div>';
  }

  container.innerHTML = html;

  // Bind actions
  container.querySelectorAll('[data-action="upgrade"]').forEach(btn => {
    btn.addEventListener('click', () => _showUpgradeConfirm(btn.dataset));
  });
  container.querySelectorAll('[data-action="stop"]').forEach(btn => {
    btn.addEventListener('click', () => _stopDriver(btn.dataset.driverId, btn));
  });
  container.querySelectorAll('[data-action="start"]').forEach(btn => {
    btn.addEventListener('click', () => _startDriver(btn.dataset.driverId, btn.dataset.image, btn));
  });
  container.querySelectorAll('[data-action="remove"]').forEach(btn => {
    btn.addEventListener('click', () => _removeDriver(btn.dataset.driverId, btn));
  });
  // Version switcher dropdowns
  container.querySelectorAll('[data-action="switch-version"]').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const wrap = btn.closest('.svc-ver-wrap');
      const dd = wrap.querySelector('.svc-ver-dropdown');
      const wasHidden = dd.classList.contains('hidden');
      // Close all others
      container.querySelectorAll('.svc-ver-dropdown').forEach(d => d.classList.add('hidden'));
      if (wasHidden) dd.classList.remove('hidden');
    });
  });
  container.querySelectorAll('.svc-ver-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      if (opt.classList.contains('current')) return;
      const { driverId, fullImage, tag, label } = opt.dataset;
      opt.closest('.svc-ver-dropdown').classList.add('hidden');
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag }],
        () => _executeDeploys([[driverId, { image: fullImage }]])
      );
    });
  });
}

function _svcGroupHTML(title, count, cls) {
  return `<div class="svc-group ${cls}">
    <div class="svc-group-header">
      <span class="svc-group-title">${title}</span>
      <span class="svc-group-count">${count}</span>
    </div>`;
}

function _svcRowHTML({ item, id, s, latestTag, currentTag, hasUpdate }) {
  const label = item._cat === 'driver' ? (item.model || item.image) : (item.name || item.image);
  const isRunning = s.running || item._cat === 'core';
  const statusDot = isRunning ? 'running' : s.status === 'error' ? 'error' : 'stopped';
  const imageBase = item.full_repo || item.image;
  const tags = item.tags || [];

  let actions = '';
  // Version switcher (dropdown)
  if (tags.length > 1) {
    const _channelLabel = (tag) => {
      if (tag.startsWith('ga.')) return '稳定';
      if (tag.startsWith('release.')) return '正式';
      if (tag.startsWith('preview.')) return '预览';
      return '';
    };
    const versionOpts = tags.map(t => {
      const fullImg = t.imageRef || (imageBase + ':' + t.tag);
      const isCurrent = currentTag && t.tag === currentTag;
      const ch = _channelLabel(t.tag);
      return `<div class="svc-ver-opt${isCurrent ? ' current' : ''}" data-driver-id="${id}" data-full-image="${fullImg}" data-tag="${t.tag}" data-label="${label}">
        <span class="svc-ver-tag">${t.tag}</span>
        ${ch ? `<span class="svc-ver-channel">${ch}</span>` : ''}
        ${isCurrent ? '<span class="svc-ver-badge">当前</span>' : ''}
      </div>`;
    }).join('');
    actions += `<div class="svc-ver-wrap">
      <button class="svc-btn svc-btn-ver" data-action="switch-version">切换版本 ▾</button>
      <div class="svc-ver-dropdown hidden">${versionOpts}</div>
    </div>`;
  }
  if (hasUpdate) {
    const latestImage = tags[0]?.imageRef || (imageBase + ':' + latestTag);
    actions += `<button class="svc-btn svc-btn-upgrade" data-action="upgrade" data-driver-id="${id}" data-current-tag="${currentTag}" data-latest-tag="${latestTag}" data-latest-image="${latestImage}" data-label="${label}">升级到 ${latestTag}</button>`;
  }
  if (item._cat === 'core') {
    // Core cannot stop itself — no stop button
  } else if (isRunning) {
    actions += `<button class="svc-btn svc-btn-stop" data-action="stop" data-driver-id="${id}">停止</button>`;
  } else {
    // Stopped: show start + remove
    const lastImage = s.running_image || s.last_deploy?.image || s.image || '';
    if (lastImage) {
      actions += `<button class="svc-btn svc-btn-start" data-action="start" data-driver-id="${id}" data-image="${lastImage}">启动</button>`;
    }
    actions += `<button class="svc-btn svc-btn-remove" data-action="remove" data-driver-id="${id}">卸载</button>`;
  }

  const versionText = currentTag || (s.running_image?.split(':').pop()) || '—';

  return `
    <div class="svc-row" id="card-${id}">
      <div class="svc-row-dot ${statusDot}" id="dot-${id}"></div>
      <div class="svc-row-info">
        <span class="svc-row-name">${label}</span>
        <span class="svc-row-version">${versionText}</span>
        ${hasUpdate ? `<span class="svc-row-arrow">→</span><span class="svc-row-new-version">${latestTag}</span>` : ''}
      </div>
      <div class="svc-row-actions">${actions}</div>
    </div>
    <div class="deploy-log hidden" id="log-${id}"></div>`;
}

// ══════════════════════════════════════════════════════════════════════════
//  TAB 2: 驱动市场
// ══════════════════════════════════════════════════════════════════════════

function _renderMarketplace() {
  const q = (document.getElementById('marketplace-search')?.value || '').trim().toLowerCase();

  // Merge all non-core categories for the marketplace
  const allItems = [
    ...(_catalog.driver || []).map(it => ({ ...it, _cat: 'driver' })),
    ...(_catalog.perception || []).map(it => ({ ...it, _cat: 'perception' })),
    ...(_catalog.core || []).map(it => ({ ...it, _cat: 'core' })),
  ];

  // Build provider list for filter chips
  const providers = [...new Set(allItems.map(it => it.provider || it.name || 'Other').filter(Boolean))];

  // Render filter chips
  const filtersEl = document.getElementById('marketplace-filters');
  filtersEl.innerHTML = `
    <button class="mp-chip${_activeFilter === null ? ' active' : ''}" data-filter="">全部</button>
    ${providers.map(p => `<button class="mp-chip${_activeFilter === p ? ' active' : ''}" data-filter="${p}">${p}</button>`).join('')}
  `;
  filtersEl.querySelectorAll('.mp-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      _activeFilter = chip.dataset.filter || null;
      _renderMarketplace();
    });
  });

  // Filter items
  let filtered = allItems;
  if (_activeFilter) {
    filtered = filtered.filter(it => (it.provider || it.name || 'Other') === _activeFilter);
  }
  if (q) {
    filtered = filtered.filter(it => {
      const model = (it.model || '').toLowerCase();
      const provider = (it.provider || '').toLowerCase();
      const name = (it.name || '').toLowerCase();
      return model.includes(q) || provider.includes(q) || name.includes(q);
    });
  }

  // Render grid
  const gridEl = document.getElementById('marketplace-grid');
  if (filtered.length === 0) {
    gridEl.innerHTML = `<div class="svc-empty">
      <div class="svc-empty-title">未找到驱动</div>
      <div class="svc-empty-hint">尝试其他搜索词或切换更新通道</div>
    </div>`;
    return;
  }

  gridEl.innerHTML = filtered.map(item => _mpCardHTML(item)).join('');

  // Bind install buttons
  gridEl.querySelectorAll('.mp-install-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleInstallDropdown(btn);
    });
  });

  // Bind version options
  gridEl.querySelectorAll('.mp-version-opt').forEach(opt => {
    opt.addEventListener('click', () => {
      const driverId = opt.dataset.driverId;
      const image = opt.dataset.fullImage;
      const label = opt.dataset.label;
      const tag = opt.dataset.tag;
      // Close dropdown
      opt.closest('.mp-card').querySelector('.mp-versions').classList.add('hidden');
      // Show confirm and deploy
      showDeployConfirmModal(
        [{ label, currentTag: '—', newTag: tag }],
        () => _executeDeploys([[driverId, { image }]])
      );
    });
  });
}

function _mpCardHTML(item) {
  const cat = item._cat;
  const label = cat === 'driver' ? (item.model || item.image) : (item.name || item.image);
  const provider = item.provider || '';
  const driverId = _driverIdForItem(item, cat);
  const s = _statuses[driverId];
  const isInstalled = s && (s.running || s.last_deploy);
  const tags = item.tags || [];
  const imageBase = item.full_repo || item.image;

  const versionOpts = tags.map(t => {
    const fullImg = t.imageRef || (imageBase + ':' + t.tag);
    return `<div class="mp-version-opt" data-driver-id="${driverId}" data-full-image="${fullImg}" data-tag="${t.tag}" data-label="${label}">
      <span class="mp-version-tag">${t.tag}</span>
      ${t.created ? `<span class="mp-version-date">${t.created.replace(/\s+\d{2}:\d{2}$/, '')}</span>` : ''}
    </div>`;
  }).join('');

  const installBtn = isInstalled
    ? `<span class="mp-installed-badge">已安装</span>`
    : tags.length > 0
      ? `<button class="mp-install-btn">安装 ▾</button>`
      : `<span class="mp-no-version">暂无版本</span>`;

  return `
    <div class="mp-card" data-driver-id="${driverId}">
      <div class="mp-card-name">${label}</div>
      ${provider ? `<div class="mp-card-provider">${provider}</div>` : ''}
      <div class="mp-card-action">${installBtn}</div>
      <div class="mp-versions hidden">${versionOpts}</div>
    </div>`;
}

function _toggleInstallDropdown(btn) {
  const card = btn.closest('.mp-card');
  const dropdown = card.querySelector('.mp-versions');
  const wasHidden = dropdown.classList.contains('hidden');

  // Close all other dropdowns
  document.querySelectorAll('.mp-versions').forEach(d => d.classList.add('hidden'));

  if (wasHidden) {
    dropdown.classList.remove('hidden');
  }
}

// Close marketplace dropdowns when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.mp-card')) {
    document.querySelectorAll('.mp-versions').forEach(d => d.classList.add('hidden'));
  }
  if (!e.target.closest('.svc-ver-wrap')) {
    document.querySelectorAll('.svc-ver-dropdown').forEach(d => d.classList.add('hidden'));
  }
});

// ══════════════════════════════════════════════════════════════════════════
//  SHARED UTILITIES
// ══════════════════════════════════════════════════════════════════════════

function _driverIdForItem(item, category) {
  if (category === 'driver') return `${item.provider}-${item.model}`;
  return item.image;
}

function _updateStatusDots() {
  for (const [id, s] of Object.entries(_statuses)) {
    const dot = document.getElementById(`dot-${id}`);
    if (dot) {
      dot.className = 'svc-row-dot ' + (s.running ? 'running' : s.status === 'error' ? 'error' : 'stopped');
    }
  }
}

function _syncFooter() {
  const footer = document.getElementById('deploy-modal-footer');
  const hint   = document.getElementById('deploy-footer-hint');
  const count  = Object.keys(_pending).length;
  if (count > 0) {
    footer.style.display = '';
    hint.textContent = `已选 ${count} 个驱动`;
  } else {
    footer.style.display = 'none';
  }
}

// ── Deploy confirm modal (shared) ─────────────────────────────────────────

/**
 * Show the unified deploy-confirm modal.
 * @param {Array<{label: string, currentTag: string, newTag: string}>} items
 * @param {Function} onConfirm - called when user clicks confirm
 */
export function showDeployConfirmModal(items, onConfirm) {
  const overlay = document.getElementById('deploy-confirm-overlay');
  const body    = document.getElementById('deploy-confirm-body');

  body.innerHTML = items.map(it => `
    <div class="deploy-confirm-item">
      <div class="deploy-confirm-item-name">${it.label}</div>
      <div class="deploy-confirm-item-versions">
        <span class="deploy-confirm-tag current">${it.currentTag || '—'}</span>
        <span class="deploy-confirm-arrow">→</span>
        <span class="deploy-confirm-tag latest">${it.newTag}</span>
      </div>
    </div>`).join('');

  overlay.classList.remove('hidden');

  const btnOk     = document.getElementById('deploy-confirm-ok');
  const btnCancel = document.getElementById('deploy-confirm-cancel');

  const cleanup = () => {
    btnOk.removeEventListener('click', doConfirm);
    btnCancel.removeEventListener('click', doCancel);
  };
  const doConfirm = () => { overlay.classList.add('hidden'); cleanup(); onConfirm(); };
  const doCancel  = () => { overlay.classList.add('hidden'); cleanup(); };

  btnOk.addEventListener('click', doConfirm);
  btnCancel.addEventListener('click', doCancel);
}

// ── Upgrade confirm ───────────────────────────────────────────────────────

function _showUpgradeConfirm({ driverId, currentTag, latestTag, latestImage, label }) {
  showDeployConfirmModal(
    [{ label, currentTag, newTag: latestTag }],
    () => _executeDeploys([[driverId, { image: latestImage }]])
  );
}

// ── Stop / Start / Remove ─────────────────────────────────────────────────

async function _stopDriver(driverId, btn) {
  btn.disabled    = true;
  btn.textContent = '停止中…';
  try {
    await fetch(`/api/drivers/${driverId}/stop`, { method: 'POST' });
  } catch (e) {
    console.error('[deploy] stop', e);
  }
  setTimeout(async () => { await _loadStatuses(); _render(); }, 1500);
}

async function _startDriver(driverId, image, btn) {
  if (!image) return;
  btn.disabled    = true;
  btn.textContent = '启动中…';
  _showDeployLog(driverId, '正在启动…');
  try {
    const res = await fetch(`/api/drivers/${driverId}/deploy`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
    const json = await res.json();
    if (json.code !== 200) {
      _appendLog(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
    } else {
      _appendLog(driverId, '容器启动中…');
      _startLogPolling(driverId);
    }
  } catch (e) {
    _appendLog(driverId, `✗ 网络错误: ${e.message}`, 'error');
  }
}

async function _removeDriver(driverId, btn) {
  if (!confirm('确定卸载此服务？将移除容器并释放空间。')) return;
  btn.disabled = true;
  btn.textContent = '卸载中…';
  try {
    await fetch(`/api/drivers/${driverId}/remove`, { method: 'POST' });
  } catch (e) {
    console.error('[deploy] remove', e);
  }
  setTimeout(async () => { await _loadStatuses(); _render(); }, 1500);
}

// ── Confirm all pending deploys ───────────────────────────────────────────

async function _confirmAll() {
  const entries = Object.entries(_pending);
  if (!entries.length) return;

  const items = entries.map(([id, { image }]) => {
    const s = _statuses[id] || {};
    const currentTag = s.running_image?.includes(':') ? s.running_image.split(':').pop() : '—';
    const newTag = image.split(':').pop();
    return { label: s.name || id, currentTag, newTag };
  });

  showDeployConfirmModal(items, () => _executeDeploys(entries));
}

async function _executeDeploys(entries) {
  const confirmBtn = document.getElementById('deploy-modal-confirm');
  if (confirmBtn) confirmBtn.disabled = true;

  for (const [driverId, { image }] of entries) {
    const isCoreDriver = (_catalog.core || []).some(item => _driverIdForItem(item, 'core') === driverId);

    if (isCoreDriver) {
      _showDeployLog(driverId, '正在启动升级…');
      try {
        const res = await fetch('/api/system/update', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const json = await res.json();
        if (json.code !== 200) {
          _appendLog(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
        } else {
          _appendLog(driverId, '升级任务已启动，拉取镜像中…');
          _startCoreUpdatePolling(driverId);
        }
      } catch (e) {
        _appendLog(driverId, `✗ 网络错误: ${e.message}`, 'error');
      }
    } else {
      _showDeployLog(driverId, '正在请求部署…');
      try {
        const res = await fetch(`/api/drivers/${driverId}/deploy`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image }),
        });
        const json = await res.json();
        if (json.code !== 200) {
          _appendLog(driverId, `✗ 错误: ${json.message || '未知错误'}`, 'error');
        } else {
          _appendLog(driverId, '容器启动中…');
          _startLogPolling(driverId);
        }
      } catch (e) {
        _appendLog(driverId, `✗ 网络错误: ${e.message}`, 'error');
      }
    }
  }

  _pending = {};
  _syncFooter();
  if (confirmBtn) confirmBtn.disabled = false;
}

// ── Deploy log (inline) ───────────────────────────────────────────────────

function _showDeployLog(driverId, msg) {
  const el = document.getElementById(`log-${driverId}`);
  if (!el) return;
  el.innerHTML = `<div class="deploy-log-line">${msg}</div>`;
  el.classList.remove('hidden');
}

function _appendLog(driverId, msg, type = '') {
  const el = document.getElementById(`log-${driverId}`);
  if (!el) return;
  const line = document.createElement('div');
  line.className = 'deploy-log-line' + (type ? ` ${type}` : '');
  line.textContent = msg;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function _startLogPolling(driverId) {
  if (_logPolls[driverId]) clearInterval(_logPolls[driverId]);

  let attempts = 0;
  _logPolls[driverId] = setInterval(async () => {
    attempts++;
    try {
      const res  = await fetch(`/api/drivers/${driverId}/status`);
      const json = await res.json();
      const data = json.data || {};
      const status = data.status || '';
      const logs   = data.logs   || '';

      const el = document.getElementById(`log-${driverId}`);
      if (el && logs) {
        const lines = logs.trim().split('\n').slice(-5);
        el.querySelectorAll('.log-output').forEach(e => e.remove());
        const pre = document.createElement('pre');
        pre.className = 'log-output';
        pre.textContent = lines.join('\n');
        el.appendChild(pre);
      }

      if (status === 'running') {
        _stopLogPolling(driverId);
        _appendLog(driverId, '✓ 运行中', 'success');
        setTimeout(() => {
          const logEl = document.getElementById(`log-${driverId}`);
          if (logEl) logEl.classList.add('hidden');
        }, 5000);
        await _loadStatuses();
        _render();
      } else if (status === 'error' || attempts > 30) {
        _stopLogPolling(driverId);
        _appendLog(driverId, `✗ ${status === 'error' ? (data.error || '启动失败') : '部署超时'}`, 'error');
      }
    } catch {
      // ignore
    }
  }, 2000);
}

function _stopLogPolling(driverId) {
  if (_logPolls[driverId]) {
    clearInterval(_logPolls[driverId]);
    delete _logPolls[driverId];
  }
}

// ── Core update polling ───────────────────────────────────────────────────

function _startCoreUpdatePolling(driverId) {
  if (_logPolls[driverId]) clearInterval(_logPolls[driverId]);

  let attempts = 0;
  _logPolls[driverId] = setInterval(async () => {
    attempts++;
    try {
      const res  = await fetch('/api/system/update-status');
      const json = await res.json();
      const data = json.data || {};

      if (data.error) {
        _stopLogPolling(driverId);
        _appendLog(driverId, `✗ 升级失败：${data.error}`, 'error');
      } else if (data.step) {
        const el = document.getElementById(`log-${driverId}`);
        if (el) {
          el.querySelectorAll('.log-output').forEach(e => e.remove());
          const pre = document.createElement('div');
          pre.className = 'log-output';
          pre.textContent = data.step;
          el.appendChild(pre);
        }
      }

      if (attempts > 90) {
        _stopLogPolling(driverId);
        _appendLog(driverId, '✗ 升级超时', 'error');
      }
    } catch {
      // 服务重启中，连接断开是正常的
    }
  }, 2000);
}
