/**
 * mobile.js — Mobile-native navigation with bottom tab bar.
 * Handles: sidebar drawer, tab switching (配置/监控/设置), overlay, activity strip.
 */

import { enterMonitorMode, exitMonitorMode } from './monitor-mode.js';

const MQ = window.matchMedia('(max-width: 768px)');
let _isMobile = MQ.matches;
let _overlay;
let _currentTab = 'configure';

export function isMobile() { return _isMobile; }

export function initMobile() {
  _overlay = document.getElementById('mobile-overlay');
  if (!_overlay) {
    _overlay = document.createElement('div');
    _overlay.id = 'mobile-overlay';
    _overlay.className = 'mobile-overlay';
    document.body.appendChild(_overlay);
  }
  _overlay.addEventListener('click', _closeAll);

  // Sidebar toggle button
  const sidebarToggle = document.getElementById('mobile-sidebar-toggle');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', _toggleSidebar);
  }

  // Tab bar
  _initTabBar();

  // Settings panel actions
  _initSettings();

  // Activity strip: tap header to expand/collapse on mobile
  const activityHeader = document.getElementById('activity-toggle');
  if (activityHeader) {
    activityHeader.addEventListener('click', _toggleActivity);
  }

  // Listen for breakpoint changes
  MQ.addEventListener('change', (e) => {
    _isMobile = e.matches;
    if (!_isMobile) {
      _closeAll();
      // Reset to configure mode view when going back to desktop
      _hideSettingsPanel();
    }
  });

  // On mobile, start with activity strip collapsed
  if (_isMobile) {
    const strip = document.getElementById('activity-strip');
    if (strip) strip.classList.remove('mobile-expanded');
  }
}

// ── Tab Bar ──────────────────────────────────────────────────────────────────

function _initTabBar() {
  const tabbar = document.getElementById('mobile-tabbar');
  if (!tabbar) return;
  tabbar.addEventListener('click', (e) => {
    const btn = e.target.closest('.tabbar-btn');
    if (!btn) return;
    const tab = btn.dataset.tab;
    if (tab === _currentTab) return;
    _switchTab(tab);
  });
}

function _switchTab(tab) {
  _currentTab = tab;

  // Update active button
  document.querySelectorAll('.tabbar-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab)
  );

  // Close sidebar if open
  closeSidebarMobile();

  const settingsPanel = document.getElementById('mobile-settings-panel');

  if (tab === 'configure') {
    exitMonitorMode();
    _hideSettingsPanel();
  } else if (tab === 'monitor') {
    _hideSettingsPanel();
    enterMonitorMode();
  } else if (tab === 'settings') {
    exitMonitorMode();
    _showSettingsPanel();
  }
}

function _showSettingsPanel() {
  const panel = document.getElementById('mobile-settings-panel');
  const app = document.getElementById('app');
  if (panel) {
    panel.classList.remove('hidden');
    panel.classList.add('active');
  }
  // Hide canvas/sidebar/monitor but keep in DOM
  app?.classList.add('settings-active');
}

function _hideSettingsPanel() {
  const panel = document.getElementById('mobile-settings-panel');
  const app = document.getElementById('app');
  if (panel) {
    panel.classList.add('hidden');
    panel.classList.remove('active');
  }
  app?.classList.remove('settings-active');
}

// ── Settings Panel ───────────────────────────────────────────────────────────

function _initSettings() {
  const panel = document.getElementById('mobile-settings-panel');
  if (!panel) return;

  panel.addEventListener('click', (e) => {
    const item = e.target.closest('.settings-item');
    if (!item) return;
    const action = item.dataset.action;
    _triggerSettingsAction(action);
  });
}

function _triggerSettingsAction(action) {
  // Programmatically click the corresponding topbar button
  const btnMap = {
    'network': 'btn-network',
    'history': 'btn-history',
    'agent-def': 'btn-agent-def',
    'skills': 'btn-skills',
    'deploy': 'btn-deploy',
  };
  const btnId = btnMap[action];
  if (btnId) {
    document.getElementById(btnId)?.click();
  }
}

// ── Sidebar Drawer ───────────────────────────────────────────────────────────

function _toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const open = sidebar.classList.toggle('mobile-open');
  if (open) {
    _overlay.classList.add('active');
  } else {
    _overlay.classList.remove('active');
  }
}

export function openSidebarMobile() {
  if (!_isMobile) return;
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.add('mobile-open');
  if (_overlay) _overlay.classList.add('active');
}

export function closeSidebarMobile() {
  if (!_isMobile) return;
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('mobile-open');
  if (_overlay) _overlay.classList.remove('active');
}

// ── Detail Panel ─────────────────────────────────────────────────────────────

export function openDetailPanelMobile() {
  if (!_isMobile) return;
  const detail = document.getElementById('detail-panel');
  if (detail) {
    detail.classList.add('mobile-open');
    _overlay.classList.add('active');
    // Close sidebar if open
    const sidebar = document.getElementById('sidebar');
    if (sidebar) sidebar.classList.remove('mobile-open');
  }
}

export function closeDetailPanelMobile() {
  if (!_isMobile) return;
  const detail = document.getElementById('detail-panel');
  if (detail) detail.classList.remove('mobile-open');
  if (_overlay) _overlay.classList.remove('active');
}

// ── Activity Strip ───────────────────────────────────────────────────────────

function _toggleActivity() {
  if (!_isMobile) return;
  const strip = document.getElementById('activity-strip');
  if (strip) strip.classList.toggle('mobile-expanded');
}

// ── Close All ────────────────────────────────────────────────────────────────

function _closeAll() {
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('mobile-open');
  const detail = document.getElementById('detail-panel');
  if (detail) detail.classList.remove('mobile-open');
  if (_overlay) _overlay.classList.remove('active');
}
