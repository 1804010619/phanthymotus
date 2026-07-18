/**
 * mobile.js — Mobile responsive navigation & drawer management.
 * Handles hamburger menu, sidebar/detail-panel drawers, overlay, activity strip toggle.
 */

const MQ = window.matchMedia('(max-width: 768px)');
let _isMobile = MQ.matches;

let _overlay;

export function isMobile() { return _isMobile; }

export function initMobile() {
  // Create overlay backdrop
  _overlay = document.getElementById('mobile-overlay');
  if (!_overlay) {
    _overlay = document.createElement('div');
    _overlay.id = 'mobile-overlay';
    _overlay.className = 'mobile-overlay';
    document.body.appendChild(_overlay);
  }
  _overlay.addEventListener('click', _closeAll);

  // Hamburger button
  const hamburger = document.getElementById('mobile-hamburger');
  if (hamburger) {
    hamburger.addEventListener('click', _toggleMenu);
  }

  // Close menu when any topbar action button is clicked
  const topbarActions = document.querySelector('.topbar-actions');
  if (topbarActions) {
    topbarActions.addEventListener('click', (e) => {
      if (e.target.classList.contains('topbar-btn')) {
        topbarActions.classList.remove('mobile-menu-open');
        if (_overlay) _overlay.classList.remove('active');
      }
    });
  }

  // Sidebar toggle button
  const sidebarToggle = document.getElementById('mobile-sidebar-toggle');
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', _toggleSidebar);
  }

  // Activity strip: tap header to expand/collapse on mobile
  const activityHeader = document.getElementById('activity-toggle');
  if (activityHeader) {
    activityHeader.addEventListener('click', _toggleActivity);
  }

  // Listen for breakpoint changes
  MQ.addEventListener('change', (e) => {
    _isMobile = e.matches;
    if (!_isMobile) _closeAll();
  });

  // On mobile, start with activity strip collapsed
  if (_isMobile) {
    const strip = document.getElementById('activity-strip');
    if (strip) strip.classList.remove('mobile-expanded');
  }
}

function _toggleMenu() {
  const actions = document.querySelector('.topbar-actions');
  if (!actions) return;
  const open = actions.classList.toggle('mobile-menu-open');
  // Close sidebar if it was open
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('mobile-open');

  if (open) {
    _overlay.classList.add('active');
  } else {
    _overlay.classList.remove('active');
  }
}

function _toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  const open = sidebar.classList.toggle('mobile-open');
  if (open) {
    _overlay.classList.add('active');
    // Close detail panel if open
    const detail = document.getElementById('detail-panel');
    if (detail) detail.classList.remove('mobile-open');
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

function _toggleActivity() {
  if (!_isMobile) return;
  const strip = document.getElementById('activity-strip');
  if (strip) strip.classList.toggle('mobile-expanded');
}

function _closeAll() {
  // Close menu
  const actions = document.querySelector('.topbar-actions');
  if (actions) actions.classList.remove('mobile-menu-open');
  // Close sidebar
  const sidebar = document.getElementById('sidebar');
  if (sidebar) sidebar.classList.remove('mobile-open');
  // Close detail panel
  const detail = document.getElementById('detail-panel');
  if (detail) detail.classList.remove('mobile-open');
  // Hide overlay
  if (_overlay) _overlay.classList.remove('active');
}
