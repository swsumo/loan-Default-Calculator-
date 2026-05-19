// Shared JS utilities for Fraud Detection Ops

// Tabs: add data-tab attribute to any .tab-btn and a matching #tab-{name} panel
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const group = btn.closest('[id$="Tabs"], .tabs');
      if (!group) return;
      group.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const panelId = 'tab-' + btn.dataset.tab;
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      const panel = document.getElementById(panelId);
      if (panel) {
        panel.classList.add('active');
        setTimeout(() => window.dispatchEvent(new Event('resize')), 80);
      }
    });
  });
});
