/* Maintenance-engineer workflow: fleet triage, advisory (planning RUL + localisation), trust-aware forecasts,
   alert handling and the inspection log. Loaded after test_script_3.js; talks to /api/fleet, /api/alerts,
   /api/engine/{id}/maintenance and /api/learning/*. */
(function () {
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const f1 = v => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(1);
  const f2 = v => (v === null || v === undefined || !isFinite(v)) ? '—' : Number(v).toFixed(2);
  const BAND_CLASS = { Normal: 'text-[#10B981]', Warning: 'text-[#F59E0B]', Alarm: 'text-[#F97316]', Emergency: 'text-[#EF4444]' };
  const SEV_CLASS = { warning: 'text-[#F59E0B]', alarm: 'text-[#F97316]', emergency: 'text-[#EF4444]' };
  const state = { enums: null, payload: null, engineId: null, user: localStorage.getItem('gz_user') || '' };

  async function api(path, opts) {
    const r = await fetch(path, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || r.status));
    return body;
  }
  const post = (path, obj) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

  /* ------------------------------------------------------------------ per-engine card */
  const card = document.createElement('div');
  card.id = 'mx-card';
  card.className = 'data-card card-shadow shrink-0 border border-outline-variant/50';
  card.innerHTML = `
    <div class="data-header flex-col items-start gap-1">
      <span class="font-label-sm text-label-sm text-on-surface-variant uppercase tracking-wider font-semibold">10 / Maintenance Workflow</span>
      <span class="text-[9px] font-label-sm text-on-surface-variant">Advisory decision support - the approved maintenance programme governs</span>
    </div>
    <div class="p-4 space-y-4 text-xs" id="mx-body"><span class="text-on-surface-variant">Select an engine…</span></div>`;
  const anchor = document.getElementById('component-cards');
  if (anchor && anchor.closest('.data-card')) anchor.closest('.data-card').after(card);

  function currentFlight() {
    try { return (typeof data !== 'undefined' && data[currentCycle]) ? data[currentCycle].cycle : null; } catch (e) { return null; }
  }

  function advisoryHtml(p) {
    const a = p.advisory;
    if (!a) return '';
    const cls = BAND_CLASS[a.band_name] || '';
    return `<div class="space-y-1">
      <div class="flex justify-between"><span class="text-on-surface-variant">Urgency (latest flight)</span><span class="font-bold ${BAND_CLASS[a.urgency_name] || ''}">${esc(a.urgency_name)}${a.urgency_driver === 'rul' && a.urgency > a.band ? ' · driven by RUL' : ''}</span></div>
      <div class="flex justify-between"><span class="text-on-surface-variant">Health band</span><span class="font-semibold ${cls}">${esc(a.band_name)}</span></div>
      <div class="flex justify-between"><span class="text-on-surface-variant">RUL estimate</span><span class="font-data-display font-semibold">${f1(a.rul)} flights</span></div>
      <div class="flex justify-between"><span class="text-on-surface-variant" title="Lower (5th-percentile) bound of the RUL interval">Planning RUL</span><span class="font-data-display font-bold">${f1(a.planning_rul)} flights</span></div>
      <div class="flex justify-between gap-2"><span class="text-on-surface-variant shrink-0">Most degraded</span><span class="font-semibold text-right">${esc(a.localisation)}</span></div>
      ${a.localisation_reliable ? '' : '<div class="text-[10px] px-2 py-1 rounded bg-[#F59E0B]/10 border border-[#F59E0B]/40 text-[#B45309]">Low-reliability localisation: this module cannot be separated from its neighbours. Confirm by borescope / EGT margin / vibration.</div>'}
      <div class="text-[10px] text-on-surface-variant leading-snug pt-1">${esc(a.guidance)}</div></div>`;
  }

  function forecastHtml(p) {
    const rows = p.forecast_display || [];
    if (!rows.length) return `<div><div class="font-semibold mb-1">Health forecasts</div><div class="text-on-surface-variant text-[10px]">No forecast passed the validation test for this model, so none is shown.</div></div>`;
    const items = rows.map(r => `<div class="border-t border-outline-variant/30 py-1">
      <div class="flex justify-between gap-2"><span class="font-semibold">${esc(r.label)}</span>
        <span class="shrink-0 px-1.5 py-0.5 rounded border text-[9px] font-bold ${r.trust === 'confident' ? 'border-[#10B981]/50 text-[#10B981] bg-[#10B981]/10' : 'border-[#F59E0B]/50 text-[#B45309] bg-[#F59E0B]/10'}">${r.trust === 'confident' ? 'RELIABLE' : 'GROUP TREND ONLY'}</span></div>
      <div class="text-[10px] font-data-display">health ${f2(r.health_now)} now → <b>${f2(r.health_forecast)}</b> in ${r.horizon} flights <span class="text-on-surface-variant">(90% range ${f2(r.lo)}–${f2(r.hi)})</span></div></div>`).join('');
    return `<div><div class="font-semibold mb-1">Health forecasts (validated only)</div>${items}</div>`;
  }

  function alertsHtml(alerts) {
    const open = alerts.filter(a => a.status === 'open');
    const rows = open.slice(-8).reverse().map(a => `<div class="flex items-start justify-between gap-2 border-t border-outline-variant/30 py-1">
      <div><span class="font-bold ${SEV_CLASS[a.severity] || ''}">${esc(a.severity.toUpperCase())}</span> <span class="text-on-surface-variant">flight ${a.cycle}</span><div class="text-[10px]">${esc(a.message)}</div></div>
      <div class="shrink-0 flex gap-1">
        <button data-alert="${a.id}" data-status="acknowledged" class="mx-ack px-1.5 py-0.5 border border-primary/40 rounded text-primary text-[10px]">Ack</button>
        <button data-alert="${a.id}" data-status="dismissed" class="mx-ack px-1.5 py-0.5 border border-outline-variant rounded text-on-surface-variant text-[10px]">Dismiss</button></div></div>`).join('');
    return `<div><div class="font-semibold mb-1">Open alerts (${open.length}${open.length > 8 ? ', latest 8 shown' : ''})</div>${rows || '<div class="text-on-surface-variant text-[10px]">None.</div>'}</div>`;
  }

  function formHtml(p, alerts) {
    const en = state.enums || { event_types: [], findings: [] };
    const opt = (arr, fmt) => arr.map(v => `<option value="${esc(v)}">${esc(fmt ? fmt(v) : v)}</option>`).join('');
    const openAlerts = alerts.filter(a => a.status === 'open').slice(-20).reverse();
    return `<div><div class="font-semibold mb-1">Log an inspection</div>
      <form id="mx-form" class="grid grid-cols-2 gap-2">
        <input name="performed_by" required placeholder="Your name / ID" value="${esc(state.user)}" class="col-span-2 border border-outline-variant/50 rounded px-2 py-1 bg-surface-container-low">
        <select name="event_type" class="border border-outline-variant/50 rounded px-1 py-1 bg-surface-container-low">${opt(en.event_types, v => v.replace(/_/g, ' '))}</select>
        <select name="finding" class="border border-outline-variant/50 rounded px-1 py-1 bg-surface-container-low" title="Did the inspection confirm what the model flagged?">${opt(en.findings, v => 'model flag ' + v.replace(/_/g, ' '))}</select>
        <select name="component" class="border border-outline-variant/50 rounded px-1 py-1 bg-surface-container-low"><option value="">module: n/a</option>${opt(p.components, c => (p.component_names || {})[c] || c)}</select>
        <input name="observed_health" type="number" step="0.01" min="0" max="1" placeholder="observed health 0-1" class="border border-outline-variant/50 rounded px-2 py-1 bg-surface-container-low">
        <select name="alert_id" class="col-span-2 border border-outline-variant/50 rounded px-1 py-1 bg-surface-container-low"><option value="">not linked to an alert</option>${openAlerts.map(a => `<option value="${a.id}">flight ${a.cycle}: ${esc(a.message.slice(0, 60))}</option>`).join('')}</select>
        <input name="action" placeholder="Action taken (e.g. wash, replace blade set)" class="col-span-2 border border-outline-variant/50 rounded px-2 py-1 bg-surface-container-low">
        <textarea name="notes" rows="2" placeholder="Notes" class="col-span-2 border border-outline-variant/50 rounded px-2 py-1 bg-surface-container-low"></textarea>
        <button class="col-span-2 py-1.5 bg-primary text-on-primary rounded font-semibold">Save to maintenance record (flight <span id="mx-flight">${currentFlight() ?? '—'}</span>)</button>
        <div id="mx-msg" class="col-span-2 text-[10px]"></div>
      </form></div>`;
  }

  function historyHtml(events) {
    const rows = events.slice(0, 8).map(e => `<div class="border-t border-outline-variant/30 py-1">
      <b>${esc(e.event_type.replace(/_/g, ' '))}</b> · ${esc(e.component || 'n/a')} · flight ${e.cycle ?? '—'} · <i>${esc(e.finding.replace(/_/g, ' '))}</i>
      <div class="text-[10px] text-on-surface-variant">${esc(e.performed_by)} · ${esc(e.performed_utc.replace('T', ' ').replace('+00:00', ' UTC'))}${e.observed_health != null ? ' · observed health ' + f2(e.observed_health) : ''}${e.action ? ' · ' + esc(e.action) : ''}${e.notes ? '<br>' + esc(e.notes) : ''}</div></div>`).join('');
    return `<div><div class="font-semibold mb-1">Maintenance history (${events.length})</div>${rows || '<div class="text-on-surface-variant text-[10px]">No events recorded.</div>'}</div>`;
  }

  async function renderEngine(p) {
    state.payload = p;
    const id = p.engine_id;
    state.engineId = id;
    const body = document.getElementById('mx-body');
    if (!body) return;
    let alerts = [], events = [];
    try {
      if (!state.enums) state.enums = (await api('/api/fleet')).enums;
      [alerts, events] = await Promise.all([api(`/api/engine/${encodeURIComponent(id)}/alerts`), api(`/api/engine/${encodeURIComponent(id)}/maintenance`)]);
    } catch (e) { body.innerHTML = `<div class="text-[#EF4444]">Maintenance data unavailable: ${esc(e.message)}</div>`; return; }
    if (state.engineId !== id) return;                                   // user switched engine meanwhile
    const keep = document.getElementById('mx-form') ? new FormData(document.getElementById('mx-form')) : null;
    body.innerHTML = advisoryHtml(p) + forecastHtml(p) + alertsHtml(alerts) + formHtml(p, alerts) + historyHtml(events);
    if (keep) ['performed_by'].forEach(k => { const el = body.querySelector(`[name=${k}]`); if (el && keep.get(k)) el.value = keep.get(k); });
    body.querySelectorAll('.mx-ack').forEach(b => b.addEventListener('click', async () => {
      const user = (body.querySelector('[name=performed_by]').value || '').trim();
      if (!user) { alert('Enter your name / ID in the inspection form first (audit trail).'); return; }
      await post(`/api/alerts/${b.dataset.alert}`, { status: b.dataset.status, user });
      renderEngine(state.payload);
    }));
    body.querySelector('#mx-form').addEventListener('submit', async ev => {
      ev.preventDefault();
      const fd = new FormData(ev.target), msg = body.querySelector('#mx-msg');
      const num = k => fd.get(k) === '' ? null : Number(fd.get(k));
      const user = fd.get('performed_by').trim();
      localStorage.setItem('gz_user', user); state.user = user;
      try {
        await post(`/api/engine/${encodeURIComponent(id)}/maintenance`, {
          performed_by: user, event_type: fd.get('event_type'), finding: fd.get('finding'), cycle: currentFlight(),
          component: fd.get('component') || null, observed_health: num('observed_health'), alert_id: num('alert_id'),
          action: fd.get('action') || null, notes: fd.get('notes') || null });
        renderEngine(state.payload);
      } catch (e) { msg.className = 'col-span-2 text-[10px] text-[#EF4444]'; msg.textContent = 'Not saved: ' + e.message; }
    });
  }

  // refresh after the existing dashboard applies a payload; keep the flight number in the form current
  if (typeof applyPayload === 'function') {
    const orig = window.applyPayload;
    window.applyPayload = function (p) { orig.apply(this, arguments); renderEngine(p); };
  }
  setInterval(() => { const el = document.getElementById('mx-flight'); if (el) el.textContent = currentFlight() ?? '—'; }, 400);

  /* ------------------------------------------------------------------ fleet triage overlay */
  const overlay = document.createElement('div');
  overlay.id = 'mx-fleet';
  overlay.style.cssText = 'display:none;position:fixed;inset:0;z-index:60;background:rgba(15,23,42,.55);';
  overlay.innerHTML = `<div class="bg-white rounded-lg shadow-2xl m-6 md:ml-[300px] max-h-[calc(100%-3rem)] overflow-auto text-xs">
    <div class="flex items-center justify-between px-4 py-3 border-b sticky top-0 bg-white z-10">
      <div><b class="text-sm">Fleet triage</b> <span class="text-on-surface-variant">ranked by urgency (worse of health band and planning-RUL level), then planning RUL (lower interval bound)</span></div>
      <div class="flex items-center gap-4">
        <label class="flex items-center gap-2" title="Replay: show the fleet at this share of each engine's recorded flights (demo engines are run-to-failure histories)">Replay point
          <input id="mx-asof" type="range" min="10" max="100" step="5" value="50"><b id="mx-asof-v">50%</b></label>
        <a id="mx-export" href="/api/learning/export.csv" class="text-primary underline">Export findings CSV</a>
        <button id="mx-close" class="px-2 py-1 border rounded">Close</button></div></div>
    <div id="mx-learn" class="px-4 py-2 text-[10px] text-on-surface-variant border-b"></div>
    <table class="w-full"><thead class="text-left text-on-surface-variant bg-surface-container-low"><tr>
      <th class="px-3 py-2">#</th><th>Engine</th><th>Urgency</th><th>Health band</th><th>RUL</th><th>Planning RUL</th><th>Most degraded</th><th>Open alerts</th><th>Inspections</th></tr></thead>
      <tbody id="mx-fleet-body"></tbody></table></div>`;
  document.body.appendChild(overlay);

  async function loadFleet() {
    const asOf = Number(document.getElementById('mx-asof').value) / 100;
    document.getElementById('mx-asof-v').textContent = Math.round(asOf * 100) + '%';
    const tb = document.getElementById('mx-fleet-body');
    try {
      const [fl, ls] = await Promise.all([api('/api/fleet?as_of=' + asOf), api('/api/learning/summary')]);
      state.enums = fl.enums;
      tb.innerHTML = fl.engines.map(e => `<tr class="border-t hover:bg-surface-container-low cursor-pointer" data-id="${esc(e.engine_id)}">
        <td class="px-3 py-1.5">${e.priority}</td><td>${esc(e.dataset || '')} · ${esc(e.engine_id)}</td>
        <td class="font-bold ${BAND_CLASS[e.urgency_name] || ''}" title="driven by ${e.urgency_driver === 'rul' ? 'planning RUL' : 'component health'}">${esc(e.urgency_name)}${e.urgency_driver === 'rul' && e.urgency > e.band ? ' (RUL)' : ''}</td>
        <td class="${BAND_CLASS[e.band_name] || ''}">${esc(e.band_name)}</td><td class="font-data-display">${f1(e.rul)}</td>
        <td class="font-data-display font-bold">${f1(e.planning_rul)}</td>
        <td>${esc(e.localisation)}${e.localisation_reliable ? '' : ' <span title="Not separable by the EKF" class="text-[#B45309]">⚠</span>'}</td>
        <td>${e.open_alerts}</td><td>${e.n_events}</td></tr>`).join('');
      tb.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => {
        const sel = document.querySelector('.engine-select');
        if (sel && [...sel.options].some(o => o.value === tr.dataset.id)) { sel.value = tr.dataset.id; sel.dispatchEvent(new Event('change')); }
        overlay.style.display = 'none';
      }));
      const p = ls.flag_precision;
      document.getElementById('mx-learn').textContent =
        `Recorded inspections: ${ls.events} (confirmed ${ls.by_finding.confirmed}, not found ${ls.by_finding.not_found}, inconclusive ${ls.by_finding.inconclusive})` +
        (p != null ? ` · flag precision ${(p * 100).toFixed(0)}%` : '') +
        (ls.health_estimate_mae != null ? ` · health-estimate MAE vs observed ${ls.health_estimate_mae.toFixed(2)} (n=${ls.n_health_checks})` : '') +
        ' · findings are kept for offline recalibration; the model is never updated automatically.';
    } catch (e) { tb.innerHTML = `<tr><td colspan="9" class="p-4 text-[#EF4444]">${esc(e.message)}</td></tr>`; }
  }
  document.getElementById('mx-asof').addEventListener('change', loadFleet);
  document.getElementById('mx-asof').addEventListener('input', () => { document.getElementById('mx-asof-v').textContent = document.getElementById('mx-asof').value + '%'; });
  document.getElementById('mx-close').addEventListener('click', () => { overlay.style.display = 'none'; });

  // sidebar entry
  const nav = document.getElementById('nav-physics');
  if (nav && nav.closest('ul')) {
    const li = document.createElement('li');
    li.innerHTML = `<a id="nav-fleet" class="nav-item flex items-center gap-md text-on-surface-variant hover:text-on-surface hover:bg-surface-container-highest px-4 py-3 mx-2 rounded-lg font-label-sm text-label-sm transition-all" href="javascript:void(0);"><span class="material-symbols-outlined text-[20px] opacity-70">format_list_numbered</span><span>Fleet &amp; Maintenance</span></a>`;
    nav.closest('ul').appendChild(li);
    li.querySelector('a').addEventListener('click', () => { overlay.style.display = 'block'; loadFleet(); });
  }
})();
