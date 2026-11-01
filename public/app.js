/* TurboZone dashboard: fleet queue -> one engine -> overview / gas path / why / maintenance.
   Talks only to the FastAPI backend (backend/app.py). No framework: plain DOM + SVG strings. */
(() => {
'use strict';

/* ------------------------------------------------------------------ constants */
const COMPS = ['fan', 'lpc', 'hpc', 'hpt', 'lpt'];                          // tracked by the Kalman filter
const AUX = ['burner', 'nozzle'];                                           // tracked by physics monitors
const ALL = ['fan', 'lpc', 'hpc', 'burner', 'hpt', 'lpt', 'nozzle'];        // every module, in gas-path order
const MODULES = [
  { key: 'fan', name: 'Fan', full: 'Fan and bypass' },
  { key: 'lpc', name: 'LPC', full: 'Low-pressure compressor' },
  { key: 'hpc', name: 'HPC', full: 'High-pressure compressor' },
  { key: 'burner', name: 'Combustor', full: 'Combustor (burner)' },
  { key: 'hpt', name: 'HPT', full: 'High-pressure turbine' },
  { key: 'lpt', name: 'LPT', full: 'Low-pressure turbine' },
  { key: 'nozzle', name: 'Nozzle', full: 'Exhaust nozzle' },
];
const MODULE_FR = [1.05, 1, 1.15, 0.8, 1, 1, 0.9];     // widths of the 2D fallback schematic
const LEVELS = ['Normal', 'Warning', 'Alarm', 'Emergency'];
const LEVEL_FILL = ['#0ca30c', '#fab219', '#ec835a', '#d03b3b'];
const LEVEL_INK = ['#006300', '#8a5a00', '#a4441b', '#b02a2a'];
const SEV_LEVEL = { warning: 1, alarm: 2, emergency: 3 };
const RUL_LEVELS = [30, 15, 5];                         // planning RUL thresholds (backend RUL_ALERTS)
const NIS_GATE = 9;
const C = { rul: '#2a55b8', pred: '#c42a6e', truth: '#8b939b', ink: '#15212d', ink2: '#485664', ink3: '#6f7c89',
            rule: '#e6eaee', axis: '#c3cbd3', up: '#2a78d6', down: '#e34948' };

/* ------------------------------------------------------------------ state */
const S = {
  fleet: [], asOf: 0.5, enums: null, model: null,
  id: null, p: null, col: null, n: 0, i: 0,
  tab: 'overview', alerts: [], events: [], alertFilter: 'open',
  cache: new Map(), explained: new Map(), playing: null, explainSeq: 0,
  user: localStorage.getItem('gz_user') || '',
  notified: new Set(), critical: {}, sound: localStorage.getItem('gz_sound') !== 'off',
  qLevels: new Set(), qText: '',
};

/* ------------------------------------------------------------------ helpers */
const $ = s => document.querySelector(s);
const msg = m => String(m || '').replace(/ -> /g, ' to ').replace(/\(<= (\d+)\)/g, '(limit $1)');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ok = v => v !== null && v !== undefined && Number.isFinite(+v);
const fx = (v, d = 1) => ok(v) ? (+v).toLocaleString('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d }) : '–';
const fi = v => ok(v) ? Math.round(+v).toLocaleString('en-GB') : '–';
const sgn = (v, d = 1) => !ok(v) ? '–' : Math.abs(v) < 0.5 * Math.pow(10, -d) ? fx(0, d) : (v > 0 ? '+' : '−') + fx(Math.abs(v), d);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `HTTP ${r.status}`);
  return body;
}
const post = (path, obj) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });

function glyph(level, size = 12) {
  const f = LEVEL_FILL[level] || LEVEL_FILL[0];
  const s = size, h = s / 2;
  const shape = [
    `<circle cx="${h}" cy="${h}" r="${h - 0.5}" fill="${f}"/>`,
    `<path d="M${h} 0.5 ${s - 0.5} ${s - 1} 0.5 ${s - 1}Z" fill="${f}"/>`,
    `<path d="M${h} 0 ${s} ${h} ${h} ${s} 0 ${h}Z" fill="${f}"/>`,
    `<rect x="0.5" y="0.5" width="${s - 1}" height="${s - 1}" rx="1.5" fill="${f}"/>`,
  ][level] || '';
  return `<svg viewBox="0 0 ${s} ${s}" width="${s}" height="${s}" aria-hidden="true">${shape}</svg>`;
}
const chip = (level, label, extra = '') =>
  `<span class="chip${level >= 2 ? ' pulse' : ''}" style="border-color:${LEVEL_FILL[level]}">${glyph(level)}<span style="color:${LEVEL_INK[level]}">${esc(label)}</span>${extra ? ` <small>${esc(extra)}</small>` : ''}</span>`;

const reduceMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

// Count a number up/down to its new value (answers a change the user caused; skipped while playing)
function tweenNumber(el, from, to, fmt, ms = 380) {
  if (!el || !ok(from) || !ok(to) || from === to || reduceMotion() || S.playing) { if (el) el.textContent = fmt(to); return; }
  const t0 = performance.now();
  const step = now => {
    const k = Math.min(1, (now - t0) / ms), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// FLIP: animate elements from their old screen position to the new one after a re-render
function flipCapture(nodes, key) { return new Map([...nodes].map(n => [n.dataset[key], n.getBoundingClientRect().top])); }
function flipPlay(nodes, key, before) {
  if (reduceMotion() || !before.size) return;
  [...nodes].forEach(n => {
    const old = before.get(n.dataset[key]), now = n.getBoundingClientRect().top;
    if (old == null) { n.animate([{ opacity: 0, transform: 'translateX(-8px)' }, { opacity: 1, transform: 'none' }], { duration: 260, easing: 'ease-out' }); return; }
    if (Math.abs(old - now) < 1) return;
    n.animate([{ transform: `translateY(${old - now}px)` }, { transform: 'none' }], { duration: 420, easing: 'cubic-bezier(.2,.8,.2,1)' });
  });
}

const tipEl = $('#tip');
function showTip(html, x, y) {
  tipEl.innerHTML = html; tipEl.hidden = false; tipEl.classList.add('on');
  const r = tipEl.getBoundingClientRect();
  tipEl.style.left = clamp(x + 14, 4, innerWidth - r.width - 4) + 'px';
  tipEl.style.top = clamp(y - r.height - 12, 4, innerHeight - r.height - 4) + 'px';
}
const hideTip = () => { tipEl.classList.remove('on'); };

function niceStep(range, target) {
  const raw = range / Math.max(1, target), p = Math.pow(10, Math.floor(Math.log10(raw)));
  return [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) || raw;
}

/* ------------------------------------------------------------------ domain logic (mirrors backend/service.py) */
const bandOf = v => (S.p.band_edges || [0.75, 0.55, 0.3]).filter(e => v < e).length;
const reliability = c => ((S.model && S.model.reliability) || {})[c] || {};
const compName = c => (MODULES.find(m => m.key === c) || {}).name || c;
function groupLabel(c) {
  if (AUX.includes(c)) return { text: compName(c), reliable: true };      // measured directly by its physics monitor
  const r = reliability(c);
  if (r.status === 'ok' && r.identifiable !== false) return { text: compName(c), reliable: true };
  const members = COMPS.filter(m => m === c || (r.merged_with || []).includes(m));
  return { text: members.map(compName).join(' / '), reliable: false };
}
const col = (k, i = S.i) => { const a = S.col[k]; return a ? a[i] : null; };

function triage(i = S.i) {
  const tracked = ALL.filter(c => S.col['hi_' + c]);
  const his = Object.fromEntries(tracked.map(c => [c, col('hi_' + c, i)]));
  const worst = tracked.reduce((a, c) => (his[c] < his[a] ? c : a), tracked[0]);
  const band = bandOf(his[worst]);
  const q05 = ok(col('rul_q05', i)) ? col('rul_q05', i) : col('rul', i);
  const planning = Math.max(0, q05);
  const rulLevel = RUL_LEVELS.filter(t => planning <= t).length;
  const urgency = Math.max(band, rulLevel);
  return { his, worst, band, planning, rul: col('rul', i), q95: col('rul_q95', i), urgency,
           driver: band >= rulLevel ? 'health' : 'rul', loc: groupLabel(worst), truth: col('truth_rul', i) };
}

/* ------------------------------------------------------------------ fleet rail */
async function loadFleet() {
  const fl = await api('/api/fleet?as_of=' + S.asOf);
  S.fleet = fl.engines; S.enums = fl.enums;
  renderBoard(); renderQueue();
}

function renderBoard() {
  const counts = [0, 0, 0, 0];
  S.fleet.forEach(e => counts[e.urgency]++);
  const total = S.fleet.length || 1;
  $('#fleet-board').innerHTML = `
    <div class="fb-head"><h2>Fleet queue</h2><span>${S.fleet.length} engines, most urgent first</span></div>
    <div class="fb-bar" role="img" aria-label="${counts.map((n, k) => `${n} ${LEVELS[k]}`).join(', ')}">
      ${[3, 2, 1, 0].map(k => counts[k] ? `<i style="flex:${counts[k] / total};background:${LEVEL_FILL[k]}" title="${counts[k]} ${LEVELS[k]}"></i>` : '').join('')}</div>
    <div class="fb-legend" role="group" aria-label="Show only">
      ${[3, 2, 1, 0].map(k => `<button data-level="${k}" aria-pressed="${S.qLevels.has(k)}" ${counts[k] ? '' : 'disabled'}
        title="Show only ${LEVELS[k]} engines">${glyph(k, 10)}<b>${counts[k]}</b><span>${LEVELS[k]}</span></button>`).join('')}</div>`;
  $('#fleet-board').querySelectorAll('[data-level]').forEach(btn => btn.addEventListener('click', () => {
    const k = +btn.dataset.level;
    S.qLevels.has(k) ? S.qLevels.delete(k) : S.qLevels.add(k);
    renderBoard(); renderQueue();
  }));
}

function renderQueue() {
  const before = flipCapture(document.querySelectorAll('#queue button[data-id]'), 'id');
  const text = S.qText.trim().toLowerCase();
  const shown = S.fleet.map((e, rank) => ({ e, rank })).filter(({ e }) =>
    (!S.qLevels.size || S.qLevels.has(e.urgency)) &&
    (!text || String(e.engine_id).toLowerCase().includes(text) || String(e.dataset || e.source || '').toLowerCase().includes(text)));
  const short = d => String(d || 'Upload').replace(/-\d+$/, '');
  // only the three most urgent Emergency engines pulse: a pulse is a signal only while it is rare
  $('#queue').innerHTML = shown.map(({ e, rank }) => {
    const used = ok(e.rul) && e.last_cycle ? clamp(e.last_cycle / (e.last_cycle + Math.max(0, e.rul)), 0, 1) : null;
    const where = e.localisation.replace(' (not separable by the EKF)', '');
    const why = e.urgency_driver === 'rul' && e.urgency > e.band ? 'Urgency set by the planning RUL' : 'Urgency set by component health';
    const tip = `${LEVELS[e.urgency]}. ${why}. Flight ${e.last_cycle}` + (used != null ? `, about ${Math.round(used * 100)}% of estimated life used.` : '.');
    return `<li><button data-id="${esc(e.engine_id)}" aria-current="${e.engine_id === S.id}" class="u${e.urgency}${e.urgency === 3 && rank < 3 ? ' pulse' : ''}" style="--u:${LEVEL_FILL[e.urgency]}" title="${esc(tip)}">
      <span class="q-glyph">${glyph(e.urgency, 14)}</span>
      <span class="q-id">${esc(e.engine_id)}<span class="q-sub">${esc(short(e.dataset || e.source))}</span><span class="sr-only">${esc(LEVELS[e.urgency])}</span></span>
      <span class="q-where">${esc(where)} most worn</span>
      <span class="q-rul"><b>${fi(e.planning_rul)}</b><span>flights</span></span>
      <span class="q-life" aria-hidden="true"><i style="width:${used == null ? 0 : (used * 100).toFixed(1)}%"></i></span></button></li>`;
  }).join('') || `<li class="q-empty"><p>No engines match.</p><button class="rail-link" id="q-clear">Clear the filters</button></li>`;
  const clear = $('#q-clear');
  if (clear) clear.addEventListener('click', () => { S.qLevels.clear(); S.qText = ''; $('#q-search').value = ''; renderBoard(); renderQueue(); });
  flipPlay(document.querySelectorAll('#queue button[data-id]'), 'id', before);
}

/* retractable rail: full panel <-> slim strip; remembered; '[' toggles */
function setRail(mini, animate = true) {
  const app = $('#app'), rail = $('#rail'), btn = $('#rail-toggle');
  const apply = () => {
    app.classList.toggle('rail-mini', mini);
    btn.setAttribute('aria-expanded', String(!mini));
    btn.title = mini ? 'Expand the fleet panel ( [ )' : 'Collapse the fleet panel ( [ )';
    btn.setAttribute('aria-label', mini ? 'Expand the fleet panel' : 'Collapse the fleet panel');
    localStorage.setItem('gz_rail', mini ? 'mini' : 'full');
  };
  if (!animate || reduceMotion()) { apply(); dispatchEvent(new Event('resize')); return; }
  rail.classList.add('switching');                     // fade the contents out, swap layout, fade back in
  setTimeout(() => { apply(); rail.classList.remove('switching'); }, 140);
  setTimeout(() => dispatchEvent(new Event('resize')), 560);   // charts and the 3D view redraw at the new width
}
$('#rail-toggle').addEventListener('click', () => setRail(!$('#app').classList.contains('rail-mini')));
document.addEventListener('keydown', ev => {
  if (ev.key === '[' && !ev.target.closest('input, select, textarea')) { ev.preventDefault(); setRail(!$('#app').classList.contains('rail-mini')); }
});
setRail(localStorage.getItem('gz_rail') === 'mini', false);

$('#q-search').addEventListener('input', ev => { S.qText = ev.target.value; renderQueue(); });
$('#q-search').addEventListener('keydown', ev => { if (ev.key === 'Escape') { ev.target.value = ''; S.qText = ''; renderQueue(); ev.target.blur(); } });
document.addEventListener('keydown', ev => {
  if (ev.key === '/' && !ev.target.closest('input, select, textarea')) { ev.preventDefault(); $('#q-search').focus(); }
});

$('#queue').addEventListener('click', ev => {
  const b = ev.target.closest('button[data-id]');
  if (!b) return;
  const e = S.fleet.find(x => x.engine_id === b.dataset.id);
  selectEngine(b.dataset.id, e ? e.last_cycle : null);
});

const asOf = $('#asof');
const paintAsOf = () => { asOf.style.setProperty('--p', ((asOf.value - asOf.min) / (asOf.max - asOf.min) * 100) + '%'); };
paintAsOf();
asOf.addEventListener('input', () => { $('#asof-v').textContent = asOf.value + '%'; paintAsOf(); });
asOf.addEventListener('change', async () => {
  S.asOf = +asOf.value / 100;
  await loadFleet();
  const e = S.fleet.find(x => x.engine_id === S.id);
  if (e) setCursorCycle(e.last_cycle);
});

function renderModelCard() {
  const m = S.model || {}, mt = m.metrics || {}, rs = mt.rul_stack || {}, iv = m.interval || {};
  const built = ((m.build || {}).trained_utc || '').slice(0, 10);
  const d = built ? new Date(built + 'T00:00:00Z').toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' }) : '';
  $('#model-card').innerHTML = `
    <div class="mc-stats">
      <div><b>±${fx(rs.rmse, 1)}</b><span>flights typical RUL error</span></div>
      <div><b>${fx((iv.coverage_test || 0) * 100, 0)}%</b><span>of test flights inside the 90% range</span></div>
    </div>
    <p class="mc-note">Trained ${esc(d)} on ${(m.files || []).length} N-CMAPSS subsets, tested on ${mt.test_engines || '?'} engines it never saw.</p>
    <button class="rail-cta" id="upload"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 11V2.5M4.5 6 8 2.5 11.5 6M2.5 10.5v3h11v-3"/></svg><span>Analyse a CSV of flights</span></button>
    <a class="rail-link" href="legacy.html">Open the previous dashboard</a>`;
  const input = Object.assign(document.createElement('input'), { type: 'file', accept: '.csv', hidden: true });
  document.body.appendChild(input);
  $('#upload').addEventListener('click', () => input.click());
  input.addEventListener('change', async () => {
    if (!input.files.length) return;
    const fd = new FormData(); fd.append('file', input.files[0]);
    $('#upload span').textContent = 'Analysing…'; $('#upload').disabled = true;
    try {
      const body = await api('/api/analyze', { method: 'POST', body: fd });
      await loadFleet();
      const first = Object.keys(body.engines)[0];
      if (first) selectEngine(first, null);
    } catch (e) { alert('The CSV could not be analysed: ' + e.message); }
    finally { $('#upload span').textContent = 'Analyse a CSV of flights'; $('#upload').disabled = false; input.value = ''; }
  });
}

/* ------------------------------------------------------------------ engine selection + cursor */
async function selectEngine(id, cycle) {
  stopPlay();
  S.id = id;
  renderQueue();
  document.querySelector('.work').classList.add('stale');
  try {
    let p = S.cache.get(id);
    if (!p) { p = await api('/api/engine/' + encodeURIComponent(id)); S.cache.set(id, p); }
    if (S.id !== id) return;
    S.p = p; S.col = p.columns; S.n = p.n_cycles;
    await refreshMaint();
  } catch (e) {
    showEmpty('This engine could not be loaded', e.message);
    return;
  } finally { document.querySelector('.work').classList.remove('stale'); }
  const idx = cycle != null ? S.col.cycle.indexOf(cycle) : -1;
  S.i = idx >= 0 ? idx : S.n - 1;
  renderEngine();
}

async function refreshMaint() {
  [S.alerts, S.events] = await Promise.all([
    api(`/api/engine/${encodeURIComponent(S.id)}/alerts`),
    api(`/api/engine/${encodeURIComponent(S.id)}/maintenance`)]);
}

function setCursor(i) {
  i = clamp(Math.round(i), 0, S.n - 1);
  if (i === S.i && $('#flight').value == S.col.cycle[i]) return;
  S.i = i;
  renderCursor();
}
function setCursorCycle(cycle) {
  const idx = S.col.cycle.indexOf(cycle);
  setCursor(idx >= 0 ? idx : S.n - 1);
}

function renderEngine() {
  if (document.body.classList.contains('booting')) setTimeout(() => document.body.classList.remove('booting'), 1400);
  const p = S.p;
  $('#empty').hidden = true;
  $('#eng-title').textContent = p.source === 'upload' ? `Uploaded engine ${p.engine_id}` : `Engine ${p.engine_id}`;
  $('#eng-sub').textContent = p.source === 'upload'
    ? `Analysed from your CSV, ${S.n} flights`
    : `Test engine ${p.unit_local ?? ''} of N-CMAPSS subset ${p.dataset}, ${S.n} flights recorded`;
  $('#flight').max = S.col.cycle[S.n - 1];
  $('#flight').min = S.col.cycle[0];
  $('#flight-max').textContent = 'of ' + S.col.cycle[S.n - 1];
  renderExplainTargets();
  renderCursor();
}

function renderCursor() {
  const t = triage();
  $('#flight').value = S.col.cycle[S.i];
  $('#eng-status').innerHTML = chip(t.urgency, LEVELS[t.urgency], t.driver === 'rul' && t.urgency > t.band ? 'on RUL' : '');
  if (S.lastUrgency != null && S.lastUrgency !== t.urgency && S.lastEngine === S.id) $('#eng-status .chip').classList.add('pop');
  S.lastUrgency = t.urgency; S.lastEngine = S.id;
  const open = S.alerts.filter(a => a.status === 'open' && a.cycle <= S.col.cycle[S.i]).length;
  $('#alert-count').textContent = open || '';
  history.replaceState(null, '', `#/${encodeURIComponent(S.id)}/${S.col.cycle[S.i]}/${S.tab}`);
  checkCritical(t);
  renderTab();
}

/* ------------------------------------------------------------------ critical-part alerts: toast + 3D highlight + beep */
const CRITICAL_LEVEL = 2;                                  // Alarm or Emergency band
const PREVENTIVE = {
  fan: 'Borescope the fan blades and check for foreign-object damage.',
  lpc: 'Borescope the LPC and consider a compressor wash.',
  hpc: 'Borescope the HPC and consider a compressor wash.',
  burner: 'Inspect fuel nozzles and combustor liner; check fuel-flow meter calibration.',
  hpt: 'Borescope the HPT blades and nozzle guide vanes; review EGT margin.',
  lpt: 'Borescope the LPT; review EGT margin and vibration trends.',
  nozzle: 'Inspect the exhaust nozzle flaps, liner and actuation.',
};

function checkCritical(t) {
  const crit = {};
  Object.entries(t.his).forEach(([c, v]) => { const l = bandOf(v); if (l >= CRITICAL_LEVEL) crit[c] = l; });
  S.critical = crit;
  const fresh = Object.entries(crit).filter(([c, l]) => !S.notified.has(`${S.id}|${c}|${l}`));
  fresh.forEach(([c, l]) => { S.notified.add(`${S.id}|${c}|${l}`); if (l === 3) S.notified.add(`${S.id}|${c}|2`); });
  e3SetCritical(crit);
  if (fresh.length) {
    notifyCritical(fresh, t);
    beep(Math.max(...fresh.map(([, l]) => l)));
  }
}

function notifyCritical(list, t) {
  const level = Math.max(...list.map(([, l]) => l)), cyc = S.col.cycle[S.i];
  const names = list.map(([c]) => compName(c));
  const title = list.length === 1 ? `${names[0]} is in ${LEVELS[level]}` : `${list.length} parts critical: ${names.join(', ')}`;
  const el = document.createElement('div');
  el.className = `toast lvl${level}`;
  el.setAttribute('role', 'alert');
  el.innerHTML = `<div class="toast-glyph">${glyph(level, 16)}</div>
    <div class="toast-body"><b>${esc(title)}</b>
      <p>Engine ${esc(S.id)}, flight ${cyc}. ${list.map(([c, l]) => `${esc(compName(c))} health ${fx(t.his[c], 2)} (${LEVELS[l]}).`).join(' ')}</p>
      <p class="toast-do">${esc(PREVENTIVE[list[0][0]] || '')}</p>
      ${S.sound && !(audio.ctx && audio.ctx.state === 'running') ? '<p class="toast-hint">Click anywhere on the page once to allow the alert sound.</p>' : ''}
      <div class="toast-acts"><button class="btn small" data-act="show">Show on engine</button><button class="btn small ghost" data-act="close">Dismiss</button></div></div>`;
  const engine = S.id, part = list[0][0];
  el.dataset.parts = list.map(([c]) => `${engine}|${c}`).join(' ');
  // an escalation replaces the older, now out-of-date toast for the same part
  [...$('#toasts').children].forEach(old => {
    if (list.some(([c]) => (old.dataset.parts || '').split(' ').includes(`${engine}|${c}`))) old.remove();
  });
  const leave = () => { if (reduceMotion()) return el.remove(); el.classList.add('out'); setTimeout(() => el.remove(), 220); };
  el.querySelector('[data-act=close]').addEventListener('click', leave);
  el.querySelector('[data-act=show]').addEventListener('click', () => {
    el.remove();
    const go = () => { if (S.tab !== 'overview') switchTab('overview'); if (E3.focus !== part) setFocus(part); $('#gaspath').scrollIntoView({ behavior: 'smooth', block: 'center' }); };
    if (S.id !== engine) selectEngine(engine, cyc).then(go); else go();
  });
  const stack = $('#toasts');
  stack.prepend(el);                                        // newest sits at the bottom, nearest the eye
  while (stack.children.length > 4) stack.lastElementChild.remove();
  if (level < 3) setTimeout(leave, 12000);                        // Emergency stays until dismissed
}

// Web Audio beep: two tones for Alarm, three higher tones for Emergency. Browsers only allow sound after the user
// has interacted with the page, so the context is created/resumed on the first click or key press.
const audio = { ctx: null };
function unlockAudio() {
  try {
    if (!audio.ctx) audio.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.ctx.state === 'suspended') audio.ctx.resume();
  } catch (e) { /* no audio available */ }
}
['pointerdown', 'keydown'].forEach(ev => document.addEventListener(ev, unlockAudio, { capture: true }));
function beep(level) {
  if (!S.sound || !audio.ctx || audio.ctx.state !== 'running') return;
  const n = level >= 3 ? 3 : 2, f = level >= 3 ? 1175 : 880, t0 = audio.ctx.currentTime + 0.02;
  for (let k = 0; k < n; k++) {
    const o = audio.ctx.createOscillator(), g = audio.ctx.createGain(), t = t0 + k * 0.2;
    o.type = 'square'; o.frequency.value = f;
    g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(0.08, t + 0.01); g.gain.linearRampToValueAtTime(0, t + 0.13);
    o.connect(g).connect(audio.ctx.destination); o.start(t); o.stop(t + 0.15);
  }
}
function renderSoundBtn() {
  const b = $('#sound');
  b.setAttribute('aria-pressed', String(S.sound));
  b.title = S.sound ? 'Alert sound on (click to mute)' : 'Alert sound muted (click to turn on)';
  $('#sound-ico').innerHTML = '<path d="M2 6h3l4-3v10l-4-3H2z" fill="currentColor" stroke="none"/>' +
    (S.sound ? '<path d="M11 5.5a3.5 3.5 0 0 1 0 5M12.8 3.8a6 6 0 0 1 0 8.4"/>' : '<path d="m11 6 4 4m0-4-4 4"/>');
}
$('#sound').addEventListener('click', () => {
  S.sound = !S.sound;
  localStorage.setItem('gz_sound', S.sound ? 'on' : 'off');
  renderSoundBtn();
  if (S.sound) { unlockAudio(); setTimeout(() => beep(2), 50); }      // confirm it works
});
renderSoundBtn();

function renderTab() {
  if (!S.p) return;
  ({ overview: renderOverview, gaspath: renderGasPathTab, why: renderWhy, maint: renderMaint })[S.tab]();
}

/* ------------------------------------------------------------------ tabs + flight controls */
document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));
function moveTabInk() {
  const b = document.querySelector('.tabs button[aria-selected="true"]'), ink = $('.tab-ink');
  if (!b || !ink) return;
  ink.style.width = b.offsetWidth + 'px';
  ink.style.transform = `translateX(${b.offsetLeft}px)`;
}
function switchTab(tab) {
  const changed = S.tab !== tab;
  S.tab = tab;
  document.querySelectorAll('.tabs button').forEach(b => b.setAttribute('aria-selected', String(b.dataset.tab === tab)));
  document.querySelectorAll('.view').forEach(v => {
    v.hidden = v.dataset.view !== tab;
    if (!v.hidden && changed) { v.classList.remove('enter'); void v.offsetWidth; v.classList.add('enter'); }
  });
  moveTabInk();
  if (S.p) renderCursor();
}
addEventListener('resize', moveTabInk);
if (document.fonts) document.fonts.ready.then(moveTabInk);
requestAnimationFrame(moveTabInk);
$('#prev').addEventListener('click', () => setCursor(S.i - 1));
$('#next').addEventListener('click', () => setCursor(S.i + 1));
$('#flight').addEventListener('change', () => {
  const c = +$('#flight').value;
  const idx = S.col.cycle.findIndex(x => x >= c);
  setCursor(idx >= 0 ? idx : S.n - 1);
});
$('#play').addEventListener('click', () => (S.playing ? stopPlay() : startPlay()));
function startPlay() {
  if (S.i >= S.n - 1) setCursor(0);
  $('#play').classList.add('playing');
  $('#play-ico').innerHTML = '<path d="M4 3h3v10H4zM9 3h3v10H9z"/>';
  S.playing = setInterval(() => { if (S.i >= S.n - 1) stopPlay(); else setCursor(S.i + 1); }, 160);
}
function stopPlay() {
  clearInterval(S.playing); S.playing = null;
  $('#play').classList.remove('playing');
  $('#play-ico').innerHTML = '<path d="M5 3v10l8-5z"/>';
}
document.addEventListener('keydown', ev => {
  if (!S.p || ev.target.closest('input, select, textarea')) return;
  if (ev.key === ' ' && ev.target.closest('button, a')) return;
  if (ev.key === 'ArrowLeft') { setCursor(S.i - 1); ev.preventDefault(); }
  else if (ev.key === 'ArrowRight') { setCursor(S.i + 1); ev.preventDefault(); }
  else if (ev.key === 'Home') setCursor(0);
  else if (ev.key === 'End') setCursor(S.n - 1);
  else if (ev.key === ' ') { S.playing ? stopPlay() : startPlay(); ev.preventDefault(); }
});

/* ------------------------------------------------------------------ overview: RUL chart */
function renderRulChart() {
  const box = $('#rul-chart');
  const W = Math.max(320, box.clientWidth), H = clamp(Math.round(W * 0.27), 210, 270);
  const m = { l: 46, r: 16, t: 24, b: 56 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const cyc = S.col.cycle, n = S.n;
  const est = S.col.rul, lo = S.col.rul_q05 || est, hi = S.col.rul_q95 || est, truth = S.col.truth_rul;
  const ymaxRaw = Math.max(...hi.filter(ok), ...est.filter(ok), ...(truth || []).filter(ok), 10);
  const step = niceStep(ymaxRaw, 5), ymax = Math.ceil(ymaxRaw / step) * step;
  const x = c => m.l + (n > 1 ? (c - cyc[0]) / (cyc[n - 1] - cyc[0]) : 0.5) * pw;
  const y = v => m.t + ph - (clamp(v, 0, ymax) / ymax) * ph;
  const line = arr => arr.map((v, k) => ok(v) ? `${k && ok(arr[k - 1]) ? 'L' : 'M'}${x(cyc[k]).toFixed(1)},${y(v).toFixed(1)}` : '').join('');
  let s = '';
  for (let v = 0; v <= ymax + 1e-9; v += step) {
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${y(v)}" y2="${y(v)}" stroke="${v ? C.rule : C.axis}"/>`;
    s += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="${C.ink3}" class="num">${v}</text>`;
  }
  RUL_LEVELS.forEach((t, k) => {
    if (t >= ymax) return;
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${y(t)}" y2="${y(t)}" stroke="${LEVEL_FILL[k + 1]}" stroke-opacity=".55"/>`;
    s += `<text x="${m.l + pw - 4}" y="${y(t) - 4}" text-anchor="end" font-size="10.5" fill="${C.ink3}">${LEVELS[k + 1]} at ${t}</text>`;
  });
  const xstep = niceStep(cyc[n - 1] - cyc[0], Math.max(4, Math.floor(pw / 90)));
  for (let c = Math.ceil(cyc[0] / xstep) * xstep; c <= cyc[n - 1]; c += xstep)
    s += `<text x="${x(c)}" y="${m.t + ph + 16}" text-anchor="middle" font-size="11" fill="${C.ink3}" class="num">${c}</text>`;
  s += `<text x="${m.l}" y="${m.t + ph + 16}" font-size="11" fill="${C.ink3}" text-anchor="end" dx="-8">Flight</text>`;
  s += `<text x="0" y="11" font-size="11" fill="${C.ink3}">Flights left</text>`;
  // interval band, then lines
  const band = cyc.map((c, k) => `${k ? 'L' : 'M'}${x(c).toFixed(1)},${y(hi[k]).toFixed(1)}`).join('') +
    cyc.map((c, k) => k).reverse().map(k => `L${x(cyc[k]).toFixed(1)},${y(lo[k]).toFixed(1)}`).join('') + 'Z';
  s += `<path d="${band}" fill="${C.rul}" fill-opacity=".12"/>`;
  if (truth && truth.some(ok)) s += `<path d="${line(truth)}" fill="none" stroke="${C.truth}" stroke-width="1.5"/>`;
  s += `<path d="${line(lo)}" fill="none" stroke="${C.pred}" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<path d="${line(est)}" fill="none" stroke="${C.rul}" stroke-width="2" stroke-linejoin="round"/>`;
  // event lane: alerts by flight, highest severity per flight
  const laneY = m.t + ph + 38;
  s += `<text x="${m.l}" y="${laneY + 4}" font-size="11" fill="${C.ink3}" text-anchor="end" dx="-8">Alerts</text>`;
  const byCycle = new Map();
  (S.alerts || []).forEach(a => { const l = SEV_LEVEL[a.severity] || 1; if (!byCycle.has(a.cycle) || byCycle.get(a.cycle) < l) byCycle.set(a.cycle, l); });
  byCycle.forEach((l, c) => { s += `<g transform="translate(${x(c) - 5},${laneY - 5})">${glyph(l, 10).replace('<svg', '<svg x="0" y="0"')}</g>`; });
  // cursor
  const cx = x(cyc[S.i]);
  s += `<line x1="${cx}" x2="${cx}" y1="${m.t}" y2="${m.t + ph}" stroke="${C.ink}" stroke-width="1"/>`;
  [[est, C.rul], [lo, C.pred]].concat(truth ? [[truth, C.truth]] : []).forEach(([a, colr]) => {
    if (ok(a[S.i])) s += `<circle cx="${cx}" cy="${y(a[S.i])}" r="4.5" fill="${colr}" stroke="#fff" stroke-width="2"/>`;
  });
  s += `<rect x="${cx - 18}" y="${m.t + ph + 4}" width="36" height="16" rx="2" fill="${C.ink}"/><text x="${cx}" y="${m.t + ph + 16}" text-anchor="middle" font-size="11" fill="#fff" font-weight="600" class="num">${cyc[S.i]}</text>`;
  s += `<line id="rul-hover" x1="0" x2="0" y1="${m.t}" y2="${m.t + ph}" stroke="${C.ink3}" stroke-width="1" visibility="hidden"/>`;
  s += `<g id="rul-hdots" visibility="hidden">${[C.rul, C.pred, C.truth].map((c, k) => `<circle data-k="${k}" r="4" fill="${c}" stroke="#fff" stroke-width="2"/>`).join('')}</g>`;
  s += `<rect id="rul-hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph + 46}" fill="transparent" style="cursor:crosshair"/>`;
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="RUL estimate, planning RUL and 90% range by flight">${s}</svg>`;

  $('#rul-legend').innerHTML = `<span><i class="key" style="border-color:${C.rul}"></i>Best estimate</span>
    <span><i class="key band"></i>90% range</span>
    <span><i class="key" style="border-color:${C.pred}"></i>Planning RUL (low end of range)</span>
    ${truth && truth.some(ok) ? `<span><i class="key" style="border-color:${C.truth}"></i>Actual (simulation truth)</span>` : ''}`;

  const hit = $('#rul-hit'), svg = box.querySelector('svg'), hov = $('#rul-hover');
  const idxAt = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * (W / r.width);
    const c = cyc[0] + ((px - m.l) / pw) * (cyc[n - 1] - cyc[0]);
    let best = 0; cyc.forEach((v, k) => { if (Math.abs(v - c) < Math.abs(cyc[best] - c)) best = k; });
    return best;
  };
  let drag = false;
  hit.addEventListener('pointerdown', ev => { drag = true; hit.setPointerCapture(ev.pointerId); setCursor(idxAt(ev)); });
  hit.addEventListener('pointerup', () => { drag = false; });
  hit.addEventListener('pointermove', ev => {
    const k = idxAt(ev);
    if (drag) { setCursor(k); return; }
    hov.setAttribute('x1', x(cyc[k])); hov.setAttribute('x2', x(cyc[k])); hov.setAttribute('visibility', 'visible');
    const dots = $('#rul-hdots');
    [est, lo, truth].forEach((arr, j) => {
      const dot = dots.querySelector(`[data-k="${j}"]`), v = arr && arr[k];
      dot.setAttribute('visibility', ok(v) ? 'visible' : 'hidden');
      if (ok(v)) { dot.setAttribute('cx', x(cyc[k])); dot.setAttribute('cy', y(Math.max(0, v))); }
    });
    dots.setAttribute('visibility', 'visible');
    const al = (S.alerts || []).filter(a => a.cycle === cyc[k]);
    showTip(`<b>Flight ${cyc[k]}</b><br>
      <i class="k" style="background:${C.rul}"></i>Estimate ${fx(est[k])}<br>
      <i class="k" style="background:${C.pred}"></i>Planning ${fx(Math.max(0, lo[k]))}<br>
      90% range ${fx(Math.max(0, lo[k]))} to ${fx(hi[k])}` +
      (truth && ok(truth[k]) ? `<br><i class="k" style="background:${C.truth}"></i>Actual ${fx(truth[k])}` : '') +
      al.map(a => `<br>${esc(msg(a.message))}`).join(''), ev.clientX, ev.clientY);
  });
  hit.addEventListener('pointerleave', () => { hideTip(); hov.setAttribute('visibility', 'hidden'); $('#rul-hdots').setAttribute('visibility', 'hidden'); });
}

/* ------------------------------------------------------------------ overview: decision panel */
function actionsFor(t) {
  const where = t.loc.text;
  const list = [
    ['Keep routine monitoring. Nothing is due on this engine yet.'],
    [`Add a trend review and borescope of the ${where} to the next scheduled check.`],
    [`Book a shop visit inside the planning RUL of ${fi(t.planning)} flights.`, `Borescope the ${where} at the next opportunity.`],
    [`Plan removal now: the planning RUL is ${fi(t.planning)} flights or less.`, `Inspect the ${where} before further flights are scheduled.`],
  ][t.urgency].slice();
  if (!t.loc.reliable && t.urgency > 0) list.push('The filter cannot separate these modules. Use EGT margin, vibration and borescope findings to pin down which one.');
  return list;
}

function renderDecision() {
  const t = triage(), cyc = S.col.cycle[S.i];
  const open = S.alerts.filter(a => a.status === 'open' && a.cycle <= cyc);
  const el = $('#decision');
  el.style.borderLeftColor = LEVEL_FILL[t.urgency];
  el.innerHTML = `
    <h2>At flight ${cyc} of ${S.col.cycle[S.n - 1]}, plan the shop visit within</h2>
    <p class="hero"><b id="hero-num">${fi(S.heroShown ?? t.planning)}</b><span>flights</span></p>
    <p class="hero-note">Low end of the 90% range. Best estimate ${fx(t.rul)} flights, range ${fx(t.planning)} to ${fx(t.q95)}.</p>
    <dl class="facts">
      <dt>Urgency</dt><dd>${chip(t.urgency, LEVELS[t.urgency], t.driver === 'rul' && t.urgency > t.band ? 'on RUL' : '')}</dd>
      <dt>Worst component health</dt><dd>${chip(t.band, LEVELS[t.band])}</dd>
      <dt>Most degraded</dt><dd>${esc(t.loc.text)} <span class="muted num">${fx(t.his[t.worst], 2)}</span></dd>
      ${ok(t.truth) ? `<dt>Actual RUL (simulation)</dt><dd class="num">${fx(t.truth)}</dd>` : ''}
    </dl>
    ${t.loc.reliable ? '' : `<p class="caution">The Kalman filter cannot tell ${esc(t.loc.text)} apart, so treat this as a group, not a single module.</p>`}
    <h2>Suggested next steps</h2>
    <ul class="todo">${actionsFor(t).map(a => `<li>${esc(a)}</li>`).join('')}</ul>
    <h2>Open alerts up to this flight (${open.length})</h2>
    <ul class="mini-alerts">${open.slice(-3).reverse().map(a => `<li>${glyph(SEV_LEVEL[a.severity] || 1)}<span><span class="muted num">Flight ${a.cycle}</span> ${esc(msg(a.message))}</span></li>`).join('') || '<li><span></span><span class="muted">None open.</span></li>'}</ul>
    <button class="btn" id="go-maint">Record an inspection</button>
    <p class="hero-note" style="margin-top:10px">Decision support only. Your approved maintenance programme governs.</p>`;
  $('#go-maint').addEventListener('click', () => switchTab('maint'));
  tweenNumber($('#hero-num'), S.heroShown, t.planning, v => fi(v));
  S.heroShown = t.planning;
}

/* ------------------------------------------------------------------ overview: gas-path strip */
function engineSchematic(t, W) {
  const H = 96, mid = 42;
  const tot = MODULE_FR.reduce((a, b) => a + b, 0);
  const xs = [0]; MODULE_FR.forEach(f => xs.push(xs[xs.length - 1] + (f / tot) * W));
  // half-heights of the flow path at each module's inlet and outlet
  const hh = { fan: [38, 35], lpc: [22, 19], hpc: [18, 11], burner: [13, 13], hpt: [12, 16], lpt: [17, 25], nozzle: [25, 18] };
  let s = `<path d="M${xs[1] - 4},${mid - 37} L${W - 4},${mid - 30} M${xs[1] - 4},${mid + 37} L${W - 4},${mid + 30}" stroke="${C.axis}" stroke-width="1.5" fill="none"/>`;
  s += `<line x1="0" x2="${W}" y1="${mid}" y2="${mid}" stroke="${C.axis}" stroke-width="1"/>`;
  MODULES.forEach((mod, k) => {
    const [a, b] = hh[mod.key];
    let x0 = xs[k] + 8, x1 = xs[k + 1] - 8;
    const level = ok(t.his[mod.key]) ? bandOf(t.his[mod.key]) : -1;
    const fill = level < 0 ? '#eef1f4' : LEVEL_FILL[level], ink = level < 0 ? C.axis : LEVEL_INK[level];
    if (mod.key === 'fan') {                            // spinner cone ahead of the fan blades
      s += `<path d="M${x0},${mid} Q${x0 + 4},${mid - 13} ${x0 + 20},${mid - 14} L${x0 + 20},${mid + 14} Q${x0 + 4},${mid + 13} ${x0},${mid}Z" fill="#dfe4e9" stroke="${C.axis}"/>`;
      x0 += 26;
    }
    s += `<path d="M${x0},${mid - a} L${x1},${mid - b} L${x1},${mid + b} L${x0},${mid + a}Z" fill="${fill}" fill-opacity="${level < 0 ? 1 : 0.3}" stroke="${level < 0 ? C.axis : fill}" stroke-width="1.5"/>`;
    if (level >= 0 && !AUX.includes(mod.key)) {
      const blades = Math.max(2, Math.round((x1 - x0) / 26));
      for (let j = 1; j <= blades; j++) {
        const f = j / (blades + 1), bx = x0 + f * (x1 - x0), bh = a + (b - a) * f - 3;
        s += `<line x1="${bx}" x2="${bx}" y1="${mid - bh}" y2="${mid + bh}" stroke="${ink}" stroke-opacity=".35" stroke-width="1.5"/>`;
      }
    } else if (mod.key === 'burner') {
      s += `<path d="M${x0 + 10},${mid - 6} q6,-8 12,0 t12,0 M${x0 + 10},${mid + 6} q6,-8 12,0 t12,0" stroke="${C.ink3}" fill="none" stroke-opacity=".6"/>`;
    }
  });
  const st = [[2, '24', 'x_T24'], [3, '30', 'x_T30'], [4, '40', 'v_T40'], [5, '48', 'x_T48'], [6, '50', 'x_T50']];
  st.forEach(([k, name, key]) => {
    const xx = k === 6 ? xs[6] - 2 : xs[k];
    s += `<line x1="${xx}" x2="${xx}" y1="${mid + 26}" y2="${mid + 38}" stroke="${C.ink3}"/>`;
    const v = col(key);
    s += `<text x="${xx}" y="${H - 2}" font-size="11.5" text-anchor="${k === 6 ? 'end' : 'middle'}" fill="${C.ink2}">` +
      `<tspan font-weight="600">T${name}</tspan>${ok(v) ? ` <tspan class="num">${fi(v)} K${key === 'v_T40' ? ' est.' : ''}</tspan>` : ''}</text>`;
  });
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Engine gas path coloured by component health, with station temperatures">${s}</svg>`;
}

function spark(c) {
  const W = 200, H = 40, a = S.col['hi_' + c], n = S.n, cyc = S.col.cycle;
  const x = k => 2 + (n > 1 ? k / (n - 1) : 0.5) * (W - 8), y = v => 3 + (1 - clamp(v, 0, 1)) * (H - 6);
  let s = (S.p.band_edges || [0.75, 0.55, 0.3]).map((e, k) => `<line x1="0" x2="${W}" y1="${y(e)}" y2="${y(e)}" stroke="${LEVEL_FILL[k + 1]}" stroke-opacity=".45"/>`).join('');
  s += `<path d="${a.map((v, k) => ok(v) ? `${k ? 'L' : 'M'}${x(k).toFixed(1)},${y(v).toFixed(1)}` : '').join('')}" fill="none" stroke="${C.ink2}" stroke-width="1.6"/>`;
  s += `<line x1="${x(S.i)}" x2="${x(S.i)}" y1="0" y2="${H}" stroke="${C.ink}" stroke-opacity=".4"/>`;
  s += `<circle cx="${x(S.i)}" cy="${y(a[S.i])}" r="4" fill="${C.ink}" stroke="#fff" stroke-width="2"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="${compName(c)} health from flight ${cyc[0]} to ${cyc[n - 1]}">${s}</svg>`;
}

function forecastLine(c) {
  const fc = (S.model && S.model.forecast) || {}, useful = (fc.useful || {})[c] || {};
  const h = (fc.horizons || []).find(h => useful[String(h)] && ok(col(`fc_${c}_h${h}`)));
  if (!h) return '';
  const v = col(`fc_${c}_h${h}`), lo = col(`fc_${c}_h${h}_lo`), hi = col(`fc_${c}_h${h}_hi`);
  return reliability(c).status === 'ok'
    ? `<p class="gp-fc" title="Forecast in ${h} flights, 90% range ${fx(lo, 2)} to ${fx(hi, 2)}">+${h} flights <b class="num">${fx(v, 2)}</b></p>`
    : `<p class="gp-fc muted" title="Forecast of the group in ${h} flights, not this module alone">Group +${h}: ${fx(v, 2)}</p>`;
}

/* ---- 3D engine: public/models/Turbojet Engine.glb ----
   The model is an STL conversion: 16 meshes that each span the whole engine, so sections are defined by position
   along the engine axis (model X, inlet at +X). Boundaries were read off a side section of the geometry:
   spinner + first stage, axial compressor (long-blade front stages / short-blade rear stages), combustor with
   fuel-injector bosses, two turbine-disc groups, variable exhaust nozzle. */
const SECTIONS = [
  { key: 'fan',    x: [160, 300],   r: 118 },
  { key: 'lpc',    x: [25, 160],    r: 112 },
  { key: 'hpc',    x: [-75, 25],    r: 106 },
  { key: 'burner', x: [-215, -75],  r: 140 },
  { key: 'hpt',    x: [-265, -215], r: 132 },
  { key: 'lpt',    x: [-330, -265], r: 146 },
  { key: 'nozzle', x: [-620, -330], r: 150 },
];
const STATIONS = [{ x: 25, name: 'T24', key: 'x_T24' }, { x: -75, name: 'T30', key: 'x_T30' },
  { x: -215, name: 'T40', key: 'v_T40', est: true }, { x: -265, name: 'T48', key: 'x_T48' }, { x: -330, name: 'T50', key: 'x_T50' }];
const ENGINE_CX = -160;                                   // axial centre of the model
const E3 = { started: false, ready: false, failed: false, meshes: [], levels: {}, focus: null,
             mode: localStorage.getItem('gz_e3mode') === 'health' ? 'health' : 'thermal' };
// Legacy dashboard thermal ramp (public/test_script_3.js tempColor): gas temperature in K -> colour
const THERMAL_STOPS = [[280, '#1e3a8a'], [400, '#0369a1'], [550, '#0891b2'], [700, '#06b6d4'], [850, '#22c55e'],
  [1000, '#eab308'], [1200, '#f97316'], [1500, '#dc2626'], [1800, '#fef3c7']];
function thermalRgb(t) {
  const st = THERMAL_STOPS;
  if (!(t > st[0][0])) return hexRgb(st[0][1]);
  if (t >= st[st.length - 1][0]) return hexRgb(st[st.length - 1][1]);
  const k = st.findIndex(x => x[0] > t), [t0, c0] = st[k - 1], [t1, c1] = st[k], f = (t - t0) / (t1 - t0);
  const a = hexRgb(c0), b = hexRgb(c1);
  return a.map((v, j) => v + (b[j] - v) * f);
}
// Gas temperature along the engine axis (model X, inlet at +X): measured station values at their stations, linear
// between them. The burner heats from T30 to T40 across the combustor; the nozzle carries T50 to the exit.
function axialTemperature(i = S.i) {
  const pts = [[300, col('t2_K', i)], [160, col('t2_K', i)], [25, col('x_T24', i)], [-75, col('x_T30', i)],
    [-120, col('x_T30', i)], [-215, col('v_T40', i)], [-265, col('x_T48', i)], [-330, col('x_T50', i)], [-620, col('x_T50', i)]]
    .filter(p => ok(p[1]));
  return x => {
    if (x >= pts[0][0]) return pts[0][1];
    for (let k = 1; k < pts.length; k++) {
      const [x1, t1] = pts[k], [x0, t0] = pts[k - 1];
      if (x >= x1) return t1 + (t0 - t1) * (x - x1) / (x0 - x1);
    }
    return pts[pts.length - 1][1];
  };
}
const AXIS_MIN = -620, AXIS_BINS = 930;                   // 1-unit bins over the model's axial extent
const hexRgb = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16) / 255);

function sectionAt(x) { const k = SECTIONS.findIndex(s => x >= s.x[0] && x < s.x[1]); return k < 0 ? (x >= 300 ? 0 : SECTIONS.length - 1) : k; }

function e3Init() {
  if (E3.started) return;
  E3.started = true;
  const host = $('#e3d');
  if (!window.THREE || !THREE.GLTFLoader || !THREE.OrbitControls) return e3Fail('The 3D library could not be loaded (no internet connection?).');
  try { E3.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
  catch (e) { return e3Fail('This browser cannot show 3D graphics.'); }
  E3.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  host.prepend(E3.renderer.domElement);
  E3.scene = new THREE.Scene();
  E3.scene.add(new THREE.HemisphereLight(0xffffff, 0x8f9aa6, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(-300, 900, -1400); E3.scene.add(key);
  const rim = new THREE.DirectionalLight(0xdfe9f2, 0.35); rim.position.set(600, -200, 900); E3.scene.add(rim);
  E3.camera = new THREE.PerspectiveCamera(22, 3, 10, 20000);
  E3.controls = new THREE.OrbitControls(E3.camera, E3.renderer.domElement);
  Object.assign(E3.controls, { enableZoom: false, enablePan: false, enableDamping: false, rotateSpeed: 0.6 });
  E3.controls.addEventListener('change', () => { e3Render(); if (!E3.framing) $('#e3d-reset').hidden = false; });
  $('#e3d-reset').addEventListener('click', () => { e3Frame(); $('#e3d-reset').hidden = true; });
  let down = null;
  E3.renderer.domElement.addEventListener('pointerdown', ev => { down = [ev.clientX, ev.clientY]; });
  E3.renderer.domElement.addEventListener('pointerup', ev => {
    if (down && Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) < 4) e3Pick(ev);
    down = null;
  });
  // Versioned URL: browsers may still cache the pre-repair .glb (its header was corrupt). Fall back to the .gltf.
  const urls = ['models/Turbojet%20Engine.glb?v=2', 'models/Turbojet%20Engine.gltf?v=2'], errors = [];
  const tryLoad = k => new THREE.GLTFLoader().load(urls[k], onModel, undefined, err => {
    errors.push(`${urls[k].split('/').pop()}: ${(err && (err.message || err.target && err.target.status)) || err}`);
    console.warn('Engine model failed to load', urls[k], err);
    if (k + 1 < urls.length) tryLoad(k + 1);
    else e3Fail(`The engine model could not be loaded (${errors.join('; ')}).`);
  });
  const onModel = gltf => {
    const mat = new THREE.MeshPhongMaterial({ vertexColors: true, shininess: 38, specular: 0x2a2f36, side: THREE.DoubleSide });
    gltf.scene.traverse(o => {
      if (!o.isMesh) return;
      const geo = o.geometry, pos = geo.attributes.position, n = pos.count;
      geo.computeVertexNormals();
      const sec = new Uint8Array(n), bin = new Uint16Array(n);
      for (let i = 0; i < n; i++) {
        const x = pos.getX(i);
        sec[i] = sectionAt(x);
        bin[i] = clamp(Math.round(x - AXIS_MIN), 0, AXIS_BINS - 1);
      }
      const colArr = new Float32Array(n * 3);
      geo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
      o.material = mat;
      E3.meshes.push({ obj: o, geo, sec, bin, colArr });
    });
    E3.scene.add(gltf.scene);
    E3.glows = new THREE.Group(); E3.scene.add(E3.glows);
    E3.ready = true;
    $('#e3d-msg').hidden = true;
    e3Frame(); e3Paint(); e3Render();
    E3.glowKey = null; e3SetCritical(S.critical || {});
  };
  tryLoad(0);
}

function e3Fail(why) {
  E3.failed = true;
  const m = $('#e3d-msg'); m.hidden = false; m.textContent = why + ' Showing a schematic instead.';
  if (E3.renderer) E3.renderer.domElement.remove();
  if (S.p) renderGasPath();
}

function e3Size() {
  const host = $('#e3d'), w = host.clientWidth, h = host.clientHeight;
  return { w, h };
}

function e3Frame() {                                   // side-on view, inlet on the left, slightly from above
  const { w, h } = e3Size();
  E3.renderer.setSize(w, h, false);
  E3.renderer.domElement.style.width = w + 'px'; E3.renderer.domElement.style.height = h + 'px';
  E3.camera.aspect = w / h;
  const vf = THREE.MathUtils.degToRad(E3.camera.fov), hf = 2 * Math.atan(Math.tan(vf / 2) * E3.camera.aspect);
  const d = Math.max(1000 / (2 * Math.tan(hf / 2)), 410 / (2 * Math.tan(vf / 2)));
  const dir = new THREE.Vector3(0.14, 0.17, -1).normalize();
  E3.controls.target.set(ENGINE_CX, 30, 0);
  E3.camera.position.copy(E3.controls.target).addScaledVector(dir, d);
  E3.camera.updateProjectionMatrix();
  E3.framing = true; E3.controls.update(); E3.framing = false;
  e3Render();
}

function e3Paint() {
  if (!E3.ready) return;
  if (E3.mode === 'thermal') {
    const pickKey = E3.hover || E3.focus;
    const tAt = axialTemperature(), focus = pickKey ? SECTIONS.find(x => x.key === pickKey) : null;
    const lut = new Float32Array(AXIS_BINS * 3);
    for (let k = 0; k < AXIS_BINS; k++) {
      const x = AXIS_MIN + k, c = thermalRgb(tAt(x));
      const dim = focus && !(x >= focus.x[0] && x < focus.x[1]) ? 0.45 : 1;   // picked section stays bright
      for (let j = 0; j < 3; j++) lut[3 * k + j] = c[j] * dim + (1 - dim) * 0.86;
    }
    E3.meshes.forEach(({ geo, bin, colArr }) => {
      for (let i = 0, n = bin.length; i < n; i++) { const o = 3 * bin[i]; colArr[3 * i] = lut[o]; colArr[3 * i + 1] = lut[o + 1]; colArr[3 * i + 2] = lut[o + 2]; }
      geo.attributes.color.needsUpdate = true;
    });
    return;
  }
  const base = [0.70, 0.73, 0.77], idle = [0.82, 0.84, 0.86];
  const cols = SECTIONS.map((s, k) => {
    const lvl = E3.levels[s.key], shade = k % 2 ? 0.93 : 1;
    if (lvl == null) return idle.map(v => v * shade);
    const pick = E3.hover || E3.focus;
    const c = hexRgb(LEVEL_FILL[lvl]), f = pick === s.key ? 0.85 : pick ? 0.3 : 0.55;
    return base.map((b, j) => (b * (1 - f) + c[j] * f) * shade);
  });
  E3.meshes.forEach(({ geo, sec, colArr }) => {
    for (let i = 0, n = sec.length; i < n; i++) { const c = cols[sec[i]]; colArr[3 * i] = c[0]; colArr[3 * i + 1] = c[1]; colArr[3 * i + 2] = c[2]; }
    geo.attributes.color.needsUpdate = true;
  });
}

function e3Legend() {
  const el = $('#e3d-legend');
  if (!el) return;
  document.querySelectorAll('[data-e3mode]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.e3mode === E3.mode)));
  $('#gp-sub').textContent = E3.mode === 'thermal'
    ? "Coloured by gas temperature at each station. Cards show each module's health; drag to turn the engine, click a section or card."
    : 'Coloured by health (1 healthy, 0 failure level). Drag to turn the engine; click a section or card.';
  if (E3.mode === 'thermal') {
    const lo = 250, hi = 1600, stops = THERMAL_STOPS.filter(x => x[0] <= hi).map(([t, c]) => `${c} ${((t - lo) / (hi - lo) * 100).toFixed(1)}%`);
    el.innerHTML = `<div class="tbar" style="background:linear-gradient(90deg, ${stops.join(', ')})"></div>
      <div class="tticks">${[300, 600, 900, 1200, 1500].map(t => `<span style="left:${((t - lo) / (hi - lo) * 100).toFixed(1)}%">${t.toLocaleString('en-GB')}</span>`).join('')}</div>
      <div class="tunit">Gas temperature, K</div>`;
  } else {
    el.innerHTML = LEVELS.map((l, k) => `<span>${glyph(k, 10)}${l}</span>`).join('');
  }
  el.className = 'e3d-legend ' + E3.mode;
}
document.querySelectorAll('[data-e3mode]').forEach(b => b.addEventListener('click', () => {
  E3.mode = b.dataset.e3mode;
  localStorage.setItem('gz_e3mode', E3.mode);
  e3Legend(); e3Paint(); e3Render();
}));

function e3Render() {
  if (!E3.ready) return;
  E3.renderer.render(E3.scene, E3.camera);
  e3Overlay();
}

function e3Project(x, y, z) {
  const v = new THREE.Vector3(x, y, z).project(E3.camera), cv = E3.renderer.domElement;
  return { x: cv.offsetLeft + (v.x + 1) / 2 * cv.clientWidth, y: cv.offsetTop + (1 - v.y) / 2 * cv.clientHeight };
}

function e3Overlay() {                                 // station labels, untracked-section labels, leader lines
  if (!E3.ready || S.tab !== 'overview') return;
  const host = $('#e3d'), gp = $('#gaspath'), hostTop = host.offsetTop;
  $('#e3d-labels').innerHTML = STATIONS.map((st, k) => {
    const top = e3Project(st.x, 160, 0), v = col(st.key), y = 20 + (k % 2) * 17;
    return `<span class="st-tick" style="left:${top.x}px;top:${y + 2}px;height:${Math.max(0, top.y - hostTop - y - 2)}px"></span>
      <span class="st-label" style="left:${top.x}px;top:${y}px" title="${st.est ? 'Estimated: T40 is not measured' : 'Measured'}"><b>${st.name}</b> ${ok(v) ? `${fi(v)} K${st.est ? '*' : ''}` : ''}</span>`;
  }).join('') + SECTIONS.filter(s => s.label).map(s => {
    const p = e3Project((s.x[0] + s.x[1]) / 2, 0, 0);
    return `<span class="sec-label" style="left:${p.x}px;top:${p.y - hostTop}px">${s.label}</span>`;
  }).join('');
  const cards = [...document.querySelectorAll('#gp-cards .gp-col')];
  const gr = gp.getBoundingClientRect();
  const svg = $('#leaders');
  svg.setAttribute('width', gp.clientWidth); svg.setAttribute('height', gp.clientHeight);
  svg.innerHTML = cards.map(card => {
    const s = SECTIONS.find(x => x.key === card.dataset.comp);
    const a = e3Project((s.x[0] + s.x[1]) / 2, -s.r * 0.92, 0);
    const r = card.getBoundingClientRect(), cx = r.left - gr.left + r.width / 2, cy = r.top - gr.top;
    const mid = Math.min(cy - 10, Math.max(a.y + 14, cy - 22));
    const on = E3.focus === s.key || E3.hover === s.key, crit = (S.critical || {})[s.key];
    const stroke = crit != null ? LEVEL_FILL[crit] : on ? C.ink : C.ink3;
    return `<path d="M${cx},${cy} V${mid} L${a.x},${a.y}" fill="none" stroke="${stroke}" stroke-width="${crit != null ? 2.2 : on ? 1.6 : 1}"/>
      <circle cx="${a.x}" cy="${a.y}" r="${crit != null ? 5 : 3.5}" fill="${stroke}" stroke="#fff" stroke-width="1.5"/>`;
  }).join('');
}

function e3Pick(ev) {
  const cv = E3.renderer.domElement, r = cv.getBoundingClientRect();
  const ndc = new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  const ray = new THREE.Raycaster(); ray.setFromCamera(ndc, E3.camera);
  const hit = ray.intersectObjects(E3.meshes.map(m => m.obj), false)[0];
  const key = hit ? SECTIONS[sectionAt(hit.point.x)].key : null;
  setFocus(key && ALL.includes(key) ? key : null);
}

function e3SetCritical(crit) {
  E3.critical = crit;
  if (!E3.ready || !E3.glows) return;
  const keyOf = Object.entries(crit).map(([c, l]) => `${c}${l}`).join(',');
  if (keyOf === E3.glowKey) return;
  E3.glowKey = keyOf;
  E3.glows.children.slice().forEach(m => { E3.glows.remove(m); m.geometry.dispose(); m.material.dispose(); });
  Object.entries(crit).forEach(([c, l]) => {
    const s = SECTIONS.find(x => x.key === c);
    if (!s) return;
    const len = s.x[1] - s.x[0], r = s.r * 1.12;
    const mat = new THREE.MeshBasicMaterial({ color: LEVEL_FILL[l], transparent: true, opacity: 0.35, side: THREE.DoubleSide, depthWrite: false });
    const shell = new THREE.Mesh(new THREE.CylinderGeometry(r, r, len, 64, 1, true), mat);
    shell.rotation.z = Math.PI / 2;                        // cylinder axis (Y) -> engine axis (X)
    shell.position.set((s.x[0] + s.x[1]) / 2, 0, 0);
    const ringMat = new THREE.MeshBasicMaterial({ color: LEVEL_FILL[l], transparent: true, opacity: 0.9, depthWrite: false });
    [s.x[0], s.x[1]].forEach(x => {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(r, 2.2, 8, 96), ringMat);
      ring.rotation.y = Math.PI / 2; ring.position.set(x, 0, 0);
      E3.glows.add(ring);
    });
    E3.glows.add(shell);
  });
  e3Pulse();
}

function e3Pulse() {                                       // animate only while something is critical
  if (E3.pulsing || !E3.glows || !E3.glows.children.length) { if (!E3.glows || !E3.glows.children.length) e3Render(); return; }
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  E3.pulsing = true;
  const step = now => {
    if (!E3.glows.children.length) { E3.pulsing = false; e3Render(); return; }
    const k = still ? 1 : 0.5 + 0.5 * Math.sin(now / 1000 * Math.PI * 2 * 1.1);
    E3.glows.children.forEach(m => { m.material.opacity = m.geometry.type === 'TorusGeometry' ? 0.5 + 0.5 * k : 0.12 + 0.3 * k; });
    E3.renderer.render(E3.scene, E3.camera);
    if (still) { E3.pulsing = false; e3Overlay(); return; }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setFocus(key) {
  E3.focus = E3.focus === key ? null : key;
  document.querySelectorAll('#gp-cards .gp-col').forEach(c => c.classList.toggle('focus', c.dataset.comp === E3.focus));
  e3Paint(); e3Render();
}

function renderGasPath() {
  const t = triage();
  $('#gp-cards').innerHTML = ALL.filter(c => S.col['hi_' + c]).map(c => {
    const mod = MODULES.find(m => m.key === c), v = t.his[c], lvl = bandOf(v), r = reliability(c), g = groupLabel(c);
    const aux = AUX.includes(c), sig = col(aux ? `hi_${c}_sigma` : `kf_hi_${c}_sigma`);
    const pct = k => `${sgn(col(k) * 100, 1)}%`;
    const note = aux
      ? `<p class="gp-note wrap" title="Change against a healthy engine at the same flight condition: combustion efficiency (η), burner pressure ratio (ΔP), nozzle effective area">${c === 'burner'
          ? `η ${pct('dev_eta_b_smooth')}, ΔP ${pct('dev_dp_b_smooth')}` : `Area ${pct('dev_a8_smooth')}`}</p>
         <p class="gp-fc muted" title="Not learned: energy balance and compressible-flow formulas with engineering limits">Physics monitor</p>`
      : (r.status === 'ok' ? `<p class="gp-note" title="The filter isolates this module; typical error on degraded flights">Isolated, ±${fx(r.val_rmse_degraded_points / 100, 2)}</p>`
        : `<p class="gp-note grouped" title="The filter cannot separate these modules">With ${esc(g.text.split(' / ').filter(x => x !== mod.name).join(', '))}</p>`) + forecastLine(c);
    return `<div class="gp-col${E3.focus === c ? ' focus' : ''}${lvl >= CRITICAL_LEVEL ? ` critical lvl${lvl}` : ''}" data-comp="${c}" tabindex="0" role="button" aria-pressed="${E3.focus === c}" aria-label="${mod.full}: health ${fx(v, 2)}, ${LEVELS[lvl]}">
      <div class="gp-top"><h3 title="${mod.full}">${mod.name}</h3>${glyph(lvl, 11)}</div>
      <div class="gp-val num">${fx(v, 2)}<small>±${fx(sig, 2)}</small></div>
      <div class="gp-state" style="color:${LEVEL_INK[lvl]}">${LEVELS[lvl]}</div>
      <div class="gp-spark">${spark(c)}</div>
      ${note}</div>`;
  }).join('');
  document.querySelectorAll('#gp-cards .gp-col').forEach(card => {
    const pick = () => setFocus(card.dataset.comp);
    const preview = on => { E3.hover = on ? card.dataset.comp : null; e3Paint(); e3Render(); };
    card.addEventListener('mouseenter', () => preview(true));
    card.addEventListener('mouseleave', () => preview(false));
    card.addEventListener('focus', () => preview(true));
    card.addEventListener('blur', () => preview(false));
    card.addEventListener('click', pick);
    card.addEventListener('keydown', ev => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); pick(); } });
  });
  E3.levels = Object.fromEntries(Object.keys(t.his).map(c => [c, bandOf(t.his[c])]));
  e3Legend();
  if (E3.failed) {
    $('#e3d').classList.add('flat');
    $('#e3d-labels').innerHTML = '';
    $('#leaders').innerHTML = '';
    const old = $('#e3d .gp-engine'); if (old) old.remove();
    $('#e3d').insertAdjacentHTML('beforeend', `<div class="gp-engine">${engineSchematic(t, Math.max(300, $('#e3d').clientWidth))}</div>`);
    return;
  }
  e3Init();
  if (E3.ready) {
    const { w, h } = e3Size();
    if (E3.renderer.domElement.clientWidth !== w || E3.renderer.domElement.clientHeight !== h) {
      E3.renderer.setSize(w, h, false);
      E3.renderer.domElement.style.width = w + 'px'; E3.renderer.domElement.style.height = h + 'px';
      E3.camera.aspect = w / h; E3.camera.updateProjectionMatrix();
    }
    e3Paint(); e3Render();
  }
}

function renderOverview() { renderRulChart(); renderDecision(); renderGasPath(); }

/* ------------------------------------------------------------------ gas path tab */
const CP_COLD = 1005, CP_HOT = 1148, G_C = 1.4, G_H = 1.33, R_AIR = 287;
function braytonAt(i) {
  const T1 = col('t2_K', i), P1 = col('x_P2', i);
  const P2 = ok(col('v_P30', i)) ? col('v_P30', i) : col('x_Ps30', i), T2 = col('x_T30', i);
  const T3 = col('v_T40', i), P3 = col('x_P40', i), T4 = col('x_T50', i), P4 = col('x_P50', i);
  if (![T1, P1, P2, T2, T3, P3, T4, P4].every(ok)) return null;
  const s2 = CP_COLD * Math.log(T2 / T1) - R_AIR * Math.log(P2 / P1);
  const s3 = s2 + CP_HOT * Math.log(T3 / T2) - R_AIR * Math.log(P3 / P2);
  const s4 = s3 + CP_HOT * Math.log(T4 / T3) - R_AIR * Math.log(P4 / P3);
  const act = { pts: [[0, T1], [s2, T2], [s3, T3], [s4, T4]] };
  act.wc = CP_COLD * (T2 - T1); act.wt = CP_HOT * (T3 - T4); act.wn = act.wt - act.wc; act.q = CP_HOT * (T3 - T2); act.eta = act.wn / act.q;
  const pr = P2 / P1, T2s = T1 * Math.pow(pr, (G_C - 1) / G_C);
  const s3i = CP_HOT * Math.log(T3 / T2s), T4s = T3 * Math.pow(pr, -(G_H - 1) / G_H);
  const ide = { pts: [[0, T1], [0, T2s], [s3i, T3], [s3i, T4s]] };
  ide.wc = CP_COLD * (T2s - T1); ide.wt = CP_HOT * (T3 - T4s); ide.wn = ide.wt - ide.wc; ide.q = CP_HOT * (T3 - T2s); ide.eta = ide.wn / ide.q;
  return { act, ide, pr, T: { T1, T2, T3, T4 } };
}

function renderBrayton() {
  const b = braytonAt(S.i), box = $('#brayton');
  if (!b) { box.innerHTML = '<p class="muted">Station data for the cycle is not available on this flight.</p>'; $('#brayton-stats').innerHTML = ''; return; }
  const W = Math.max(320, box.clientWidth), H = clamp(Math.round(W * 0.62), 260, 420);
  const m = { l: 52, r: 120, t: 24, b: 40 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const all = b.act.pts.concat(b.ide.pts);
  const smin = Math.min(...all.map(p => p[0])), smax = Math.max(...all.map(p => p[0]));
  const tmin = Math.min(...all.map(p => p[1])), tmax = Math.max(...all.map(p => p[1]));
  const sp = (smax - smin) * 0.08 || 50, tstep = niceStep(tmax - tmin, 5);
  const t0 = Math.floor((tmin - 30) / tstep) * tstep, t1 = Math.ceil((tmax + 30) / tstep) * tstep;
  const x = s => m.l + ((s - (smin - sp)) / (smax - smin + 2 * sp)) * pw, y = T => m.t + ph - ((T - t0) / (t1 - t0)) * ph;
  let s = '';
  for (let v = t0; v <= t1; v += tstep) {
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${y(v)}" y2="${y(v)}" stroke="${v === t0 ? C.axis : C.rule}"/>`;
    s += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="${C.ink3}" class="num">${fi(v)}</text>`;
  }
  s += `<text x="0" y="11" font-size="11" fill="${C.ink3}">Temperature (K)</text>`;
  s += `<text x="${m.l + pw}" y="${H - 8}" text-anchor="end" font-size="11" fill="${C.ink3}">Entropy change from inlet (J/kg·K)</text>`;
  const sstep = niceStep(smax - smin + 2 * sp, 5);
  for (let v = Math.ceil((smin - sp) / sstep) * sstep; v <= smax + sp; v += sstep)
    s += `<text x="${x(v)}" y="${m.t + ph + 16}" text-anchor="middle" font-size="11" fill="${C.ink3}" class="num">${fi(Math.abs(v) < 1e-9 ? 0 : v)}</text>`;
  const path = pts => pts.map((p, k) => `${k ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('') + 'Z';
  s += `<path d="${path(b.ide.pts)}" fill="none" stroke="${C.truth}" stroke-width="2" stroke-linejoin="round"/>`;
  s += `<path d="${path(b.act.pts)}" fill="${C.rul}" fill-opacity=".1" stroke="${C.rul}" stroke-width="2" stroke-linejoin="round"/>`;
  const names = ['Station 2, fan inlet', 'Station 30, HPC exit', 'Station 40, burner exit (est.)', 'Station 50, LPT exit'];
  const anchor = ['start', 'end', 'start', 'start'], dx = [10, -10, 10, 10], dy = [16, -16, -4, 0];
  b.act.pts.forEach((p, k) => {
    s += `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="4.5" fill="${C.rul}" stroke="#fff" stroke-width="2"/>`;
    s += `<text x="${x(p[0]) + dx[k]}" y="${y(p[1]) + dy[k]}" text-anchor="${anchor[k]}" font-size="11.5" fill="${C.ink2}"><tspan font-weight="600">${names[k]}</tspan><tspan x="${x(p[0]) + dx[k]}" dy="14" class="num">${fi(p[1])} K</tspan></text>`;
  });
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Measured core cycle against the isentropic ideal">${s}</svg>`;
  $('#brayton-legend').innerHTML = `<span><i class="key" style="border-color:${C.rul}"></i>Measured core cycle, flight ${S.col.cycle[S.i]}</span><span><i class="key" style="border-color:${C.truth}"></i>Isentropic ideal, same pressure ratio</span>`;
  const b0 = braytonAt(0);
  const d = (v, v0, k = 1, dp = 1) => b0 && S.i > 0 ? `<small>${sgn((v - v0) * k, dp)} since flight ${S.col.cycle[0]}</small>` : '';
  $('#brayton-stats').innerHTML = `
    <dl class="stat"><dt>Thermal efficiency</dt><dd class="num">${fx(b.act.eta * 100, 1)}%${d(b.act.eta, b0 && b0.act.eta, 100)}</dd></dl>
    <dl class="stat"><dt>Ideal at this pressure ratio</dt><dd class="num">${fx(b.ide.eta * 100, 1)}%</dd></dl>
    <dl class="stat"><dt>Net specific work</dt><dd class="num">${fi(b.act.wn / 1000)} kJ/kg${d(b.act.wn, b0 && b0.act.wn, 0.001, 0)}</dd></dl>
    <dl class="stat"><dt>HPC efficiency (proxy)</dt><dd class="num">${fx(col('p_eta_hpc') * 100, 1)}%${S.i > 0 ? `<small>${sgn((col('p_eta_hpc') - col('p_eta_hpc', 0)) * 100, 1)} since flight ${S.col.cycle[0]}</small>` : ''}</dd></dl>`;
}

const UNIT = s => s[0] === 'T' ? ['K', 0] : s[0] === 'P' ? ['kPa', 1] : s[0] === 'N' ? ['rpm', 0] : ['kg/s', 3];
const FLOW_ORDER = ['P2', 'P21', 'P15', 'P24', 'T24', 'Ps30', 'T30', 'P40', 'T48', 'P50', 'T50', 'Nf', 'Nc', 'Wf'];
function renderStations() {
  const sens = FLOW_ORDER.filter(s => S.p.sensors.includes(s));
  const res = sens.map(s => col('res_' + s)).filter(ok);
  const lim = Math.max(0.25, ...res.map(Math.abs));
  $('#stations').innerHTML = `<table class="grid"><thead><tr><th>Sensor</th><th class="r">Measured</th><th class="r">Healthy engine</th><th class="r">Difference</th><th>Lower <span style="float:right">Higher</span></th><th class="r">Filter check</th></tr></thead><tbody>` +
    sens.map(s => {
      const [u, dp] = UNIT(s), r = col('res_' + s), gated = col('gated_' + s), nis = col('nis_' + s);
      const w = ok(r) ? Math.abs(r) / lim * 50 : 0;
      return `<tr class="${s === 'Nf' ? 'sep' : ''}"><td class="sname"><b>${s}</b><span>${esc(S.p.sensor_info[s])}</span></td>
        <td class="r">${fx(col('x_' + s), dp)} ${u}</td><td class="r muted">${fx(col('ideal_' + s), dp)} ${u}</td>
        <td class="r">${sgn(r, 2)}%</td>
        <td><div class="divbar"><i></i><b style="${r >= 0 ? 'left:50%' : `left:${50 - w}%`};width:${w}%"></b></div></td>
        <td class="r">${gated ? `<span class="flag" style="color:#b02a2a">${glyph(3, 10)} Rejected</span>` : '<span class="muted">Accepted</span>'} <span class="muted num">${fx(nis, 1)}</span></td></tr>`;
    }).join('') + `</tbody></table><p class="chart-note">Healthy engine: the baseline model's reading for a new engine at this flight's altitude, Mach number and throttle. Bars are scaled to ±${fx(lim, 2)}%. Filter total for this flight ${fx(col('nis_total'), 1)}.</p>`;
}

function renderEkf() {
  const pct = v => ok(v) ? sgn(v * 100, 2) + '%' : '–';
  $('#ekf').innerHTML = `<table class="grid"><thead><tr><th>Module</th><th class="r">Efficiency change</th><th class="r">Flow change</th><th class="r">Health</th><th>Filter can isolate it</th></tr></thead><tbody>` +
    COMPS.map(c => {
      const T = c === 'fan' ? 'fan' : c.toUpperCase(), r = reliability(c), g = groupLabel(c);
      return `<tr><td><b>${compName(c)}</b></td>
        <td class="r">${pct(col(`kf_${T}_eff_mod`))} <span class="muted">±${fx(col(`kf_${T}_eff_mod_sigma`) * 100, 2)}</span></td>
        <td class="r">${pct(col(`kf_${T}_flow_mod`))} <span class="muted">±${fx(col(`kf_${T}_flow_mod_sigma`) * 100, 2)}</span></td>
        <td class="r">${fx(col('kf_hi_' + c), 2)} <span class="muted">±${fx(col(`kf_hi_${c}_sigma`), 2)}</span></td>
        <td>${r.status === 'ok' ? 'Yes' : `<span style="color:#8a5a00">No, grouped with ${esc(g.text.split(' / ').filter(x => x !== compName(c)).join(', '))}</span>`}</td></tr>`;
    }).join('') + `</tbody></table>`;
}

function renderMonitors() {
  const mon = (S.p && S.p.monitors) || {};
  const rows = [['eta_b', 'burner'], ['dp_b', 'burner'], ['a8', 'nozzle']].map(([m, comp]) => {
    const d = col(`dev_${m}_smooth`), h = col(`hi_${m}`), lvl = bandOf(h), lim = mon[m] || {};
    return `<tr><td><b>${esc(lim.label || m)}</b><span class="muted"> ${compName(comp)}</span></td>
      <td class="r">${sgn(d * 100, 2)}%</td><td class="r muted">±${fx((lim.tolerance || 0) * 100, 1)}%</td>
      <td class="r muted">${lim.direction === 'both' ? '±' : '−'}${fx((lim.failure_limit || 0) * 100, 0)}%</td>
      <td class="r"><span style="color:${LEVEL_INK[lvl]}">${glyph(lvl, 10)} ${fx(h, 2)}</span></td></tr>`;
  }).join('');
  const npr = col('phys_npr'), t40p = col('phys_t40_K'), t40v = col('v_T40');
  $('#monitors').innerHTML = `<table class="grid"><thead><tr><th>Monitor</th><th class="r">Change vs healthy</th><th class="r">Tolerance</th><th class="r">Failure limit</th><th class="r">Health</th></tr></thead><tbody>${rows}</tbody></table>
    <dl class="kv" style="margin-top:12px">
      <dt>Turbine inlet T40 from the spool power balance (measured sensors)</dt><dd>${fi(t40p)} K</dd>
      <dt>Simulator's T40 (reference only, not an input)</dt><dd>${fi(t40v)} K</dd>
      <dt>Core nozzle pressure ratio</dt><dd>${fx(npr, 2)}, ${npr >= 1.85 ? 'choked' : 'subsonic exit'}</dd></dl>
    <p class="chart-note">The spool balance leaves out turbine cooling air, so its T40 reads low by a near-constant factor. Each monitor compares like with like against a healthy engine, so that offset cancels.</p>`;
}

async function renderPhysicsCheck() {
  if (!S.physics) {
    $('#physics').innerHTML = '<p class="muted">Running the physics check on every flight of the test engines…</p>';
    try { S.physics = await api('/api/physics/check'); }
    catch (e) { $('#physics').innerHTML = `<p class="muted">The physics check could not run: ${esc(e.message)}</p>`; return; }
  }
  const r = S.physics;
  $('#physics-sub').textContent = `${r.passed} of ${r.total} checks pass on ${r.flights.toLocaleString('en-GB')} flights of ${r.engines} test engines. Hover a check for what it tests.`;
  $('#physics').innerHTML = `<ul class="checks">${r.checks.map(c => `<li title="${esc(c.detail || '')}">
      ${glyph(c.pass ? 0 : 3, 11)}<div><b>${esc(c.name)}</b><span>${esc(c.value)}</span></div></li>`).join('')}</ul>`;
}

function renderGasPathTab() { renderBrayton(); renderStations(); renderEkf(); renderMonitors(); renderPhysicsCheck(); }

/* ------------------------------------------------------------------ why tab */
function renderExplainTargets() {
  const sel = $('#explain-target'), cur = sel.value || 'rul';
  sel.innerHTML = `<option value="rul">Remaining useful life</option>` + COMPS.map(c => `<option value="hi_${c}">${compName(c)} health</option>`).join('');
  sel.value = cur;
}
$('#explain-target').addEventListener('change', () => renderWhy());

function featureLabel(name) {
  const st = (S.model && S.model.states) || [];
  const human = t => t.replace(/_eff_mod/g, ' efficiency').replace(/_flow_mod/g, ' flow').replace(/\+/g, ' + ').replace(/^fan/, 'Fan');
  if (name === 'cycle') return 'Flights flown (engine age)';
  if (name === 'exposure_h') return 'Hours flown (engine age)';
  let m;
  if ((m = name.match(/^res_(\w+)/))) return `${m[1]} vs healthy engine`;
  if ((m = name.match(/^ekf_(\d+)/))) return `Filter state: ${human(st[+m[1]] || 'state ' + m[1])}`;
  if ((m = name.match(/^innov_(\w+)/))) return `${m[1]} filter innovation`;
  if ((m = name.match(/^m_(\w+)/))) return `${m[1]} measured`;
  if ((m = name.match(/^p_(\w+)/))) return `Physics ratio ${m[1].replace(/_/g, ' ')}`;
  if (/^(alt|Mach|TRA|T2)/.test(name)) return `Flight condition: ${name.replace(/_mean|_std|_max|_min/, '')}`;
  return name;
}

function divBars(rows, unit, dp, downLabel, upLabel) {
  const lim = Math.max(1e-9, ...rows.map(r => Math.abs(r.value)));
  return `<div class="bars">` + rows.map(r => {
    const w = Math.abs(r.value) / lim * 50;
    return `<span class="lab" title="${esc(r.title || r.label)}">${esc(r.label)}</span>
      <span class="bar-track"><i></i><b class="${r.value < 0 ? 'down' : 'up'}" style="${r.value < 0 ? `left:${50 - w}%` : 'left:50%'};width:${w}%;background:${r.value < 0 ? C.down : C.up}"></b></span>
      <span class="val">${sgn(r.value, dp)}${unit}</span>`;
  }).join('') + `</div><div class="bars-key"><span><i style="background:${C.down}"></i>${downLabel}</span><span><i style="background:${C.up}"></i>${upLabel}</span></div>`;
}

let whyTimer = null;
function renderWhy() {
  renderTrust();
  const target = $('#explain-target').value || 'rul', cyc = S.col.cycle[S.i], seq = ++S.explainSeq;
  const isRul = target === 'rul', unit = isRul ? '' : '', dp = isRul ? 1 : 3;
  $('#explain-pred').innerHTML = `<span class="muted">Explaining flight ${cyc}</span>`;
  const key = `${S.id}|${cyc}|${target}`;
  const wait = '<p class="muted">Working out the explanation. This re-runs the pipeline once per sensor and takes a few seconds.</p>';
  if (!S.explained.has(key)) {
    ['#drivers', '#occlusion', '#blend'].forEach(s => { if (!$(s).innerHTML.trim()) $(s).innerHTML = wait; $(s).classList.add('stale'); });
  }
  clearTimeout(whyTimer);
  whyTimer = setTimeout(async () => {
    let ex = S.explained.get(key);
    try { if (!ex) { ex = await api(`/api/engine/${encodeURIComponent(S.id)}/explain/${cyc}?target=${target}`); S.explained.set(key, ex); } }
    catch (e) { ['#drivers', '#occlusion', '#blend'].forEach(s => $(s).classList.remove('stale')); $('#drivers').innerHTML = `<p class="muted">No explanation for this flight: ${esc(e.message)}</p>`; return; }
    if (seq !== S.explainSeq) return;
    ['#drivers', '#occlusion', '#blend'].forEach(s => $(s).classList.remove('stale'));
    $('#explain-pred').innerHTML = isRul
      ? `Prediction at flight ${cyc}: <b>${fx(ex.prediction, 1)} flights</b> remaining`
      : `${compName(target.slice(3))} health at flight ${cyc}: <b>${fx(ex.prediction, 3)}</b>`;
    const down = isRul ? 'Shortens predicted life' : 'Lowers health', up = isRul ? 'Extends predicted life' : 'Raises health';
    if (ex.tree_shap) {
      $('#drivers-sub').textContent = 'TreeSHAP of the four models, weighted as the blend weights them, in flights of RUL. A group combines a signal with its recent trend.';
      $('#drivers').innerHTML = divBars(ex.tree_shap.groups.slice(0, 9).map(g => ({ label: featureLabel(g.name), title: g.name, value: g.value })), '', dp, down, up);
    } else {
      $('#drivers-sub').textContent = '';
      $('#drivers').innerHTML = `<p>${esc(ex.source || 'This value')} comes from the Kalman filter's physics model, not from a learned model, so there are no model drivers to show. The sensor view shows what the filter reacted to.</p>`;
    }
    const occ = ((ex.occlusion || {}).sensors || []).slice(0, 9);
    $('#occlusion').innerHTML = occ.length ? divBars(occ.map(o => ({ label: `${o.sensor} ${o.description}`, value: o.value })), '', dp, down, up) : '<p class="muted">Not available.</p>';
    const cb = ex.contributions;
    const mt = (S.model && S.model.metrics) || {};
    $('#blend').innerHTML = cb ? `<table class="grid"><thead><tr><th>Model</th><th class="r">Its RUL</th><th class="r">Share of the blend</th><th class="r">Test error (RMSE)</th></tr></thead><tbody>` +
      cb.models.map(r => `<tr><td>${esc({ randomforest: 'Random forest', xgboost: 'XGBoost', lightgbm: 'LightGBM', catboost: 'CatBoost' }[r.name] || r.name)}</td>
        <td class="r">${fx(r.prediction, 1)}</td><td class="r">${sgn(r.contribution, 1)}</td><td class="r">${fx((mt['rul_' + r.name] || {}).rmse, 2)}</td></tr>`).join('') +
      `<tr class="sep"><td>Ridge blend (baseline ${fx(cb.intercept, 1)})</td><td class="r"><b>${fx(ex.prediction, 1)}</b></td><td></td><td class="r">${fx((mt.rul_stack || {}).rmse, 2)}</td></tr></tbody></table>
      <p class="chart-note">The four models agree closely (correlation above 0.99), so the blend adds little over the best single model. Share = Ridge weight × the model's scaled prediction.</p>`
      : `<p class="muted">Component health is not blended from models.</p>`;
  }, S.explained.has(key) ? 0 : 350);
}

function renderTrust() {
  const m = S.model || {}, mt = m.metrics || {}, rs = mt.rul_stack || {}, iv = m.interval || {};
  const phase = (mt.by_phase || []).slice().sort((a, b) => (a.group > b.group ? 1 : -1));
  const plabel = g => ({ 'early (RUL>60)': 'Early life (more than 60 left)', 'mid (20<RUL<=60)': 'Mid life (20 to 60 left)', 'end_of_life (RUL<=20)': 'End of life (20 or fewer left)' }[g] || g);
  $('#trust').innerHTML = `<dl class="kv">
      <dt>Typical RUL error on unseen engines (RMSE)</dt><dd>${fx(rs.rmse, 1)} flights</dd>
      <dt>Average absolute error</dt><dd>${fx(rs.mae, 1)} flights</dd>
      <dt>90% range held on test flights</dt><dd>${fx((iv.coverage_test || 0) * 100, 0)}%</dd>
      <dt>Average width of that range</dt><dd>${fx(iv.mean_width_test, 0)} flights</dd></dl>
    <table class="grid" style="margin-top:12px"><thead><tr><th>Life stage</th><th class="r">Error</th><th class="r">Bias</th></tr></thead><tbody>
      ${phase.map(r => `<tr><td>${esc(plabel(r.group))}</td><td class="r">${fx(r.rmse, 1)}</td><td class="r">${sgn(r.bias, 1)}</td></tr>`).join('')}</tbody></table>
    <ul class="fine">
      <li>Positive bias means the model says more life is left than there was. It does this in mid life, so plan from the low end of the range.</li>
      <li>Only LPC and HPC health can be isolated by the filter. Fan, HPT and LPT are reported as groups.</li>
      <li>Trained and tested on NASA's N-CMAPSS simulation. Real fleet data would need recalibration first.</li></ul>`;
}

/* ------------------------------------------------------------------ maintenance tab */
document.querySelectorAll('[data-af]').forEach(b => b.addEventListener('click', () => {
  S.alertFilter = b.dataset.af;
  document.querySelectorAll('[data-af]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
  renderAlerts();
}));

function renderAlerts() {
  const cyc = S.col.cycle[S.i];
  const later = S.alerts.filter(a => a.cycle > cyc).length;
  const rows = S.alerts.filter(a => a.cycle <= cyc && (S.alertFilter === 'all' || a.status === 'open')).slice().reverse();
  $('#alerts').innerHTML = rows.map(a => {
    const future = false;
    return `<div class="alert-row ${future ? 'future' : ''}">${glyph(SEV_LEVEL[a.severity] || 1)}
      <span class="when">Flight ${a.cycle}</span>
      <span>${esc(msg(a.message))}${future ? ' <span class="muted">(after the selected flight)</span>' : ''}
        ${a.status !== 'open' ? `<br><span class="handled">${a.status === 'acknowledged' ? 'Acknowledged' : 'Dismissed'} by ${esc(a.handled_by)} on ${esc((a.handled_utc || '').slice(0, 16).replace('T', ' '))} UTC</span>` : ''}</span>
      <span class="acts">${a.status === 'open' && !future ? `<button class="btn small ghost" data-ack="${a.id}" data-st="acknowledged">Acknowledge</button><button class="btn small ghost" data-ack="${a.id}" data-st="dismissed">Dismiss</button>` : ''}</span></div>`;
  }).join('') + (later ? `<p class="chart-note">${later} more alert${later === 1 ? ' was' : 's were'} raised after flight ${cyc}. Move the flight forward to see ${later === 1 ? 'it' : 'them'}.</p>` : '') ||
    `<p class="muted">No ${S.alertFilter === 'open' ? 'open ' : ''}alerts on this engine up to flight ${cyc}.</p>`;
  $('#alerts').querySelectorAll('[data-ack]').forEach(b => b.addEventListener('click', async () => {
    const user = ($('#log-form [name=performed_by]') || {}).value || S.user;
    if (!user.trim()) { formMsg('Enter your name in the form first, so the alert record shows who handled it.', true); $('#log-form [name=performed_by]').focus(); return; }
    try {
      await post(`/api/alerts/${b.dataset.ack}`, { status: b.dataset.st, user: user.trim() });
      const row = b.closest('.alert-row');
      if (row && !reduceMotion() && S.alertFilter === 'open') { row.classList.add('leaving'); await new Promise(r => setTimeout(r, 240)); }
      await refreshMaint(); renderAlerts(); renderCursor();
    } catch (e) { formMsg('The alert was not updated: ' + e.message, true); }
  }));
}

function formMsg(text, err) {
  const el = $('#log-form .form-msg');
  if (el) { el.textContent = text; el.className = 'form-msg ' + (err ? 'err' : 'ok'); }
}

function renderForm() {
  const f = $('#log-form'), en = S.enums || { event_types: [], findings: [] };
  const keep = f.querySelector('[name]') ? Object.fromEntries(new FormData(f)) : null;
  const cyc = S.col.cycle[S.i], t = triage();
  const openNow = S.alerts.filter(a => a.status === 'open' && a.cycle <= cyc).slice().reverse();
  const nice = v => v.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());
  const findLabel = { confirmed: 'Confirmed what the model flagged', not_found: 'Found nothing wrong', inconclusive: 'Inconclusive' };
  f.innerHTML = `
    <label>Your name or staff ID<input name="performed_by" required value="${esc(S.user)}"></label>
    <label>Flight<input value="${cyc}" disabled class="num"></label>
    <label>What was done<select name="event_type">${en.event_types.map(v => `<option value="${v}">${nice(v)}</option>`).join('')}</select></label>
    <label>Module<select name="component"><option value="">Not module-specific</option>${ALL.map(c => `<option value="${c}" ${c === t.worst ? 'selected' : ''}>${compName(c)}</option>`).join('')}</select></label>
    <label>Result<select name="finding">${en.findings.map(v => `<option value="${v}">${findLabel[v] || nice(v)}</option>`).join('')}</select></label>
    <label>Observed health, 0 to 1 (optional)<input name="observed_health" type="number" min="0" max="1" step="0.01"></label>
    <label class="wide">Linked alert<select name="alert_id"><option value="">None</option>${openNow.map(a => `<option value="${a.id}">Flight ${a.cycle}: ${esc(msg(a.message))}</option>`).join('')}</select></label>
    <label class="wide">Action taken<input name="action" placeholder="For example: compressor wash, blade blend repair"></label>
    <label class="wide">Notes<textarea name="notes"></textarea></label>
    <div class="wide"><button class="btn" type="submit">Save inspection</button></div>
    <p class="form-msg" role="status"></p>`;
  if (keep) Object.entries(keep).forEach(([k, v]) => { const el = f.querySelector(`[name="${k}"]`); if (el && v && k !== 'alert_id') el.value = v; });
}

$('#log-form').addEventListener('submit', async ev => {
  ev.preventDefault();
  const fd = new FormData(ev.target), num = k => (fd.get(k) === '' || fd.get(k) == null ? null : Number(fd.get(k)));
  const user = (fd.get('performed_by') || '').trim();
  S.user = user; localStorage.setItem('gz_user', user);
  try {
    await post(`/api/engine/${encodeURIComponent(S.id)}/maintenance`, {
      performed_by: user, event_type: fd.get('event_type'), finding: fd.get('finding'), cycle: S.col.cycle[S.i],
      component: fd.get('component') || null, observed_health: num('observed_health'), alert_id: num('alert_id'),
      action: fd.get('action') || null, notes: fd.get('notes') || null });
    await refreshMaint();
    ev.target.reset();
    renderMaint(); renderCursor();
    formMsg('Inspection saved to the maintenance record.', false);
    loadFleet();
  } catch (e) { formMsg('Not saved: ' + e.message, true); }
});

async function renderMaint() {
  renderAlerts(); renderForm();
  $('#history').innerHTML = S.events.map(e => `<div class="event"><b>${esc(e.event_type.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase()))}</b>${e.component ? ` on the ${esc(compName(e.component))}` : ''}, flight ${e.cycle ?? '–'}: ${esc({ confirmed: 'confirmed the flag', not_found: 'nothing found', inconclusive: 'inconclusive' }[e.finding] || e.finding)}${e.observed_health != null ? `, observed health ${fx(e.observed_health, 2)}` : ''}
    ${e.action ? `<br>${esc(e.action)}` : ''}${e.notes ? `<br>${esc(e.notes)}` : ''}
    <div class="meta">${esc(e.performed_by)}, ${esc((e.performed_utc || '').slice(0, 16).replace('T', ' '))} UTC</div></div>`).join('') || '<p class="muted">No inspections recorded on this engine yet.</p>';
  try {
    const ls = await api('/api/learning/summary');
    $('#learn').textContent = `Across the fleet: ${ls.events} inspection${ls.events === 1 ? '' : 's'} recorded` +
      (ls.flag_precision != null ? `, the model's flag was confirmed ${fx(ls.flag_precision * 100, 0)}% of the time` : '') +
      (ls.health_estimate_mae != null ? `, health estimates were off by ${fx(ls.health_estimate_mae, 2)} on average (${ls.n_health_checks} checks)` : '') +
      '. These findings are kept for recalibrating the model offline; the model never changes on its own.';
  } catch (e) { $('#learn').textContent = ''; }
}

/* ------------------------------------------------------------------ startup */
function showEmpty(title, msg) {
  const el = $('#empty');
  el.hidden = false;
  el.innerHTML = `<h2>${esc(title)}</h2><p>${esc(msg)}</p>`;
  document.querySelectorAll('.view').forEach(v => { v.hidden = true; });
}

let resizeT = null;
addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => S.p && renderTab(), 120); });

async function init() {
  let health;
  try { health = await api('/api/health'); }
  catch (e) {
    // Opened as a file or from another dev server: the API (and the 3D model) only work when the backend serves
    // the page. If the backend is up on its usual port, go there.
    const home = 'http://localhost:8000/index.html' + location.hash;
    const elsewhere = location.protocol === 'file:' || location.port !== '8000';
    if (elsewhere) {
      try {
        await fetch('http://localhost:8000/api/health', { mode: 'no-cors', cache: 'no-store' });
        showEmpty('Opening the dashboard on the backend', 'This page has to be served by the backend. Taking you to ' + home);
        location.replace(home);
        return;
      } catch (e2) { /* backend not reachable either */ }
    }
    showEmpty('The backend is not running', (elsewhere ? 'This page was opened directly, so it cannot reach the model. ' : '') +
      'Start the backend from the project folder with .venv\\Scripts\\python -m backend, then open ' + home);
    return;
  }
  if (!health.bundle_loaded) { showEmpty('No trained model is loaded', health.detail || 'Copy a model bundle into backend/models and restart the backend.'); return; }
  S.model = health.model;
  renderModelCard();
  const [, hid, hcyc, htab] = (location.hash.match(/^#\/([^/]+)\/(\d+)\/(\w+)/) || []);
  if (htab && ['overview', 'gaspath', 'why', 'maint'].includes(htab)) switchTab(htab);
  try { await loadFleet(); }
  catch (e) { showEmpty('The fleet could not be loaded', e.message); return; }
  const first = S.fleet.find(e => e.engine_id === decodeURIComponent(hid || '')) || S.fleet[0];
  if (!first) { showEmpty('No engines yet', 'Analyse a CSV of flights to add one.'); return; }
  selectEngine(first.engine_id, hcyc ? +hcyc : first.last_cycle);
}
// follow links that only change the hash (#/engine/flight/tab), e.g. pasted into the address bar
addEventListener('hashchange', () => {
  const [, hid, hcyc, htab] = location.hash.match(/^#\/([^/]+)\/(\d+)\/(\w+)/) || [];
  if (!hid || !S.p) return;
  const id = decodeURIComponent(hid), cyc = +hcyc;
  if (htab && htab !== S.tab && ['overview', 'gaspath', 'why', 'maint'].includes(htab)) switchTab(htab);
  if (id !== S.id) selectEngine(id, cyc);
  else if (cyc !== S.col.cycle[S.i]) setCursorCycle(cyc);
});

init();
})();
