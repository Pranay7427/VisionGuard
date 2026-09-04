/**
 * VisionGuard — Cyber Command Center Frontend Controller
 * WebSockets, Interactive Canvas Drawing, Web Audio Alarms, Chart.js Analytics, and Media Gallery.
 */

// Global State
const state = {
    ws: null,
    config: null,
    telemetry: null,
    audioEnabled: true,
    activeTab: 'traffic',
    currentSeverityFilter: 'ALL',
    eventsList: [],
    
    // Interactive Canvas Drawing State
    drawingMode: null, // null | 'zone' | 'tripwire'
    drawingPoints: [],
    mousePos: { x: 0, y: 0 },
    editingZoneId: null,
    editingTripwireId: null,

    // Charts
    charts: {
        traffic: null,
        classes: null,
        hourly: null,
        zones: null
    },
    trafficHistory: {
        labels: [],
        peopleIn: [],
        vehiclesIn: [],
        outflow: []
    }
};

// Audio Synthesizer Context
let audioCtx = null;

function getAudioContext() {
    if (!audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
            audioCtx = new AudioContext();
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
    return audioCtx;
}

// ---------------------------------------------------------
// Initialization on DOM Load
// ---------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    initWebSocket();
    loadConfig();
    initCanvas();
    initCharts();
    loadEvents();
    initPeriodicUpdates();
});

// ---------------------------------------------------------
// WebSocket Connection & Real-Time Telemetry
// ---------------------------------------------------------
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log('[VisionGuard WS] Telemetry stream connected.');
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleTelemetry(data);
        } catch (e) {
            console.error('[WS Parse Error]', e);
        }
    };

    state.ws.onclose = () => {
        console.warn('[VisionGuard WS] Telemetry stream closed. Reconnecting in 2s...');
        setTimeout(initWebSocket, 2000);
    };

    state.ws.onerror = (err) => {
        console.error('[VisionGuard WS Error]', err);
    };
}

function handleTelemetry(t) {
    state.telemetry = t;

    // Update Top HUD Pills
    document.getElementById('hudFps').textContent = t.fps.toFixed(1);
    document.getElementById('hudCpu').textContent = `${t.cpu_usage}%`;
    document.getElementById('hudRam').textContent = `${t.ram_usage}%`;

    // Update Live Stats Strip
    document.getElementById('statInCount').textContent = t.counts.total_in;
    document.getElementById('statOutCount').textContent = t.counts.total_out;
    document.getElementById('statActiveTracks').textContent = t.active_tracks_count;

    // Calculate total active breaches
    let totalBreaches = 0;
    for (const zId in t.active_breaches) {
        totalBreaches += t.active_breaches[zId].length;
    }
    document.getElementById('statBreachesCount').textContent = totalBreaches;

    // Update Recording State on Record Button
    const recordBtn = document.getElementById('btnRecord');
    const recordText = document.getElementById('recordBtnText');
    if (t.is_recording) {
        recordBtn.classList.add('recording');
        recordText.textContent = 'RECORDING';
    } else {
        recordBtn.classList.remove('recording');
        recordText.textContent = 'RECORD';
    }

    // Process new incoming events for sound and feed
    if (t.recent_events && t.recent_events.length > 0) {
        t.recent_events.forEach(ev => {
            if (!state.eventsList.some(e => e.id === ev.id)) {
                state.eventsList.unshift(ev);
                prependEventCard(ev);
                playEventAudio(ev.severity, ev.event_type);
            }
        });
    }

    // Update Real-time Traffic Timeline Chart data point every 2s
    updateTrafficChartPoint(t);
}

// ---------------------------------------------------------
// Web Audio Alarm Synthesizer
// ---------------------------------------------------------
function playEventAudio(severity, eventType) {
    if (!state.audioEnabled) return;
    const ctx = getAudioContext();
    if (!ctx) return;

    try {
        const now = ctx.currentTime;
        if (severity === 'CRITICAL' || eventType === 'ZONE_INTRUSION') {
            // Pulsing two-tone alarm siren
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(800, now);
            osc.frequency.linearRampToValueAtTime(1200, now + 0.15);
            osc.frequency.linearRampToValueAtTime(800, now + 0.3);
            osc.frequency.linearRampToValueAtTime(1200, now + 0.45);
            gain.gain.setValueAtTime(0.18, now);
            gain.gain.linearRampToValueAtTime(0.01, now + 0.5);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.5);
        } else if (severity === 'WARNING' || eventType === 'ZONE_DWELL_BREACH') {
            // Warning dual beep
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(650, now);
            osc.frequency.setValueAtTime(750, now + 0.1);
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.linearRampToValueAtTime(0.01, now + 0.25);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.25);
        } else if (eventType === 'TRIPWIRE_CROSSED') {
            // Gentle high-tech crossing chirp
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(980, now);
            osc.frequency.exponentialRampToValueAtTime(1400, now + 0.08);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.linearRampToValueAtTime(0.001, now + 0.09);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.09);
        }
    } catch (e) {
        console.warn('[Audio Alert Error]', e);
    }
}

function toggleAudioAlarm() {
    state.audioEnabled = !state.audioEnabled;
    const btn = document.getElementById('btnAudioToggle');
    const text = document.getElementById('soundBtnText');
    if (state.audioEnabled) {
        btn.classList.add('active');
        text.textContent = 'ALARM ON';
        getAudioContext(); // Initialize
    } else {
        btn.classList.remove('active');
        text.textContent = 'ALARM OFF';
    }
}

// ---------------------------------------------------------
// System Configuration & Settings
// ---------------------------------------------------------
async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        state.config = await res.json();
        applyConfigToUI(state.config);
    } catch (e) {
        console.error('[Load Config Error]', e);
    }
}

function applyConfigToUI(cfg) {
    document.getElementById('toggleMotion').checked = cfg.motion_detection_enabled;
    document.getElementById('toggleTracking').checked = cfg.object_tracking_enabled;
    document.getElementById('togglePeopleCount').checked = cfg.people_counting_enabled;
    document.getElementById('toggleVehicleCount').checked = cfg.vehicle_counting_enabled;
    document.getElementById('toggleIntrusion').checked = cfg.intrusion_detection_enabled;
    document.getElementById('toggleAutoRecord').checked = cfg.auto_recording_enabled;

    document.getElementById('detectorModelSelect').value = cfg.detector_model;
    document.getElementById('motionSensitivity').value = cfg.motion_sensitivity;
    document.getElementById('valMotionSens').textContent = `${cfg.motion_sensitivity}%`;
    document.getElementById('minContourArea').value = cfg.min_contour_area;
    document.getElementById('valMinArea').textContent = `${cfg.min_contour_area} px`;

    // Overlays
    setHudBtnState('btnHudBoxes', cfg.show_bounding_boxes);
    setHudBtnState('btnHudTracks', cfg.show_track_ids);
    setHudBtnState('btnHudTrails', cfg.show_tracking_trails);
    setHudBtnState('btnHudHeatmap', cfg.show_heatmap);
    setHudBtnState('btnHudMask', cfg.show_motion_mask);
    setHudBtnState('btnHudZones', cfg.show_zones);
    setHudBtnState('btnHudLines', cfg.show_tripwires);

    renderZonesList();
    renderTripwiresList();
}

function setHudBtnState(btnId, isActive) {
    const btn = document.getElementById(btnId);
    if (btn) {
        if (isActive) btn.classList.add('active');
        else btn.classList.remove('active');
    }
}

async function updateSettings() {
    if (!state.config) return;

    state.config.motion_detection_enabled = document.getElementById('toggleMotion').checked;
    state.config.object_tracking_enabled = document.getElementById('toggleTracking').checked;
    state.config.people_counting_enabled = document.getElementById('togglePeopleCount').checked;
    state.config.vehicle_counting_enabled = document.getElementById('toggleVehicleCount').checked;
    state.config.intrusion_detection_enabled = document.getElementById('toggleIntrusion').checked;
    state.config.auto_recording_enabled = document.getElementById('toggleAutoRecord').checked;

    state.config.detector_model = document.getElementById('detectorModelSelect').value;
    state.config.motion_sensitivity = parseInt(document.getElementById('motionSensitivity').value);
    state.config.min_contour_area = parseInt(document.getElementById('minContourArea').value);

    document.getElementById('valMotionSens').textContent = `${state.config.motion_sensitivity}%`;
    document.getElementById('valMinArea').textContent = `${state.config.min_contour_area} px`;

    try {
        await fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(state.config)
        });
    } catch (e) {
        console.error('[Update Settings Error]', e);
    }
}

async function toggleHudLayer(layer) {
    if (!state.config) return;

    if (layer === 'boxes') state.config.show_bounding_boxes = !state.config.show_bounding_boxes;
    if (layer === 'tracks') state.config.show_track_ids = !state.config.show_track_ids;
    if (layer === 'trails') state.config.show_tracking_trails = !state.config.show_tracking_trails;
    if (layer === 'heatmap') state.config.show_heatmap = !state.config.show_heatmap;
    if (layer === 'mask') state.config.show_motion_mask = !state.config.show_motion_mask;
    if (layer === 'zones') state.config.show_zones = !state.config.show_zones;
    if (layer === 'lines') state.config.show_tripwires = !state.config.show_tripwires;

    applyConfigToUI(state.config);
    await updateSettings();
}

// ---------------------------------------------------------
// Video Source Switching & Uploads
// ---------------------------------------------------------
async function changeVideoSource() {
    const val = document.getElementById('videoSourceSelect').value;
    const [sourceType, param] = val.split(':');

    const form = new FormData();
    form.append('source_type', sourceType);
    form.append('param', param || '');

    try {
        await fetch('/api/source', { method: 'POST', body: form });
        // Refresh stream image src
        const img = document.getElementById('liveStreamImg');
        img.src = `/api/stream?t=${Date.now()}`;
    } catch (e) {
        console.error('[Change Source Error]', e);
    }
}

function openUploadModal() {
    document.getElementById('videoFileInput').click();
}

async function uploadVideoFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const form = new FormData();
    form.append('file', file);

    try {
        const res = await fetch('/api/upload', { method: 'POST', body: form });
        const data = await res.json();
        if (data.status === 'success') {
            const img = document.getElementById('liveStreamImg');
            img.src = `/api/stream?t=${Date.now()}`;
        }
    } catch (e) {
        console.error('[Upload Video Error]', e);
    }
}

function onStreamLoaded() {
    resizeCanvas();
}

function onStreamError() {
    console.warn('[Live Stream] Waiting for stream frames...');
    setTimeout(() => {
        const img = document.getElementById('liveStreamImg');
        img.src = `/api/stream?t=${Date.now()}`;
    }, 1500);
}

// ---------------------------------------------------------
// Interactive Canvas Studio (Draw Zones & Tripwires)
// ---------------------------------------------------------
function initCanvas() {
    const canvas = document.getElementById('overlayCanvas');
    window.addEventListener('resize', resizeCanvas);

    canvas.addEventListener('mousemove', (e) => {
        if (!state.drawingMode) return;
        const rect = canvas.getBoundingClientRect();
        state.mousePos = {
            x: (e.clientX - rect.left) / rect.width,
            y: (e.clientY - rect.top) / rect.height
        };
        renderDrawing();
    });

    canvas.addEventListener('click', (e) => {
        if (!state.drawingMode) return;
        const rect = canvas.getBoundingClientRect();
        const pt = {
            x: parseFloat(((e.clientX - rect.left) / rect.width).toFixed(3)),
            y: parseFloat(((e.clientY - rect.top) / rect.height).toFixed(3))
        };

        if (state.drawingMode === 'zone') {
            // Check if clicking close to first point to close polygon
            if (state.drawingPoints.length >= 3) {
                const first = state.drawingPoints[0];
                const dist = Math.hypot(pt.x - first.x, pt.y - first.y);
                if (dist < 0.04) {
                    finishCurrentDrawing();
                    return;
                }
            }
            state.drawingPoints.push(pt);
            renderDrawing();
        } else if (state.drawingMode === 'tripwire') {
            state.drawingPoints.push(pt);
            renderDrawing();
            if (state.drawingPoints.length >= 2) {
                finishCurrentDrawing();
            }
        }
    });

    canvas.addEventListener('dblclick', () => {
        if (state.drawingMode === 'zone' && state.drawingPoints.length >= 3) {
            finishCurrentDrawing();
        }
    });
}

function resizeCanvas() {
    const img = document.getElementById('liveStreamImg');
    const canvas = document.getElementById('overlayCanvas');
    if (img && canvas) {
        canvas.width = img.clientWidth || 960;
        canvas.height = img.clientHeight || 540;
    }
}

function startDrawingZone() {
    state.drawingMode = 'zone';
    state.drawingPoints = [];
    const canvas = document.getElementById('overlayCanvas');
    canvas.classList.add('interactive');
    document.getElementById('drawInstructionBar').style.display = 'flex';
    document.getElementById('drawInstructionText').textContent = 'Click to place polygon vertices. Click near start point or double-click to finish.';
}

function startDrawingTripwire() {
    state.drawingMode = 'tripwire';
    state.drawingPoints = [];
    const canvas = document.getElementById('overlayCanvas');
    canvas.classList.add('interactive');
    document.getElementById('drawInstructionBar').style.display = 'flex';
    document.getElementById('drawInstructionText').textContent = 'Click Start Point, then click End Point to place counting tripwire.';
}

function cancelCurrentDrawing() {
    state.drawingMode = null;
    state.drawingPoints = [];
    const canvas = document.getElementById('overlayCanvas');
    canvas.classList.remove('interactive');
    document.getElementById('drawInstructionBar').style.display = 'none';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function finishCurrentDrawing() {
    const mode = state.drawingMode;
    const pts = [...state.drawingPoints];
    cancelCurrentDrawing();

    if (mode === 'zone' && pts.length >= 3) {
        state.editingZoneId = `zone_${Date.now()}`;
        state.tempPolygon = pts;
        document.getElementById('modalZoneName').value = `Zone ${state.config.zones.length + 1}`;
        document.getElementById('zoneModal').style.display = 'flex';
    } else if (mode === 'tripwire' && pts.length >= 2) {
        state.editingTripwireId = `tw_${Date.now()}`;
        state.tempTripwire = { start: pts[0], end: pts[1] };
        document.getElementById('modalTwName').value = `Line ${state.config.tripwires.length + 1}`;
        document.getElementById('tripwireModal').style.display = 'flex';
    }
}

function renderDrawing() {
    const canvas = document.getElementById('overlayCanvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!state.drawingMode || state.drawingPoints.length === 0) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.strokeStyle = state.drawingMode === 'zone' ? '#ef4444' : '#00f2fe';
    ctx.fillStyle = state.drawingMode === 'zone' ? 'rgba(239, 68, 68, 0.2)' : 'rgba(0, 242, 254, 0.2)';
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(state.drawingPoints[0].x * w, state.drawingPoints[0].y * h);
    for (let i = 1; i < state.drawingPoints.length; i++) {
        ctx.lineTo(state.drawingPoints[i].x * w, state.drawingPoints[i].y * h);
    }
    // Rubber-band line to mouse
    ctx.lineTo(state.mousePos.x * w, state.mousePos.y * h);

    if (state.drawingMode === 'zone' && state.drawingPoints.length >= 2) {
        ctx.closePath();
        ctx.fill();
    }
    ctx.stroke();

    // Draw vertex handles
    state.drawingPoints.forEach((p, idx) => {
        ctx.fillStyle = idx === 0 ? '#10b981' : '#00f2fe';
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.stroke();
    });
}

// ---------------------------------------------------------
// Zone & Tripwire Management Modals & REST
// ---------------------------------------------------------
function closeZoneModal() {
    document.getElementById('zoneModal').style.display = 'none';
}

function closeTripwireModal() {
    document.getElementById('tripwireModal').style.display = 'none';
}

async function saveCurrentZoneConfig() {
    const zone = {
        id: state.editingZoneId,
        name: document.getElementById('modalZoneName').value,
        polygon: state.tempPolygon,
        color: document.getElementById('modalZoneColor').value,
        severity: document.getElementById('modalZoneSeverity').value,
        dwell_threshold_sec: parseFloat(document.getElementById('modalZoneDwell').value),
        enabled: true
    };

    try {
        const res = await fetch('/api/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(zone)
        });
        const data = await res.json();
        state.config.zones = data.zones;
        renderZonesList();
        closeZoneModal();
    } catch (e) {
        console.error('[Save Zone Error]', e);
    }
}

async function saveCurrentTripwireConfig() {
    const tw = {
        id: state.editingTripwireId,
        name: document.getElementById('modalTwName').value,
        start: state.tempTripwire.start,
        end: state.tempTripwire.end,
        color: document.getElementById('modalTwColor').value,
        direction: document.getElementById('modalTwDirection').value,
        enabled: true
    };

    try {
        const res = await fetch('/api/tripwires', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(tw)
        });
        const data = await res.json();
        state.config.tripwires = data.tripwires;
        renderTripwiresList();
        closeTripwireModal();
    } catch (e) {
        console.error('[Save Tripwire Error]', e);
    }
}

async function deleteZone(zoneId) {
    try {
        const res = await fetch(`/api/zones/${zoneId}`, { method: 'DELETE' });
        const data = await res.json();
        state.config.zones = data.zones;
        renderZonesList();
    } catch (e) {
        console.error('[Delete Zone Error]', e);
    }
}

async function deleteTripwire(twId) {
    try {
        const res = await fetch(`/api/tripwires/${twId}`, { method: 'DELETE' });
        const data = await res.json();
        state.config.tripwires = data.tripwires;
        renderTripwiresList();
    } catch (e) {
        console.error('[Delete Tripwire Error]', e);
    }
}

function renderZonesList() {
    const container = document.getElementById('zonesList');
    if (!container || !state.config) return;

    if (state.config.zones.length === 0) {
        container.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">No security zones configured.</span>';
        return;
    }

    container.innerHTML = state.config.zones.map(z => `
        <div class="item-chip">
            <span style="color:${z.color}; font-weight:600;">${z.name} (${z.severity})</span>
            <div class="item-actions">
                <button class="btn-icon-del" onclick="deleteZone('${z.id}')" title="Delete Zone">&times;</button>
            </div>
        </div>
    `).join('');
}

function renderTripwiresList() {
    const container = document.getElementById('tripwiresList');
    if (!container || !state.config) return;

    if (state.config.tripwires.length === 0) {
        container.innerHTML = '<span class="text-muted" style="font-size:0.75rem;">No tripwires configured.</span>';
        return;
    }

    container.innerHTML = state.config.tripwires.map(tw => `
        <div class="item-chip line-chip">
            <span style="color:${tw.color}; font-weight:600;">${tw.name} (${tw.direction})</span>
            <div class="item-actions">
                <button class="btn-icon-del" onclick="deleteTripwire('${tw.id}')" title="Delete Line">&times;</button>
            </div>
        </div>
    `).join('');
}

// ---------------------------------------------------------
// Instant Actions (Manual Snapshot, Record, Reset)
// ---------------------------------------------------------
async function triggerManualSnapshot() {
    try {
        const res = await fetch('/api/snapshot', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'success') {
            playEventAudio('INFO', 'SNAPSHOT_TAKEN');
        }
    } catch (e) {
        console.error('[Snapshot Error]', e);
    }
}

async function toggleManualRecording() {
    const isRec = state.telemetry && state.telemetry.is_recording;
    try {
        if (isRec) {
            await fetch('/api/record/stop', { method: 'POST' });
        } else {
            await fetch('/api/record/start', { method: 'POST' });
        }
    } catch (e) {
        console.error('[Record Toggle Error]', e);
    }
}

async function resetCounters() {
    try {
        await fetch('/api/counts/reset', { method: 'POST' });
    } catch (e) {
        console.error('[Reset Counters Error]', e);
    }
}

// ---------------------------------------------------------
// Real-Time Event Audit Log Feed
// ---------------------------------------------------------
async function loadEvents() {
    try {
        const res = await fetch('/api/events?limit=30');
        const data = await res.json();
        state.eventsList = data.events;
        renderEventsFeed();
    } catch (e) {
        console.error('[Load Events Error]', e);
    }
}

function filterEvents(sev) {
    state.currentSeverityFilter = sev;
    document.querySelectorAll('.filter-chip').forEach(chip => {
        chip.classList.toggle('active', chip.textContent === sev);
    });
    renderEventsFeed();
}

function renderEventsFeed() {
    const container = document.getElementById('eventsList');
    if (!container) return;

    const filtered = state.eventsList.filter(ev => {
        if (state.currentSeverityFilter === 'ALL') return true;
        return ev.severity === state.currentSeverityFilter;
    });

    if (filtered.length === 0) {
        container.innerHTML = '<span class="text-muted text-center" style="font-size:0.75rem; padding:1rem;">No events recorded.</span>';
        return;
    }

    container.innerHTML = filtered.map(ev => createEventCardHTML(ev)).join('');
}

function prependEventCard(ev) {
    const container = document.getElementById('eventsList');
    if (!container) return;

    if (state.currentSeverityFilter !== 'ALL' && ev.severity !== state.currentSeverityFilter) {
        return;
    }

    const card = document.createElement('div');
    card.innerHTML = createEventCardHTML(ev);
    if (container.firstChild) {
        container.insertBefore(card.firstElementChild, container.firstChild);
    } else {
        container.appendChild(card.firstElementChild);
    }
}

function createEventCardHTML(ev) {
    const timeStr = ev.formatted_time ? ev.formatted_time.split(' ')[1] : new Date(ev.timestamp * 1000).toLocaleTimeString();
    let mediaHTML = '';

    if (ev.snapshot_url) {
        mediaHTML = `
            <div class="event-media-preview" onclick="openMediaModal('image', '${ev.snapshot_url}', '${ev.event_type}')">
                <img src="${ev.snapshot_url}" alt="Snapshot" loading="lazy">
            </div>
        `;
    } else if (ev.recording_url) {
        mediaHTML = `
            <button class="cyber-btn-small mt-3" onclick="openMediaModal('video', '${ev.recording_url}', '${ev.event_type}')">
                Play Video Clip (.MP4)
            </button>
        `;
    }

    return `
        <div class="event-card ${ev.severity}">
            <div class="event-top">
                <span class="event-type-badge text-${ev.severity === 'CRITICAL' ? 'rose' : ev.severity === 'WARNING' ? 'amber' : 'cyan'}">
                    ${ev.event_type}
                </span>
                <span class="event-time">${timeStr}</span>
            </div>
            <div class="event-desc">${ev.details || ''}</div>
            ${mediaHTML}
        </div>
    `;
}

async function clearEventLogs() {
    try {
        await fetch('/api/events', { method: 'DELETE' });
        state.eventsList = [];
        renderEventsFeed();
    } catch (e) {
        console.error('[Clear Logs Error]', e);
    }
}

function exportCsv() {
    window.location.href = '/api/events/export/csv';
}

// ---------------------------------------------------------
// Chart.js Real-Time Analytics
// ---------------------------------------------------------
function initCharts() {
    // 1. Traffic Timeline Chart
    const ctxTraffic = document.getElementById('trafficTimelineChart');
    if (ctxTraffic) {
        state.charts.traffic = new Chart(ctxTraffic, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'People Inflow',
                        data: [],
                        borderColor: '#00f2fe',
                        backgroundColor: 'rgba(0, 242, 254, 0.1)',
                        fill: true,
                        tension: 0.35
                    },
                    {
                        label: 'Vehicle Inflow',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.35
                    },
                    {
                        label: 'Total Outflow',
                        data: [],
                        borderColor: '#f59e0b',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.35
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }, beginAtZero: true }
                }
            }
        });
    }

    // 2. Object Classification Doughnut
    const ctxClasses = document.getElementById('classesDoughnutChart');
    if (ctxClasses) {
        state.charts.classes = new Chart(ctxClasses, {
            type: 'doughnut',
            data: {
                labels: ['Pedestrians', 'Cars / Vehicles', 'General Motion'],
                datasets: [{
                    data: [12, 8, 3],
                    backgroundColor: ['#00f2fe', '#10b981', '#f59e0b'],
                    borderWidth: 1,
                    borderColor: '#050811'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } }
                }
            }
        });
    }

    // 3. Hourly Traffic Volume Bar
    const ctxHourly = document.getElementById('hourlyVolumeChart');
    if (ctxHourly) {
        state.charts.hourly = new Chart(ctxHourly, {
            type: 'bar',
            data: {
                labels: ['08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00'],
                datasets: [{
                    label: 'Traffic Volume',
                    data: [24, 45, 68, 92, 110, 85, 76, 95],
                    backgroundColor: 'rgba(0, 242, 254, 0.4)',
                    borderColor: '#00f2fe',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } }, beginAtZero: true }
                }
            }
        });
    }

    // 4. Zone Violations Bar Chart
    const ctxZones = document.getElementById('zoneViolationsChart');
    if (ctxZones) {
        state.charts.zones = new Chart(ctxZones, {
            type: 'bar',
            data: {
                labels: ['Restricted Courtyard', 'Perimeter Fence', 'Server Room', 'Loading Dock'],
                datasets: [{
                    label: 'Intrusion Breaches',
                    data: [4, 1, 0, 2],
                    backgroundColor: 'rgba(239, 68, 68, 0.5)',
                    borderColor: '#ef4444',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b', font: { size: 9 } }, beginAtZero: true }
                }
            }
        });
    }
}

let lastChartUpdate = 0;
function updateTrafficChartPoint(t) {
    const now = Date.now();
    if (now - lastChartUpdate < 2000) return;
    lastChartUpdate = now;

    if (state.charts.traffic) {
        const timeLabel = new Date().toLocaleTimeString().split(' ')[0];
        const chart = state.charts.traffic;

        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
            chart.data.datasets[1].data.shift();
            chart.data.datasets[2].data.shift();
        }

        chart.data.labels.push(timeLabel);
        chart.data.datasets[0].data.push(t.counts.people_in);
        chart.data.datasets[1].data.push(t.counts.vehicles_in);
        chart.data.datasets[2].data.push(t.counts.total_out);
        chart.update('none');
    }
}

function switchAnalyticsTab(tab) {
    state.activeTab = tab;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');

    if (tab === 'traffic') {
        document.getElementById('tabTraffic').style.display = 'block';
    } else if (tab === 'classes') {
        document.getElementById('tabClasses').style.display = 'block';
    } else if (tab === 'zones') {
        document.getElementById('tabZones').style.display = 'block';
        updateZoneStatsChart();
    } else if (tab === 'gallery') {
        document.getElementById('tabGallery').style.display = 'block';
        loadRecordingsGallery();
    }
    event.target.classList.add('active');
}

async function updateZoneStatsChart() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (state.charts.zones && data.event_stats && data.event_stats.zone_counts) {
            const zCounts = data.event_stats.zone_counts;
            state.charts.zones.data.labels = Object.keys(zCounts).length ? Object.keys(zCounts) : ['No Zones'];
            state.charts.zones.data.datasets[0].data = Object.values(zCounts).length ? Object.values(zCounts) : [0];
            state.charts.zones.update();
        }
    } catch (e) {
        console.error('[Update Zone Chart Error]', e);
    }
}

// ---------------------------------------------------------
// Media Gallery (Recordings & Snapshots Playback)
// ---------------------------------------------------------
async function loadRecordingsGallery() {
    document.getElementById('btnShowRecordings').classList.add('active');
    document.getElementById('btnShowSnapshots').classList.remove('active');
    const grid = document.getElementById('mediaGalleryGrid');
    grid.innerHTML = '<span class="text-muted">Loading recorded clips...</span>';

    try {
        const res = await fetch('/api/recordings');
        const data = await res.json();
        if (data.recordings.length === 0) {
            grid.innerHTML = '<span class="text-muted">No video recordings saved yet. Intrusion breaches will automatically appear here.</span>';
            return;
        }

        grid.innerHTML = data.recordings.map(rec => `
            <div class="media-card" onclick="openMediaModal('video', '${rec.url}', '${rec.filename}')">
                <div class="media-thumb" style="display:flex; align-items:center; justify-content:center; background:#0f172a; color:#ef4444;">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
                </div>
                <div class="media-meta">
                    <span style="color:#fff; font-weight:700;">${rec.filename}</span>
                    <span>${rec.created_at} | ${rec.size_mb} MB</span>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('[Load Recordings Error]', e);
    }
}

async function loadSnapshotsGallery() {
    document.getElementById('btnShowRecordings').classList.remove('active');
    document.getElementById('btnShowSnapshots').classList.add('active');
    const grid = document.getElementById('mediaGalleryGrid');
    grid.innerHTML = '<span class="text-muted">Loading snapshots...</span>';

    try {
        const res = await fetch('/api/snapshots');
        const data = await res.json();
        if (data.snapshots.length === 0) {
            grid.innerHTML = '<span class="text-muted">No snapshots captured yet. Click "SNAPSHOT" button above to capture one.</span>';
            return;
        }

        grid.innerHTML = data.snapshots.map(s => `
            <div class="media-card" onclick="openMediaModal('image', '${s.url}', '${s.filename}')">
                <img src="${s.url}" class="media-thumb" alt="Snapshot">
                <div class="media-meta">
                    <span style="color:#fff; font-weight:700;">${s.filename}</span>
                    <span>${s.created_at} | ${s.size_kb} KB</span>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('[Load Snapshots Error]', e);
    }
}

function openMediaModal(type, url, title) {
    const modal = document.getElementById('mediaModal');
    const modalTitle = document.getElementById('mediaModalTitle');
    const content = document.getElementById('mediaModalContent');

    modalTitle.textContent = title || 'Incident Media';
    if (type === 'video') {
        content.innerHTML = `
            <video src="${url}" controls autoplay style="width:100%; max-height:480px;"></video>
            <div class="mt-3"><a href="${url}" download class="cyber-btn-outline" style="display:inline-flex;">Download MP4</a></div>
        `;
    } else {
        content.innerHTML = `
            <img src="${url}" alt="Snapshot" style="width:100%; max-height:480px; object-fit:contain;">
            <div class="mt-3"><a href="${url}" download class="cyber-btn-outline" style="display:inline-flex;">Download High-Res Snapshot</a></div>
        `;
    }
    modal.style.display = 'flex';
}

function closeMediaModal() {
    const modal = document.getElementById('mediaModal');
    const content = document.getElementById('mediaModalContent');
    content.innerHTML = '';
    modal.style.display = 'none';
}

function initPeriodicUpdates() {
    // Periodically update chart statistics
    setInterval(() => {
        if (state.activeTab === 'zones') {
            updateZoneStatsChart();
        }
    }, 5000);
}
