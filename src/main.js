/**
 * Air Canvas - Logic Module
 */

// --- Constants & Config ---
const W = 800;
const H = 600;
const PINCH_ON = 0.05;
const PINCH_OFF = 0.12; // Increased for better latching
const HAND_LOSS_GRACE_FRAMES = 10; // Frames to wait before breaking stroke on hand loss
const MAX_HISTORY = 50;

// Kalman Filter constants
const Q = 0.01; // Process noise
const R = 0.1;  // Measurement noise

// --- DOM Elements ---
const videoEl = document.getElementById('inputVideo');
const baseCanvas = document.getElementById('baseCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const cursorDot = document.getElementById('cursorDot');
const statusPill = document.getElementById('statusPill');
const undoBtn = document.getElementById('undoBtn');
const redoBtn = document.getElementById('redoBtn');
const saveBtn = document.getElementById('saveBtn');
const clearBtn = document.getElementById('clearBtn');
const brushSizeInput = document.getElementById('brushSize');
const sizeVal = document.getElementById('sizeVal');
const pressureToggle = document.getElementById('pressureToggle');

// Debug elements
const dbgX = document.getElementById('dbgX');
const dbgY = document.getElementById('dbgY');
const dbgDist = document.getElementById('dbgDist');
const dbgConf = document.getElementById('dbgConf');
const dbgPres = document.getElementById('dbgPres');
const dbgUndo = document.getElementById('dbgUndo');

// --- Canvas Contexts ---
const bCtx = baseCanvas.getContext('2d');
const oCtx = outputCanvas.getContext('2d');

// --- State ---
let color = '#3b82f6';
let mode = 'draw';
let baseSize = 8;
let usePressure = true;

// Kalman smoothing state
let sx = 0, sy = 0, vx = 0, vy = 0;
let kx = 1, ky = 1;
let initialized = false;

// Interaction state
let pinching = false;
let pinchConfidence = 0;
let lastX = 0, lastY = 0;
let undoStack = [];
let redoStack = [];
let latestResults = null;
let handLossCounter = 0;

// --- Initialization ---
function init() {
  // Initial canvas state
  bCtx.fillStyle = '#ffffff'; // White background for light mode
  bCtx.fillRect(0, 0, W, H);
  
  bCtx.lineCap = 'round';
  bCtx.lineJoin = 'round';
  
  setupHands();
  setupCamera();
  bindEvents();
  rafLoop();
}

// --- Hand Tracking Setup ---
function setupHands() {
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.75,
    minTrackingConfidence: 0.75
  });

  hands.onResults((results) => {
    latestResults = results;
    if (!statusPill.classList.contains('ready')) {
      statusPill.textContent = 'Ready — pinch to draw';
      statusPill.className = 'status-pill ready';
    }
  });

  window.handsInstance = hands;
}

async function setupCamera() {
  const camera = new Camera(videoEl, {
    onFrame: async () => {
      if (window.handsInstance) {
        await window.handsInstance.send({ image: videoEl });
      }
    },
    width: 640,
    height: 480
  });

  try {
    await camera.start();
  } catch (err) {
    statusPill.textContent = '⚠ Camera denied';
    statusPill.style.background = 'rgba(239, 68, 68, 0.15)';
    statusPill.style.color = '#fca5a5';
  }
}

// --- Main Logic Loop ---
function rafLoop() {
  if (latestResults) {
    processResults(latestResults);
    latestResults = null;
  }
  requestAnimationFrame(rafLoop);
}

function processResults(results) {
  // Clear skeleton canvas
  oCtx.save();
  oCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  oCtx.translate(outputCanvas.width, 0);
  oCtx.scale(-1, 1);

  if (!results.multiHandLandmarks?.length) {
    handLossCounter++;
    if (handLossCounter > HAND_LOSS_GRACE_FRAMES) {
      handleNoHands();
    } else if (pinching) {
      // If we were pinching, try to interpolate or at least don't break the stroke yet
      // We keep the last position (sx, sy) as is
      updateCursor(sx, sy, baseSize, true);
    }
    oCtx.restore();
    return;
  }

  handLossCounter = 0; // Reset counter on hand detection

  const lm = results.multiHandLandmarks[0];
  
  // Draw skeleton
  drawConnectors(oCtx, lm, HAND_CONNECTIONS, { color: 'rgba(16, 185, 129, 0.6)', lineWidth: 2 });
  drawLandmarks(oCtx, lm, { color: '#ef4444', lineWidth: 1, radius: 2 });
  oCtx.restore();

  // Extract key landmarks
  const thumbTip = lm[4];
  const indexTip = lm[8];

  // Map to canvas coords (x is mirrored)
  const rawX = (1 - indexTip.x) * W;
  const rawY = indexTip.y * H;

  // Apply Kalman smoothing
  if (!initialized) {
    sx = rawX; sy = rawY; initialized = true;
  }
  const filteredX = kalmanStep(rawX, sx, vx, kx);
  const filteredY = kalmanStep(rawY, sy, vy, ky);
  sx = filteredX.s; vx = filteredX.v; kx = filteredX.k;
  sy = filteredY.s; vy = filteredY.v; ky = filteredY.k;

  // Pinch Detection
  const dist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, (thumbTip.z - indexTip.z) * 0.5);

  if (dist < PINCH_ON) {
    pinchConfidence = Math.min(1, pinchConfidence + 0.4); // Faster turn-on
  } else if (dist > PINCH_OFF) {
    pinchConfidence = Math.max(0, pinchConfidence - 0.15); // Slower turn-off
  }

  // Hysteresis latching: once pinching, stay pinching longer
  const threshold = pinching ? 0.3 : 0.7; 
  const isPinching = pinchConfidence > threshold;

  // Pressure mapping
  const pressure = usePressure 
    ? Math.max(0.4, Math.min(2.0, 1.2 - (dist / PINCH_OFF) + 0.5))
    : 1.0;
  const strokeWidth = baseSize * pressure;

  updateDebug(sx, sy, dist, pinchConfidence, pressure);
  updateCursor(sx, sy, strokeWidth, isPinching);

  // Drawing logic
  if (isPinching && !pinching) {
    // Start stroke
    saveState();
    lastX = sx; lastY = sy;
    drawPoint(sx, sy, strokeWidth);
  } else if (isPinching && pinching) {
    // Continue stroke
    drawSegment(lastX, lastY, sx, sy, strokeWidth);
    lastX = sx; lastY = sy;
  }

  pinching = isPinching;
}

function handleNoHands() {
  pinchConfidence = Math.max(0, pinchConfidence - 0.1);
  if (pinchConfidence === 0) pinching = false;
  cursorDot.style.opacity = '0';
}

// --- Utilities ---
function kalmanStep(measured, state, vel, k) {
  const predicted = state + vel;
  const kGain = k / (k + R);
  const updated = predicted + kGain * (measured - predicted);
  const newK = (1 - kGain) * k + Q;
  const newVel = updated - state;
  return { s: updated, v: newVel, k: newK };
}

function drawPoint(x, y, w) {
  if (mode === 'erase') {
    bCtx.save();
    bCtx.globalCompositeOperation = 'destination-out';
    bCtx.beginPath();
    bCtx.arc(x, y, w, 0, Math.PI * 2);
    bCtx.fill();
    bCtx.restore();
  } else {
    bCtx.beginPath();
    bCtx.arc(x, y, w / 2, 0, Math.PI * 2);
    bCtx.fillStyle = color;
    bCtx.fill();
  }
}

function drawSegment(x1, y1, x2, y2, w) {
  bCtx.save();
  if (mode === 'erase') {
    bCtx.globalCompositeOperation = 'destination-out';
    bCtx.lineWidth = w * 2.5;
  } else {
    bCtx.strokeStyle = color;
    bCtx.lineWidth = w;
  }
  bCtx.beginPath();
  bCtx.moveTo(x1, y1);
  bCtx.lineTo(x2, y2);
  bCtx.stroke();
  bCtx.restore();
}

function updateCursor(x, y, w, active) {
  const xPct = (x / W) * 100;
  const yPct = (y / H) * 100;
  
  cursorDot.style.left = `${xPct}%`;
  cursorDot.style.top = `${yPct}%`;
  cursorDot.style.opacity = '1';
  
  const size = w * 2 + 4;
  cursorDot.style.width = `${size}px`;
  cursorDot.style.height = `${size}px`;
  
  if (active) {
    cursorDot.style.background = mode === 'erase' ? 'rgba(255,255,255,0.2)' : color;
    cursorDot.style.border = '2px solid #fff';
    cursorDot.style.transform = 'translate(-50%, -50%) scale(1.1)';
  } else {
    cursorDot.style.background = 'transparent';
    cursorDot.style.border = `2px solid ${mode === 'erase' ? 'rgba(255,255,255,0.4)' : color}`;
    cursorDot.style.transform = 'translate(-50%, -50%) scale(1)';
  }
}

function updateDebug(x, y, dist, conf, pres) {
  dbgX.textContent = Math.round(x);
  dbgY.textContent = Math.round(y);
  dbgDist.textContent = dist.toFixed(3);
  dbgConf.textContent = conf.toFixed(2);
  dbgPres.textContent = pres.toFixed(2);
}

// --- State Management ---
function saveState() {
  const snapshot = bCtx.getImageData(0, 0, W, H);
  undoStack.push(snapshot);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  updateHistoryUI();
}

function updateHistoryUI() {
  undoBtn.disabled = undoStack.length === 0;
  redoBtn.disabled = redoStack.length === 0;
  dbgUndo.textContent = undoStack.length;
}

function undo() {
  if (!undoStack.length) return;
  const current = bCtx.getImageData(0, 0, W, H);
  redoStack.push(current);
  bCtx.putImageData(undoStack.pop(), 0, 0);
  updateHistoryUI();
}

function redo() {
  if (!redoStack.length) return;
  const snapshot = redoStack.pop();
  undoStack.push(bCtx.getImageData(0, 0, W, H));
  bCtx.putImageData(snapshot, 0, 0);
  updateHistoryUI();
}

// --- Event Binding ---
function bindEvents() {
  // Swatches
  document.querySelectorAll('.swatch').forEach(sw => {
    sw.addEventListener('click', () => {
      document.querySelectorAll('.swatch').forEach(s => s.classList.remove('active'));
      sw.classList.add('active');
      mode = sw.dataset.mode;
      if (sw.dataset.color !== 'erase') {
        color = sw.dataset.color;
      }
    });
  });

  // Brush settings
  brushSizeInput.addEventListener('input', e => {
    baseSize = parseInt(e.target.value);
    sizeVal.textContent = baseSize;
  });

  pressureToggle.addEventListener('change', e => {
    usePressure = e.target.checked;
  });

  // Actions
  undoBtn.addEventListener('click', undo);
  redoBtn.addEventListener('click', redo);
  
  clearBtn.addEventListener('click', () => {
    saveState();
    bCtx.fillStyle = '#ffffff';
    bCtx.fillRect(0, 0, W, H);
  });

  saveBtn.addEventListener('click', () => {
    const dataURL = baseCanvas.toDataURL('image/png', 1.0);
    const filename = `air-canvas-export-${new Date().toISOString().replace(/:/g, '-')}.png`;
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataURL;
    link.click();
  });

  // Shortcuts
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
      e.preventDefault(); undo();
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key === 'z' && e.shiftKey))) {
      e.preventDefault(); redo();
    }
    if (e.key === 'Delete') {
      saveState();
      bCtx.fillStyle = '#ffffff';
      bCtx.fillRect(0, 0, W, H);
    }
  });
}

// Start the app
init();
