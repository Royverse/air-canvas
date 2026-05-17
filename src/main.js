/**
 * Air Canvas - Logic Module
 * ─────────────────────────────────────────────────────────────
 * Gesture Intent Engine v2
 *
 * Core insight: the user has TWO distinct hand-movement intents
 * that share the same physical space:
 *
 *   DRAW    — pinch + controlled movement → paint a stroke
 *   RELOCATE — open hand + fast sweep     → move to new spot
 *
 * The old code used a simple pinchConfidence ramp with slow
 * hysteresis, which caused three problems:
 *
 *   1. TRAILING TAILS — when releasing a pinch, the confidence
 *      took ~5 frames to drop below threshold. During those
 *      frames the user was already moving, so an unwanted
 *      "tail" was drawn at the end of every stroke.
 *
 *   2. CONNECTING LINES — if the user's fingers drifted close
 *      during a fast relocation, a spurious pinch was detected
 *      and a line was drawn connecting the old and new spots.
 *
 *   3. LAGGY STROKE START — the Kalman filter was still converging
 *      from the old position when the new pinch started, pulling
 *      the first few pixels of the new stroke toward the old one.
 *
 * The v2 engine fixes all three with:
 *
 *   A) Velocity-aware pinch gating — fast hand movement
 *      aggressively suppresses pinch confidence, so the system
 *      never thinks you're drawing while you're relocating.
 *
 *   B) Stroke-start stabilisation buffer — when a new pinch is
 *      detected, the engine waits a few frames for the position
 *      to settle before committing any ink. False-positive
 *      micro-pinches during relocation are filtered out because
 *      they don't sustain long enough.
 *
 *   C) Kalman reset on spatial jumps — if the new stroke start
 *      is far from the last draw position, the Kalman filter is
 *      hard-reset to the raw measurement, eliminating the
 *      "pulled toward old position" lag.
 * ─────────────────────────────────────────────────────────────
 */

// ═══════════════════════════════════════════════════════════════
//  CONSTANTS & CONFIG
// ═══════════════════════════════════════════════════════════════

const CANVAS_BG = '#F9F8F6'; // Neumorphic surface
const W = 800;
const H = 600;
const MAX_HISTORY = 50;

// ── Pinch detection thresholds ──
const PINCH_ON  = 0.05;  // finger distance to trigger pinch
const PINCH_OFF = 0.12;  // finger distance to release pinch

// ── Pinch confidence ramp rates ──
const CONF_RISE  =  0.35;  // per-frame increase when fingers close
const CONF_DECAY = -0.18;  // per-frame decrease when fingers apart (idle)

// ── Velocity-aware pinch suppression ──
//    When the hand is moving fast, we assume RELOCATE intent and
//    crush pinch confidence much harder, preventing the trailing
//    tail and the mid-flight false-positive pinch.
const VELOCITY_EMA_ALPHA     = 0.35;  // exponential moving average smoothing
const VELOCITY_SUPPRESS_LOW  = 8;     // px/frame — start suppressing above this
const VELOCITY_SUPPRESS_HIGH = 25;    // px/frame — maximum suppression above this
const CONF_DECAY_FAST        = -0.55; // per-frame decay when hand is moving fast

// ── Stroke-start stabilisation ──
//    When a new pinch is detected, we buffer for STAB_FRAMES
//    before committing ink.  If the pinch is released during
//    the buffer, nothing is drawn (micro-pinch rejection).
const STAB_FRAMES = 3;

// ── Jump detection ──
//    If the distance from the last draw position to the new
//    pinch position exceeds JUMP_PX, we hard-reset the Kalman
//    filter so the stroke starts exactly where the hand is.
const JUMP_PX = 40;

// ── Hand loss ──
const HAND_LOSS_GRACE_FRAMES = 8;

// ── Kalman filter ──
const Q = 0.01; // process noise
const R = 0.1;  // measurement noise

// ═══════════════════════════════════════════════════════════════
//  DOM ELEMENTS
// ═══════════════════════════════════════════════════════════════

const videoEl        = document.getElementById('inputVideo');
const baseCanvas     = document.getElementById('baseCanvas');
const outputCanvas   = document.getElementById('outputCanvas');
const cursorDot      = document.getElementById('cursorDot');
const statusPill     = document.getElementById('statusPill');
const undoBtn        = document.getElementById('undoBtn');
const redoBtn        = document.getElementById('redoBtn');
const saveBtn        = document.getElementById('saveBtn');
const clearBtn       = document.getElementById('clearBtn');
const brushSizeInput = document.getElementById('brushSize');
const sizeVal        = document.getElementById('sizeVal');
const pressureToggle = document.getElementById('pressureToggle');

// Debug readouts
const dbgX    = document.getElementById('dbgX');
const dbgY    = document.getElementById('dbgY');
const dbgDist = document.getElementById('dbgDist');
const dbgConf = document.getElementById('dbgConf');
const dbgPres = document.getElementById('dbgPres');
const dbgUndo = document.getElementById('dbgUndo');

// ═══════════════════════════════════════════════════════════════
//  CANVAS CONTEXTS
// ═══════════════════════════════════════════════════════════════

const bCtx = baseCanvas.getContext('2d');
const oCtx = outputCanvas.getContext('2d');

// ═══════════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════════

// Drawing mode
let color       = '#3b82f6';
let mode        = 'draw';
let baseSize    = 8;
let usePressure = true;

// Kalman smoothing (position + velocity + gain, per axis)
let sx = 0, sy = 0;   // smoothed position
let vx = 0, vy = 0;   // estimated velocity
let kx = 1, ky = 1;   // Kalman gain
let initialized = false;

// Hand velocity (exponential moving average of frame-to-frame
// displacement of the smoothed position)
let prevSx = 0, prevSy = 0;   // previous frame's smoothed pos
let handSpeed = 0;             // EMA of displacement magnitude

// Pinch state machine
let pinching        = false;   // true = we are actively drawing
let pinchConfidence = 0;       // [0, 1] ramp

// Stroke start stabilisation
let isStabilising   = false;   // true during the stabilisation window
let stabFrames      = 0;       // frames elapsed in current stab window
let stabSaved       = false;   // whether we already saved undo state for this stroke

// Drawing position
let lastX = 0, lastY = 0;

// History
let undoStack = [];
let redoStack = [];

// Frame pipeline
let latestResults    = null;
let handLossCounter  = 0;

// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════

function init() {
  bCtx.fillStyle = CANVAS_BG;
  bCtx.fillRect(0, 0, W, H);
  bCtx.lineCap  = 'round';
  bCtx.lineJoin = 'round';

  setupHands();
  setupCamera();
  bindEvents();
  rafLoop();
}

// ═══════════════════════════════════════════════════════════════
//  HAND TRACKING
// ═══════════════════════════════════════════════════════════════

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
      statusPill.className  = 'status-pill ready';
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
    statusPill.textContent   = '⚠ Camera denied';
    statusPill.style.background = 'rgba(239, 68, 68, 0.15)';
    statusPill.style.color      = '#fca5a5';
  }
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════

function rafLoop() {
  if (latestResults) {
    processResults(latestResults);
    latestResults = null;
  }
  requestAnimationFrame(rafLoop);
}

// ═══════════════════════════════════════════════════════════════
//  PROCESS RESULTS  — the heart of the intent engine
// ═══════════════════════════════════════════════════════════════

function processResults(results) {
  // ── Draw skeleton overlay ──
  oCtx.save();
  oCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  oCtx.translate(outputCanvas.width, 0);
  oCtx.scale(-1, 1);

  if (!results.multiHandLandmarks?.length) {
    handLossCounter++;
    if (handLossCounter > HAND_LOSS_GRACE_FRAMES) {
      handleNoHands();
    } else if (pinching) {
      updateCursor(sx, sy, baseSize, true);
    }
    oCtx.restore();
    return;
  }

  handLossCounter = 0;

  const lm = results.multiHandLandmarks[0];
  drawConnectors(oCtx, lm, HAND_CONNECTIONS, { color: 'rgba(16, 185, 129, 0.6)', lineWidth: 2 });
  drawLandmarks(oCtx, lm, { color: '#ef4444', lineWidth: 1, radius: 2 });
  oCtx.restore();

  // ── Extract landmarks ──
  const thumbTip = lm[4];
  const indexTip = lm[8];

  const rawX = (1 - indexTip.x) * W;
  const rawY = indexTip.y * H;

  // ── Kalman smoothing ──
  if (!initialized) {
    sx = rawX; sy = rawY;
    prevSx = rawX; prevSy = rawY;
    initialized = true;
  }

  const fX = kalmanStep(rawX, sx, vx, kx);
  const fY = kalmanStep(rawY, sy, vy, ky);
  sx = fX.s;  vx = fX.v;  kx = fX.k;
  sy = fY.s;  vy = fY.v;  ky = fY.k;

  // ────────────────────────────────────────────────────────────
  //  A) HAND VELOCITY (EMA of displacement per frame)
  //
  //     This is the single most important signal for telling
  //     DRAW from RELOCATE.  A drawing hand moves slowly and
  //     deliberately.  A relocating hand sweeps fast.
  // ────────────────────────────────────────────────────────────
  const dx = sx - prevSx;
  const dy = sy - prevSy;
  const instantSpeed = Math.sqrt(dx * dx + dy * dy);
  handSpeed = VELOCITY_EMA_ALPHA * instantSpeed
            + (1 - VELOCITY_EMA_ALPHA) * handSpeed;
  prevSx = sx;
  prevSy = sy;

  // ────────────────────────────────────────────────────────────
  //  PINCH CONFIDENCE — velocity-modulated
  // ────────────────────────────────────────────────────────────
  const dist = Math.hypot(
    thumbTip.x - indexTip.x,
    thumbTip.y - indexTip.y,
    (thumbTip.z - indexTip.z) * 0.5
  );

  if (dist < PINCH_ON) {
    // ── Fingers are close — ramp up, BUT only if not relocating ──
    //    When moving fast, even close fingers are probably just
    //    a natural hand shape during the sweep.
    const suppressFactor = velocitySuppression(handSpeed);
    const effectiveRise  = CONF_RISE * (1 - suppressFactor);
    pinchConfidence = Math.min(1, pinchConfidence + effectiveRise);

  } else if (dist > PINCH_OFF) {
    // ── Fingers are apart — decay, faster when moving ──
    const suppressFactor = velocitySuppression(handSpeed);
    const effectiveDecay = lerp(CONF_DECAY, CONF_DECAY_FAST, suppressFactor);
    pinchConfidence = Math.max(0, pinchConfidence + effectiveDecay);
  }
  // (Between PINCH_ON and PINCH_OFF: hold steady — dead zone)

  // ── Hysteresis thresholds ──
  const threshold  = pinching ? 0.25 : 0.65;
  const isPinching = pinchConfidence > threshold;

  // ── Pressure mapping ──
  const pressure = usePressure
    ? Math.max(0.4, Math.min(2.0, 1.2 - (dist / PINCH_OFF) + 0.5))
    : 1.0;
  const strokeWidth = baseSize * pressure;

  updateDebug(sx, sy, dist, pinchConfidence, pressure);
  updateCursor(sx, sy, strokeWidth, isPinching);

  // ────────────────────────────────────────────────────────────
  //  DRAWING STATE MACHINE
  //
  //  States:
  //    HOVER        — !isPinching, !pinching
  //    STAB_ENTER   — isPinching, !pinching (just started pinch)
  //    STABILISING  — isPinching, isStabilising (waiting for settle)
  //    DRAWING      — isPinching, pinching, !isStabilising
  //    STAB_ABORT   — !isPinching during stabilisation (micro-pinch)
  // ────────────────────────────────────────────────────────────

  if (isPinching && !pinching) {
    // ╔═══════════════════════════════════════════════════════╗
    // ║  STAB_ENTER — potential new stroke                   ║
    // ║                                                      ║
    // ║  Don't draw anything yet.  Enter the stabilisation   ║
    // ║  buffer so we can reject micro-pinches and let the   ║
    // ║  Kalman filter settle at the new position.           ║
    // ╚═══════════════════════════════════════════════════════╝

    isStabilising = true;
    stabFrames    = 0;
    stabSaved     = false;

    // Jump detection → Kalman reset
    const jumpDist = Math.hypot(sx - lastX, sy - lastY);
    if (jumpDist > JUMP_PX || !lastX) {
      // Hard-reset Kalman to raw measurement so the new stroke
      // starts exactly where the hand is, not where it was.
      sx = rawX;  sy = rawY;
      vx = 0;     vy = 0;
      kx = 1;     ky = 1;
      prevSx = sx;  prevSy = sy;
      handSpeed = 0;
    }

  } else if (isPinching && pinching && isStabilising) {
    // ╔═══════════════════════════════════════════════════════╗
    // ║  STABILISING — pinch is holding, count frames        ║
    // ╚═══════════════════════════════════════════════════════╝

    stabFrames++;

    // Track hand position during stabilisation (don't draw)
    lastX = sx;
    lastY = sy;

    if (stabFrames >= STAB_FRAMES) {
      // Stabilisation complete — commit to drawing
      isStabilising = false;

      if (!stabSaved) {
        saveState();
        stabSaved = true;
      }

      // Draw the initial point at the settled position
      drawPoint(sx, sy, strokeWidth);
    }

  } else if (isPinching && pinching && !isStabilising) {
    // ╔═══════════════════════════════════════════════════════╗
    // ║  DRAWING — steady pinch, paint segments              ║
    // ║                                                      ║
    // ║  Additional safety: if hand velocity spikes above    ║
    // ║  the suppression threshold while we're drawing, it   ║
    // ║  means the user started relocating but the pinch     ║
    // ║  hasn't released yet.  Suppress drawing.             ║
    // ╚═══════════════════════════════════════════════════════╝

    if (handSpeed < VELOCITY_SUPPRESS_HIGH) {
      drawSegment(lastX, lastY, sx, sy, strokeWidth);
    }
    // (When velocity is too high, we skip drawing this frame
    //  but don't break the stroke — the confidence decay will
    //  handle transitioning out of DRAWING state naturally.)

    lastX = sx;
    lastY = sy;

  } else if (!isPinching) {
    // ╔═══════════════════════════════════════════════════════╗
    // ║  HOVER / STAB_ABORT                                  ║
    // ╚═══════════════════════════════════════════════════════╝

    if (isStabilising) {
      // Pinch released during stabilisation → micro-pinch, discard
      isStabilising = false;
      stabFrames    = 0;
    }
  }

  pinching = isPinching;
}

// ═══════════════════════════════════════════════════════════════
//  VELOCITY SUPPRESSION
//
//  Returns a factor in [0, 1] indicating how much the current
//  hand speed should suppress pinch confidence.
//
//    0 = hand is slow/still → no suppression (drawing)
//    1 = hand is fast       → full suppression (relocating)
//
//  Uses a smooth ramp between LOW and HIGH thresholds.
// ═══════════════════════════════════════════════════════════════

function velocitySuppression(speed) {
  if (speed <= VELOCITY_SUPPRESS_LOW)  return 0;
  if (speed >= VELOCITY_SUPPRESS_HIGH) return 1;
  // Smooth hermite interpolation for a natural transition
  const t = (speed - VELOCITY_SUPPRESS_LOW)
          / (VELOCITY_SUPPRESS_HIGH - VELOCITY_SUPPRESS_LOW);
  return t * t * (3 - 2 * t); // smoothstep
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// ═══════════════════════════════════════════════════════════════
//  NO HANDS
// ═══════════════════════════════════════════════════════════════

function handleNoHands() {
  pinchConfidence = Math.max(0, pinchConfidence - 0.15);
  if (pinchConfidence === 0) {
    pinching      = false;
    isStabilising = false;
  }
  cursorDot.style.opacity = '0';
}

// ═══════════════════════════════════════════════════════════════
//  KALMAN FILTER
// ═══════════════════════════════════════════════════════════════

function kalmanStep(measured, state, vel, k) {
  const predicted = state + vel;
  const kGain     = k / (k + R);
  const updated   = predicted + kGain * (measured - predicted);
  const newK      = (1 - kGain) * k + Q;
  const newVel    = updated - state;
  return { s: updated, v: newVel, k: newK };
}

// ═══════════════════════════════════════════════════════════════
//  DRAWING PRIMITIVES
// ═══════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════
//  CURSOR
// ═══════════════════════════════════════════════════════════════

function updateCursor(x, y, w, active) {
  const xPct = (x / W) * 100;
  const yPct = (y / H) * 100;

  cursorDot.style.left    = `${xPct}%`;
  cursorDot.style.top     = `${yPct}%`;
  cursorDot.style.opacity = '1';

  const size = w * 2 + 4;
  cursorDot.style.width  = `${size}px`;
  cursorDot.style.height = `${size}px`;

  if (active) {
    cursorDot.style.background = mode === 'erase' ? 'rgba(255,255,255,0.2)' : color;
    cursorDot.style.border     = '2px solid #fff';
    cursorDot.style.transform  = 'translate(-50%, -50%) scale(1.1)';
  } else {
    cursorDot.style.background = 'transparent';
    cursorDot.style.border     = `2px solid ${mode === 'erase' ? 'rgba(255,255,255,0.4)' : color}`;
    cursorDot.style.transform  = 'translate(-50%, -50%) scale(1)';
  }
}

// ═══════════════════════════════════════════════════════════════
//  DEBUG
// ═══════════════════════════════════════════════════════════════

function updateDebug(x, y, dist, conf, pres) {
  dbgX.textContent    = Math.round(x);
  dbgY.textContent    = Math.round(y);
  dbgDist.textContent = dist.toFixed(3);
  dbgConf.textContent = conf.toFixed(2);
  dbgPres.textContent = pres.toFixed(2);
}

// ═══════════════════════════════════════════════════════════════
//  UNDO / REDO
// ═══════════════════════════════════════════════════════════════

function saveState() {
  const snapshot = bCtx.getImageData(0, 0, W, H);
  undoStack.push(snapshot);
  if (undoStack.length > MAX_HISTORY) undoStack.shift();
  redoStack = [];
  updateHistoryUI();
}

function updateHistoryUI() {
  undoBtn.disabled    = undoStack.length === 0;
  redoBtn.disabled    = redoStack.length === 0;
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

// ═══════════════════════════════════════════════════════════════
//  EVENT BINDING
// ═══════════════════════════════════════════════════════════════

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
    bCtx.fillStyle = CANVAS_BG;
    bCtx.fillRect(0, 0, W, H);
  });

  saveBtn.addEventListener('click', () => {
    const dataURL  = baseCanvas.toDataURL('image/png', 1.0);
    const filename = `air-canvas-export-${new Date().toISOString().replace(/:/g, '-')}.png`;
    const link     = document.createElement('a');
    link.download  = filename;
    link.href      = dataURL;
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
      bCtx.fillStyle = CANVAS_BG;
      bCtx.fillRect(0, 0, W, H);
    }
  });
}

// ═══════════════════════════════════════════════════════════════
//  START
// ═══════════════════════════════════════════════════════════════

init();
