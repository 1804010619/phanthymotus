/**
 * camera-stream.js — browser camera capture, JPEG encode, WebSocket send.
 *
 * The visual counterpart to mic-stream.js: a video device on whatever machine
 * has the dashboard open becomes a live image source for the robot. Frames go
 * out as JPEG and the server republishes them on `/remote_control/camera` as
 * `image/jpeg` — the format the on-robot camera drivers emit — so any vision
 * card consumes it with no special case.
 *
 *   import { toggleCameraStream, isCameraActive } from './camera-stream.js';
 *   await toggleCameraStream(wsUrl, onStateChange, { fps: 12 });
 */
let _stream = null;
let _ws = null;
let _timer = null;
let _active = false;
let _video = null;
let _canvas = null;
let _rvfcArmed = false;
let _sent = 0;
let _startedAt = 0;

export const DEFAULTS = { width: 640, height: 480, fps: 12, quality: 0.7 };

/** Frames per second clamped to something a browser can actually sustain while
 *  also JPEG-encoding each frame on the main thread. */
export function clampFps(fps) {
  // `null` is checked separately because `Number(null)` is 0, not NaN — so a
  // card whose fps has never been set would otherwise stream at 1 fps, which
  // looks like a stall and is not what "unset" means.
  if (fps === null || fps === undefined || fps === '') return DEFAULTS.fps;
  const value = Number(fps);
  if (!Number.isFinite(value)) return DEFAULTS.fps;
  return Math.min(30, Math.max(1, Math.round(value)));
}

/**
 * Size the capture canvas to the camera's real aspect ratio.
 *
 * Forcing 640x480 on a 16:9 webcam would letterbox or stretch, and a stretched
 * body breaks the pose card's angles — the geometry rules are invariant to
 * scale and rotation but not to anisotropic scaling, which is what a wrong
 * aspect ratio is.
 */
export function fitCapture(videoWidth, videoHeight, target = DEFAULTS.width) {
  if (!(videoWidth > 0 && videoHeight > 0)) {
    return { width: DEFAULTS.width, height: DEFAULTS.height };
  }
  const scale = target / Math.max(videoWidth, videoHeight);
  return {
    width: Math.max(2, Math.round(videoWidth * scale / 2) * 2),
    height: Math.max(2, Math.round(videoHeight * scale / 2) * 2),
  };
}

export function isCameraActive() {
  return _active;
}

/** Frames actually sent per second since the stream started.
 *
 * Reported because the configured rate and the achieved rate can differ by an
 * order of magnitude — a backgrounded tab clamps timers to 1 Hz — and a stream
 * running at a twelfth of its setting is otherwise indistinguishable from a
 * slow robot. */
export function achievedFps() {
  if (!_active || !_startedAt) return 0;
  const seconds = (performance.now() - _startedAt) / 1000;
  return seconds > 0.5 ? _sent / seconds : 0;
}

/** List the browser's video inputs, for the card's device picker. */
export async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter(d => d.kind === 'videoinput')
      .map(d => ({ deviceId: d.deviceId, label: d.label || 'Camera' }));
  } catch {
    return [];
  }
}

/**
 * Toggle the browser camera.
 * @param {string} wsUrl        e.g. wss://host/ws/camera
 * @param {(active:boolean)=>void} onStateChange
 * @param {{fps?:number, deviceId?:string, width?:number, quality?:number}} opts
 */
export async function toggleCameraStream(wsUrl, onStateChange, opts = {}) {
  if (_active) {
    _stopCamera();
    onStateChange(false);
    return;
  }

  const fps = clampFps(opts.fps ?? DEFAULTS.fps);
  const quality = opts.quality ?? DEFAULTS.quality;
  const target = opts.width ?? DEFAULTS.width;

  try {
    _stream = await navigator.mediaDevices.getUserMedia({
      video: opts.deviceId
        ? { deviceId: { exact: opts.deviceId } }
        : { facingMode: 'user' },
      audio: false,
    });

    _video = document.createElement('video');
    _video.srcObject = _stream;
    _video.muted = true;
    _video.playsInline = true;
    await _video.play();

    const size = fitCapture(_video.videoWidth, _video.videoHeight, target);
    _canvas = document.createElement('canvas');
    _canvas.width = size.width;
    _canvas.height = size.height;
    const ctx = _canvas.getContext('2d');

    _ws = new WebSocket(wsUrl);
    _ws.binaryType = 'arraybuffer';
    await new Promise((resolve, reject) => {
      _ws.onopen = resolve;
      _ws.onerror = () => reject(new Error('WebSocket connection failed'));
      setTimeout(() => reject(new Error('WS timeout')), 5000);
    });

    let sending = false;
    const interval = 1000 / fps;
    let lastSent = 0;

    const sendFrame = () => {
      if (!_ws || _ws.readyState !== WebSocket.OPEN || !_video) return;
      // Skip rather than queue. A frame that arrives late is worse than one
      // that never arrives: the action rules measure velocity between frames,
      // so a backlog delivered in a burst reads as motion that did not happen.
      if (sending || _ws.bufferedAmount > 1 << 20) return;
      const now = performance.now();
      if (now - lastSent < interval * 0.9) return;
      lastSent = now;
      sending = true;
      _sent++;
      ctx.drawImage(_video, 0, 0, _canvas.width, _canvas.height);
      _canvas.toBlob(async (blob) => {
        try {
          if (blob && _ws && _ws.readyState === WebSocket.OPEN) {
            _ws.send(await blob.arrayBuffer());
          }
        } finally {
          sending = false;
        }
      }, 'image/jpeg', quality);
    };

    // Driven by the video's own decoded frames where the browser offers it.
    //
    // setInterval alone is not enough: browsers clamp timers in a hidden or
    // background tab to **once per second**, so the stream silently collapses
    // to 1 fps the moment the dashboard is not the foreground tab. Measured on
    // a live session: frames arriving 1003 ms apart, p95 1052 — not jitter, a
    // 1 Hz timer. At that rate tracks expire between frames and the action
    // model never accumulates the seconds of history it needs, so the activity
    // channel can never fire.
    //
    // requestVideoFrameCallback fires per decoded frame and is throttled far
    // less aggressively. The interval stays as a fallback for browsers without
    // it, and `sendFrame` rate-limits either way.
    if (typeof _video.requestVideoFrameCallback === 'function') {
      const onFrame = () => {
        if (!_active && _rvfcArmed) return;
        sendFrame();
        if (_video) _video.requestVideoFrameCallback(onFrame);
      };
      _rvfcArmed = true;
      _video.requestVideoFrameCallback(onFrame);
    } else {
      _timer = setInterval(sendFrame, interval);
    }

    _active = true;
    _sent = 0;
    _startedAt = performance.now();
    onStateChange(true);
  } catch (err) {
    _stopCamera();
    onStateChange(false);
    throw err;
  }
}

function _stopCamera() {
  if (_timer) { clearInterval(_timer); _timer = null; }
  if (_ws) { try { _ws.close(); } catch { /* already closed */ } _ws = null; }
  if (_stream) { _stream.getTracks().forEach(t => t.stop()); _stream = null; }
  if (_video) { _video.srcObject = null; _video = null; }
  _canvas = null;
  _rvfcArmed = false;
  _active = false;
}
