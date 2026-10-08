/**
 * camera-stream.js — 浏览器摄像头采集 + JPEG 编码 + WebSocket 发送
 *
 * The visual counterpart to mic-stream.js, and built the same way for the same
 * reason: a card that needs a live sensor cannot be tested from a file.
 *
 * `remote_image` uploads one picture, which is enough to exercise a detector and
 * useless for anything temporal — waving is motion that comes back, a fall is a
 * transition, and neither exists in a single frame. A machine with no camera
 * (every Orin test rig) therefore had no way to test those at all.
 *
 * Frames go out as JPEG over a WebSocket and the server republishes them on
 * `/remote_control/camera` as `image/jpeg`, which is exactly what the vision
 * cards already take as input — so a browser webcam wires into vop, pose, face
 * or ocr with no special case anywhere.
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

/** List the browser's video inputs, for the card's device picker. */
export async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter(d => d.kind === 'videoinput')
      .map(d => ({ deviceId: d.deviceId, label: d.label || '摄像头' }));
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
      _ws.onerror = () => reject(new Error('WebSocket 连接失败'));
      setTimeout(() => reject(new Error('WS timeout')), 5000);
    });

    let sending = false;
    _timer = setInterval(() => {
      if (!_ws || _ws.readyState !== WebSocket.OPEN || !_video) return;
      // Skip rather than queue. A frame that arrives late is worse than a frame
      // that never arrives: the action rules measure velocity between frames, so
      // a backlog delivered in a burst reads as motion that did not happen.
      if (sending || _ws.bufferedAmount > 1 << 20) return;
      sending = true;
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
    }, 1000 / fps);

    _active = true;
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
  _active = false;
}
