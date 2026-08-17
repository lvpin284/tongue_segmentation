"""Real-time ultrasound tongue segmentation.

Live inference pipeline built on top of the TongueSegSAM backend. It grabs frames
from a live source (camera / capture card / RTSP-HTTP stream / video file), runs
per-frame segmentation, and renders a mask overlay + smoothed centerline with an
on-screen FPS/latency HUD.

Design highlights (see the module docstring blocks below):
  * A dedicated grabber thread always keeps only the *latest* frame so inference
    never processes stale frames -> lowest possible end-to-end latency.
  * FP16 autocast + cudnn.benchmark + inference_mode for max GPU throughput.
  * A reusable 4-channel "prototype prior" (same as run_vis.py) instead of zeros,
    which gives noticeably better masks than test_video.py.
  * Temporal EMA smoothing on the probability map to remove per-frame flicker.

Examples
--------
    # Webcam / capture card (device index 0), show a live window
    python realtime_seg.py --source 0

    # RTSP ultrasound stream, no GUI (headless server), record annotated output
    python realtime_seg.py --source rtsp://... --no-display --record out.mp4

    # Replay a file as if it were a live stream
    python realtime_seg.py --source ../testvedio/muler_vedio/vedio2/x.mp4
"""

import os
import sys
import time
import csv
import math
import queue
import socket
import argparse
import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Network protocol constants (must match NetworkServerExample.py /
# NetworkImageUploader.cs). The C# client sends raw pixel bytes per frame
# terminated by a fixed separator; no image encoding is used.
# --------------------------------------------------------------------------- #
NET_FRAME_SEP = b"--FRAME_END--"
NET_IMG_W, NET_IMG_H, NET_IMG_CH = 512, 512, 3

from models.model_dict import get_model  # noqa: E402
from utils.config import get_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Model helpers
# --------------------------------------------------------------------------- #
def load_checkpoint_state_dict(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def adapt_state_dict_to_model(state, model):
    """Reconcile input-channel count mismatches between checkpoint and model.

    Older tongue checkpoints were trained with a 4-channel prior, while the
    current prompt encoder expects 5 channels (extra temporal channel that the
    forward pass zero-pads). For conv weights whose only difference is the
    in-channel dim, zero-pad (or truncate) so loading succeeds and the extra
    channel stays zero -> identical behaviour to the original model.
    """
    model_state = model.state_dict()
    adapted = {}
    for k, v in state.items():
        if k in model_state:
            tv = model_state[k]
            if v.shape != tv.shape and v.dim() == tv.dim() == 4 and \
                    v.shape[0] == tv.shape[0] and v.shape[2:] == tv.shape[2:]:
                c_ckpt, c_model = v.shape[1], tv.shape[1]
                if c_ckpt < c_model:
                    pad = torch.zeros(
                        (v.shape[0], c_model - c_ckpt, *v.shape[2:]),
                        dtype=v.dtype, device=v.device)
                    v = torch.cat([v, pad], dim=1)
                    print(f"[model] padded '{k}' in-channels {c_ckpt}->{c_model}", flush=True)
                elif c_ckpt > c_model:
                    v = v[:, :c_model]
                    print(f"[model] truncated '{k}' in-channels {c_ckpt}->{c_model}", flush=True)
        adapted[k] = v
    return adapted


def resolve_checkpoint_path(explicit_checkpoint, checkpoint_dir):
    if explicit_checkpoint:
        p = os.path.abspath(explicit_checkpoint)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    model_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {model_dir}")
    candidates = [os.path.join(model_dir, n) for n in os.listdir(model_dir) if n.endswith(".pth")]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint (*.pth) found under: {model_dir}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def build_model(args, device):
    opt = get_config(args.task)
    opt.device = device
    model = get_model(args.modelname, args=args, opt=opt).to(device)
    ckpt = resolve_checkpoint_path(args.checkpoint, args.checkpoint_dir)
    state = load_checkpoint_state_dict(ckpt, device)
    state = adapt_state_dict_to_model(state, model)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[model] using segmentation checkpoint: {ckpt}", flush=True)
    return model


def build_prototype_prior(img_gray):
    """4-channel weak prior fed to the prompt encoder (raw / smooth / edge / Otsu).

    Identical formulation to run_vis.py; gives better masks than an all-zero prior.
    """
    img_f = img_gray.astype(np.float32) / 255.0
    ch1 = img_f
    ch2 = cv2.GaussianBlur(img_f, (0, 0), 2.0)
    sx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    ch3 = cv2.normalize(np.abs(sx) + np.abs(sy), None, 0.0, 1.0, cv2.NORM_MINMAX)
    _, th = cv2.threshold((img_f * 255).astype(np.uint8), 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ch4 = th.astype(np.float32) / 255.0
    return np.stack([ch1, ch2, ch3, ch4], axis=0)


# --------------------------------------------------------------------------- #
# Inference engine
# --------------------------------------------------------------------------- #
class SegEngine:
    """Wraps a single-frame forward pass with all real-time optimizations."""

    def __init__(self, model, device, input_size=256, use_fp16=True, ema=0.5, threshold=0.5):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.use_fp16 = use_fp16 and device.type == "cuda"
        self.ema = float(ema)
        self.threshold = float(threshold)
        self._prob_state = None  # EMA state at input_size resolution

        c = input_size // 2
        self.pt = (
            torch.tensor([[[c, c]]], dtype=torch.float32, device=device),
            torch.tensor([[1]], dtype=torch.int64, device=device),
        )

    def reset(self):
        self._prob_state = None

    @torch.inference_mode()
    def infer_prob(self, gray_resized):
        """gray_resized: uint8 HxW at input_size. Returns float prob map [0,1] at input_size."""
        img_tensor = (
            torch.from_numpy(gray_resized).to(self.device).float().div_(255.0)
            .unsqueeze(0).unsqueeze(0)
        )
        prior = build_prototype_prior(gray_resized)
        cls_sim = torch.from_numpy(prior).unsqueeze(0).to(self.device, dtype=torch.float32)

        autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16)
        with autocast:
            out = self.model(img_tensor, self.pt, bbox=None, cls_sim_avg_label_input=cls_sim)
            logits = out["masks"]  # [1,1,256,256]
            prob = torch.sigmoid(logits.float())[0, 0]

        prob_np = prob.detach().cpu().numpy()

        if self.ema > 0.0:
            if self._prob_state is None:
                self._prob_state = prob_np
            else:
                self._prob_state = self.ema * self._prob_state + (1.0 - self.ema) * prob_np
            prob_np = self._prob_state
        return prob_np

    def warmup(self, iters=5):
        dummy = np.zeros((self.input_size, self.input_size), dtype=np.uint8)
        for _ in range(iters):
            self.infer_prob(dummy)
        self.reset()
        if self.device.type == "cuda":
            torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Threaded frame grabber (keep-latest to minimize latency)
# --------------------------------------------------------------------------- #
class FrameGrabber(threading.Thread):
    def __init__(self, source, pace_fps=0.0, loop=False):
        super().__init__(daemon=True)
        self.source = source
        # pace_fps > 0 paces reads to a target fps (needed for file sources so a
        # video is played back in real time instead of decoded instantly).
        self.pace_fps = float(pace_fps)
        self.loop = bool(loop)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        # Small internal buffer helps drop stale frames on live cameras/streams.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stopped = threading.Event()

    def run(self):
        dt = 1.0 / self.pace_fps if self.pace_fps > 0 else 0.0
        next_t = time.time()
        while not self._stopped.is_set():
            ret, frame = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._stopped.set()
                break
            with self._lock:
                self._latest = frame
                self._seq += 1
            if dt > 0:
                next_t += dt
                sleep_left = next_t - time.time()
                if sleep_left > 0:
                    time.sleep(sleep_left)
                else:
                    next_t = time.time()

    def read_latest(self):
        with self._lock:
            if self._latest is None:
                return None, -1
            return self._latest, self._seq

    def stopped(self):
        return self._stopped.is_set()

    def stop(self):
        self._stopped.set()
        try:
            self.cap.release()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Threaded network frame grabber (TCP server, same protocol as
# NetworkServerExample.py / NetworkImageUploader.cs)
# --------------------------------------------------------------------------- #
class NetworkFrameGrabber(threading.Thread):
    """Acts as the passive TCP server that the C# uploader connects to.

    It receives raw pixel bytes terminated by ``NET_FRAME_SEP``, reshapes each
    complete frame to ``(H, W, C)`` and keeps only the *latest* one so inference
    never processes stale frames. Exposes the same ``read_latest`` / ``stop`` /
    ``stopped`` interface as :class:`FrameGrabber` so it drops into ``main``
    unchanged. On client disconnect it keeps listening for a new connection.
    """

    BUF_SIZE = 65536

    def __init__(self, host="0.0.0.0", port=9005,
                 img_w=NET_IMG_W, img_h=NET_IMG_H, img_ch=NET_IMG_CH,
                 order="rgb"):
        super().__init__(daemon=True)
        self.host = host
        self.port = int(port)
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.img_ch = int(img_ch)
        self.frame_bytes = self.img_w * self.img_h * self.img_ch
        self.order = order.lower()  # pixel byte order sent by the client
        # Interface parity with FrameGrabber.
        self.width = self.img_w
        self.height = self.img_h
        self.fps = 0.0  # unknown for a push stream; main() falls back to 25.0
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._stopped = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        self._sock.settimeout(1.0)  # so accept() can react to stop()
        print(f"[net] listening on {self.host}:{self.port}, "
              f"expecting {self.img_w}x{self.img_h}x{self.img_ch} "
              f"({self.frame_bytes} bytes/frame, {self.order})", flush=True)

    def _decode_frame(self, frame_data):
        arr = np.frombuffer(frame_data, dtype=np.uint8).reshape(
            self.img_h, self.img_w, self.img_ch)
        if self.img_ch == 3 and self.order == "rgb":
            # Downstream pipeline (cv2) works in BGR; convert once here.
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif self.img_ch == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return arr

    def run(self):
        while not self._stopped.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            print(f"[net] client connected: {addr}", flush=True)
            conn.settimeout(1.0)
            buffer = b""
            try:
                while not self._stopped.is_set():
                    try:
                        chunk = conn.recv(self.BUF_SIZE)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk
                    # Keep only the most recent complete frame in the buffer.
                    while NET_FRAME_SEP in buffer:
                        frame_data, buffer = buffer.split(NET_FRAME_SEP, 1)
                        if len(frame_data) != self.frame_bytes:
                            continue
                        frame = self._decode_frame(frame_data)
                        with self._lock:
                            self._latest = frame
                            self._seq += 1
            except OSError:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
                print("[net] client disconnected, waiting for a new connection...", flush=True)

    def read_latest(self):
        with self._lock:
            if self._latest is None:
                return None, -1
            return self._latest, self._seq

    def stopped(self):
        return self._stopped.is_set()

    def stop(self):
        self._stopped.set()
        try:
            self._sock.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Web live viewer (multi-panel MJPEG dashboard)
#   Row 1: raw video            | segmented tongue video
#   Row 2: M-mode (30 deg)      | grayscale       | frame-diff w/ arrows
#   Row 3: predicted tongue-base-collapse probability (ECG-style strip)
# --------------------------------------------------------------------------- #
WEB_CHANNELS = ("raw", "seg", "mmode", "gray", "flow", "prob")

_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>\u820c\u9762\u8d85\u58f0\u5b9e\u65f6\u5206\u6790</title>
<style>
  :root{--bg:#0c121c;--panel:#141d2c;--line:#26344a;--txt:#e8eef7;--muted:#8aa0bd;--accent:#22c55e;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,'PingFang SC','Microsoft YaHei',Arial,sans-serif}
  header{padding:12px 18px;background:#0f1826;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
  header h1{font-size:17px;font-weight:600;margin:0}
  header .tag{font-size:12px;color:var(--muted)}
  .grid{display:grid;gap:12px;padding:12px;max-width:1500px;margin:0 auto;
        grid-template-columns:repeat(6,1fr);
        grid-template-areas:
          "raw raw raw seg seg seg"
          "mmode mmode gray gray flow flow"
          "prob prob prob prob prob prob";}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
  .card h2{font-size:13px;font-weight:600;margin:0;padding:8px 12px;color:var(--txt);border-bottom:1px solid var(--line);background:#101a29}
  .card h2 small{color:var(--muted);font-weight:400;margin-left:6px}
  .card .body{padding:8px;display:flex;justify-content:center;align-items:center;background:#000}
  .card img{width:100%;height:auto;display:block;border-radius:4px}
  .a-raw{grid-area:raw}.a-seg{grid-area:seg}.a-mmode{grid-area:mmode}
  .a-gray{grid-area:gray}.a-flow{grid-area:flow}.a-prob{grid-area:prob}
  @media(max-width:900px){.grid{grid-template-columns:1fr;grid-template-areas:"raw" "seg" "mmode" "gray" "flow" "prob";}}
</style></head>
<body>
  <header>
    <h1>\u820c\u9762\u8d85\u58f0\u5b9e\u65f6\u5206\u6790\u9762\u677f</h1>
    <span class="tag">TongueSegSAM \u00b7 \u5b9e\u65f6\u5206\u5272 + \u8fd0\u52a8\u5206\u6790 + \u4e8b\u4ef6\u9884\u6d4b</span>
  </header>
  <div class="grid">
    <div class="card a-raw"><h2>\u539f\u59cb\u89c6\u9891<small>raw</small></h2><div class="body"><img src="/raw.mjpg"></div></div>
    <div class="card a-seg"><h2>\u5206\u5272\u820c\u9762\u89c6\u9891<small>segmentation</small></h2><div class="body"><img src="/seg.mjpg"></div></div>
    <div class="card a-mmode"><h2>M\u8d85\u56fe\u50cf<small>\u4ece\u53f3\u5f80\u5de6 30\u00b0</small></h2><div class="body"><img src="/mmode.mjpg"></div></div>
    <div class="card a-gray"><h2>\u7070\u5ea6\u65b9\u5757\u683c\u5b50 + \u7070\u5ea6\u66f2\u7ebf<small>blocks + mean curve</small></h2><div class="body"><img src="/gray.mjpg"></div></div>
    <div class="card a-flow"><h2>\u5e27\u95f4\u5dee\u5206<small>\u8fd0\u52a8\u7bad\u5934</small></h2><div class="body"><img src="/flow.mjpg"></div></div>
    <div class="card a-prob"><h2>\u9884\u6d4b\u820c\u540e\u5760\u4e8b\u4ef6\u6982\u7387<small>\u5b9e\u65f6\u6982\u7387\u66f2\u7ebf</small></h2><div class="body"><img src="/prob.mjpg"></div></div>
  </div>
</body></html>""".encode("utf-8")


class StreamState:
    """Thread-safe holder for the latest JPEG frame of each named channel."""

    def __init__(self, jpeg_quality=80):
        self._cond = threading.Condition()
        self._frames = {}  # name -> jpeg bytes
        self._seq = 0
        self._enc = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    def update(self, images):
        """images: dict name -> BGR ndarray (None entries are skipped)."""
        encoded = {}
        for name, img in images.items():
            if img is None:
                continue
            ok, buf = cv2.imencode(".jpg", img, self._enc)
            if ok:
                encoded[name] = buf.tobytes()
        if not encoded:
            return
        with self._cond:
            self._frames.update(encoded)
            self._seq += 1
            self._cond.notify_all()

    def wait_next(self, last_seq, name, timeout=1.0):
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            return self._seq, self._frames.get(name)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silence per-request logging
            pass

        def handle_one_request(self):
            # Browsers/curl closing an MJPEG connection raise reset/broken-pipe
            # errors while reading the next request line; swallow them quietly.
            try:
                super().handle_one_request()
            except (ConnectionResetError, BrokenPipeError):
                self.close_connection = True

        def _serve_index(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)

        def _serve_stream(self, name):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    last, data = state.wait_next(last, name)
                    if data is None:
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(data)).encode())
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._serve_index()
                return
            # /<name>.mjpg  ->  channel stream
            name = self.path.lstrip("/").split(".")[0].split("?")[0]
            if name in WEB_CHANNELS:
                self._serve_stream(name)
            else:
                self.send_error(404)

    return Handler


def start_web_server(state, host, port):
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_overlay(frame, mask, draw_centerline=True):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    overlay[mask == 1] = (0, 255, 0)
    final = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    centerline = None
    if draw_centerline:
        midpoints = []
        for x in range(w):
            ys = np.where(mask[:, x] == 1)[0]
            if len(ys) > 0:
                midpoints.append([x, int(np.mean(ys))])
        if len(midpoints) > 10:
            pts = np.array(midpoints, np.int32)
            xc, yc = pts[:, 0], pts[:, 1]
            z = np.polyfit(xc, yc, 3)
            p = np.poly1d(z)
            smooth = np.column_stack((xc, p(xc))).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(final, [smooth], False, (0, 0, 255), 3)
            xs100 = np.linspace(xc.min(), xc.max(), 100)
            centerline = np.column_stack((xs100, p(xs100)))
    return final, centerline


def draw_hud(img, fps, latency_ms, area_ratio):
    txt = f"FPS {fps:5.1f} | lat {latency_ms:5.1f} ms | tongue {area_ratio*100:4.1f}%"
    cv2.rectangle(img, (0, 0), (max(360, len(txt) * 10), 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# Derived web panels: M-mode, optical-flow arrows, probability strip
# --------------------------------------------------------------------------- #
class MModeBuilder:
    """Classic ultrasound M-mode: sample intensities along a fixed scan line
    every frame and stack the columns left->right to visualize motion over time.

    The scan line starts near the fan apex (top-center) and goes down toward the
    left at ``angle_deg`` from vertical ("from right to left, 30 degrees").
    """

    def __init__(self, height=260, width=380, angle_deg=30.0):
        self.height = int(height)
        self.width = int(width)
        self.angle = math.radians(angle_deg)
        self.buf = np.zeros((self.height, self.width), np.uint8)

    def line_geometry(self, w, h):
        x0 = int(w * 0.5)
        y0 = int(h * 0.06)
        length = int(h * 0.9)
        dx = -math.sin(self.angle)   # toward the left
        dy = math.cos(self.angle)    # downward
        t = np.linspace(0.0, length, self.height)
        xs = np.clip((x0 + dx * t).astype(np.int32), 0, w - 1)
        ys = np.clip((y0 + dy * t).astype(np.int32), 0, h - 1)
        return (x0, y0), xs, ys

    def push(self, gray):
        h, w = gray.shape
        _, xs, ys = self.line_geometry(w, h)
        col = gray[ys, xs].astype(np.uint8)
        self.buf = np.roll(self.buf, -1, axis=1)
        self.buf[:, -1] = col

    def image(self):
        vis = cv2.applyColorMap(self.buf, cv2.COLORMAP_BONE)
        cv2.line(vis, (self.width - 1, 0), (self.width - 1, self.height), (0, 255, 255), 1)
        cv2.putText(vis, "M-mode 30deg  (time ->)", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1, cv2.LINE_AA)
        return vis


class FlowArrows:
    """Farneback optical flow between consecutive frames, drawn as motion arrows.

    Returns the arrow overlay plus a scalar motion-energy (mean flow magnitude)
    reused by the event-probability estimator.
    """

    def __init__(self, small_w=320, step=20):
        self.small_w = int(small_w)
        self.step = int(step)
        self.prev = None

    def panel(self, gray_full):
        h, w = gray_full.shape
        sw = self.small_w
        sh = max(1, int(h * sw / w))
        small = cv2.resize(gray_full, (sw, sh))
        base = cv2.cvtColor((small * 0.55).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self.prev is None or self.prev.shape != small.shape:
            self.prev = small
            return base, 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = small
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion = float(mag.mean())
        step = self.step
        for y in range(step // 2, sh, step):
            for x in range(step // 2, sw, step):
                fx, fy = flow[y, x]
                m = (fx * fx + fy * fy) ** 0.5
                if m < 0.5:
                    continue
                ex, ey = int(x + fx * 3.0), int(y + fy * 3.0)
                g = int(min(255, 90 + m * 45))
                cv2.arrowedLine(base, (x, y), (ex, ey), (0, g, 255), 1, tipLength=0.4)
        return base, motion


class ProbStrip:
    """Scrolling probability curve rendered like an ECG strip."""

    def __init__(self, width=1200, height=220, maxlen=900, threshold=0.5):
        self.w = int(width)
        self.h = int(height)
        self.thr = float(threshold)
        self.vals = deque(maxlen=int(maxlen))

    def push(self, p):
        self.vals.append(float(np.clip(p, 0.0, 1.0)))

    def image(self):
        img = np.full((self.h, self.w, 3), (16, 22, 32), np.uint8)
        for gx in range(0, self.w, 40):
            cv2.line(img, (gx, 0), (gx, self.h), (30, 42, 58), 1)
        for gy in range(0, self.h, 30):
            cv2.line(img, (0, gy), (self.w, gy), (30, 42, 58), 1)
        ty = int(self.h - self.thr * (self.h - 1))
        cv2.line(img, (0, ty), (self.w, ty), (70, 90, 120), 1)
        cv2.putText(img, f"thr={self.thr:.2f}", (self.w - 90, ty - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 150, 190), 1, cv2.LINE_AA)
        vals = list(self.vals)
        if len(vals) >= 2:
            n = len(vals)
            xs = np.linspace(0, self.w - 1, n).astype(np.int32)
            ys = (self.h - 1 - np.array(vals) * (self.h - 1)).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, (0, 230, 120), 2, cv2.LINE_AA)
            last = vals[-1]
            col = (0, 80, 255) if last >= self.thr else (0, 230, 120)
            cv2.circle(img, (int(xs[-1]), int(ys[-1])), 4, col, -1)
            cv2.putText(img, f"P(collapse)={last:0.2f}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
            if last >= self.thr:
                cv2.putText(img, "EVENT", (self.w - 130, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 80, 255), 2, cv2.LINE_AA)
        return img


class EventProbEstimator:
    """Live proxy for tongue-base-collapse (glossoptosis) probability.

    Heuristic placeholder: collapse tends to coincide with the tongue sitting
    lower/posterior (centerline mean-y high within the recent range) together
    with reduced motion (airflow drop). Output is EMA-smoothed to [0,1]. Swap in
    the trained temporal transformer here for a model-based probability.
    """

    def __init__(self, ema=0.9):
        self.ema = float(ema)
        self.p = 0.0
        self.y_hist = deque(maxlen=120)

    def update(self, area_ratio, centerline, motion):
        cy = None
        if centerline is not None and len(centerline) > 0:
            cy = float(np.mean(centerline[:, 1]))
            self.y_hist.append(cy)
        drop = 0.0
        if cy is not None and len(self.y_hist) > 15:
            lo, hi = min(self.y_hist), max(self.y_hist)
            if hi - lo > 1e-6:
                drop = (cy - lo) / (hi - lo)
        low_motion = float(np.clip(1.0 - motion / 1.5, 0.0, 1.0))
        score = 0.6 * drop + 0.4 * low_motion
        raw = 1.0 / (1.0 + math.exp(-(score - 0.55) * 6.0))
        self.p = self.ema * self.p + (1.0 - self.ema) * raw
        return self.p


def make_gray_panel(gray, block_size=16, draw_grid=True):
    """Pixelate the grayscale frame into a grid of small square blocks.

    Each block is the mean intensity of that patch (INTER_AREA downsample), then
    nearest-neighbour upscaled so the panel looks like a mosaic of tiles.
    """
    h, w = gray.shape[:2]
    bs = max(2, int(block_size))
    nw = max(1, w // bs)
    nh = max(1, h // bs)
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(blocks, cv2.COLOR_GRAY2BGR)
    if draw_grid:
        # thin separators so each square reads as its own tile
        line = (48, 58, 78)
        for x in range(0, w, bs):
            cv2.line(vis, (x, 0), (x, h - 1), line, 1)
        for y in range(0, h, bs):
            cv2.line(vis, (0, y), (w - 1, y), line, 1)
    cv2.putText(vis, f"blocks {bs}px", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 220, 255), 1, cv2.LINE_AA)
    return vis


class GrayPanel:
    """Block-mosaic grayscale view with a scrolling mean-intensity curve below.

    The curve tracks the overall (whole-frame) mean gray level over time, so
    global brightness changes are visible at a glance under the mosaic.
    """

    def __init__(self, block_size=16, hist_len=300, strip_ratio=0.30):
        self.block_size = int(block_size)
        self.strip_ratio = float(strip_ratio)
        self.vals = deque(maxlen=int(hist_len))

    def panel(self, gray):
        self.vals.append(float(gray.mean()))
        mosaic = make_gray_panel(gray, block_size=self.block_size)
        h, w = gray.shape[:2]
        sh = max(48, int(h * self.strip_ratio))
        strip = np.full((sh, w, 3), (16, 22, 32), np.uint8)
        for gx in range(0, w, 40):
            cv2.line(strip, (gx, 0), (gx, sh), (30, 42, 58), 1)
        for gy in range(0, sh, 20):
            cv2.line(strip, (0, gy), (w, gy), (30, 42, 58), 1)
        vals = list(self.vals)
        if len(vals) >= 2:
            n = len(vals)
            xs = np.linspace(0, w - 1, n).astype(np.int32)
            ys = (sh - 1 - (np.array(vals) / 255.0) * (sh - 1)).astype(np.int32)
            pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
            cv2.polylines(strip, [pts], False, (255, 200, 80), 2, cv2.LINE_AA)
            cv2.circle(strip, (int(xs[-1]), int(ys[-1])), 3, (255, 230, 120), -1)
            cv2.putText(strip, f"mean gray {vals[-1]:5.1f}", (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 120), 1, cv2.LINE_AA)
        cv2.line(strip, (0, 0), (w - 1, 0), (60, 76, 100), 1)
        return np.vstack([mosaic, strip])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_source(s):
    return int(s) if s.isdigit() else s


def is_network_source(s):
    return isinstance(s, str) and s.lower().startswith("tcp://")


def build_grabber(args):
    """Return a grabber exposing read_latest/stop/stopped/width/height/fps."""
    if is_network_source(args.source):
        parsed = urlparse(args.source)
        host = parsed.hostname or "0.0.0.0"
        port = parsed.port or 9005
        return NetworkFrameGrabber(
            host=host, port=port,
            img_w=args.net_width, img_h=args.net_height, img_ch=args.net_channels,
            order=args.net_order,
        )
    src = parse_source(args.source)
    # For local video files, pace playback to the file fps so it *simulates* a
    # live stream (e.g. the P005 ultrasound clips) instead of decoding instantly.
    is_file = isinstance(src, str) and os.path.isfile(src)
    pace_fps = 0.0
    loop = False
    if is_file and not args.no_pace:
        probe = cv2.VideoCapture(src)
        file_fps = probe.get(cv2.CAP_PROP_FPS) or 0.0
        probe.release()
        if file_fps and file_fps > 1:
            pace_fps = file_fps * max(0.1, args.playback_speed)
        loop = args.loop
    return FrameGrabber(src, pace_fps=pace_fps, loop=loop)


def main():
    parser = argparse.ArgumentParser(description="Real-time ultrasound tongue segmentation")
    # Model / checkpoint
    parser.add_argument("--modelname", default="TongueSegSAM", type=str)
    parser.add_argument("-encoder_input_size", "--encoder-input-size", type=int, default=256)
    parser.add_argument("-low_image_size", "--low-image-size", type=int, default=128)
    parser.add_argument("--task", default="Cardiac_multi_plane_test")
    parser.add_argument("--vit_name", type=str, default="vit_b")
    parser.add_argument("--sam_ckpt", type=str, default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="../save/Tongue/")
    # Realtime I/O
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (e.g. 0), video file path, RTSP/HTTP stream URL, "
                             "or tcp://IP:PORT to receive the NetworkImageUploader raw stream "
                             "(e.g. tcp://0.0.0.0:9005).")
    parser.add_argument("--net-width", type=int, default=NET_IMG_W,
                        help="Frame width for tcp:// source (must match the sender).")
    parser.add_argument("--net-height", type=int, default=NET_IMG_H,
                        help="Frame height for tcp:// source (must match the sender).")
    parser.add_argument("--net-channels", type=int, default=NET_IMG_CH,
                        help="Channels per pixel for tcp:// source (1 or 3).")
    parser.add_argument("--net-order", type=str, default="rgb", choices=["rgb", "bgr"],
                        help="Pixel byte order sent by the tcp:// client.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--ema", type=float, default=0.5,
                        help="Temporal smoothing factor in [0,1); higher=smoother, more lag. 0 disables.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--flip", action="store_true", help="Horizontally flip the input (mirror).")
    parser.add_argument("--max-fps", type=float, default=0.0, help="Cap processing FPS (0 = unlimited).")
    parser.add_argument("--no-centerline", dest="centerline", action="store_false", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false", default=True)
    parser.add_argument("--record", type=str, default="", help="Path to save annotated MP4 (optional).")
    parser.add_argument("--snapshot-dir", type=str, default="",
                        help="Directory to periodically save annotated PNG frames (robust to hard exits).")
    parser.add_argument("--snapshot-every", type=int, default=15,
                        help="Save one annotated PNG every N processed frames when --snapshot-dir is set.")
    parser.add_argument("--web", action="store_true",
                        help="Serve a live web page (left: raw video, right: segmentation) via MJPEG.")
    parser.add_argument("--web-host", type=str, default="0.0.0.0",
                        help="Bind address for the web viewer.")
    parser.add_argument("--web-port", type=int, default=8000,
                        help="Port for the web viewer.")
    parser.add_argument("--web-quality", type=int, default=80,
                        help="JPEG quality [1-100] for the web streams.")
    parser.add_argument("--mmode-angle", type=float, default=30.0,
                        help="M-mode scan-line angle in degrees (right-to-left from vertical).")
    parser.add_argument("--gray-block-size", type=int, default=16,
                        help="Pixel size of each square tile in the grayscale mosaic panel.")
    parser.add_argument("--no-pace", action="store_true",
                        help="Disable real-time pacing for file sources (decode as fast as possible).")
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="Playback speed multiplier for paced file sources (1.0 = real time).")
    parser.add_argument("--loop", action="store_true",
                        help="Loop file sources when they reach the end.")
    parser.add_argument("--csv", type=str, default="", help="Path to save centerline points CSV (optional).")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = build_model(args, device)
    engine = SegEngine(
        model, device,
        input_size=args.encoder_input_size,
        use_fp16=args.fp16,
        ema=args.ema,
        threshold=args.threshold,
    )
    print("[engine] warming up ...", flush=True)
    engine.warmup(iters=5)

    grabber = build_grabber(args)
    grabber.start()
    time.sleep(0.3)  # let the first frame arrive
    W, H = grabber.width, grabber.height
    src_fps = grabber.fps if grabber.fps and grabber.fps > 1 else 25.0
    print(f"[source] {args.source}  {W}x{H} @ {src_fps:.1f}fps (reported)", flush=True)

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, src_fps, (W, H))
        print(f"[record] -> {args.record}", flush=True)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["ts_ms", "frame_seq"] + [f"{c}{i}" for i in range(100) for c in ("x", "y")]
        csv_writer.writerow(header)
        print(f"[csv] -> {args.csv}", flush=True)

    if args.snapshot_dir:
        os.makedirs(args.snapshot_dir, exist_ok=True)
        print(f"[snapshot] every {args.snapshot_every} frames -> {args.snapshot_dir}", flush=True)

    stream_state = None
    web_httpd = None
    mmode = flow_arrows = prob_strip = prob_est = gray_panel = None
    if args.web:
        stream_state = StreamState(jpeg_quality=args.web_quality)
        web_httpd = start_web_server(stream_state, args.web_host, args.web_port)
        shown = args.web_host if args.web_host != "0.0.0.0" else "<this-machine-ip>"
        print(f"[web] live viewer at http://{shown}:{args.web_port}/", flush=True)
        mmode = MModeBuilder(angle_deg=args.mmode_angle)
        flow_arrows = FlowArrows()
        prob_strip = ProbStrip(threshold=args.threshold)
        prob_est = EventProbEstimator()
        gray_panel = GrayPanel(block_size=args.gray_block_size)

    ema_fps = 0.0
    last_seq = -1
    min_dt = 1.0 / args.max_fps if args.max_fps > 0 else 0.0
    start = time.time()
    processed = 0
    print("[run] press 'q' in the window to quit (Ctrl+C in headless mode)", flush=True)

    try:
        while True:
            if grabber.stopped() and grabber.read_latest()[0] is None:
                break
            frame, seq = grabber.read_latest()
            if frame is None or seq == last_seq:
                if grabber.stopped():
                    break
                time.sleep(0.001)
                continue
            last_seq = seq

            t0 = time.time()
            if args.flip:
                frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_rs = cv2.resize(gray, (args.encoder_input_size, args.encoder_input_size))
            prob = engine.infer_prob(gray_rs)

            prob_full = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            mask = (prob_full > args.threshold).astype(np.uint8)
            area_ratio = float(mask.mean())

            final, centerline = render_overlay(frame, mask, draw_centerline=args.centerline)

            dt = time.time() - t0
            inst_fps = 1.0 / max(dt, 1e-6)
            ema_fps = inst_fps if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst_fps
            draw_hud(final, ema_fps, dt * 1000.0, area_ratio)

            if writer is not None:
                writer.write(final)
            if args.snapshot_dir and processed % max(1, args.snapshot_every) == 0:
                cv2.imwrite(os.path.join(args.snapshot_dir, f"snap_{processed:06d}.png"), final)
            if stream_state is not None:
                flow_img, motion = flow_arrows.panel(gray)
                mmode.push(gray)
                prob_val = prob_est.update(area_ratio, centerline, motion)
                prob_strip.push(prob_val)
                stream_state.update({
                    "raw": frame,
                    "seg": final,
                    "mmode": mmode.image(),
                    "gray": gray_panel.panel(gray),
                    "flow": flow_img,
                    "prob": prob_strip.image(),
                })
            if csv_writer is not None and centerline is not None:
                row = [f"{(time.time()-start)*1000:.1f}", seq]
                for px, py in centerline:
                    row += [f"{px:.2f}", f"{py:.2f}"]
                csv_writer.writerow(row)

            if args.display:
                cv2.imshow("TongueSeg realtime", final)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            processed += 1
            if min_dt > 0:
                sleep_left = min_dt - (time.time() - t0)
                if sleep_left > 0:
                    time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[run] interrupted", flush=True)
    finally:
        grabber.stop()
        if web_httpd is not None:
            web_httpd.shutdown()
        if writer is not None:
            writer.release()
        if csv_file is not None:
            csv_file.close()
        if args.display:
            cv2.destroyAllWindows()
        elapsed = time.time() - start
        avg = processed / elapsed if elapsed > 0 else 0.0
        print(f"[done] processed {processed} frames in {elapsed:.1f}s (avg {avg:.1f} FPS)", flush=True)


if __name__ == "__main__":
    main()
