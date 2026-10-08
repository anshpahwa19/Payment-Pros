"""4x super-resolution of the EPOS device (RGB + alpha) with Real-ESRGAN x4plus on CPU via ncnn.

Usage: python3 -I sr.py <device.webp> <model_dir> <out_dir>
Writes rgb_sr.npy (float32 0..1, HxWx3), alpha_sr.npy (float32 0..1, HxW) and rgb_ext.png.
"""
import sys, time
import numpy as np, cv2, ncnn
from PIL import Image

src, model_dir, out_dir = sys.argv[1:4]
D = np.array(Image.open(src).convert("RGBA")).astype(np.float32) / 255.0
rgb, a = D[..., :3], D[..., 3]
H, W = a.shape
S = 4


def push_pull_fill(rgb, w):
    """Fill pixels with low weight using a push-pull pyramid so edge colours bleed outward
    (avoids the black transparent background contaminating SR near the silhouette)."""
    levels = []
    c, ww = rgb * w[..., None], w.copy()
    while min(ww.shape) > 4:
        levels.append((c, ww))
        h2, w2 = (ww.shape[0] + 1) // 2, (ww.shape[1] + 1) // 2
        c = cv2.resize(c, (w2, h2), interpolation=cv2.INTER_AREA)
        ww = cv2.resize(ww, (w2, h2), interpolation=cv2.INTER_AREA)
    col = c / np.maximum(ww, 1e-6)[..., None]
    for c, ww in reversed(levels):
        up = cv2.resize(col, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        col = c / np.maximum(ww, 1e-6)[..., None]
        k = np.clip(ww, 0, 1)[..., None]
        col = np.where(ww[..., None] > 1e-6, k * col + (1 - k) * up, up)
    return col


wgt = np.clip((a - 0.5) * 2, 0, 1)  # trust colours only where mostly opaque
rgb_ext = rgb * wgt[..., None] + push_pull_fill(rgb, wgt) * (1 - wgt[..., None])
Image.fromarray((np.clip(rgb_ext, 0, 1) * 255 + 0.5).astype(np.uint8)).save(f"{out_dir}/rgb_ext.png")

net = ncnn.Net()
net.opt.use_vulkan_compute = False
net.opt.num_threads = 4
net.load_param(f"{model_dir}/realesrgan-x4plus.param")
net.load_model(f"{model_dir}/realesrgan-x4plus.bin")


def run(img):  # img HxWx3 float 0..1 -> (4H)x(4W)x3
    m = ncnn.Mat(np.ascontiguousarray(img.transpose(2, 0, 1)).astype(np.float32))
    ex = net.create_extractor()
    ex.input("data", m)
    _, out = ex.extract("output")
    return np.array(out).transpose(1, 2, 0)


def sr_tiled(img, need, tile=192, pad=16):
    """need(y0,y1,x0,x1) -> bool decides whether a tile must be processed; others get bicubic."""
    out = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    t0, n = time.time(), 0
    tiles = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    for y, x in tiles:
        y1, x1 = min(y + tile, H), min(x + tile, W)
        if not need(y, y1, x, x1):
            continue
        inp = padded[y:y1 + 2 * pad, x:x1 + 2 * pad]
        o = run(inp)
        out[y * S:y1 * S, x * S:x1 * S] = o[pad * S:pad * S + (y1 - y) * S, pad * S:pad * S + (x1 - x) * S]
        n += 1
        print(f"tile {n} ({y},{x}) {time.time() - t0:.0f}s", flush=True)
    return np.clip(out, 0, 1)


# RGB: every tile that touches the object (incl. a margin so dilated edge colours are right)
a_dil = cv2.dilate((a > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
rgb_sr = sr_tiled(rgb_ext.astype(np.float32), lambda y0, y1, x0, x1: a_dil[y0:y1, x0:x1].any())
np.save(f"{out_dir}/rgb_sr.npy", rgb_sr.astype(np.float32))

# Alpha: only tiles containing a silhouette edge; solid tiles are exact 0/1 anyway
edge = cv2.dilate(((a > 0) & (a < 1)).astype(np.uint8) | (cv2.morphologyEx((a > .5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))), np.ones((9, 9), np.uint8))
a3 = np.repeat(a[..., None], 3, -1).astype(np.float32)
a_sr = sr_tiled(a3, lambda y0, y1, x0, x1: edge[y0:y1, x0:x1].any()).mean(-1)
np.save(f"{out_dir}/alpha_sr.npy", a_sr.astype(np.float32))
print("done")
