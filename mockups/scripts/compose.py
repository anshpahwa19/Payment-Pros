"""Composite the Payment Pros EPOS mockup.

Usage: python3 -I compose.py <images_dir> <sr_dir> <corners.npy> <out_dir>
"""
import sys
import numpy as np, cv2
from PIL import Image

img_dir, sr_dir, corners_path, out_dir = sys.argv[1:5]
S = 4                      # SR scale
CANVAS = 5120              # final square canvas
FW, FH = 7680, 4320        # flat screen space (16:9), 2x supersampled
LOGO_W_FRAC = 0.50         # logo width as fraction of screen width

f32 = np.float32


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(f32)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(f32)


# ---------------------------------------------------------------- device (HR)
src = np.array(Image.open(f"{img_dir}/2.webp").convert("RGBA")).astype(f32) / 255
H0, W0 = src.shape[:2]
lr_ext = np.array(Image.open(f"{sr_dir}/rgb_ext.png")).astype(f32) / 255
rgb = np.load(f"{sr_dir}/rgb_sr.npy").astype(f32)
alpha = np.clip(np.load(f"{sr_dir}/alpha_sr.npy").astype(f32), 0, 1)

# Back-projection: pull the low frequencies of the SR result back onto the source so that,
# viewed at source resolution, the hardware is unchanged; SR only contributes edge sharpness.
for _ in range(3):
    err = lr_ext - cv2.resize(rgb, (W0, H0), interpolation=cv2.INTER_AREA)
    rgb = rgb + cv2.resize(err, (W0 * S, H0 * S), interpolation=cv2.INTER_CUBIC)
rgb = np.clip(rgb, 0, 1)
for _ in range(2):
    err = src[..., 3] - cv2.resize(alpha, (W0, H0), interpolation=cv2.INTER_AREA)
    alpha = np.clip(alpha + cv2.resize(err, (W0 * S, H0 * S), interpolation=cv2.INTER_CUBIC), 0, 1)
# Solid interior / empty exterior stay exactly solid / empty
a_lr = src[..., 3]
solid = cv2.erode((a_lr >= 1).astype(np.uint8), np.ones((5, 5), np.uint8))
empty = cv2.erode((a_lr <= 0).astype(np.uint8), np.ones((5, 5), np.uint8))
solid = cv2.resize(solid, (W0 * S, H0 * S), interpolation=cv2.INTER_NEAREST).astype(bool)
empty = cv2.resize(empty, (W0 * S, H0 * S), interpolation=cv2.INTER_NEAREST).astype(bool)
alpha[solid] = 1
alpha[empty] = 0
alpha[alpha < 1.5 / 255] = 0
alpha[alpha > 253.5 / 255] = 1
Hh, Wh = alpha.shape

# ---------------------------------------------------------------- screen geometry
corners = np.load(corners_path).astype(np.float64)        # TL, TR, BR, BL (source px, continuous)
cHR = corners * S


def poly_mask(pts, shape, ss=4):
    """Anti-aliased polygon coverage via supersampled rasterisation (bbox-limited)."""
    x0, y0 = np.floor(pts.min(0)).astype(int) - 2
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
    m = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    p = np.round((pts - [x0, y0]) * ss * 16).astype(np.int32)
    cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_8, shift=4)
    m = cv2.resize(m.astype(f32) / 255, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    out = np.zeros(shape, f32)
    out[y0:y1, x0:x1] = m
    return out


# pixel centres are at +0.5 in continuous coords; cv2 fillPoly treats integer coords as pixel centres
screen_mask = poly_mask(cHR - 0.5, (Hh, Wh))

# Remove the grey placeholder (incl. its anti-aliased rim) and refill that rim with bezel colour
rim = cv2.dilate((screen_mask > 0).astype(np.uint8), np.ones((15, 15), np.uint8)).astype(f32)
wgt = 1 - rim


def push_pull(img, w):
    levels = []
    c, ww = img * w[..., None], w.copy()
    while min(ww.shape) > 4:
        levels.append((c, ww))
        h2, w2 = (ww.shape[0] + 1) // 2, (ww.shape[1] + 1) // 2
        c = cv2.resize(c, (w2, h2), interpolation=cv2.INTER_AREA)
        ww = cv2.resize(ww, (w2, h2), interpolation=cv2.INTER_AREA)
    col = c / np.maximum(ww, 1e-6)[..., None]
    for c, ww in reversed(levels):
        up = cv2.resize(col, (ww.shape[1], ww.shape[0]), interpolation=cv2.INTER_LINEAR)
        cur = c / np.maximum(ww, 1e-6)[..., None]
        k = np.clip(ww, 0, 1)[..., None]
        col = np.where(ww[..., None] > 1e-6, k * cur + (1 - k) * up, up)
    return col


# limit the fill to a band around the screen for speed
ys, xs = np.where(rim > 0)
by0, by1, bx0, bx1 = ys.min() - 64, ys.max() + 64, xs.min() - 64, xs.max() + 64
band = rgb[by0:by1, bx0:bx1]
filled = push_pull(band, wgt[by0:by1, bx0:bx1])
r = rim[by0:by1, bx0:bx1, None]
rgb[by0:by1, bx0:bx1] = band * (1 - r) + filled * r

# ---------------------------------------------------------------- glass region (bezel + active area)
# dark bezel pixels connected to the screen, inside the silver frame
L_lr = src[..., :3].mean(-1)
dark = ((L_lr < 0.16) & (a_lr > 0.99)).astype(np.uint8)
scr_lr = poly_mask(corners - 0.5, (H0, W0))
seed = cv2.dilate((scr_lr > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8))
n, lab = cv2.connectedComponents(dark | seed)
ids = np.unique(lab[seed > 0]); ids = ids[ids > 0]
glass_lr = np.isin(lab, ids).astype(np.uint8)
glass_lr = cv2.morphologyEx(glass_lr, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
cs, _ = cv2.findContours(glass_lr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
glass_lr = np.zeros_like(glass_lr)
cv2.drawContours(glass_lr, [max(cs, key=cv2.contourArea)], -1, 1, -1)
glass = cv2.resize(glass_lr.astype(f32), (Wh, Hh), interpolation=cv2.INTER_LINEAR)
glass = cv2.GaussianBlur(glass, (0, 0), 2.0) * alpha
glass = np.maximum(glass, screen_mask)

# ---------------------------------------------------------------- flat screen content (linear light)
bg = np.array(Image.open(f"{img_dir}/1.jpg").convert("RGB")).astype(f32) / 255
bg = cv2.resize(bg, (1920, 1080), interpolation=cv2.INTER_CUBIC)       # fill the 16:9 panel
bg = cv2.GaussianBlur(bg, (0, 0), 6)                                     # remove JPEG blocking
bg = cv2.resize(bg, (FW, FH), interpolation=cv2.INTER_CUBIC)
flat = srgb_to_lin(np.clip(bg, 0, 1))

logo = np.array(Image.open(f"{img_dir}/3.png").convert("RGBA")).astype(f32) / 255
la = logo[..., 3]
PURPLE = np.array([146, 102, 255], f32) / 255
WHITE = np.ones(3, f32)
# decompose each pixel into purple / white coverage using the green channel
g_pm = logo[..., 1] * la
w_cov = np.clip((g_pm - PURPLE[1] * la) / (1 - PURPLE[1]), 0, 1)
w_cov = np.minimum(w_cov, la)

lw = int(round(FW * LOGO_W_FRAC))
lh = int(round(lw * logo.shape[0] / logo.shape[1]))
scale = lw / logo.shape[1]
pad = 8
def up_sharp(cov):
    c = cv2.copyMakeBorder(cov, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    big = cv2.resize(c, (lw + int(2 * pad * scale), lh + int(2 * pad * scale)), interpolation=cv2.INTER_LANCZOS4)
    k = scale / 1.6                                       # ~1.6 px AA ramp at supersampled res
    big = np.clip((big - 0.5) * k + 0.5, 0, 1)
    o = int(pad * scale)
    return big[o:o + lh, o:o + lw]
A = up_sharp(la)
Wc = np.minimum(up_sharp(w_cov), A)
Pc = np.clip(A - Wc, 0, 1)
logo_pm_lin = Pc[..., None] * srgb_to_lin(PURPLE) + Wc[..., None] * srgb_to_lin(WHITE)

lx, ly = (FW - lw) // 2, (FH - lh) // 2
reg = flat[ly:ly + lh, lx:lx + lw]
flat[ly:ly + lh, lx:lx + lw] = logo_pm_lin + reg * (1 - A[..., None])

# soft emissive bloom so the logo reads as light from the panel, not a sticker
glow_src = np.zeros_like(flat)
glow_src[ly:ly + lh, lx:lx + lw] = logo_pm_lin
small = cv2.resize(glow_src, (FW // 8, FH // 8), interpolation=cv2.INTER_AREA)
glow = cv2.GaussianBlur(small, (0, 0), 9) * 0.55 + cv2.GaussianBlur(small, (0, 0), 3) * 0.35
glow = cv2.resize(glow, (FW, FH), interpolation=cv2.INTER_CUBIC)
flat = flat + glow * 0.16

# ---------------------------------------------------------------- warp flat content onto the screen
src_q = np.array([[0, 0], [FW, 0], [FW, FH], [0, FH]], np.float64)
x0, y0 = np.floor(cHR.min(0)).astype(int) - 4
x1, y1 = np.ceil(cHR.max(0)).astype(int) + 4
dst_q = (cHR - [x0, y0]) * 2                     # 2x supersampled target (continuous coords)
Hm = cv2.getPerspectiveTransform(src_q.astype(np.float32), dst_q.astype(np.float32))
# cv2 maps pixel centres: shift continuous->centre convention
T_in = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
T_out = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
Hm = T_out @ Hm @ T_in
warped = cv2.warpPerspective(flat, Hm, ((x1 - x0) * 2, (y1 - y0) * 2), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
warped = cv2.resize(warped, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
screen_lin = np.zeros((Hh, Wh, 3), f32)
screen_lin[y0:y1, x0:x1] = warped

rgb_lin = srgb_to_lin(rgb)
m = screen_mask[..., None]
rgb_lin = screen_lin * m + rgb_lin * (1 - m)

# ---------------------------------------------------------------- glass reflections (in device space)
yy, xx = np.mgrid[0:Hh, 0:Wh].astype(f32)
cx, cy = cHR[:, 0].mean(), cHR[:, 1].mean()
sw = np.linalg.norm(cHR[1] - cHR[0])
# broad diagonal sheen from the upper-left key light
d = ((xx - cx) * np.cos(np.radians(35)) + (yy - cy) * np.sin(np.radians(35))) / sw
sheen = np.exp(-((d + 0.30) / 0.22) ** 2) * 0.050
# faint secondary streak
sheen += np.exp(-((d + 0.02) / 0.035) ** 2) * 0.012
# gentle top-to-bottom falloff typical of a glossy panel under studio softbox
v = (yy - cHR[:, 1].min()) / (cHR[:, 1].max() - cHR[:, 1].min())
sheen += np.clip(1 - v, 0, 1) ** 2 * 0.018
sheen *= glass
refl = sheen[..., None]
rgb_lin = 1 - (1 - rgb_lin) * (1 - refl)          # screen blend
rgb = lin_to_srgb(rgb_lin)

# ---------------------------------------------------------------- contact shadow
# bottom profile of the base (columns whose lowest opaque pixel is near the floor)
cols = np.where((src[..., 3] > 0.5).any(0))[0]
low = np.array([np.where(src[:, x, 3] > 0.5)[0].max() for x in cols])
base_cols = cols[low > 1000]
foot = np.zeros((H0, W0), f32)
for x, yb in zip(cols, low):
    if yb > 1000:
        foot[yb - 10:yb + 1, x] = 1          # thin band hugging the bottom edge
foot = cv2.resize(foot, (Wh, Hh), interpolation=cv2.INTER_LINEAR)
foot = np.roll(foot, 3 * S, axis=0)
contact = cv2.GaussianBlur(foot, (0, 0), sigmaX=5 * S, sigmaY=2.0 * S)
ambient = cv2.GaussianBlur(foot, (0, 0), sigmaX=30 * S, sigmaY=8 * S)
contact /= contact.max(); ambient /= ambient.max()
shadow_a = np.clip(contact * 0.55 + ambient * 0.22, 0, 1)
shadow_a[shadow_a < 1 / 255] = 0

# ---------------------------------------------------------------- assemble on the final canvas
def place(img_hr):
    ys, xs = np.where(alpha > 0)
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    ox = (CANVAS - (bx1 - bx0)) // 2 - bx0
    oy = (CANVAS - (by1 - by0)) // 2 - by0
    shape = (CANVAS, CANVAS) + img_hr.shape[2:]
    out = np.zeros(shape, f32)
    sy0, sx0 = max(0, -oy), max(0, -ox)
    sy1, sx1 = min(Hh, CANVAS - oy), min(Wh, CANVAS - ox)
    out[sy0 + oy:sy1 + oy, sx0 + ox:sx1 + ox] = img_hr[sy0:sy1, sx0:sx1]
    return out


def finish(with_shadow):
    a_dev = alpha
    if with_shadow:
        a_out = a_dev + shadow_a * (1 - a_dev)
        rgb_out = rgb * a_dev[..., None] / np.maximum(a_out, 1e-6)[..., None]  # shadow is black
    else:
        a_out, rgb_out = a_dev, rgb
    A_ = place(a_out); C_ = place(rgb_out)
    rng = np.random.default_rng(7)
    dither = (rng.random(C_.shape, dtype=f32) - rng.random(C_.shape, dtype=f32))   # TPDF, ±1 LSB
    C8 = np.clip(np.round(C_ * 255 + dither), 0, 255).astype(np.uint8)
    A8 = np.clip(np.round(A_ * 255), 0, 255).astype(np.uint8)
    C8[A8 == 0] = 0
    return np.dstack([C8, A8])


out_main = finish(True)
Image.fromarray(out_main, "RGBA").save(f"{out_dir}/payment-pros-epos-mockup-4k.png", optimize=True)
out_ns = finish(False)
Image.fromarray(out_ns, "RGBA").save(f"{out_dir}/payment-pros-epos-mockup-4k-no-shadow.png", optimize=True)
print("saved", out_main.shape)
