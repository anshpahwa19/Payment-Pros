import sys, numpy as np, cv2
from PIL import Image
D = np.array(Image.open(sys.argv[1])); ad = D[..., 3]
L = D[..., :3].astype(float).mean(-1)
rgb = D[..., :3].astype(int)
g = ((np.abs(rgb - 95) <= 8).all(-1) & (ad == 255)).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(g, 8)
i = np.argmax(st[1:, 4]) + 1
m = (lab == i)
ys, xs = np.where(m)
x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
GREY = 95.0
def sub(prof_vals, idx_in, idx_out):
    # subpixel crossing where value = midpoint between grey and bezel value
    pass
top, bot, left, right = [], [], [], []
for x in range(x0, x1 + 1):
    col = np.where(m[:, x])[0]
    if len(col) < 50: continue
    a, b = col.min(), col.max()
    # top edge: crossing between bezel (above) and grey
    for (edge, step, out) in ((a, -1, top), (b, 1, bot)):
        bez = L[edge + 3 * step, x]
        v = L[edge + step, x]
        frac = np.clip((GREY - v) / max(GREY - bez, 1), 0, 1)  # fraction of pixel covered by bezel
        # pixel edge+step has coverage (1-frac) by grey -> boundary location
        pos = edge + step * (0.5 + (1 - frac)) if step > 0 else edge - (1 - frac) - 0.5 + 1
        out.append((x, (edge + 0.5 * step) + step * (1 - frac) if True else pos))
for y in range(y0, y1 + 1):
    row = np.where(m[y])[0]
    if len(row) < 50: continue
    a, b = row.min(), row.max()
    for (edge, step, out) in ((a, -1, left), (b, 1, right)):
        bez = L[y, edge + 3 * step]
        v = L[y, edge + step]
        frac = np.clip((GREY - v) / max(GREY - bez, 1), 0, 1)
        out.append(((edge + 0.5 * step) + step * (1 - frac), y))
def fitline(pts, horiz):
    pts = np.array(pts, float)
    # use the central 70% to avoid rounded corners
    k = len(pts); pts = pts[int(k * .15):int(k * .85)]
    if horiz:
        A = np.c_[pts[:, 0], np.ones(len(pts))]; c, *_ = np.linalg.lstsq(A, pts[:, 1], rcond=None)
        res = pts[:, 1] - A @ c
        return ('h', c, np.abs(res).max(), res.std())
    else:
        A = np.c_[pts[:, 1], np.ones(len(pts))]; c, *_ = np.linalg.lstsq(A, pts[:, 0], rcond=None)
        res = pts[:, 0] - A @ c
        return ('v', c, np.abs(res).max(), res.std())
lines = {k: fitline(v, k in ('top', 'bot')) for k, v in dict(top=top, bot=bot, left=left, right=right).items()}
for k, v in lines.items(): print(k, v)
def inter(h, v):
    # y = a x + b ; x = c y + d
    a, b = h[1]; c, d = v[1]
    y = (a * d + b) / (1 - a * c); x = c * y + d; return (x, y)
# pixel-center coords -> continuous coords (+0.5)
corners = [inter(lines['top'], lines['left']), inter(lines['top'], lines['right']), inter(lines['bot'], lines['right']), inter(lines['bot'], lines['left'])]
corners = np.array(corners) + 0.5
print('corners (TL,TR,BR,BL), continuous px coords:\n', corners)
np.save(sys.argv[2], corners)
# corner rounding check: distance from corner to nearest grey pixel
for (cx, cy) in corners:
    d = np.sqrt((xs + .5 - cx) ** 2 + (ys + .5 - cy) ** 2).min(); print('nearest grey to corner', round(d, 2))
