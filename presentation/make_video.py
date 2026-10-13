"""Animated, narrated explainer video (PIL frames + gTTS narration -> ffmpeg).
usage: python animate.py WORKDIR NPZ LOGDIR OUT.mp4 [preview SCENE STEP T]"""
import math, os, re, subprocess, sys, wave
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

OUT, NPZ, LOGS, FINAL = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1920, 1080, 25
BG = (247, 245, 240); NAVY = (27, 42, 73); TEAL = (15, 139, 141); ORANGE = (224, 122, 31)
GREY = (107, 114, 128); LINE = (214, 211, 204); WHITE = (255, 255, 255)
TEAL_L = (221, 241, 241); ORANGE_L = (252, 232, 213); GREEN = (46, 140, 87); RED = (200, 70, 60)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_fc = {}
def F(size, bold=False):
    k = (int(size), bold)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONTB if bold else FONT, int(size))
    return _fc[k]
def T(d, xy, s, size=36, color=NAVY, bold=False, anchor="la"): d.text(xy, s, font=F(size, bold), fill=color, anchor=anchor)
def wrap(s, size, maxw, bold=False):
    lines, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if F(size, bold).getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w
    return lines + ([cur] if cur else [])
def P(d, x, y, s, size=36, color=NAVY, maxw=1600, lh=1.35, bold=False, anchor="la"):
    for line in wrap(s, size, maxw, bold): T(d, (x, y), line, size, color, bold, anchor); y += int(size * lh)
    return y
def box(d, x, y, w, h, fill=WHITE, outline=LINE, r=22, width=3): d.rounded_rectangle([x, y, x + w, y + h], r, fill=fill, outline=outline, width=width)
def card(d, x, y, w, h, title, sub=None, fill=WHITE, outline=LINE, ts=38, ss=28, tcol=NAVY, scol=GREY, width=3):
    box(d, x, y, w, h, fill, outline, width=width)
    lt = wrap(title, ts, w - 40, True); ls = wrap(sub, ss, w - 40) if sub else []
    total = len(lt) * ts * 1.25 + (len(ls) * ss * 1.3 + 10 if ls else 0); cy = y + (h - total) / 2
    for l in lt: T(d, (x + w / 2, cy), l, ts, tcol, True, "ma"); cy += ts * 1.25
    cy += 10
    for l in ls: T(d, (x + w / 2, cy), l, ss, scol, False, "ma"); cy += ss * 1.3
def arrow(d, x1, y1, x2, y2, color=GREY, width=5, head=20):
    d.line([x1, y1, x2, y2], fill=color, width=width); a = math.atan2(y2 - y1, x2 - x1)
    d.polygon([(x2, y2), (x2 - head * math.cos(a - .45), y2 - head * math.sin(a - .45)), (x2 - head * math.cos(a + .45), y2 - head * math.sin(a + .45))], fill=color)
def pill(d, x, y, s, fill=TEAL_L, color=TEAL, size=30):
    w = F(size, True).getlength(s) + 48; d.rounded_rectangle([x, y, x + w, y + size + 28], (size + 28) // 2, fill=fill)
    T(d, (x + 24, y + 13), s, size, color, True); return x + w
def ease(t, start=0.0, dur=1.0):
    u = min(1.0, max(0.0, (t - start) / dur)); return u * u * (3 - 2 * u)
def lerp(a, b, u): return a + (b - a) * u
def mix(c1, c2, u): return tuple(int(lerp(a, b, u)) for a, b in zip(c1, c2))
def dot(d, x, y, r, col): d.ellipse([x - r, y - r, x + r, y + r], fill=col)

data = np.load(NPZ); X, Y = data["train_images"], data["train_labels"].ravel()
ORGANS = ["Liver", "Right kidney", "Left kidney", "Right femur", "Left femur", "Bladder", "Heart", "Right lung", "Left lung", "Spleen", "Pancreas"]
SAMPLE = {c: X[np.where(Y == c)[0][0]] for c in range(11)}
_tc = {}
def tile(c, idx, size):
    k = (c, idx, size)
    if k not in _tc: _tc[k] = Image.fromarray(SAMPLE[c][idx].astype(np.uint8)).resize((size, size), Image.BICUBIC).convert("RGB")
    return _tc[k]
def curve(name): return [float(m) for m in re.findall(r"val acc ([0-9.]+)", open(f"{LOGS}/{name}.log").read())]
VAL_B, VAL_V = curve("baseline"), curve("vlm")

# ======================= scenes: fn(img, d, k, t) =======================
def s_title(img, d, k, t):
    d.rectangle([0, 0, W, H], fill=NAVY)
    for i in range(14):  # drifting dots
        x = (i * 173 + t * 22 * (1 + i % 3)) % (W + 200) - 100; y = 120 + (i * 97) % 860
        dot(d, x, y, 4 + i % 4, (45, 70, 110))
    T(d, (160, 250), "DEEP LEARNING MINI PROJECT  ·  EXPLAINED SIMPLY", 32, (150, 210, 210), True)
    y = P(d, 160, 320, "Design of a Vision-Language Model for 3D Medical Image Analysis", 76, WHITE, 1550, 1.2, True)
    T(d, (160, y + 40), "3D CT scans  ·  Deep learning  ·  Vision-language models", 38, (210, 220, 235))
    if k >= 1:
        x = 160
        for s in ["The data", "Signals & filters", "The baseline", "The improvement", "The results", "Summary"]:
            x = pill(d, x, y + 150, s, (40, 62, 100), (230, 240, 250), 28) + 18

def s_onesentence(img, d, k, t):
    box(d, 160, 240, 1600, 200, TEAL_L, TEAL)
    P(d, 960, 285, "Teach a computer to look at a 3D CT scan and say which organ it shows.", 50, NAVY, 1450, 1.3, True, "ma")
    if k >= 1: card(d, 160, 520, 700, 260, "Model 1: Baseline", "A standard deep learning model (3D CNN)")
    if k >= 2:
        card(d, 1060, 520, 700, 260, "Model 2: Vision-Language Model", "Also understands words — our improvement", ORANGE_L, ORANGE)
        T(d, (960, 650), "vs", 56, GREY, True, "mm")
    if k >= 3:
        box(d, 160, 830, 1600, 110, WHITE, LINE); T(d, (960, 885), "Then compare: which one recognises organs better?", 40, TEAL, True, "mm")

def s_ct(img, d, k, t):
    P(d, 120, 250, "A CT machine takes many X-ray images of the body, slice by slice — like slicing a loaf of bread.", 38, NAVY, 820)
    for i, s in enumerate([4, 8, 12, 16, 20, 24]):
        u = ease(t, 0.6 + i * 0.55, 0.5) if k == 0 else 1
        if u <= 0: continue
        x, y = 1080 + i * 55, 230 + i * 55 - (1 - u) * 160
        d.rectangle([x - 6, y - 6, x + 306, y + 306], fill=WHITE, outline=NAVY, width=3); img.paste(tile(0, s, 300), (int(x), int(y)))
    if k >= 1:
        P(d, 120, 470, "Stacking all the slices gives a 3D picture of the inside of the body, called a volume.", 38, NAVY, 820)
        T(d, (1080 + 5 * 55 + 150, 230 + 5 * 55 + 340), "Slices stacked = 3D volume", 30, TEAL, True, "ma")
    if k >= 2:
        box(d, 120, 680, 820, 270, ORANGE_L, ORANGE); T(d, (160, 712), "In our project", 32, ORANGE, True)
        P(d, 160, 765, "Each volume is a cube of 28 × 28 × 28 points. Each point is a voxel — a 3D pixel, i.e. a 3D array of samples.", 36, NAVY, 740)

def s_dataset(img, d, k, t):
    P(d, 120, 235, "OrganMNIST3D — a free, public dataset from the MedMNIST benchmark (Scientific Data, 2023).", 34, NAVY, 700)
    for i, (a, b) in enumerate([("1,742", "CT volumes"), ("11", "organ classes"), ("28×28×28", "voxels each"), ("201", "real CT scans used")]):
        if k >= 1 + (i >= 2):
            x, y = 120 + (i % 2) * 360, 420 + (i // 2) * 230; box(d, x, y, 330, 200)
            T(d, (x + 165, y + 50), a, 58 if len(a) < 6 else 46, TEAL, True, "ma"); T(d, (x + 165, y + 135), b, 28, GREY, False, "ma")
    if k >= 2:
        for c in range(11):
            u = ease(t, c * 0.35, 0.4) if k == 2 else 1
            if u <= 0: continue
            s = int(150 * (0.6 + 0.4 * u)); x, y = 900 + (c % 4) * 245, 230 + (c // 4) * 255
            ox = x + 40 + (150 - s) // 2; oy = y + (150 - s) // 2
            img.paste(tile(c, 14, s), (ox, oy)); d.rectangle([ox, oy, ox + s, oy + s], outline=NAVY, width=2)
            if u > 0.6: T(d, (x + 115, y + 162), ORGANS[c], 26, NAVY, True, "ma")
        T(d, (1880, 1000), "Real images from the dataset (middle slice of each volume)", 24, GREY, False, "ra")

def s_split(img, d, k, t):
    parts = [("Training", 971, TEAL, "The textbook — the model learns from these"), ("Validation", 161, ORANGE, "The practice test — picks the best version"),
             ("Testing", 610, NAVY, "The final exam — never seen while learning")]
    P(d, 120, 240, "The 1,742 volumes are split into three groups — like a student preparing for an exam.", 38, NAVY, 1680)
    x = 120
    for i, (name, n, col, desc) in enumerate(parts):
        w = 1680 * n / 1742
        if k >= i + 1:
            u = ease(t, 0, 0.9) if k == i + 1 else 1
            d.rectangle([x, 400, x + max(2, (w - 6) * u), 520], fill=col)
            if u > 0.8: T(d, (x + w / 2, 460), f"{n}", 44, WHITE, True, "mm")
            y = 600 + i * 130; d.rectangle([120, y + 8, 160, y + 48], fill=col)
            T(d, (190, y), f"{name}: {n} volumes", 38, NAVY, True); T(d, (190, y + 50), desc, 30, GREY)
        x += w

def s_ml(img, d, k, t):
    box(d, 120, 240, 1680, 190, WHITE, LINE)
    T(d, (160, 265), "Classic signal processing", 30, GREY, True)
    for i, s in enumerate(["Input signal", "Filter designed by an engineer", "Output"]):
        card(d, 160 + i * 560, 315, 480, 90, s, None, WHITE, LINE, 30)
        if i: arrow(d, 160 + i * 560 - 75, 360, 160 + i * 560 - 8, 360)
    if k >= 1:
        box(d, 120, 460, 1680, 190, TEAL_L, TEAL); T(d, (160, 485), "Machine learning", 30, TEAL, True)
        for i, s in enumerate(["Many examples + answers", "Computer LEARNS the filter", "Output"]):
            card(d, 160 + i * 560, 535, 480, 90, s, None, WHITE, TEAL, 30)
            if i: arrow(d, 160 + i * 560 - 75, 580, 160 + i * 560 - 8, 580, TEAL)
    if k >= 2:
        cx, cy, r = 1340, 860, 125
        steps = ["Look", "Guess", "Check error", "Adjust"]; pts = [(cx - 330, cy), (cx - 110, cy), (cx + 110, cy), (cx + 330, cy)]
        for i, (p, s) in enumerate(zip(pts, steps)):
            card(d, p[0] - 100, cy - 45, 200, 90, s, None, WHITE, TEAL, 30)
            if i < 3: arrow(d, p[0] + 100, cy, pts[i + 1][0] - 100 + 2, cy, TEAL, 4, 14)
        d.line([pts[3][0], cy + 45, pts[3][0], cy + 85, pts[0][0], cy + 85, pts[0][0], cy + 47], fill=TEAL, width=4)
        arrow(d, pts[0][0], cy + 70, pts[0][0], cy + 47, TEAL, 4, 14)
        ph = (t * 0.6) % 1.0  # moving highlight around the loop
        pathx = [pts[0][0], pts[3][0], pts[3][0], pts[0][0], pts[0][0]]; pathy = [cy - 60, cy - 60, cy + 85, cy + 85, cy - 60]
        seg = ph * 4; i = int(seg); f = seg - i
        dot(d, lerp(pathx[i], pathx[i + 1], f), lerp(pathy[i], pathy[i + 1], f), 11, ORANGE)
        P(d, 120, 690, "Same idea as an adaptive (LMS) filter: compare output with the desired answer, measure the error, adjust the coefficients — repeat thousands of times.", 30, NAVY, 700)

SIG = np.concatenate([np.zeros(10), np.ones(14), np.zeros(8), np.linspace(0, 1, 6), np.ones(4), np.zeros(8)]) + 0.06 * np.sin(np.arange(50) * 2.3)
KER = np.array([-1, 0, 1])
def s_conv(img, d, k, t):
    # 1-D convolution animation
    x0, y0, dx, amp = 140, 430, 22, 120
    T(d, (x0, 250), "Input signal x[n]", 28, GREY, True)
    n = len(SIG)
    pts = [(x0 + i * dx, y0 - SIG[i] * amp) for i in range(n)]
    d.line(pts, fill=NAVY, width=4)
    for p in pts: dot(d, p[0], p[1], 4, NAVY)
    T(d, (x0, 530), "Output y[n] = x[n] ∗ h[n]   (kernel h = [−1, 0, 1], an edge detector)", 26, GREY, True)
    out = np.convolve(SIG, KER[::-1], mode="same")
    pos = int(min(n - 2, 1 + (t - 1.0) * 9)) if k == 0 else n - 2
    if k > 0 or t > 1.0:
        yo = 690
        opts = [(x0 + i * dx, yo - out[i] * 90) for i in range(1, pos + 1)]
        d.line([x0, yo, x0 + (n - 1) * dx, yo], fill=LINE, width=2)
        if len(opts) > 1: d.line(opts, fill=ORANGE, width=4)
        for p in opts: dot(d, p[0], p[1], 4, ORANGE)
        if k == 0 and pos < n - 2:
            kx = x0 + (pos - 1) * dx
            d.rounded_rectangle([kx - 14, y0 - amp - 30, kx + 2 * dx + 14, y0 + 30], 10, outline=ORANGE, width=4)
            T(d, (kx + dx, y0 + 62), "kernel slides →", 24, ORANGE, True, "ma")
            d.line([x0 + pos * dx, y0 + 30, x0 + pos * dx, yo - out[pos] * 90], fill=ORANGE, width=2)
    box(d, 1700 - 380, 300, 480, 200, ORANGE_L, ORANGE)
    P(d, 1340, 320, "At every position: multiply and add. Edges in the input give spikes in the output.", 28, NAVY, 430)
    if k >= 1:
        box(d, 1320, 530, 480, 400, WHITE, TEAL, width=3)
        T(d, (1560, 550), "Same on a CT slice (2D)", 26, TEAL, True, "ma")
        img.paste(tile(0, 14, 280), (1420, 600)); d.rectangle([1420, 600, 1700, 880], outline=NAVY, width=2)
        step = int(t * 8) if k == 1 else 300
        gx = step % 26; gy = (step // 26 * 4 + 2) % 26
        cs = 280 / 28
        d.rectangle([1420 + gx * cs, 600 + gy * cs, 1420 + (gx + 3) * cs, 600 + (gy + 3) * cs], outline=ORANGE, width=4)
        T(d, (1560, 895), "3 × 3 window slides over pixels", 22, GREY, False, "ma")
    if k >= 2:
        box(d, 120, 850, 1160, 150, TEAL_L, TEAL)
        P(d, 150, 870, "A CNN does exactly this in 3D with 3 × 3 × 3 kernels. The big difference: the kernel coefficients are NOT designed by hand — the network LEARNS them from data.", 28, NAVY, 1100, bold=True)

def s_cnn(img, d, k, t):
    P(d, 120, 235, "Model 1 is a 3D Convolutional Neural Network (3D CNN) — the standard method, our baseline.", 36, NAVY, 1680)
    y = 400
    img.paste(tile(0, 14, 170), (120, y)); d.rectangle([120, y, 290, y + 170], outline=NAVY, width=3)
    T(d, (205, y + 185), "CT volume", 28, NAVY, True, "ma")
    xs = []
    if k >= 1:
        subs = ["edges", "textures", "shapes", "organ parts"]
        for i in range(4):
            x = 360 + i * 230; hh = 170 - i * 25; xs.append(x)
            glow = k == 1 and t < 900 and int(t * 1.5) % 4 == i
            box(d, x, y + (170 - hh) / 2, 190, hh, ORANGE_L if glow else TEAL_L, ORANGE if glow else TEAL, width=4 if glow else 3)
            T(d, (x + 95, y + 70), f"Block {i + 1}", 28, NAVY, True, "ma")
            T(d, (x + 95, y + 105), "32 64 128 256".split()[i] + " filters", 20, GREY, False, "ma")
            if k >= 2: T(d, (x + 95, y + 200), subs[i], 26, TEAL, True, "ma")
            arrow(d, x - 40 if i else 300, y + 85, x - 6, y + 85, GREY, 4, 16)
        P(d, 120, 660, "Each block = learned 3 × 3 × 3 filters + a non-linearity + down-sampling (like decimation).", 32, NAVY, 1680)
    if k >= 2: P(d, 120, 715, "Early blocks find simple edges; later blocks combine them into shapes and organs. The scan becomes 256 numbers.", 32, NAVY, 1680)
    if k >= 3:
        arrow(d, 1290, y + 85, 1330, y + 85, GREY, 4, 16); card(d, 1330, y + 20, 200, 130, "Final layer", "256 → 11 scores", WHITE, LINE, 28, 24)
        arrow(d, 1530, y + 85, 1570, y + 85, GREY, 4, 16); card(d, 1570, y + 20, 230, 130, "Answer", "highest score: Liver", ORANGE_L, ORANGE, 30, 24)
        for j in range(5):  # data flowing through the network
            ph = (t * 0.45 + j / 5) % 1; x = lerp(290, 1570, ph)
            dot(d, x, y + 85, 9, ORANGE)
        box(d, 120, 830, 1680, 110, WHITE, LINE)
        T(d, (960, 885), "About 1.17 million adjustable numbers (parameters) — all learned from data", 34, TEAL, True, "mm")

def s_train(img, d, k, t):
    P(d, 120, 240, "Training: compare the guess with the correct organ. The loss is the error signal — a number saying how wrong the guess was.", 34, NAVY, 1680)
    items = [("Loss (error)", "Cross-entropy"), ("Update rule", "Adam optimiser"), ("Step size", "Learning rate 0.001"),
             ("Batch", "32 scans at a time"), ("Epochs", "50 full passes"), ("Extra variety", "Random flips")]
    if k >= 1:
        for i, (a, b) in enumerate(items):
            x, y = 120 + (i % 2) * 400, 380 + (i // 2) * 150
            box(d, x, y, 380, 130); T(d, (x + 25, y + 22), a, 24, GREY); T(d, (x + 25, y + 65), b, 30, NAVY, True)
    if k >= 2:
        gx, gy, gw, gh = 1000, 380, 800, 440
        box(d, gx - 20, gy - 20, gw + 60, gh + 130)
        T(d, (gx, gy - 5), "Real training run: validation accuracy per epoch", 24, GREY, True)
        for v in (0.2, 0.4, 0.6, 0.8, 1.0):
            yy = gy + gh - v * (gh - 40); d.line([gx + 60, yy, gx + gw, yy], fill=LINE, width=1); T(d, (gx + 50, yy), f"{int(v * 100)}%", 20, GREY, False, "rm")
        upto = int(min(50, 1 + t * 10)) if k == 2 else 50
        for vals, col, name, ly in [(VAL_B, NAVY, "Baseline", 0), (VAL_V, ORANGE, "Vision-language", 1)]:
            p = [(gx + 60 + i * (gw - 70) / 49, gy + gh - vals[i] * (gh - 40)) for i in range(upto)]
            if len(p) > 1: d.line(p, fill=col, width=4)
            d.rectangle([gx + 80 + ly * 260, gy + gh + 50, gx + 110 + ly * 260, gy + gh + 64], fill=col); T(d, (gx + 120 + ly * 260, gy + gh + 57), name, 22, NAVY, True, "lm")
        T(d, (gx + 60 + (gw - 70) / 2, gy + gh + 20), f"epoch {upto} / 50", 22, GREY, False, "ma")
    if k >= 3:
        box(d, 120, 850, 780, 140, TEAL_L, TEAL)
        P(d, 150, 870, "After every epoch we check the practice test and keep the best version of the model.", 28, NAVY, 720, bold=True)

def s_limit(img, d, k, t):
    P(d, 120, 250, "Limitation: the baseline only learns class numbers.", 46, NAVY, 1680, bold=True)
    card(d, 220, 400, 600, 300, "Organ = 0", "To the baseline, the liver is just “number 0”", WHITE, LINE, 60, 32)
    T(d, (960, 550), "→", 90, GREY, True, "mm")
    card(d, 1100, 400, 600, 300, "“liver” = ?", "It never learns what the word means", ORANGE_L, ORANGE, 60, 32)
    if k >= 1:
        box(d, 220, 780, 1480, 120, TEAL_L, TEAL); T(d, (960, 840), "This is where vision-language models help", 42, TEAL, True, "mm")

def s_vlm(img, d, k, t):
    P(d, 120, 240, "A vision-language model learns from pictures and words together, so a picture sits close to the sentence that describes it.", 34, NAVY, 1000)
    box(d, 1200, 230, 600, 360); T(d, (1500, 255), "Shared space", 28, GREY, True, "ma")
    u = ease(t, 0.5, 1.8) if k == 0 else 1
    ix, sx = lerp(1230, 1260, u), lerp(1500, 1430, u); iy, sy = lerp(470, 340, u), lerp(310, 360, u)
    img.paste(tile(0, 14, 110), (int(ix), int(iy))); d.rectangle([ix, iy, ix + 110, iy + 110], outline=NAVY, width=2)
    box(d, sx, sy, 330, 70, ORANGE_L, ORANGE, 16, 2); T(d, (sx + 165, sy + 35), "“A CT scan of the liver”", 22, NAVY, True, "mm")
    if u > 0.95: d.line([1370, 395, 1430, 395], fill=GREEN, width=4); T(d, (1500, 480), "close together = match", 26, GREEN, True, "ma")
    if k >= 1: card(d, 120, 640, 520, 260, "CLIP (2021)", "Learned from millions of internet images and their captions")
    if k >= 2: card(d, 690, 640, 560, 260, "Med3DVLM (2026) — base paper", "Same idea for 3D scans, but needs 120,000 pairs and 8 big GPUs", WHITE, NAVY, 32)
    if k >= 3: card(d, 1300, 640, 500, 260, "Our project", "A small version of the same idea — runs on one computer", ORANGE_L, ORANGE)

def s_ourvlm(img, d, k, t):
    P(d, 120, 230, "Our modification: keep the same 3D CNN, but replace its final layer with sentence matching.", 34, NAVY, 1680)
    y1, y2, ym = 420, 720, 570
    img.paste(tile(0, 14, 140), (120, y1 - 70)); d.rectangle([120, y1 - 70, 260, y1 + 70], outline=NAVY, width=3)
    T(d, (190, y1 + 85), "CT volume", 24, NAVY, True, "ma")
    arrow(d, 270, y1, 320, y1); card(d, 320, y1 - 70, 260, 140, "3D CNN", "same as baseline")
    if k >= 1:
        box(d, 120, y2 - 80, 460, 160, ORANGE_L, ORANGE); T(d, (350, y2 - 55), "11 organ sentences", 30, ORANGE, True, "ma")
        for i, s in enumerate(["“A CT scan of the liver.”", "“A CT scan of the heart.” …"]): T(d, (350, y2 - 5 + i * 40), s, 26, NAVY, False, "ma")
    if k >= 2: arrow(d, 580, y2, 640, y2, ORANGE); card(d, 640, y2 - 70, 330, 140, "ClinicalBERT", "text → 768 numbers (frozen)", ORANGE_L, ORANGE, 32, 24)
    if k >= 3: arrow(d, 580, y1, 640, y1, TEAL); card(d, 640, y1 - 70, 330, 140, "Projection", "256 → 768 numbers", TEAL_L, TEAL, 34, 26)
    if k >= 4:
        card(d, 1100, ym - 90, 330, 180, "Similarity", "compare scan with all 11 sentences", TEAL_L, TEAL, 36, 26, width=4)
        arrow(d, 970, y1, 1100, ym - 40, TEAL); arrow(d, 970, y2, 1100, ym + 40, ORANGE); arrow(d, 1430, ym, 1500, ym)
        card(d, 1500, ym - 90, 300, 180, "Answer", "best-matching sentence: Liver", WHITE, NAVY, 38, 26)
        for j in range(3):
            ph = (t * 0.5 + j / 3) % 1
            if t > 900: continue
            if ph < 0.5: u = ph / 0.5; dot(d, lerp(970, 1100, u), lerp(y1, ym - 40, u), 8, TEAL); dot(d, lerp(970, 1100, u), lerp(y2, ym + 40, u), 8, ORANGE)
            else: u = (ph - 0.5) / 0.5; dot(d, lerp(1430, 1500, u), ym, 8, NAVY)

def s_cos(img, d, k, t):
    P(d, 120, 240, "Think of each list of 768 numbers as a vector — an arrow pointing in some direction.", 36, NAVY, 820)
    ox, oy = 1150, 860
    for ang, name, col in [(47, "liver", GREEN), (78, "heart", GREY), (12, "spleen", GREY)]:
        a = math.radians(ang); arrow(d, ox, oy, ox + 560 * math.cos(a), oy - 560 * math.sin(a), col, 5, 24)
        T(d, (ox + 560 * math.cos(a) - 15, oy - 560 * math.sin(a) - 25), f"“{name}”", 30, col, True, "rm" if name == "liver" else "mm")
    sa = math.radians(lerp(25, 44, ease(t, 0.5, 2.5)) if k == 2 else (25 if k < 2 else 44))
    arrow(d, ox, oy, ox + 520 * math.cos(sa), oy - 520 * math.sin(sa), TEAL, 9, 32); T(d, (ox + 545 * math.cos(sa) + 10, oy - 545 * math.sin(sa)), "Scan", 36, TEAL, True, "lm")
    if k >= 1:
        box(d, 120, 400, 820, 230, WHITE, LINE)
        T(d, (150, 425), "cos θ = (a · b) / (|a| |b|)", 40, NAVY, True)
        P(d, 150, 490, "This is just normalised correlation, as in communication systems: 1 = perfectly correlated, 0 = unrelated.", 30, NAVY, 760)
    if k >= 2:
        box(d, 120, 670, 820, 190, TEAL_L, TEAL)
        P(d, 150, 700, "The model picks the organ whose sentence vector is most correlated with the scan vector → here, liver.", 32, NAVY, 760, bold=True)

def s_siglip(img, d, k, t):
    P(d, 120, 240, "Training uses the SigLIP loss, taken from the base paper Med3DVLM.", 38, NAVY, 900)
    cx, cy = 1420, 640
    img.paste(tile(0, 14, 120), (cx - 60, cy - 60)); d.rectangle([cx - 60, cy - 60, cx + 60, cy + 60], outline=NAVY, width=3)
    others = ["heart", "spleen", "bladder", "pancreas", "right lung", "left lung", "left kidney", "right kidney", "right femur", "left femur"]
    if k >= 1:
        u = (0.5 - 0.5 * math.cos(min(t, 6) * 1.2)) if k == 1 else 1
        u = ease(t, 0.3, 3.0) if k == 1 else 1
        for i, n in enumerate(others):
            a = math.radians(-50 + i * 28); r = lerp(230, 330, u)
            x, y = cx + r * 1.03 * math.cos(a), cy + r * math.sin(a)
            arrow(d, cx + 95 * math.cos(a), cy + 95 * math.sin(a), x - 60 * math.cos(a), y - 60 * math.sin(a), RED, 4, 18); T(d, (x, y), n, 24, RED, True, "mm")
        a = math.radians(-112); r = lerp(330, 150, u); x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        d.line([x, y + 20, cx + 62 * math.cos(a), cy + 62 * math.sin(a)], fill=GREEN, width=7); T(d, (x, y), "liver ✓", 36, GREEN, True, "mm")
        P(d, 120, 400, "For each scan: pull it towards its correct sentence (green) and push it away from the other 10 (red).", 34, NAVY, 820)
    if k >= 2:
        box(d, 120, 640, 820, 190, TEAL_L, TEAL)
        P(d, 150, 670, "After many epochs, every scan sits closest to the sentence of its own organ.", 34, NAVY, 760, bold=True)

def s_problem(img, d, k, t):
    def fan(x, y, angles, col, L=280):
        for a in angles: a = math.radians(a); arrow(d, x, y, x + L * math.cos(a), y - L * math.sin(a), col, 4, 16)
    tight = [60 + (i - 5) * 1.2 for i in range(11)]; spread = [i * 360 / 11 + 8 for i in range(11)]
    box(d, 120, 250, 820, 640, WHITE, LINE); T(d, (530, 285), "Before: sentences look the same", 32, RED, True, "ma")
    fan(530, 720, tight, RED); T(d, (530, 790), "similarity between sentences = 0.985", 28, NAVY, True, "ma")
    if k >= 1: T(d, (530, 835), "accuracy only 66.6%", 30, RED, True, "ma")
    if k == 2 or k == 3:
        # DC-offset analogy
        box(d, 980, 250, 820, 640, WHITE, ORANGE, width=4); T(d, (1390, 285), "Like a signal with a big DC offset", 32, ORANGE, True, "ma")
        u = ease(t, 2.0, 2.0) if k == 2 else 1
        for j, col in enumerate([NAVY, TEAL, ORANGE]):
            base = lerp(560, 560, u); dc = lerp(170, 0, u)
            pts = [(1040 + i * 14, 640 - dc - 30 * math.sin(i * 0.25 + j * 2.1) - j * 6) for i in range(50)]
            d.line(pts, fill=col, width=4)
        d.line([1030, 640, 1760, 640], fill=GREY, width=2); T(d, (1765, 640), "0", 24, GREY, False, "lm")
        T(d, (1390, 760), "remove the mean (DC)" if u > 0.5 else "common part hides the differences", 28, NAVY, True, "ma")
        T(d, (1390, 800), "→ the differences become visible" if u > 0.5 else "", 28, GREEN, True, "ma")
    if k >= 4:
        box(d, 980, 250, 820, 640, WHITE, TEAL, width=4); T(d, (1390, 285), "After centring: clearly different", 32, GREEN, True, "ma")
        u = ease(t, 0.3, 2.2) if k == 4 else 1
        fan(1390, 555, [lerp(a, b, u) for a, b in zip(tight, spread)], mix(RED, GREEN, u), 190)
        if u > 0.9: T(d, (1390, 790), "similarity = −0.10 (distinct)", 28, NAVY, True, "ma"); T(d, (1390, 835), "accuracy rose to 73.3%", 30, GREEN, True, "ma")
    if k >= 5:
        box(d, 120, 920, 1680, 80, ORANGE_L, ORANGE); T(d, (960, 960), "Key takeaway: the model was actually built, tested and improved", 30, ORANGE, True, "mm")

def s_results(img, d, k, t):
    P(d, 120, 230, "Final results — 50 epochs, tested on 610 volumes the models never saw", 40, NAVY, 1680, bold=True)
    base, bh = 820, 440
    if k >= 1:
        u = ease(t, 0.2, 1.5) if k == 1 else 1
        for i, (name, acc, col) in enumerate([("Baseline 3D CNN", 80.5, NAVY), ("Vision-Language Model", 80.0, ORANGE)]):
            x = 200 + i * 360; h = bh * acc / 100 * u
            d.rectangle([x, base - h, x + 240, base], fill=col)
            T(d, (x + 120, base - h - 60), f"{acc * u:.1f}%", 48, col, True, "ma"); T(d, (x + 120, base + 20), name, 28, NAVY, True, "ma")
        d.line([160, base, 940, base], fill=GREY, width=3); T(d, (160, 360), "Test accuracy", 30, GREY, True)
    if k >= 2:
        card(d, 1060, 340, 360, 200, "AUC 0.981", "baseline", WHITE, LINE, 44); card(d, 1460, 340, 360, 200, "AUC 0.979", "vision-language model", WHITE, LINE, 44)
        T(d, (1440, 560), "AUC: 0.5 = guessing, 1.0 = perfect (like an ROC curve in detection theory)", 22, GREY, False, "ma")
    if k >= 3: card(d, 1060, 620, 760, 170, "≈ 7 minutes each", "trained on an ordinary computer, no graphics card", TEAL_L, TEAL, 44)
    if k >= 4:
        box(d, 120, 900, 1680, 100, ORANGE_L, ORANGE)
        T(d, (960, 950), "Finding: the VLM is as accurate as the standard model — and it also understands organ names", 29, NAVY, True, "mm")

CM = np.array([[66,0,0,0,0,0,0,0,0,3,0],[0,49,13,1,0,1,0,0,0,3,1],[0,24,39,2,0,0,0,0,0,4,0],[0,1,0,50,13,0,0,0,0,0,1],
               [0,1,0,35,29,0,0,0,0,0,0],[2,0,0,1,2,60,0,0,0,1,0],[2,0,0,0,0,0,25,0,0,1,0],[0,0,0,0,0,0,0,21,0,0,0],
               [0,0,0,0,0,0,0,3,18,0,0],[0,4,0,0,0,0,0,0,0,65,0],[0,2,0,0,0,1,0,0,0,0,66]])
def s_confusion(img, d, k, t):
    P(d, 120, 230, "A confusion matrix shows where the model makes mistakes. Rows = true organ, columns = model's answer.", 32, NAVY, 760)
    short = ["Liver", "R kidney", "L kidney", "R femur", "L femur", "Bladder", "Heart", "R lung", "L lung", "Spleen", "Pancreas"]
    x0, y0, c = 1100, 300, 58; rown = CM / CM.sum(1, keepdims=True)
    shown = int(min(11, t * 3)) if k == 0 else 11
    for i in range(11):
        T(d, (x0 - 12, y0 + i * c + c / 2), short[i], 20, NAVY, True, "rm")
        if i >= shown: continue
        for j in range(11):
            v = rown[i, j]; col = tuple(int(255 - (255 - q) * v) for q in TEAL) if i == j else tuple(int(255 - (255 - q) * min(1, v * 2)) for q in RED)
            d.rectangle([x0 + j * c, y0 + i * c, x0 + (j + 1) * c - 2, y0 + (i + 1) * c - 2], fill=col)
            if CM[i, j]: T(d, (x0 + j * c + c / 2, y0 + i * c + c / 2), str(CM[i, j]), 20, WHITE if v > 0.45 else NAVY, True, "mm")
    for j in range(11): T(d, (x0 + j * c + c / 2, y0 - 14), short[j].replace(" ", "")[:4], 16, NAVY, True, "ms")
    T(d, (x0 + 11 * c / 2, y0 + 11 * c + 20), "Diagonal (teal) = correct answers", 24, TEAL, True, "ma")
    if k >= 1:
        pulse = 3 + int(3 * (1 + math.sin(t * 5))) if k == 1 else 5
        for (i, j) in [(2, 1), (1, 2), (4, 3), (3, 4)]:
            d.rectangle([x0 + j * c - 3, y0 + i * c - 3, x0 + (j + 1) * c + 1, y0 + (i + 1) * c + 1], outline=ORANGE, width=pulse)
        box(d, 120, 520, 760, 330, ORANGE_L, ORANGE)
        P(d, 150, 550, "Most mistakes: left vs right kidney, and left vs right femur.", 34, NAVY, 700, bold=True)
        P(d, 150, 680, "These pairs are mirror images and look almost the same — an understandable mistake.", 30, NAVY, 700)

def s_papers(img, d, k, t):
    for i, (tag, name, venue, txt) in enumerate([
        ("BASE PAPER 1", "Med3DVLM", "IEEE Journal of Biomedical and Health Informatics, 2026", "Showed that matching 3D scans with text works very well. We use its image-text matching idea and its SigLIP loss."),
        ("BASE PAPER 2", "Diagnostic Captioning", "IEEE Trans. Pattern Analysis & Machine Intelligence, 2025", "Showed that aligning image features with text helps models catch fine medical details. Supports why our modification is useful.")]):
        if k >= i + 1:
            x = 120 + i * 860; box(d, x, 260, 820, 600, WHITE, NAVY)
            T(d, (x + 40, 295), tag, 26, TEAL, True); P(d, x + 40, 340, name, 44, NAVY, 740, bold=True)
            P(d, x + 40, 400, venue, 28, GREY, 740); P(d, x + 40, 490, txt, 32, NAVY, 740)
    if k == 0: T(d, (960, 560), "Two base papers support this project", 48, NAVY, True, "mm")

def s_present(img, d, k, t):
    items = [("Problem & objective", "Recognise organs in 3D CT with a simple VLM"), ("Dataset", "OrganMNIST3D: 1,742 samples, 11 classes"),
             ("Methodology", "Baseline 3D CNN (convolution = learned filters)"), ("Modification", "Vision-language model: scan ↔ organ sentences"),
             ("Results", "80.5% vs 80.0%, AUC ≈ 0.98, confusion matrix"), ("Literature", "Med3DVLM and Diagnostic Captioning"), ("Conclusion", "Small VLM works as well as the standard model")]
    for i, (a, b) in enumerate(items[:[2, 4, 7][min(k, 2)]]):
        x, y = 120 + (i % 2) * 860, 260 + (i // 2) * 175; box(d, x, y, 820, 145)
        d.ellipse([x + 25, y + 37, x + 95, y + 107], fill=TEAL); T(d, (x + 60, y + 72), str(i + 1), 36, WHITE, True, "mm")
        T(d, (x + 120, y + 25), a, 34, NAVY, True); P(d, x + 120, y + 78, b, 26, GREY, 680)

def s_qa(img, d, k, t):
    qa = [("Why not use Med3DVLM directly?", "It needs 120,000 scans and 8 big GPUs. We use the same idea at a size one computer can train."),
          ("Why a VLM if accuracy is the same?", "It also learns the meaning of words, and can later use new descriptions such as full radiology reports."),
          ("What is a convolution layer?", "A bank of small 3D filters, like FIR filters — but their coefficients are learned from data."),
          ("What problem was found while building it?", "The organ sentences looked almost identical. Removing their mean (like removing DC offset) fixed it.")]
    for i, (q, a) in enumerate(qa):
        if k >= i:
            y = 250 + i * 190; box(d, 120, y, 1680, 170)
            T(d, (160, y + 25), "Q: " + q, 32, TEAL, True); P(d, 160, y + 82, "A: " + a, 28, NAVY, 1600)

def s_summary(img, d, k, t):
    for i, (a, b) in enumerate([("Objective", "Recognise organs in 3D CT scans using a simple vision-language model"),
                                ("Data & method", "1,742 volumes, 11 classes. 3D CNN baseline, modified to match scans with organ sentences"),
                                ("Result", "Both reach ≈ 80% accuracy — the small VLM works as well as the standard model")]):
        if k >= i: card(d, 120 + i * 570, 300, 530, 460, a, b, [TEAL_L, WHITE, ORANGE_L][i], [TEAL, LINE, ORANGE][i], 42, 32)
    if k >= 2: T(d, (960, 880), "Thank you", 44, TEAL, True, "mm")

A = "all"
SCENES = [
 ("Welcome", "", s_title, {0: A, 1: A}, [
  "Design of a Vision-Language Model for 3D Medical Image Analysis. The goal of this project is to make a computer look at a 3D CT scan of the body, and recognise which organ it shows.",
  "We will go step by step: the data, the core idea of convolution, the baseline model, the improved vision-language model, and finally the results."]),
 ("1 · The big picture", "The project in one sentence", s_onesentence, {}, [
  "Here is the whole project in one sentence. We teach a computer to look at a 3D CT scan, and tell us which organ it shows.",
  "We build this in two ways. The first is a standard deep learning model. We call it the baseline.",
  "The second is an improved model that also understands words. This is called a vision-language model, and it is our modification.",
  "Finally, we compare the two models, to see how well each one recognises the organs."]),
 ("2 · The data", "What is a CT scan?", s_ct, {0: 4.2}, [
  "First, what is a CT scan? A CT machine takes many X-ray images of the body, slice by slice, just like slicing a loaf of bread.",
  "When we stack all these slices together, we get a 3D picture of the inside of the body. This 3D picture is called a volume.",
  "In our project, each volume is a cube of 28 by 28 by 28 points, called voxels. For an engineer, it is simply a three-dimensional array of samples, just like a sampled signal, but in three dimensions."]),
 ("2 · The data", "Our dataset: OrganMNIST3D", s_dataset, {2: 4.5}, [
  "We use a free, public dataset called OrganMNIST3D. It is part of a well-known benchmark called MedMNIST, published in the journal Scientific Data in 2023.",
  "It contains 1,742 small CT volumes, cut out from 201 real abdominal CT scans of patients.",
  "Each volume shows one of 11 organs. So, the number of classes is 11. They are: liver, right kidney, left kidney, right femur, left femur, bladder, heart, right lung, left lung, spleen and pancreas. These are real images from the dataset."]),
 ("2 · The data", "Splitting the data", s_split, {1: 1.2, 2: 1.2, 3: 1.2}, [
  "The volumes are divided into three groups, just like a student preparing for an exam.",
  "971 volumes are used for training. This is like the textbook. The model learns from these.",
  "161 volumes are used for validation. This is like a practice test. We use it to pick the best version of the model.",
  "And 610 volumes are used for testing. This is the final exam. The model never sees these while learning, so the test score is fair and honest."]),
 ("3 · Machine learning", "What is machine learning?", s_ml, {2: A}, [
  "Now, what is machine learning? In classic signal processing, an engineer designs a filter by hand, choosing its coefficients, and the filter turns an input signal into an output.",
  "In machine learning, we do not design the filter. Instead, we give the computer many examples together with the correct answers, and the computer learns the filter coefficients by itself.",
  "This is the same idea as an adaptive filter, such as the LMS algorithm. The model looks at an example, makes a guess, measures the error against the correct answer, and adjusts its coefficients a little to reduce that error. It repeats this loop thousands of times."]),
 ("3 · Machine learning", "Convolution — the core operation", s_conv, {0: 6.5, 1: A}, [
  "The main building block of the model is convolution, the same operation used in signals and systems. A small kernel slides along the input signal. At every position, we multiply and add. Here the kernel is minus one, zero, one, an edge detector, so every edge in the input creates a spike in the output.",
  "Exactly the same operation works on images. Here, a small 3 by 3 window slides over a real CT slice, and at every position, it multiplies and adds the pixel values.",
  "A convolutional neural network, or CNN, does this in three dimensions, using 3 by 3 by 3 kernels. The big difference from classic filtering is this: the kernel coefficients are not designed by hand. The network learns them from the training data."]),
 ("4 · The baseline model", "Model 1: the 3D CNN (baseline)", s_cnn, {1: A, 3: A}, [
  "Our first model is a 3D convolutional neural network, or 3D CNN. This is the standard method, so we call it the baseline.",
  "The CT volume passes through four blocks. Each block is a bank of learned 3D filters, 32, then 64, 128 and 256 of them, followed by a simple non-linear step and down-sampling, which is similar to decimation.",
  "The early blocks find simple things like edges. The later blocks combine them into shapes, and then into parts of organs. At the end, the whole scan is summarised as 256 numbers.",
  "A final layer turns these 256 numbers into 11 scores, one for each organ, and the highest score is the answer. In total, this model has about 1.17 million adjustable numbers, called parameters, and all of them are learned from data."]),
 ("4 · The baseline model", "How the model learns", s_train, {2: 5.5}, [
  "During training, the model's guess is compared with the correct organ. The difference is measured by the loss, here called cross-entropy. The loss works like an error signal.",
  "The Adam optimiser then updates the parameters to reduce the error, with a step size, called the learning rate, of 0.001. We show 32 scans at a time, flip them randomly for variety, and repeat for 50 epochs. One epoch is one full pass over all 971 training volumes.",
  "This graph is from our real training run. It shows the accuracy on the practice test after each epoch. Both models start low, and quickly climb to around 85 to 90 percent.",
  "After every epoch, we keep the version of the model that scored best on the practice test."]),
 ("5 · The modification", "The limitation of the baseline", s_limit, {}, [
  "But the baseline has one limitation. It only learns class numbers. To the model, the liver is just number zero. It never learns what the word liver actually means.",
  "This is exactly where vision-language models help."]),
 ("5 · The modification", "What is a vision-language model?", s_vlm, {0: 2.5}, [
  "A vision-language model learns from pictures and words together. It places pictures and sentences in the same space, so that a picture moves close to the sentence that describes it.",
  "A famous example is CLIP, from 2021, which learned from millions of internet images and their captions.",
  "The first base paper, Med3DVLM, published in 2026, applies this idea to 3D medical scans. It is very powerful, but it needs 120,000 scan and report pairs, and eight large graphics cards. That is too heavy for a mini project.",
  "So, our project builds a small and simple version of the same idea, which can be trained on one ordinary computer."]),
 ("5 · The modification", "Model 2: our vision-language model", s_ourvlm, {4: A}, [
  "Here is our modification. We keep the same 3D CNN, but we replace its final layer with sentence matching.",
  "First, we write one simple sentence for each organ. For example: A CT scan of the liver. This gives us 11 sentences.",
  "A ready-made language model called ClinicalBERT, which was trained on hospital notes, turns each sentence into a vector of 768 numbers. We do not retrain it. It stays frozen.",
  "On the image side, a small projection layer turns the scan's 256 numbers into 768 numbers too, so that the scan and the sentences can be compared directly.",
  "Finally, we measure how similar the scan is to each of the 11 sentences. The most similar sentence gives the answer."]),
 ("5 · The modification", "How similarity works", s_cos, {2: 3.2}, [
  "How do we measure similarity? Think of each list of 768 numbers as a vector, an arrow pointing in some direction.",
  "We use cosine similarity. For an engineer, this is simply normalised correlation, exactly like the correlation between two signals in communication systems. A value of one means perfectly correlated, and zero means unrelated.",
  "The model picks the organ whose sentence vector is most correlated with the scan vector. During training, the scan vector is rotated towards the correct sentence, here, the liver."]),
 ("5 · The modification", "How the new model learns", s_siglip, {1: 3.5}, [
  "To train this model, we use the SigLIP loss, which we take from the base paper, Med3DVLM.",
  "For every scan, it pulls the scan towards its correct sentence, shown in green, and pushes it away from the other ten sentences, shown in red.",
  "After many epochs, every scan ends up closest to the sentence of its own organ."]),
 ("6 · Building it", "A problem we found, and the fix", s_problem, {2: 4.5, 4: 2.8}, [
  "When we actually built this model, we found a problem. The 11 sentences differ by only one word, so ClinicalBERT gave them almost identical vectors. Their similarity was 0.985, very close to a perfect one.",
  "Because all the sentences looked the same, the model got confused, and its accuracy was only 66.6 percent.",
  "In signal processing terms, this is like several signals riding on a large DC offset. The big common part hides the small differences between them. The fix is the same: remove the mean.",
  "So we subtract the average of the 11 sentence vectors from each one. In machine learning, this is called centring.",
  "After centring, the sentence vectors point in clearly different directions, and the accuracy rose to 73.3 percent in the same short test.",
  "This is a key takeaway. The model was not only designed on paper. It was actually built, tested, and improved."]),
 ("7 · The results", "Final results", s_results, {1: 2.0}, [
  "Now, the final results. Both models were trained for 50 epochs, and tested on the 610 test volumes that they had never seen before.",
  "The baseline 3D CNN reached 80.5 percent accuracy. Our vision-language model reached 80.0 percent. They are practically equal.",
  "We also measured the AUC, the area under the ROC curve, a standard measure from detection theory. 0.5 means guessing, and 1 means perfect. The scores were 0.981 and 0.979. Both are excellent.",
  "Each model trained in only about 7 minutes, on an ordinary computer without a graphics card.",
  "So the main finding is this: the small vision-language model matches the standard model's accuracy, while also learning the meaning of the organ names."]),
 ("7 · The results", "Where does the model make mistakes?", s_confusion, {0: 4.0, 1: A}, [
  "This table is called a confusion matrix. It shows where the model makes mistakes. The rows are the true organs, and the columns are the model's answers. The teal diagonal shows the correct answers.",
  "Most mistakes are between the left and right kidney, and between the left and right femur. These pairs are mirror images, and look almost the same in small scans. So this mistake is very understandable."]),
 ("8 · Literature", "The two base papers", s_papers, {}, [
  "The project builds on two base papers.",
  "Base paper one is Med3DVLM, in the IEEE Journal of Biomedical and Health Informatics, 2026. It showed that matching 3D scans with text works very well. We use its image-text matching idea, and its SigLIP loss.",
  "Base paper two is Diagnostic Captioning, in IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. It showed that aligning image features with text helps a model catch fine medical details. It supports why our modification is useful."]),
 ("9 · Presenting", "Presentation outline", s_present, {}, [
  "The project can be presented in seven parts. One: the problem and the objective. Two: the dataset, with 1,742 samples and 11 classes.",
  "Three: the methodology, which is the baseline 3D CNN. Four: the modification, which is the vision-language model.",
  "Five: the results, with the confusion matrix. Six: the literature, meaning the two base papers. And seven: the conclusion."]),
 ("9 · Presenting", "Common questions", s_qa, {}, [
  "Question: why not use Med3DVLM directly? Answer: it needs 120,000 scans and eight large graphics cards. We use the same idea at a size that one computer can train.",
  "Question: why use a vision-language model if the accuracy is the same? Answer: it also learns the meaning of words, and later it can use new descriptions, such as full radiology reports.",
  "Question: what is a convolution layer? Answer: it is a bank of small 3D filters, like FIR filters, but their coefficients are learned from data instead of designed by hand.",
  "Question: what problem was found while building it? Answer: the organ sentences looked almost identical to the text model, and removing their mean, just like removing a DC offset, fixed it."]),
 ("Summary", "In short", s_summary, {}, [
  "To summarise. The objective is to recognise organs in 3D CT scans, using a simple vision-language model.",
  "The data is 1,742 CT volumes with 11 organ classes. The method is a 3D CNN baseline, modified to match each scan with organ sentences.",
  "And the result: both models reach about 80 percent accuracy. So the small vision-language model works as well as the standard model. Thank you for watching."]),
]

def render(si, k, t):
    sec, title, fn, _, _ = SCENES[si]
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    if title:
        T(d, (120, 80), sec.upper(), 28, TEAL, True); T(d, (120, 120), title, 58, NAVY, True); d.line([120, 205, 1800, 205], fill=LINE, width=3)
    if k >= 0: fn(img, d, k, t)
    if si:
        d.rectangle([0, H - 14, W, H], fill=LINE); d.rectangle([0, H - 14, int(W * (si + 1) / len(SCENES)), H], fill=TEAL)
    return img

if len(sys.argv) > 5 and sys.argv[5] == "preview":
    si, k, t = int(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])
    render(si, k, t).save(f"{OUT}/preview_{si}_{k}_{t}.png"); sys.exit(0)

# ---------- narration ----------
steps = [(si, k, txt) for si, sc in enumerate(SCENES) for k, txt in enumerate(sc[4])]
from kokoro_onnx import Kokoro
import soundfile as sf
KOK = Kokoro(os.environ["KOKORO_DIR"] + "/kokoro-v1.0.onnx", os.environ["KOKORO_DIR"] + "/voices-v1.0.bin")
def tts(job):
    si, k, txt = job; raw, wav = f"{OUT}/k_{si:02d}_{k}.wav", f"{OUT}/a_{si:02d}_{k}.wav"
    import hashlib; marker = f"{OUT}/k_{si:02d}_{k}." + hashlib.md5(txt.encode()).hexdigest()[:10]
    if not (os.path.exists(wav) and os.path.exists(marker)):
        samples, sr = KOK.create(txt.replace("Med3DVLM", "Med 3D V L M").replace("SigLIP", "Sig-lip").replace("ClinicalBERT", "Clinical BERT").replace("OrganMNIST3D", "Organ M-NIST 3D").replace("MedMNIST", "Med M-NIST").replace("3D", "3-D"), voice="af_heart", speed=1.0, lang="en-us")
        sf.write(raw, samples, sr)
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-ar", "44100", "-ac", "1", wav], check=True, capture_output=True)
        open(marker, "w").close()
    return wav
for j in steps: tts(j)

PAD, REVEAL, SCENE_FADE = 0.8, 0.7, 0.6
plan, audio = [], []
for si, k, _ in steps:
    with wave.open(f"{OUT}/a_{si:02d}_{k}.wav") as w:
        frames = w.readframes(w.getnframes()); rate = w.getframerate()
    lead = 0.5 if k == 0 else 0.15
    D = lead + len(frames) / 2 / rate + PAD; n = int(math.ceil(D * FPS))
    samples = int(n / FPS * rate)
    audio.append(b"\x00\x00" * int(lead * rate) + frames + b"\x00\x00" * max(0, samples - int(lead * rate) - len(frames) // 2))
    audio[-1] = audio[-1][:samples * 2]
    plan.append((si, k, n))
with wave.open(f"{OUT}/narration.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(b"".join(audio))
total = sum(n for _, _, n in plan); print("total minutes", round(total / FPS / 60, 2), flush=True)

# ---------- video ----------
proc = subprocess.Popen(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                         "-i", f"{OUT}/narration.wav", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", FINAL], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
prev_final = Image.new("RGB", (W, H), BG); done = 0
for si, k, n in plan:
    anim = SCENES[si][3].get(k, 0)
    final = render(si, k, 999.0)
    if k == 0:
        mask = None
    else:
        diff = ImageChops.difference(prev_final, final).convert("L").point(lambda v: 255 if v > 10 else 0)
        mask = diff.filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(3))
    still = final.tobytes()
    for f in range(n):
        t = f / FPS
        animating = anim == A or (anim and t < anim)
        cur = render(si, k, t) if animating else final
        rv = SCENE_FADE if k == 0 else REVEAL
        if t < rv:
            a = ease(t, 0, rv)
            if k == 0:
                frame = Image.blend(prev_final, cur, a)
            else:
                off = int((1 - a) * 28)
                shifted = Image.new("RGB", (W, H), BG); shifted.paste(cur, (0, off))
                m = Image.new("L", (W, H), 0); m.paste(mask, (0, off))
                mixed = Image.blend(prev_final, shifted, a)
                frame = Image.composite(mixed, cur, ImageChops.lighter(mask, m))
            proc.stdin.write(frame.tobytes())
        else:
            proc.stdin.write(cur.tobytes() if animating else still)
    prev_final = final; done += n
    print(f"scene {si} step {k} done ({done}/{total} frames)", flush=True)
proc.stdin.close(); proc.wait()
print("done", FINAL)
