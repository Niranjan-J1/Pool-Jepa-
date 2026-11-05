"""Polished 2-D renderer for demo clips (record_shot.py). Presentation only.

Perception never sees these frames: the encoder reads the plain 432x240 env.render() frames it
was trained on, and the clip shows exactly that input in a camera inset. This module draws the
same simulator state at 1920x1080:
  - felt with spotlight, cushion shadow, nap and vignette; wood rails with grain, bevel and
    sights; decorative corner pockets (the simulated table has none);
  - balls shaded as spheres with drop shadows; a robot sprite with an orange nose and striker;
    a fading trail behind the moving cue ball;
  - overlays: target crosshair, perceived-cue reticle, aim line, the fan of imagined shots
    (one straight ray per candidate force, from the cue ball to the landing point the shot
    model predicts; the model predicts landing points, not paths, and every candidate is fired
    along the same heading), stage labels that fade in, and a result card;
  - the camera inset: the encoder's input (camera frame minus the empty-table background,
    rescaled from [-1, 1] to grey levels) with the encoder's learned keypoints on it (cue-ball
    keypoints highlighted) and the decoded cue-ball position.
Layout: the band above the table holds the stage label / result card (left) and the inset
(right); nothing is drawn over the playing surface except the in-world overlays.
"""
import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from env.sim_env import (BALL_COLORS, BALL_RADIUS, CUSHION_THICKNESS, RENDER_HEIGHT, RENDER_MARGIN,
                         RENDER_WIDTH, ROBOT_LENGTH, ROBOT_WIDTH, TABLE_LENGTH, TABLE_WIDTH)

W, H = 1920, 1080
INNER_X = TABLE_LENGTH / 2 - CUSHION_THICKNESS   # cushion faces = where balls bounce
INNER_Y = TABLE_WIDTH / 2 - CUSHION_THICKNESS
CUSH = 0.012     # m of green cushion rubber drawn outside the face
RAIL = 0.052     # m from the face to the outer edge of the wood
SS = 3           # supersampling for sprites
BALL_NAMES = ["cue", "red", "blue", "yellow"]
TRAIL_FRAMES = 40
TABLE_TOP = 390  # px; the band above holds labels and the camera inset
GOLD = (255, 205, 60)
CYAN = (70, 225, 255)
MAGENTA = (255, 80, 200)
ENC_STRIDE = 8   # encoder feature map = image / 8 (three stride-2 convs)


def _font(size, bold=False):
    try:
        from matplotlib import font_manager
        path = font_manager.findfont(font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"))
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _opt(d, key, default):
    """d[key], or default when the key is missing OR its value is None (dict.get only covers
    a missing key, and the overlay dicts pass explicit Nones)."""
    v = d.get(key)
    return default if v is None else v


def _resize(a, size):
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.BICUBIC))


def lerp_state(a, b, t):
    th = a["robot"][2] + ((b["robot"][2] - a["robot"][2] + math.pi) % (2 * math.pi) - math.pi) * t
    rob = tuple(a["robot"][i] + (b["robot"][i] - a["robot"][i]) * t for i in range(2)) + (th,)
    return {"robot": rob, "balls": a["balls"] + (b["balls"] - a["balls"]) * t}


def interpolate(states, slowmo):
    """Physics states (one per 60 Hz step) -> (state, index of the nearest physics step) per
    output frame, `slowmo` frames per step, linearly interpolated between steps."""
    if slowmo <= 1.0 or len(states) < 2:
        return [(s, i) for i, s in enumerate(states)]
    out, n = [], len(states) - 1
    for j in range(int(n * slowmo) + 1):
        t = j / slowmo
        i = min(int(t), n - 1)
        out.append((lerp_state(states[i], states[i + 1], t - i), int(round(t))))
    return out


def world_to_cam(xy):
    """Metres -> pixel in the 432x240 sim camera frame."""
    hw = TABLE_LENGTH / 2 + RENDER_MARGIN
    hh = hw * RENDER_HEIGHT / RENDER_WIDTH
    return ((xy[0] + hw) / (2 * hw) * RENDER_WIDTH, (hh - xy[1]) / (2 * hh) * RENDER_HEIGHT)


def keypoint_to_cam(kp):
    """Spatial-softmax keypoint in [-1, 1] feature-map coordinates -> camera pixel."""
    fw, fh = RENDER_WIDTH // ENC_STRIDE, RENDER_HEIGHT // ENC_STRIDE
    return ((kp[0] + 1) / 2 * (fw - 1) * ENC_STRIDE + ENC_STRIDE / 2 - 0.5,
            (kp[1] + 1) / 2 * (fh - 1) * ENC_STRIDE + ENC_STRIDE / 2 - 0.5)


class DemoRenderer:
    def __init__(self, pockets=True, seed=0):
        self.S = min(1700 / (2 * (INNER_X + RAIL)), (1030 - TABLE_TOP) / (2 * (INNER_Y + RAIL)))
        self.cx, self.cy = W / 2, TABLE_TOP + (INNER_Y + RAIL) * self.S
        self.rng = np.random.default_rng(seed)
        self.backdrop, self.bg = self._background(pockets)
        self.balls = {n: self._ball_sprite(BALL_COLORS[n]) for n in BALL_NAMES}
        self.ball_shadow = self._shadow_sprite(BALL_RADIUS * 2, BALL_RADIUS * 2, round_frac=1.0)
        self.robot = self._robot_sprite()
        self.robot_shadow = self._shadow_sprite(ROBOT_LENGTH, ROBOT_WIDTH, round_frac=0.25)
        self.f_title, self.f_sub = _font(54, True), _font(26)
        self.f_small, self.f_cap = _font(22), _font(26, True)
        self.f_huge, self.f_card = _font(84, True), _font(34)
        self.reset()

    def reset(self):
        """Forget the trail and the current stage (call between shots)."""
        self.trail = deque()
        self.stage, self.stage_start, self.n = None, 0, 0

    # ---------------------------------------------------------------- geometry
    def px(self, xy):
        return self.cx + float(xy[0]) * self.S, self.cy - float(xy[1]) * self.S

    def m(self, metres):
        return metres * self.S

    # ---------------------------------------------------------------- static art
    def _background(self, pockets):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        wx, wy = (xx - self.cx) / self.S, (self.cy - yy) / self.S
        ax, ay = np.abs(wx), np.abs(wy)

        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        img = np.array([30, 34, 41], np.float32) * (1.08 - 0.5 * np.clip(r / 1.4, 0, 1))[..., None]
        backdrop = Image.fromarray(np.clip(img * (1 - 0.28 * np.clip(r - 0.55, 0, 1))[..., None], 0, 255)
                                   .astype(np.uint8), "RGB").convert("RGBA")

        bx, by, rc = INNER_X + RAIL, INNER_Y + RAIL, 0.03

        def rounded(px_, py_, hx, hy, rad):
            qx, qy = np.maximum(px_ - (hx - rad), 0), np.maximum(py_ - (hy - rad), 0)
            return np.hypot(qx, qy) - rad   # <= 0 inside

        sh = Image.fromarray((rounded(np.abs(wx - 0.006), np.abs(wy + 0.012), bx, by, rc) <= 0)
                             .astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(28))
        img *= (1 - 0.6 * np.asarray(sh, np.float32) / 255)[..., None]

        outer = rounded(ax, ay, bx, by, rc) <= 0
        felt = (ax <= INNER_X) & (ay <= INNER_Y)
        cush = (ax <= INNER_X + CUSH) & (ay <= INNER_Y + CUSH) & ~felt
        wood = outer & ~felt & ~cush

        # wood: grain along each rail, bevel highlight outside, shadow toward the cushion
        g_h = _resize(self.rng.random((H // 3, 18)), (W, H))
        g_v = _resize(self.rng.random((18, W // 3)), (W, H))
        horiz = (ay - INNER_Y) >= (ax - INNER_X)
        grain = np.where(horiz, g_h, g_v)
        fine = _resize(self.rng.random((H // 2, W // 2)), (W, H))
        wood_col = np.array([118, 66, 33], np.float32) * (0.72 + 0.28 * grain + 0.06 * fine)[..., None]
        d_out = -rounded(ax, ay, bx, by, rc)
        d_in = np.minimum(np.abs(ay - INNER_Y - CUSH) * horiz + 9 * ~horiz,
                          np.abs(ax - INNER_X - CUSH) * ~horiz + 9 * horiz)
        bevel = 1 + 0.35 * np.exp(-np.clip(d_out, 0, None) / 0.0025) - 0.35 * np.exp(-np.clip(d_in, 0, None) / 0.004)
        light = 1 + 0.07 * wy / (INNER_Y + RAIL)
        img = np.where(wood[..., None], wood_col * (bevel * light)[..., None], img)

        # cushions
        t = np.clip(np.maximum(ax - INNER_X, ay - INNER_Y) / CUSH, 0, 1)
        cush_col = np.array([18, 86, 50], np.float32) * (0.7 + 0.45 * np.sin(np.pi * (0.25 + 0.6 * t)))[..., None]
        img = np.where(cush[..., None], cush_col, img)

        # felt
        spot = np.exp(-((wx / 0.42) ** 2 + (wy / 0.24) ** 2))
        edge = np.minimum(INNER_X - ax, INNER_Y - ay)
        shade = (0.80 + 0.30 * spot) * (1 - 0.38 * np.exp(-np.clip(edge, 0, None) / 0.007))
        nap = 1 + 0.025 * self.rng.standard_normal((H, W)).astype(np.float32)
        felt_col = np.array([24, 112, 64], np.float32) * (shade * nap)[..., None]
        img = np.where(felt[..., None], felt_col, img)

        # sights
        mid = INNER_X + CUSH + (RAIL - CUSH) * 0.45
        midy = INNER_Y + CUSH + (RAIL - CUSH) * 0.45
        sights = [(INNER_X * k / 4, s * midy) for k in (-3, -2, -1, 1, 2, 3) for s in (-1, 1)] + \
                 [(s * mid, INNER_Y * k / 2) for k in (-1, 0, 1) for s in (-1, 1)]
        for sx, sy in sights:
            a = np.clip((0.0034 - np.hypot(wx - sx, wy - sy)) / 0.0008, 0, 1)
            img = img * (1 - a[..., None]) + np.array([236, 226, 200], np.float32) * a[..., None]

        # decorative corner pockets, centred in the cushion corner so they barely reach the felt
        if pockets:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    d = np.hypot(wx - sx * (INNER_X + 0.011), wy - sy * (INNER_Y + 0.011))
                    rim = np.clip((0.021 - d) / 0.0012, 0, 1)
                    img = img * (1 - rim[..., None]) + np.array([46, 30, 22], np.float32) * rim[..., None]
                    hole = np.clip((0.0165 - d) / 0.0012, 0, 1)
                    hole_col = np.array([8, 8, 9], np.float32) * (0.5 + 1.5 * np.clip(d / 0.0165, 0, 1))[..., None]
                    img = img * (1 - hole[..., None]) + hole_col * hole[..., None]

        img *= (1 - 0.28 * np.clip(r - 0.55, 0, 1))[..., None]
        return backdrop, Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    def _ball_sprite(self, color):
        r = self.m(BALL_RADIUS)
        n = int(math.ceil(2 * r + 4))
        big = n * SS
        yy, xx = np.mgrid[0:big, 0:big].astype(np.float32) + 0.5
        R, c = r * SS, big / 2
        dx, dy = (xx - c) / R, (yy - c) / R
        d2 = dx * dx + dy * dy
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        L = np.array([-0.42, -0.58, 0.70]); L /= np.linalg.norm(L)
        Hv = L + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
        diff = np.clip(dx * L[0] + dy * L[1] + nz * L[2], 0, 1)
        spec = np.clip(dx * Hv[0] + dy * Hv[1] + nz * Hv[2], 0, 1) ** 70
        bounce = np.clip(dy, 0, 1) * 0.12
        base = np.array(color, np.float32) * 255
        rgb = base * (0.22 + 0.86 * diff + bounce)[..., None] * (0.82 + 0.18 * nz)[..., None]
        rgb = rgb + 255 * 0.9 * spec[..., None]
        alpha = np.clip((1 - np.sqrt(d2)) * R, 0, 1) * 255
        rgba = np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA").resize((n, n), Image.LANCZOS)

    def _shadow_sprite(self, lx, ly, round_frac):
        pad = 24
        w, h = int(self.m(lx)) + 2 * pad, int(self.m(ly)) + 2 * pad
        img = Image.new("L", (w, h), 0)
        rad = round_frac * min(self.m(lx), self.m(ly)) / 2
        ImageDraw.Draw(img).rounded_rectangle([pad, pad, w - pad, h - pad], radius=rad, fill=120)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.putalpha(img.filter(ImageFilter.GaussianBlur(7)))
        return out

    def _robot_sprite(self):
        u = self.S * SS                       # px per m, supersampled
        L, Wd = ROBOT_LENGTH * u, ROBOT_WIDTH * u
        pad = int(0.006 * u)
        img = Image.new("RGBA", (int(L) + 2 * pad, int(Wd) + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, y0 = pad, pad
        for side in (0, 1):                   # wheels, inside the footprint
            wy0 = y0 + (0 if side == 0 else Wd - 0.011 * u)
            d.rounded_rectangle([x0 + 0.018 * u, wy0, x0 + L - 0.018 * u, wy0 + 0.011 * u],
                                radius=0.003 * u, fill=(22, 22, 25, 255))
            for k in range(9):
                tx = x0 + 0.020 * u + k * (L - 0.040 * u) / 8
                d.line([tx, wy0 + 0.001 * u, tx, wy0 + 0.010 * u], fill=(48, 48, 54, 255), width=max(1, int(0.0008 * u)))
        d.rounded_rectangle([x0, y0 + 0.008 * u, x0 + L, y0 + Wd - 0.008 * u], radius=0.010 * u,
                            fill=(46, 50, 58, 255), outline=(20, 22, 26, 255), width=int(0.0012 * u))
        d.rounded_rectangle([x0 + 0.008 * u, y0 + 0.013 * u, x0 + L - 0.016 * u, y0 + Wd - 0.013 * u],
                            radius=0.007 * u, fill=(84, 92, 106, 255))
        d.rounded_rectangle([x0 + 0.010 * u, y0 + 0.015 * u, x0 + L - 0.018 * u, y0 + 0.020 * u],
                            radius=0.002 * u, fill=(120, 130, 146, 255))
        mx, my, ms = x0 + 0.014 * u, y0 + Wd / 2 - 0.009 * u, 0.018 * u     # tracking marker
        d.rectangle([mx, my, mx + ms, my + ms], fill=(245, 245, 245, 255))
        cell = ms / 4
        d.rectangle([mx + cell * 0.5, my + cell * 0.5, mx + ms - cell * 0.5, my + ms - cell * 0.5], fill=(20, 20, 20, 255))
        for i, j in ((1, 1), (2, 2)):
            d.rectangle([mx + i * cell, my + j * cell, mx + (i + 1) * cell - 1, my + (j + 1) * cell - 1], fill=(245, 245, 245, 255))
        nx, cy = x0 + L - 0.014 * u, y0 + Wd / 2                              # nose + striker
        d.polygon([(nx - 0.016 * u, cy - 0.012 * u), (nx, cy), (nx - 0.016 * u, cy + 0.012 * u),
                   (nx - 0.010 * u, cy)], fill=(255, 140, 36, 255))
        d.rectangle([x0 + L - 0.010 * u, cy - 0.004 * u, x0 + L + 0.001 * u, cy + 0.004 * u],
                    fill=(196, 200, 208, 255), outline=(90, 94, 102, 255))
        return img.resize((img.width // SS, img.height // SS), Image.LANCZOS)

    # ---------------------------------------------------------------- helpers
    def _paste_center(self, canvas, sprite, xy, dx=0, dy=0):
        x, y = self.px(xy)
        canvas.alpha_composite(sprite, (int(round(x - sprite.width / 2 + dx)), int(round(y - sprite.height / 2 + dy))))

    @staticmethod
    def _rot(sprite, theta):
        return sprite.rotate(math.degrees(theta), resample=Image.BICUBIC, expand=True)

    @staticmethod
    def _dashed(d, a, b, fill, width, dash=14, gap=10):
        L = math.dist(a, b)
        if L < 1:
            return
        ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
        t = 0.0
        while t < L:
            e = min(t + dash, L)
            d.line([(a[0] + ux * t, a[1] + uy * t), (a[0] + ux * e, a[1] + uy * e)], fill=fill, width=width)
            t = e + gap

    @staticmethod
    def cost_color(cost, alpha=235):
        t = float(np.clip(cost / 0.08, 0, 1))      # 0 cm green -> 4 cm yellow -> >= 8 cm red
        if t < 0.5:
            k = t / 0.5
            c = (60 + 190 * k, 220 - 10 * k, 120 - 60 * k)
        else:
            k = (t - 0.5) / 0.5
            c = (250 - 10 * k, 210 - 140 * k, 60)
        return tuple(int(v) for v in c) + (int(alpha),)

    @staticmethod
    def _panel(d, box, alpha=1.0):
        d.rounded_rectangle(box, radius=14, fill=(12, 14, 18, int(180 * alpha)),
                            outline=(255, 255, 255, int(40 * alpha)), width=2)

    def _inset(self, top, d, ins):
        """Camera inset (top right): the encoder input with its keypoints."""
        cam = Image.fromarray(ins["image"]).convert("RGBA")          # 432x240, encoder resolution
        scale = 1.0
        kd = ImageDraw.Draw(cam)
        for k, kp in enumerate(ins["kps"]):
            x, y = keypoint_to_cam(kp)
            if ins["cue_sel"][k]:
                kd.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=CYAN + (255,), outline=(0, 0, 0, 255))
            else:
                kd.ellipse([x - 2.2, y - 2.2, x + 2.2, y + 2.2], fill=(255, 255, 255, 150))
        if ins.get("cue") is not None:
            x, y = world_to_cam(ins["cue"])
            kd.ellipse([x - 10, y - 10, x + 10, y + 10], outline=CYAN + (255,), width=2)
        iw, ih = int(cam.width * scale), int(cam.height * scale)
        cap1 = "what the AI sees (raw camera)"
        cap2 = "camera minus the empty table = encoder input"
        cap3 = "dots: learned keypoints  ·  cyan: cue ball"
        bw = max(iw, d.textlength(cap2, font=self.f_small), d.textlength(cap3, font=self.f_small))
        x0, y0 = W - bw - 40, 26
        d.rounded_rectangle([x0 - 12, y0 - 12, x0 + bw + 12, y0 + ih + 104], radius=12, fill=(12, 14, 18, 215),
                            outline=(255, 255, 255, 40), width=2)
        top.alpha_composite(cam, (int(x0 + (bw - iw) / 2), y0))
        d.ellipse([x0, y0 + ih + 16, x0 + 12, y0 + ih + 28], fill=(235, 60, 60, 255))   # "live" dot
        d.text((x0 + 20, y0 + ih + 8), cap1, font=self.f_cap, fill=(255, 255, 255, 255))
        d.text((x0, y0 + ih + 44), cap2, font=self.f_small, fill=(205, 212, 222, 230))
        d.text((x0, y0 + ih + 70), cap3, font=self.f_small, fill=(205, 212, 222, 230))

    # ---------------------------------------------------------------- frames
    def card(self, title, sub="", alpha=1.0, note=""):
        """Full-frame title card (montage)."""
        img = self.backdrop.copy()
        top = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(top)
        a = int(255 * alpha)
        tw = d.textlength(title, font=self.f_huge)
        d.text((W / 2 - tw / 2, H / 2 - 110), title, font=self.f_huge, fill=(255, 255, 255, a))
        if sub:
            sw = d.textlength(sub, font=self.f_card)
            d.text((W / 2 - sw / 2, H / 2 + 10), sub, font=self.f_card, fill=(210, 216, 226, a))
        if note:
            nw = d.textlength(note, font=self.f_small)
            d.text((W / 2 - nw / 2, H / 2 + 80), note, font=self.f_small, fill=(170, 178, 190, a))
        wm = "Pool-JEPA  ·  simulation"
        d.text((W - d.textlength(wm, font=self.f_small) - 40, H - 46), wm, font=self.f_small, fill=(255, 255, 255, 110))
        img.alpha_composite(top)
        return np.asarray(img.convert("RGB"))

    def frame(self, state, ov):
        """state: {"robot": (x, y, th), "balls": (4, 2)}. ov keys (all optional):
        stage, sub: label text; target: xy; target_alpha; cue_est: xy; cue_alpha; aim_line;
        fan: dict(start, ends (N, 2), costs (N,), shown (int), chosen (int), lock (0..1),
             only_chosen (bool: draw just the chosen ray));
        trail: bool; burst: 0..1; inset: dict(image, kps, cue_sel, cue);
        result: dict(predicted, actual, miss, travel, force, n_forces, alpha)."""
        self.n += 1
        if ov.get("stage") != self.stage:
            self.stage, self.stage_start = ov.get("stage"), self.n
        img = self.bg.copy()
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        balls = np.asarray(state["balls"])
        r_px = self.m(BALL_RADIUS)

        # trail
        if ov.get("trail"):
            self.trail.append((tuple(balls[0]), self.n))
        while self.trail and self.n - self.trail[0][1] > TRAIL_FRAMES:
            self.trail.popleft()
        pts = list(self.trail)
        for (p0, n0), (p1, _) in zip(pts, pts[1:]):
            age = (self.n - n0) / TRAIL_FRAMES
            d.line([self.px(p0), self.px(p1)], fill=(235, 245, 255, int(170 * (1 - age) ** 1.6)),
                   width=max(1, int(r_px * 1.5 * (1 - age) + 1)))

        # target crosshair
        if ov.get("target") is not None:
            ta = _opt(ov, "target_alpha", 1.0)
            tx, ty = self.px(ov["target"])
            rr = 2.1 * r_px * (1 + 0.10 * math.sin(2 * math.pi * self.n / 70))
            col = GOLD + (int(255 * ta),)
            d.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], outline=col, width=4)
            d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=col)
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.line([tx + sx * rr * 0.55, ty + sy * rr * 0.55, tx + sx * rr * 1.5, ty + sy * rr * 1.5], fill=col, width=4)

        if ov.get("aim_line") and ov.get("cue_est") is not None:
            self._dashed(d, self.px(ov["cue_est"]), self.px(ov["target"]),
                         (255, 255, 255, int(150 * _opt(ov, "cue_alpha", 1.0))), 3)

        # fan of imagined shots: one straight ray per candidate force to its predicted landing point
        fan = ov.get("fan")
        if fan is not None:
            s = self.px(fan["start"])
            lock = _opt(fan, "lock", 0.0)
            order =np.argsort(-np.asarray(fan["costs"]))            # worst first, best on top
            for j in ([] if fan.get("only_chosen") else order):
                if j >= fan["shown"] or j == fan.get("chosen"):
                    continue
                e = self.px(fan["ends"][j])
                a = 95 * (1 - 0.6 * lock)
                d.line([s, e], fill=self.cost_color(fan["costs"][j], a), width=2)
                d.ellipse([e[0] - 4, e[1] - 4, e[0] + 4, e[1] + 4], fill=self.cost_color(fan["costs"][j], 2 * a))
        img.alpha_composite(layer)

        # shadows, balls, robot
        rx, ry, th = state["robot"]
        self._paste_center(img, self._rot(self.robot_shadow, th), (rx, ry), 7, 11)
        for p in balls:
            self._paste_center(img, self.ball_shadow, p, 0.35 * r_px, 0.55 * r_px)
        self._paste_center(img, self._rot(self.robot, th), (rx, ry))
        for name, p in zip(BALL_NAMES, balls):
            self._paste_center(img, self.balls[name], p)

        top = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(top)
        # the chosen imagined shot brightens and locks in
        if fan is not None and fan.get("chosen") is not None and fan["chosen"] < fan["shown"]:
            j, lock = fan["chosen"], _opt(fan, "lock", 0.0)
            s, e = self.px(fan["start"]), self.px(fan["ends"][j])
            base = self.cost_color(fan["costs"][j], 255)
            col = tuple(int(base[i] + (255 - base[i]) * lock) for i in range(3))
            if lock > 0:
                d.line([s, e], fill=col + (int(70 * lock),), width=int(4 + 14 * lock))   # glow
            d.line([s, e], fill=col + (int(110 + 145 * max(lock, 0.0)),), width=int(2 + 3 * lock))
            rr = r_px * (0.5 + 0.8 * lock)
            d.ellipse([e[0] - rr, e[1] - rr, e[0] + rr, e[1] + rr], outline=col + (255,), width=3)
            if lock >= 1.0:
                # lock_frame is None while the lock-in ramp is still running: the pulse starts now
                pulse = ((self.n - _opt(fan, "lock_frame", self.n)) % 45) / 45
                rp = r_px * (1.3 + 1.6 * pulse)
                d.ellipse([e[0] - rp, e[1] - rp, e[0] + rp, e[1] + rp], outline=(255, 255, 255, int(200 * (1 - pulse))), width=3)
        if ov.get("burst") is not None:
            b = ov["burst"]
            x, y = self.px(balls[0])
            rr = r_px * (1.4 + 3.0 * b)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(255, 255, 255, int(220 * (1 - b))), width=5)
        if ov.get("cue_est") is not None:
            ca = _opt(ov, "cue_alpha", 1.0)
            x, y = self.px(ov["cue_est"])
            rr = 1.75 * r_px
            col = CYAN + (int(255 * ca),)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=col, width=4)
            for k in range(4):
                a0 = math.radians(45 + 90 * k)
                d.line([x + rr * 1.15 * math.cos(a0), y + rr * 1.15 * math.sin(a0),
                        x + rr * 1.55 * math.cos(a0), y + rr * 1.55 * math.sin(a0)], fill=col, width=4)

        # result: markers on the table + card in the label slot
        res = ov.get("result")
        if res is not None:
            ra = _opt(res, "alpha", 1.0)
            pp, aa = self.px(res["predicted"]), self.px(res["actual"])
            self._dashed(d, pp, aa, (255, 255, 255, int(160 * ra)), 3, dash=8, gap=6)
            rr = 1.25 * r_px
            d.ellipse([pp[0] - rr, pp[1] - rr, pp[0] + rr, pp[1] + rr], outline=(255, 255, 255, int(255 * ra)), width=4)
            rr = 1.6 * r_px
            d.ellipse([aa[0] - rr, aa[1] - rr, aa[0] + rr, aa[1] + rr], outline=MAGENTA + (int(255 * ra),), width=5)
            title = f"Landed {100 * res['miss']:.1f} cm from the target"
            sub = (f"target {100 * res['travel']:.1f} cm away  ·  strength {res['force']:.2f}, best of "
                   f"{res['n_forces']} imagined shots")
            text_w = max(d.textlength(title, font=self.f_title), d.textlength(sub, font=self.f_sub))
            legend_w = 54 + d.textlength("imagined landing", font=self.f_small)
            box = [32, 30, 32 + 34 + text_w + 40 + legend_w + 24, 30 + 210]
            self._panel(d, box, ra)
            d.text((box[0] + 34, box[1] + 18), "Result", font=self.f_sub, fill=(205, 212, 222, int(255 * ra)))
            d.text((box[0] + 34, box[1] + 58), title, font=self.f_title, fill=(255, 255, 255, int(255 * ra)))
            d.text((box[0] + 36, box[1] + 140), sub, font=self.f_sub, fill=(205, 212, 222, int(255 * ra)))
            lx = box[0] + 34 + text_w + 40
            d.ellipse([lx, box[1] + 70, lx + 22, box[1] + 92], outline=(255, 255, 255, int(255 * ra)), width=3)
            d.text((lx + 34, box[1] + 66), "imagined landing", font=self.f_small, fill=(255, 255, 255, int(230 * ra)))
            d.ellipse([lx, box[1] + 106, lx + 22, box[1] + 128], outline=MAGENTA + (int(255 * ra),), width=4)
            d.text((lx + 34, box[1] + 102), "actual landing", font=self.f_small, fill=(255, 255, 255, int(230 * ra)))
        elif ov.get("stage"):
            a = float(np.clip((self.n - self.stage_start) / 18, 0, 1))
            off = 24 * (1 - a) ** 2
            tw = max(d.textlength(ov["stage"], font=self.f_title), d.textlength(_opt(ov, "sub", ""), font=self.f_sub))
            self._panel(d, [32 - off, 30, 32 - off + tw + 56, 30 + 128], a)
            d.text((60 - off, 44), ov["stage"], font=self.f_title, fill=(255, 255, 255, int(255 * a)))
            if ov.get("sub"):
                d.text((62 - off, 112), ov["sub"], font=self.f_sub, fill=(205, 212, 222, int(255 * a)))

        if ov.get("inset") is not None:
            self._inset(top, d, ov["inset"])
        wm = "Pool-JEPA  ·  simulation"
        d.text((W - d.textlength(wm, font=self.f_small) - 40, H - 46), wm, font=self.f_small, fill=(255, 255, 255, 110))
        img.alpha_composite(top)
        return np.asarray(img.convert("RGB"))
