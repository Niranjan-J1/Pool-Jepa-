"""Polished 2-D renderer for demo clips (record_shot.py). Presentation only.

Perception never sees these frames: the encoder still reads the plain 432x240 env.render()
frames it was trained on (record_shot.py shows those as an inset). This module only draws the
same simulator state more nicely at 1920x1080:
  - felt with a soft spotlight, cushion shadow, grain and vignette; wood rails with grain,
    bevel and sights; decorative corner pockets (the simulated table has none);
  - balls shaded as spheres (diffuse + specular) with soft drop shadows;
  - a robot sprite (wheels, top plate, marker, orange nose and striker) with a shadow;
  - a fading trail behind the moving cue ball;
  - overlays: target crosshair, perceived-cue ring, aim line, imagined landing points,
    stage labels that fade in, and a result card.
A frame is drawn from a state dict {"robot": (x, y, theta), "balls": (4, 2) array in the
order cue, red, blue, yellow} and an overlay dict (keys documented in DemoRenderer.frame).
"""
import math
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from env.sim_env import (BALL_COLORS, BALL_RADIUS, CUSHION_THICKNESS, ROBOT_LENGTH, ROBOT_WIDTH,
                         TABLE_LENGTH, TABLE_WIDTH)

W, H = 1920, 1080
INNER_X = TABLE_LENGTH / 2 - CUSHION_THICKNESS   # cushion faces = where balls bounce
INNER_Y = TABLE_WIDTH / 2 - CUSHION_THICKNESS
CUSH = 0.012     # m of green cushion rubber drawn outside the face
RAIL = 0.052     # m from the face to the outer edge of the wood
SS = 3           # supersampling for sprites
BALL_NAMES = ["cue", "red", "blue", "yellow"]
TRAIL_FRAMES = 40
GOLD = (255, 205, 60)
CYAN = (70, 225, 255)
MAGENTA = (255, 80, 200)


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


def _resize(a, size):
    """float array -> resized float array via PIL (bicubic)."""
    img = Image.fromarray(a.astype(np.float32), mode="F").resize(size, Image.BICUBIC)
    return np.asarray(img)


def lerp_state(a, b, t):
    th = a["robot"][2] + ((b["robot"][2] - a["robot"][2] + math.pi) % (2 * math.pi) - math.pi) * t
    rob = tuple(a["robot"][i] + (b["robot"][i] - a["robot"][i]) * t for i in range(2)) + (th,)
    return {"robot": rob, "balls": a["balls"] + (b["balls"] - a["balls"]) * t}


def interpolate(states, slowmo):
    """Physics states (one per 60 Hz step) -> one state per output frame, `slowmo` frames per
    step, linearly interpolated between steps."""
    if slowmo <= 1.0 or len(states) < 2:
        return list(states)
    out, n = [], len(states) - 1
    for j in range(int(n * slowmo) + 1):
        t = j / slowmo
        i = min(int(t), n - 1)
        out.append(lerp_state(states[i], states[i + 1], t - i))
    return out


class DemoRenderer:
    def __init__(self, pockets=True, seed=0):
        # Table fills y 260..1020; the band above holds the stage label, encoder inset and result
        # card, the strip below the HUD text, so nothing is drawn over the playing surface.
        self.S = min(1700 / (2 * (INNER_X + RAIL)), 760 / (2 * (INNER_Y + RAIL)))  # px per m
        self.cx, self.cy = W / 2, 260 + (INNER_Y + RAIL) * self.S
        self.rng = np.random.default_rng(seed)
        self.bg = self._background(pockets)
        self.balls = {n: self._ball_sprite(BALL_COLORS[n]) for n in BALL_NAMES}
        self.ball_shadow = self._shadow_sprite(BALL_RADIUS * 2, BALL_RADIUS * 2, round_frac=1.0)
        self.robot = self._robot_sprite()
        self.robot_shadow = self._shadow_sprite(ROBOT_LENGTH, ROBOT_WIDTH, round_frac=0.25)
        self.f_title, self.f_sub = _font(54, True), _font(26)
        self.f_small, self.f_card, self.f_big = _font(22), _font(28), _font(60, True)
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

        # backdrop: dark slate with a soft radial falloff
        r = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        img = np.array([30, 34, 41], np.float32) * (1.08 - 0.5 * np.clip(r / 1.4, 0, 1))[..., None]

        # table drop shadow
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
        d_out = -rounded(ax, ay, bx, by, rc)                          # m inside the outer edge
        d_in = np.minimum(np.abs(ay - INNER_Y - CUSH) * horiz + 9 * ~horiz,
                          np.abs(ax - INNER_X - CUSH) * ~horiz + 9 * horiz)
        bevel = 1 + 0.35 * np.exp(-d_out / 0.0025) - 0.35 * np.exp(-d_in / 0.004)
        light = 1 + 0.07 * wy / (INNER_Y + RAIL)
        img = np.where(wood[..., None], wood_col * (bevel * light)[..., None], img)

        # cushions: darker green, lit toward the top ridge
        t = np.clip(np.maximum(ax - INNER_X, ay - INNER_Y) / CUSH, 0, 1)
        cush_col = np.array([18, 86, 50], np.float32) * (0.7 + 0.45 * np.sin(np.pi * (0.25 + 0.6 * t)))[..., None]
        img = np.where(cush[..., None], cush_col, img)

        # felt: spotlight, cushion shadow, vignette, fine nap
        spot = np.exp(-((wx / 0.42) ** 2 + (wy / 0.24) ** 2))
        edge = np.minimum(INNER_X - ax, INNER_Y - ay)
        shade = (0.80 + 0.30 * spot) * (1 - 0.38 * np.exp(-np.clip(edge, 0, None) / 0.007))
        nap = 1 + 0.025 * self.rng.standard_normal((H, W)).astype(np.float32)
        felt_col = np.array([24, 112, 64], np.float32) * (shade * nap)[..., None]
        img = np.where(felt[..., None], felt_col, img)

        # sights (diamonds) on the rails
        mid = INNER_X + CUSH + (RAIL - CUSH) * 0.45
        midy = INNER_Y + CUSH + (RAIL - CUSH) * 0.45
        sights = [(INNER_X * k / 4, s * midy) for k in (-3, -2, -1, 1, 2, 3) for s in (-1, 1)] + \
                 [(s * mid, INNER_Y * k / 2) for k in (-1, 0, 1) for s in (-1, 1)]
        for sx, sy in sights:
            d = np.hypot(wx - sx, wy - sy)
            a = np.clip((0.0034 - d) / 0.0008, 0, 1)
            img = img * (1 - a[..., None]) + np.array([236, 226, 200], np.float32) * a[..., None]

        # decorative corner pockets
        if pockets:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    # centred in the cushion corner so they barely reach into the playing area
                    # (a ball at the corner never looks half-pocketed)
                    px_, py_ = sx * (INNER_X + 0.011), sy * (INNER_Y + 0.011)
                    d = np.hypot(wx - px_, wy - py_)
                    rim = np.clip((0.021 - d) / 0.0012, 0, 1)
                    img = img * (1 - rim[..., None]) + np.array([46, 30, 22], np.float32) * rim[..., None]
                    hole = np.clip((0.0165 - d) / 0.0012, 0, 1)
                    depth = np.clip(d / 0.0165, 0, 1)
                    hole_col = np.array([8, 8, 9], np.float32) * (0.5 + 1.5 * depth)[..., None]
                    img = img * (1 - hole[..., None]) + hole_col * hole[..., None]

        # vignette over everything
        img *= (1 - 0.28 * np.clip(r - 0.55, 0, 1))[..., None]
        out = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB").convert("RGBA")

    def _ball_sprite(self, color):
        r = self.m(BALL_RADIUS)
        n = int(math.ceil(2 * r + 4))
        big = n * SS
        yy, xx = np.mgrid[0:big, 0:big].astype(np.float32) + 0.5
        R, c = r * SS, big / 2
        dx, dy = (xx - c) / R, (yy - c) / R                      # image y points down
        d2 = dx * dx + dy * dy
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        L = np.array([-0.42, -0.58, 0.70]); L /= np.linalg.norm(L)
        Hv = L + np.array([0, 0, 1.0]); Hv /= np.linalg.norm(Hv)
        diff = np.clip(dx * L[0] + dy * L[1] + nz * L[2], 0, 1)
        spec = np.clip(dx * Hv[0] + dy * Hv[1] + nz * Hv[2], 0, 1) ** 70
        bounce = np.clip(dy, 0, 1) * 0.12                          # light from the felt below
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
        img = img.filter(ImageFilter.GaussianBlur(7))
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.putalpha(img)
        return out

    def _robot_sprite(self):
        s = self.S * SS
        L, Wd = ROBOT_LENGTH * s, ROBOT_WIDTH * s
        pad = int(0.006 * s)
        img = Image.new("RGBA", (int(L) + 2 * pad, int(Wd) + 2 * pad), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        x0, y0 = pad, pad
        u = s  # px per m (supersampled)
        # wheels (inside the footprint, along both sides)
        for side in (0, 1):
            wy0 = y0 + (0 if side == 0 else Wd - 0.011 * u)
            d.rounded_rectangle([x0 + 0.018 * u, wy0, x0 + L - 0.018 * u, wy0 + 0.011 * u],
                                radius=0.003 * u, fill=(22, 22, 25, 255))
            for k in range(9):   # tread lines
                tx = x0 + 0.020 * u + k * (L - 0.040 * u) / 8
                d.line([tx, wy0 + 0.001 * u, tx, wy0 + 0.010 * u], fill=(48, 48, 54, 255), width=max(1, int(0.0008 * u)))
        # chassis
        d.rounded_rectangle([x0, y0 + 0.008 * u, x0 + L, y0 + Wd - 0.008 * u], radius=0.010 * u,
                            fill=(46, 50, 58, 255), outline=(20, 22, 26, 255), width=int(0.0012 * u))
        # top plate
        d.rounded_rectangle([x0 + 0.008 * u, y0 + 0.013 * u, x0 + L - 0.016 * u, y0 + Wd - 0.013 * u],
                            radius=0.007 * u, fill=(84, 92, 106, 255))
        d.rounded_rectangle([x0 + 0.010 * u, y0 + 0.015 * u, x0 + L - 0.018 * u, y0 + 0.020 * u],
                            radius=0.002 * u, fill=(120, 130, 146, 255))   # plate highlight
        # tracking marker on the back half
        mx, my, ms = x0 + 0.014 * u, y0 + Wd / 2 - 0.009 * u, 0.018 * u
        d.rectangle([mx, my, mx + ms, my + ms], fill=(245, 245, 245, 255))
        cell = ms / 4
        pattern = [(1, 1), (2, 1), (1, 2), (2, 2), (0, 3), (3, 0)]
        d.rectangle([mx + cell * 0.5, my + cell * 0.5, mx + ms - cell * 0.5, my + ms - cell * 0.5], fill=(20, 20, 20, 255))
        for i, j in pattern[:4]:
            if (i + j) % 2 == 0:
                d.rectangle([mx + i * cell, my + j * cell, mx + (i + 1) * cell - 1, my + (j + 1) * cell - 1],
                            fill=(245, 245, 245, 255))
        # nose: orange chevron pointing forward (+x), and the striker at the front edge
        nx = x0 + L - 0.014 * u
        cy = y0 + Wd / 2
        d.polygon([(nx - 0.016 * u, cy - 0.012 * u), (nx, cy), (nx - 0.016 * u, cy + 0.012 * u),
                   (nx - 0.010 * u, cy)], fill=(255, 140, 36, 255))
        d.rectangle([x0 + L - 0.010 * u, cy - 0.004 * u, x0 + L + 0.001 * u, cy + 0.004 * u],
                    fill=(196, 200, 208, 255), outline=(90, 94, 102, 255))
        n = (img.width // SS, img.height // SS)
        return img.resize(n, Image.LANCZOS)

    # ---------------------------------------------------------------- per frame
    def _paste_center(self, canvas, sprite, xy, dx=0, dy=0):
        x, y = self.px(xy)
        canvas.alpha_composite(sprite, (int(round(x - sprite.width / 2 + dx)), int(round(y - sprite.height / 2 + dy))))

    def _rot(self, sprite, theta):
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
    def _cost_color(cost, alpha=235):
        t = float(np.clip(cost / 0.08, 0, 1))      # 0 cm green -> 4 cm yellow -> >= 8 cm red
        if t < 0.5:
            k = t / 0.5
            c = (60 + 190 * k, 220 - 10 * k, 120 - 60 * k)
        else:
            k = (t - 0.5) / 0.5
            c = (250 - 10 * k, 210 - 140 * k, 60)
        return tuple(int(v) for v in c) + (alpha,)

    def _panel(self, d, box, alpha=1.0):
        d.rounded_rectangle(box, radius=14, fill=(12, 14, 18, int(175 * alpha)),
                            outline=(255, 255, 255, int(40 * alpha)), width=2)

    def frame(self, state, ov):
        """state: {"robot": (x, y, th), "balls": (4, 2)}. ov keys (all optional except stage):
        stage, sub: label text; target: xy; target_alpha; cue_est: xy; cue_alpha; aim_line: bool;
        candidates: (points (N, 2), costs (N,)); chosen: xy; chosen_text; trail: bool (cue ball
        leaves a trail this frame); burst: 0..1 strike flash; hud: str; inset: (h, w, 3) image;
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
            a = int(170 * (1 - age) ** 1.6)
            w = max(1, int(r_px * 1.5 * (1 - age) + 1))
            d.line([self.px(p0), self.px(p1)], fill=(235, 245, 255, a), width=w)

        # target crosshair (on the felt, under the balls)
        if ov.get("target") is not None:
            ta = ov.get("target_alpha", 1.0)
            tx, ty = self.px(ov["target"])
            pulse = 1 + 0.10 * math.sin(2 * math.pi * self.n / 70)
            rr = 2.1 * r_px * pulse
            col = GOLD + (int(255 * ta),)
            d.ellipse([tx - rr, ty - rr, tx + rr, ty + rr], outline=col, width=4)
            d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=col)
            for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.line([tx + sx * rr * 0.55, ty + sy * rr * 0.55, tx + sx * rr * 1.5, ty + sy * rr * 1.5], fill=col, width=4)

        # aim line from the perceived cue ball to the target
        if ov.get("aim_line") and ov.get("cue_est") is not None:
            self._dashed(d, self.px(ov["cue_est"]), self.px(ov["target"]),
                         (255, 255, 255, int(150 * ov.get("cue_alpha", 1.0))), 3)

        # imagined landing points
        if ov.get("candidates") is not None:
            cp, cc = ov["candidates"]
            for p, c in zip(cp, cc):
                x, y = self.px(p)
                rr = 0.42 * r_px
                d.ellipse([x - rr * 1.9, y - rr * 1.9, x + rr * 1.9, y + rr * 1.9], fill=self._cost_color(c, 45))
                d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=self._cost_color(c))
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
        # strike flash
        if ov.get("burst") is not None:
            b = ov["burst"]
            x, y = self.px(balls[0])
            rr = r_px * (1.4 + 3.0 * b)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(255, 255, 255, int(220 * (1 - b))), width=5)
        # perceived cue ring
        if ov.get("cue_est") is not None:
            ca = ov.get("cue_alpha", 1.0)
            x, y = self.px(ov["cue_est"])
            rr = 1.75 * r_px
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=CYAN + (int(255 * ca),), width=4)
            for k in range(4):   # corner ticks: a "detection" reticle
                a0 = math.radians(45 + 90 * k)
                d.line([x + rr * 1.15 * math.cos(a0), y + rr * 1.15 * math.sin(a0),
                        x + rr * 1.55 * math.cos(a0), y + rr * 1.55 * math.sin(a0)], fill=CYAN + (int(255 * ca),), width=4)
            d.text((x + rr * 1.6, y - rr * 1.9), "perceived", font=self.f_small, fill=CYAN + (int(230 * ca),))
        # chosen imagined landing point
        if ov.get("chosen") is not None:
            x, y = self.px(ov["chosen"])
            rr = 1.25 * r_px
            d.ellipse([x - rr * 1.6, y - rr * 1.6, x + rr * 1.6, y + rr * 1.6], outline=(255, 255, 255, 80), width=8)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(255, 255, 255, 255), width=4)
            if ov.get("chosen_text"):
                d.text((x + rr * 1.8, y + rr * 0.6), ov["chosen_text"], font=self.f_small, fill=(255, 255, 255, 240))

        # result markers + card
        res = ov.get("result")
        if res is not None:
            ra = res.get("alpha", 1.0)
            pp, aa = self.px(res["predicted"]), self.px(res["actual"])
            self._dashed(d, pp, aa, (255, 255, 255, int(160 * ra)), 3, dash=8, gap=6)
            rr = 1.25 * r_px
            d.ellipse([pp[0] - rr, pp[1] - rr, pp[0] + rr, pp[1] + rr], outline=(255, 255, 255, int(255 * ra)), width=4)
            rr = 1.6 * r_px
            d.ellipse([aa[0] - rr, aa[1] - rr, aa[0] + rr, aa[1] + rr], outline=MAGENTA + (int(255 * ra),), width=5)
            title = f"Landed {100 * res['miss']:.1f} cm from the target"
            sub = (f"target {100 * res['travel']:.1f} cm away  ·  force {res['force']:.2f}, best of "
                   f"{res['n_forces']} imagined shots")
            text_w = max(d.textlength(title, font=self.f_title), d.textlength(sub, font=self.f_sub))
            legend_w = 20 + 34 + d.textlength("imagined landing", font=self.f_small)
            bw, bh = text_w + legend_w + 34 + 60 + 34, 150
            box = [W / 2 - bw / 2, 50, W / 2 + bw / 2, 50 + bh]
            self._panel(d, box, ra)
            d.text((box[0] + 34, box[1] + 22), title, font=self.f_title, fill=(255, 255, 255, int(255 * ra)))
            d.text((box[0] + 36, box[1] + 96), sub, font=self.f_sub, fill=(220, 225, 232, int(255 * ra)))
            d.line([box[0] + 34 + text_w + 30, box[1] + 24, box[0] + 34 + text_w + 30, box[3] - 24],
                   fill=(255, 255, 255, int(50 * ra)), width=2)
            lx = box[0] + 34 + text_w + 60
            d.ellipse([lx, box[1] + 30, lx + 22, box[1] + 52], outline=(255, 255, 255, int(255 * ra)), width=3)
            d.text((lx + 34, box[1] + 26), "imagined landing", font=self.f_small, fill=(255, 255, 255, int(230 * ra)))
            d.ellipse([lx, box[1] + 66, lx + 22, box[1] + 88], outline=MAGENTA + (int(255 * ra),), width=4)
            d.text((lx + 34, box[1] + 62), "actual landing", font=self.f_small, fill=(255, 255, 255, int(230 * ra)))

        # camera inset: what the encoder actually sees
        if ov.get("inset") is not None:
            raw = Image.fromarray(ov["inset"])
            ins = raw.resize((raw.width * 3 // 4, raw.height * 3 // 4), Image.LANCZOS).convert("RGBA")
            iw, ih = ins.size
            cap = f"encoder input · {raw.width}x{raw.height} px"
            bw = max(iw, d.textlength(cap, font=self.f_small))
            x0, y0 = W - bw - 48, 30
            d.rounded_rectangle([x0 - 10, y0 - 10, x0 + bw + 10, y0 + ih + 44], radius=10, fill=(12, 14, 18, 190))
            top.alpha_composite(ins, (int(x0 + (bw - iw) / 2), y0))
            d.text((x0, y0 + ih + 10), cap, font=self.f_small, fill=(220, 225, 232, 230))

        # stage label (fades and slides in on change)
        if ov.get("stage"):
            a = float(np.clip((self.n - self.stage_start) / 18, 0, 1))
            off = 24 * (1 - a) ** 2
            tw = max(d.textlength(ov["stage"], font=self.f_title), d.textlength(ov.get("sub", ""), font=self.f_sub))
            self._panel(d, [32 - off, 30, 32 - off + tw + 56, 30 + 128], a)
            d.text((60 - off, 44), ov["stage"], font=self.f_title, fill=(255, 255, 255, int(255 * a)))
            if ov.get("sub"):
                d.text((62 - off, 112), ov["sub"], font=self.f_sub, fill=(205, 212, 222, int(255 * a)))
        if ov.get("hud"):
            d.text((40, H - 46), ov["hud"], font=self.f_small, fill=(205, 212, 222, 220))
        wm = "Pool-JEPA  ·  simulation"
        d.text((W - d.textlength(wm, font=self.f_small) - 40, H - 46), wm, font=self.f_small, fill=(255, 255, 255, 110))
        img.alpha_composite(top)
        return np.asarray(img.convert("RGB"))
