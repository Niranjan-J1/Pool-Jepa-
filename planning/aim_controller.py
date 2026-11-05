"""Stage 1 of the hierarchical planner: drive the robot to the firing pose for a target.

The solenoid fires along the robot heading, so to send the cue ball at `target` the robot
must point along cue -> target with its front a few cm behind the cue ball:

    phi    = atan2(target - cue)                                   desired heading
    centre = cue - (ROBOT_LENGTH / 2 + gap) * [cos phi, sin phi]   desired robot centre

This is a plain closed-loop controller on decoded positions, not a learned planner: the cue
position comes from the encoder's cue keypoints (linear readout, cem.CueKeypoints) and the
robot pose from the simulator (a stand-in for the marker tracker on hardware). Each control
step it recomputes the firing pose from the latest estimates and commands the wheels.

Getting there without hitting anything. A differential-drive robot heading straight for a
point on the far side of the ball would hit it, and the table is narrow (the robot centre has
~+/-9 cm of room across it), so the controller plans a path for the robot CENTRE on a 5 mm
grid. The robot is treated as a disc of its half-diagonal, so it can turn in place anywhere on
the path; cells are free when that disc clears the cushions and the known balls. The path is
re-planned every step from the latest estimates (cheap: ~7k cells), and the controller
    1. if the robot starts somewhere the disc does not fit (e.g. squeezed against a cushion),
       backs straight out along its heading (no turning) to the nearest free spot;
    2. follows the path to the firing-pose centre (turn toward a waypoint a few cm ahead,
       then drive; forward or in reverse, whichever needs less turning);
    3. turns in place to phi, and reports done once position and heading are within tolerance.
Targets whose firing pose leaves no room to turn (the disc does not fit there) are treated as
unreachable; `firing_pose_reachable` checks this. Balls passed as `other_balls` are avoided;
if they block every path they are dropped from the obstacles (and may get pushed).
"""
import heapq
import math
from dataclasses import dataclass

import numpy as np

from env.sim_env import (BALL_RADIUS, CUSHION_THICKNESS, MAX_WHEEL_SPEED, ROBOT_LENGTH,
                         ROBOT_WIDTH, TABLE_LENGTH, TABLE_WIDTH, WHEEL_SEPARATION)

STRIKE_GAP = 0.05                    # robot front -> cue centre when firing (shot model trained on 3-7 cm)
HALF_DIAG = math.hypot(ROBOT_LENGTH / 2, ROBOT_WIDTH / 2)   # 0.05 m: robot treated as this disc
CLEARANCE = 0.003                    # m, extra margin around the disc
GRID = 0.005                         # m, path-planning cell size
INNER_X = TABLE_LENGTH / 2 - CUSHION_THICKNESS   # cushion faces
INNER_Y = TABLE_WIDTH / 2 - CUSHION_THICKNESS
assert HALF_DIAG + BALL_RADIUS + CLEARANCE < ROBOT_LENGTH / 2 + STRIKE_GAP, \
    "robot could not turn in place at the firing pose"


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def firing_pose(cue_xy, target_xy, gap: float = STRIKE_GAP):
    """(x, y, heading) that fires the cue ball from cue_xy toward target_xy."""
    d = np.asarray(target_xy, float) - np.asarray(cue_xy, float)
    phi = math.atan2(d[1], d[0])
    r = ROBOT_LENGTH / 2 + gap
    return (cue_xy[0] - r * math.cos(phi), cue_xy[1] - r * math.sin(phi), phi)


def robot_corners(x, y, th):
    c, s = math.cos(th), math.sin(th)
    return [(x + c * lx - s * ly, y + s * lx + c * ly)
            for lx in (-ROBOT_LENGTH / 2, ROBOT_LENGTH / 2) for ly in (-ROBOT_WIDTH / 2, ROBOT_WIDTH / 2)]


def box_clear(x, y, th, balls, margin: float = 0.002) -> bool:
    """The robot rectangle at (x, y, th) is inside the cushions and clear of every ball."""
    if not all(abs(cx) <= INNER_X - margin and abs(cy) <= INNER_Y - margin
               for cx, cy in robot_corners(x, y, th)):
        return False
    c, s = math.cos(th), math.sin(th)
    for bx, by in balls:
        dx, dy = bx - x, by - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        # distance from ball centre to the rectangle
        ex = max(abs(lx) - ROBOT_LENGTH / 2, 0.0)
        ey = max(abs(ly) - ROBOT_WIDTH / 2, 0.0)
        if math.hypot(ex, ey) < BALL_RADIUS + margin:
            return False
    return True


def disc_free(x, y, balls, margin: float = CLEARANCE) -> bool:
    """Robot centre at (x, y) can turn in place without touching a cushion or any ball."""
    r = HALF_DIAG + margin
    if abs(x) > INNER_X - r or abs(y) > INNER_Y - r:
        return False
    return all(math.hypot(x - bx, y - by) >= r + BALL_RADIUS for bx, by in balls)


def firing_pose_reachable(cue_xy, target_xy, other_balls=(), gap: float = STRIKE_GAP) -> bool:
    """The robot has room to turn in place at the firing pose (so Stage 1 can get there)."""
    x, y, _ = firing_pose(cue_xy, target_xy, gap)
    return disc_free(x, y, [tuple(cue_xy), *map(tuple, other_balls)])


class GridPlanner:
    """Shortest 8-connected path for the robot centre over disc-free cells."""

    def __init__(self, cell: float = GRID):
        self.cell = cell
        self.nx = int(2 * INNER_X / cell) + 1
        self.ny = int(2 * INNER_Y / cell) + 1
        self.xs = -INNER_X + cell * np.arange(self.nx)
        self.ys = -INNER_Y + cell * np.arange(self.ny)
        self._gx, self._gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        r = HALF_DIAG + CLEARANCE
        self._walls = (np.abs(self._gx) <= INNER_X - r) & (np.abs(self._gy) <= INNER_Y - r)
        self._steps = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]

    def free_grid(self, balls):
        free = self._walls.copy()
        for bx, by in balls:
            free &= np.hypot(self._gx - bx, self._gy - by) >= HALF_DIAG + CLEARANCE + BALL_RADIUS
        return free

    def to_cell(self, p):
        return (int(np.clip(round((p[0] + INNER_X) / self.cell), 0, self.nx - 1)),
                int(np.clip(round((p[1] + INNER_Y) / self.cell), 0, self.ny - 1)))

    def to_xy(self, c):
        return np.array([self.xs[c[0]], self.ys[c[1]]])

    def path(self, free, start_xy, goal_xy):
        """Cells from start to goal, or None if unreachable. The goal cell is forced free. A start
        cell that is not free (robot within the margin of an obstacle, but still able to turn)
        is joined to the nearest free cell within 2 cm, which the robot drives to first."""
        s, g = self.to_cell(start_xy), self.to_cell(goal_xy)
        free = free.copy()
        free[g] = True
        if not free[s]:
            k = int(0.02 / self.cell)
            best = None
            for i in range(max(0, s[0] - k), min(self.nx, s[0] + k + 1)):
                for j in range(max(0, s[1] - k), min(self.ny, s[1] + k + 1)):
                    d = math.hypot(i - s[0], j - s[1])
                    if free[i, j] and d <= k and (best is None or d < best[0]):
                        best = (d, (i, j))
            if best is None:
                return None
            rest = self.path(free, self.to_xy(best[1]), goal_xy)
            return None if rest is None else [s] + rest
        dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
        while pq:
            d, c = heapq.heappop(pq)
            if c == g:
                out = [c]
                while c in prev:
                    c = prev[c]
                    out.append(c)
                return out[::-1]
            if d > dist[c]:
                continue
            for dx, dy, w in self._steps:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny and free[n]):
                    continue
                if dx and dy and not (free[c[0] + dx, c[1]] or free[c[0], c[1] + dy]):
                    continue  # no squeezing diagonally between two blocked cells
                nd = d + w
                if nd < dist.get(n, math.inf):
                    dist[n], prev[n] = nd, c
                    heapq.heappush(pq, (nd, n))
        return None


def wheels(v: float, omega: float):
    """Body speed (m/s) and yaw rate (rad/s) -> (left_pwm, right_pwm), scaled into [-1, 1]
    keeping their ratio so the commanded path curvature is preserved."""
    vl = v - omega * WHEEL_SEPARATION / 2
    vr = v + omega * WHEEL_SEPARATION / 2
    scale = max(1.0, abs(vl) / MAX_WHEEL_SPEED, abs(vr) / MAX_WHEEL_SPEED)
    return vl / MAX_WHEEL_SPEED / scale, vr / MAX_WHEEL_SPEED / scale


@dataclass
class AimConfig:
    gap: float = STRIKE_GAP
    lookahead: float = 0.03                # m along the path to the steering waypoint
    pos_tol: float = 0.003                 # m, firing-pose centre
    head_tol: float = math.radians(0.25)   # final heading
    turn_first: float = math.radians(20)   # turn in place before driving if heading error exceeds this
    k_v: float = 3.0                       # 1/s, speed per metre of distance
    v_max: float = 0.15                    # m/s
    k_w: float = 8.0                       # 1/s, yaw rate per radian of heading error
    w_max: float = 6.0                     # rad/s


class AimController:
    """reset(), then step(robot_pose, cue_xy, target_xy) every control step until done."""

    def __init__(self, cfg: AimConfig = AimConfig()):
        self.cfg = cfg
        self.grid = GridPlanner()
        self.reset()

    def reset(self):
        self.final = False      # on the firing-pose centre, turning to the heading
        self.back_out = None    # (x, y) target of a straight back-out move, latched once chosen

    def _go_to(self, pose, wp, allow_turn=True):
        """Drive toward waypoint wp, forward or in reverse (whichever needs less turning)."""
        c = self.cfg
        x, y, th = pose
        dx, dy = wp[0] - x, wp[1] - y
        dist = math.hypot(dx, dy)
        e = wrap(math.atan2(dy, dx) - th)
        sign = 1.0
        if abs(e) > math.pi / 2:
            e, sign = wrap(e - math.pi), -1.0
        if not allow_turn:
            return sign * min(c.v_max, c.k_v * dist) * math.cos(e), 0.0
        w = float(np.clip(c.k_w * e, -c.w_max, c.w_max))
        v = 0.0 if abs(e) > c.turn_first else sign * min(c.v_max, c.k_v * dist) * math.cos(e)
        return v, w

    def _choose_back_out(self, pose, balls, free):
        """Nearest free-cell spot reachable by driving straight along +/- heading."""
        x, y, th = pose
        u = np.array([math.cos(th), math.sin(th)])
        for s in np.arange(0.005, 0.20, 0.005):
            for sign in (-1.0, 1.0):        # prefer backing away (the robot usually faces the cue)
                ok = True
                for t in np.arange(0.005, s + 1e-9, 0.005):
                    px, py = np.array([x, y]) + sign * t * u
                    if not box_clear(px, py, th, balls):
                        ok = False
                        break
                p = np.array([x, y]) + sign * s * u
                if ok and disc_free(p[0], p[1], balls):
                    return p
        return None

    def step(self, robot_pose, cue_xy, target_xy, other_balls=()):
        """Returns (left_pwm, right_pwm, done, info)."""
        c = self.cfg
        x, y, th = robot_pose
        cue = np.asarray(cue_xy, float)
        fx, fy, phi = firing_pose(cue, target_xy, c.gap)
        pos_err = math.hypot(fx - x, fy - y)
        head_err = wrap(phi - th)
        info = {"pos_err": pos_err, "head_err": head_err, "firing_pose": (fx, fy, phi)}

        if self.final and pos_err > 3 * c.pos_tol:
            self.final = False                  # knocked off the pose (or the cue estimate moved)
        if not self.final and pos_err <= c.pos_tol:
            self.final = True
            self.back_out = None

        done = False
        if self.final:
            info["phase"] = "turn"
            v, w = 0.0, float(np.clip(c.k_w * head_err, -c.w_max, c.w_max))
            if abs(head_err) <= c.head_tol:
                v, w, done = 0.0, 0.0, True
        else:
            balls = [tuple(cue), *map(tuple, other_balls)]
            free = self.grid.free_grid(balls)
            if len(other_balls) and not free[self.grid.to_cell((fx, fy))]:
                balls, free = [tuple(cue)], self.grid.free_grid([tuple(cue)])  # a ball sits on the firing pose
            # Back-out only reacts to the cue ball and cushions; other balls just shape the path
            # (reacting to them too made the robot oscillate when one sat next to it).
            if self.back_out is None and not disc_free(x, y, [tuple(cue)], margin=0.0):
                self.back_out = self._choose_back_out(robot_pose, [tuple(cue)], free)
            if self.back_out is not None:
                info["phase"] = "back_out"
                v, w = self._go_to(robot_pose, self.back_out, allow_turn=False)
                along = (self.back_out[0] - x) * math.cos(th) + (self.back_out[1] - y) * math.sin(th)
                if abs(along) < 0.002:   # straight move only, so judge arrival along the heading
                    self.back_out = None
            else:
                path = self.grid.path(free, (x, y), (fx, fy))
                if path is None and len(balls) > 1:   # other balls block every route: ignore them
                    path = self.grid.path(self.grid.free_grid([tuple(cue)]), (x, y), (fx, fy))
                if path is None:
                    info["phase"] = "blocked"
                    v, w = 0.0, 0.0
                else:
                    k = int(round(c.lookahead / self.grid.cell))
                    wp = np.array([fx, fy]) if len(path) <= k + 1 else self.grid.to_xy(path[k])
                    info["phase"] = "approach" if len(path) <= k + 1 else "path"
                    v, w = self._go_to(robot_pose, wp)

        left, right = wheels(v, w)
        return left, right, done, info
