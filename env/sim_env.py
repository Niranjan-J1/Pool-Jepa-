import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pymunk

from env.base_env import Action, BaseEnv, Observation

# Table dimensions: 26-inch table scaled to meters
TABLE_LENGTH = 26 * 0.0254  # ~0.6604 m
TABLE_WIDTH = TABLE_LENGTH / 2  # ~0.3302 m
CUSHION_THICKNESS = 0.02

# Robot dimensions (approx 14cm x 20cm scaled down for this table)
ROBOT_WIDTH = 0.06
ROBOT_LENGTH = 0.08
ROBOT_MASS = 0.5
WHEEL_SEPARATION = ROBOT_WIDTH
MAX_WHEEL_SPEED = 0.3  # m/s at |pwm| = 1 -> max yaw rate 2 * 0.3 / 0.06 = 10 rad/s

# Ball properties
BALL_RADIUS = 0.01
BALL_MASS = 0.02

# Physics tuning (legacy defaults; kept so a no-args SimEnv() reproduces old behaviour)
BALL_ELASTICITY = 0.85
BALL_FRICTION = 0.4
CUSHION_ELASTICITY = 0.7
CUSHION_FRICTION = 0.3
LINEAR_DAMPING = 0.97
ANGULAR_DAMPING = 0.95
# Rolling resistance of the cloth: constant deceleration on every moving ball. Without it
# (damping alone keeps 97% of speed per second) balls never come to rest. With 0.5 m/s^2,
# behind_cue strikes settle in ~1-2 s (median 70 steps, cushions absorb energy too) and the
# cue ball travels 1-56 cm (median 16 cm).
ROLLING_DECEL = 0.5  # m/s^2

# Multi-ball defaults (used only when n_object_balls is given, i.e. not legacy mode)
OBJECT_BALL_ELASTICITY = 0.95     # ball-ball rebound: near-elastic, clean momentum transfer
MULTIBALL_CUSHION_ELASTICITY = 0.75  # cushion rebound: lively but lossy
# Multiball rails use a thinner collision radius than the legacy 2 cm so a ball can roll close
# enough to a corner for a modest pocket to catch it (the rail's thickness forms the pocket jaws).
MULTIBALL_CUSHION_RADIUS = 0.006
POCKET_RADIUS = 2.8 * BALL_RADIUS  # 0.028 m capture radius; a ball centre inside this is potted
CUE_DEFAULT_SPOT = (0.0, 0.0)      # where the cue ball respawns after a scratch
MAX_OBJECT_BALLS = 7

# Solenoid
MAX_STRIKE_IMPULSE = 0.05

# "behind_cue" reset: robot front placed behind the cue ball, roughly aimed at it
SETUP_DISTANCE_RANGE = (0.03, 0.07)       # m, robot front -> cue ball center
SETUP_AIM_OFFSET_RANGE = (-math.pi / 4, math.pi / 4)  # heading offset from the aim line
SETUP_MAX_TRIES = 1000

# Rendering. The view spans the table plus a margin; frames are table-shaped (not square).
# At 432 px across a 0.7204 m view -> 600 px/m -> balls are 12 px across.
RENDER_WIDTH = 432
RENDER_HEIGHT = 240
RENDER_MARGIN = 0.03  # m beyond the table on each side (x); y margin is set by the aspect ratio
FELT_COLOR = "#1a6c2a"   # multiball felt (same green as the legacy inner table)
RAIL_COLOR = "#5c3a1a"   # multiball rail band (same brown as the legacy cushion outline)

# Simulation
DT = 1 / 60.0
SUBSTEPS = 10
SETTLE_VELOCITY_THRESHOLD = 0.005

BALL_COLORS = {
    "cue": (1.0, 1.0, 1.0),
    "red": (0.9, 0.15, 0.15),
    "blue": (0.15, 0.3, 0.9),
    "yellow": (0.95, 0.85, 0.1),
}

# Legacy ball set: ids 0..3 are cue, red, blue, yellow (the pre-multiball layout).
LEGACY_BALL_NAMES = ["cue", "red", "blue", "yellow"]

# Distinct colours for object balls id 1..7 in multiball mode (id 0 / cue stays white).
OBJECT_COLORS = [
    (0.90, 0.15, 0.15),  # 1 red
    (0.15, 0.30, 0.90),  # 2 blue
    (0.95, 0.85, 0.10),  # 3 yellow
    (0.55, 0.20, 0.75),  # 4 purple
    (0.95, 0.50, 0.10),  # 5 orange
    (0.10, 0.70, 0.70),  # 6 teal
    (0.95, 0.35, 0.65),  # 7 pink
]

CUE_ID = 0


def pocket_centers() -> list[tuple[float, float]]:
    """The six pocket centres (metres): four corners and the midpoints of the two long rails."""
    hl, hw = TABLE_LENGTH / 2, TABLE_WIDTH / 2
    return [(-hl, -hw), (hl, -hw), (-hl, hw), (hl, hw), (0.0, -hw), (0.0, hw)]


def _center_in_play(x: float, y: float, clearance: float) -> bool:
    """True if a ball centre is inside the cushions with `clearance` to spare."""
    lim_x = TABLE_LENGTH / 2 - CUSHION_THICKNESS - clearance
    lim_y = TABLE_WIDTH / 2 - CUSHION_THICKNESS - clearance
    return abs(x) <= lim_x and abs(y) <= lim_y


def _clear_of_pockets(x: float, y: float, pocket_radius: float) -> bool:
    """True if a ball centre is outside every pocket mouth (so it would not be potted at rest)."""
    keep = pocket_radius + BALL_RADIUS
    return all(math.hypot(x - px, y - py) > keep for px, py in pocket_centers())


def random_layout(n, rng, *, cue_pos=None, pocket_radius=POCKET_RADIUS, avoid=None,
                  min_sep=2 * BALL_RADIUS + 0.005, max_tries=5000):
    """Generate a legal layout of the cue ball plus `n` object balls.

    All positions are in metres (the table's native units, matching obs["ball_positions"]).
    Legal means: inside the cushions (not touching), clear of every pocket mouth, mutually
    non-overlapping (centres at least `min_sep` apart), and outside the `avoid(pt)` region
    (used for the robot's start area). `rng` is a `random.Random` (or the `random` module).

    Returns an ordered dict {0: cue, 1: obj, ..., n: obj}. If `cue_pos` is given it is used
    verbatim for id 0; otherwise the cue is sampled too. Raises RuntimeError if it cannot
    place every ball within `max_tries`.
    """
    if not 0 <= n <= MAX_OBJECT_BALLS:
        raise ValueError(f"n must be in [0, {MAX_OBJECT_BALLS}], got {n}")
    clearance = BALL_RADIUS + 0.003

    def legal(pt, placed):
        x, y = pt
        if not _center_in_play(x, y, clearance):
            return False
        if not _clear_of_pockets(x, y, pocket_radius):
            return False
        if avoid is not None and avoid(pt):
            return False
        return all(math.hypot(x - qx, y - qy) >= min_sep for qx, qy in placed)

    def sample_one(placed):
        lim_x = TABLE_LENGTH / 2 - CUSHION_THICKNESS - clearance
        lim_y = TABLE_WIDTH / 2 - CUSHION_THICKNESS - clearance
        for _ in range(max_tries):
            pt = (rng.uniform(-lim_x, lim_x), rng.uniform(-lim_y, lim_y))
            if legal(pt, placed):
                return pt
        raise RuntimeError("random_layout: could not place a ball; loosen constraints or n")

    placed = []
    layout = {}
    if cue_pos is None:
        cue_pos = sample_one(placed)
    layout[CUE_ID] = (float(cue_pos[0]), float(cue_pos[1]))
    placed.append(layout[CUE_ID])
    for i in range(1, n + 1):
        pt = sample_one(placed)
        layout[i] = pt
        placed.append(pt)
    return layout


class SimEnv(BaseEnv):
    def __init__(self, n_object_balls: int | None = None, *,
                 ball_elasticity: float | None = None,
                 cushion_elasticity: float | None = None,
                 pocket_radius: float = POCKET_RADIUS,
                 cue_spawn: tuple[float, float] = CUE_DEFAULT_SPOT):
        """Pool-table simulator.

        n_object_balls:
            None (default) -> LEGACY mode: byte-for-byte the old behaviour. The table holds
                the cue plus three object balls (red, blue, yellow), there are no pockets, and
                the observation has exactly the original keys. Everything downstream that calls
                SimEnv() with no arguments is unaffected.
            int in [0, 7] -> MULTIBALL mode: cue (id 0) plus this many object balls (ids 1..n),
                six pockets with potting/scratch, and extra observation keys (ball_positions_cm,
                potted, scratch, n_balls_on_table) added alongside the originals.

        ball_elasticity / cushion_elasticity: override the rebound coefficients. Defaults are
            the legacy values (0.85 / 0.7) in legacy mode and the livelier (0.95 / 0.75) in
            multiball mode.
        pocket_radius: a ball whose centre comes within this of a pocket centre is potted.
        cue_spawn: where the cue ball is replaced after a scratch.
        """
        self._legacy = n_object_balls is None
        if self._legacy:
            self._n_object_balls = 3
        else:
            if not 0 <= int(n_object_balls) <= MAX_OBJECT_BALLS:
                raise ValueError(f"n_object_balls must be in [0, {MAX_OBJECT_BALLS}]")
            self._n_object_balls = int(n_object_balls)

        self._ball_elasticity = (
            ball_elasticity if ball_elasticity is not None
            else (BALL_ELASTICITY if self._legacy else OBJECT_BALL_ELASTICITY))
        self._cushion_elasticity = (
            cushion_elasticity if cushion_elasticity is not None
            else (CUSHION_ELASTICITY if self._legacy else MULTIBALL_CUSHION_ELASTICITY))
        self._cushion_radius = CUSHION_THICKNESS if self._legacy else MULTIBALL_CUSHION_RADIUS
        self._pocket_radius = pocket_radius
        self._pockets = pocket_centers()
        self._pocket_r2 = pocket_radius * pocket_radius
        # A ball with |x| < near_x and |y| < near_y is too far from every pocket to be potted;
        # this interior early-out keeps the per-substep check cheap.
        self._near_x = TABLE_LENGTH / 2 - pocket_radius
        self._near_y = TABLE_WIDTH / 2 - pocket_radius
        self._cue_spawn = (float(cue_spawn[0]), float(cue_spawn[1]))

        self._space: pymunk.Space | None = None
        self._robot_body: pymunk.Body | None = None
        self._robot_shape: pymunk.Shape | None = None
        self._ball_bodies: dict[int, pymunk.Body] = {}
        self._ball_shapes: dict[int, pymunk.Shape] = {}
        self._ball_colors: dict[int, tuple] = {}
        self._potted: list[int] = []
        self._pot_events: list[tuple[int, int]] = []   # (ball_id, pocket index), cue included
        self._scratch = False
        self._fig = None
        self._ax = None
        self.last_setup: dict = {}
        self.last_strike_connected = False

    # ---- construction helpers -------------------------------------------------

    def _create_space(self) -> pymunk.Space:
        space = pymunk.Space()
        space.gravity = (0, 0)
        space.damping = LINEAR_DAMPING
        return space

    def _add_walls(self, space: pymunk.Space):
        hw = TABLE_WIDTH / 2
        hl = TABLE_LENGTH / 2
        # Solid rectangle in both modes, so a ball can never leave the table (no tunnelling or
        # escaping through a pocket -- potting is handled by the sensor check, not a gap). Legacy
        # keeps the original 2 cm collision radius for byte-identical behaviour; multiball uses a
        # thinner rail so a ball can roll close enough to the corners for a modest pocket radius
        # to catch it.
        t = self._cushion_radius
        wall_segments = [
            ((-hl, -hw), (hl, -hw)),   # bottom
            ((-hl, hw), (hl, hw)),     # top
            ((-hl, -hw), (-hl, hw)),   # left
            ((hl, -hw), (hl, hw)),     # right
        ]
        for a, b in wall_segments:
            seg = pymunk.Segment(space.static_body, a, b, t)
            seg.elasticity = self._cushion_elasticity
            seg.friction = CUSHION_FRICTION
            seg.collision_type = 0
            space.add(seg)

    def _add_robot(self, space: pymunk.Space, pos: tuple[float, float], angle: float):
        mass = ROBOT_MASS
        moment = pymunk.moment_for_box(mass, (ROBOT_LENGTH, ROBOT_WIDTH))
        body = pymunk.Body(mass, moment)
        body.position = pos
        body.angle = angle
        shape = pymunk.Poly.create_box(body, (ROBOT_LENGTH, ROBOT_WIDTH))
        shape.elasticity = 0.2
        shape.friction = 0.8
        shape.collision_type = 1
        space.add(body, shape)
        self._robot_body = body
        self._robot_shape = shape

    def _add_ball(self, space: pymunk.Space, ball_id: int, pos: tuple[float, float], color: tuple):
        moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        body = pymunk.Body(BALL_MASS, moment)
        body.position = pos
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = self._ball_elasticity
        shape.friction = BALL_FRICTION
        shape.collision_type = 2
        space.add(body, shape)
        self._ball_bodies[ball_id] = body
        self._ball_shapes[ball_id] = shape
        self._ball_colors[ball_id] = color

    def _random_table_pos(self, margin: float = 0.05) -> tuple[float, float]:
        hl = TABLE_LENGTH / 2 - margin
        hw = TABLE_WIDTH / 2 - margin
        return (random.uniform(-hl, hl), random.uniform(-hw, hw))

    # ---- reset ----------------------------------------------------------------

    def reset(self, setup: str = "default", layout=None, seed: int | None = None,
              aim: float | None = None) -> Observation:
        """Reset the table.

        setup="default":    robot at a fixed spot, cue ball at a fixed spot, object balls random.
        setup="behind_cue": cue ball at a random spot; robot front 3-7 cm behind it along a
                            random aim line, heading within +/-45 deg of that line; object balls
                            random and non-overlapping.

        layout (multiball mode only): a dict {id: (x, y)} in metres (0 = cue, 1.. = object balls).
            When given it is used verbatim instead of sampling; must list the cue and exactly
            n_object_balls object balls. When None a seeded legal layout is generated.
        seed: seed the layout RNG in multiball mode (reproducible without touching global random).
        aim: (multiball, setup="behind_cue" only) aim angle in radians, robot -> cue direction.
            None keeps the random aim. The random aim is still drawn and discarded when aim is
            given, so every other draw (cue position, distance, offset, layout) is identical to
            the aim=None case for the same seed.
        """
        if aim is not None and (self._legacy or setup != "behind_cue"):
            raise ValueError("aim is only supported for multiball mode with setup='behind_cue'")
        self._space = self._create_space()
        self._ball_bodies.clear()
        self._ball_shapes.clear()
        self._ball_colors.clear()
        self._potted = []
        self._pot_events = []
        self._scratch = False

        self._add_walls(self._space)

        if self._legacy:
            self._reset_legacy(setup)
        else:
            self._reset_multiball(setup, layout, seed, aim)

        self.last_strike_connected = False
        return self._get_observation()

    def _reset_legacy(self, setup: str):
        """The original 4-ball reset, unchanged (same RNG draw order and values)."""
        if setup == "default":
            robot_pos, robot_angle = (-TABLE_LENGTH / 4, 0.0), 0.0
            ball_pos = {"cue": (0.0, 0.0)}
            for name in ("red", "blue", "yellow"):
                ball_pos[name] = self._random_table_pos()
            self.last_setup = {}
        elif setup == "behind_cue":
            robot_pos, robot_angle, ball_pos = self._sample_behind_cue_setup()
        else:
            raise ValueError(f"Unknown setup: {setup}")

        self._add_robot(self._space, robot_pos, robot_angle)
        for ball_id, name in enumerate(LEGACY_BALL_NAMES):
            self._add_ball(self._space, ball_id, ball_pos[name], BALL_COLORS[name])

    def _reset_multiball(self, setup: str, layout, seed, aim=None):
        rng = random.Random(seed) if seed is not None else random
        n = self._n_object_balls

        if setup == "default":
            robot_pos, robot_angle = (-TABLE_LENGTH / 4, 0.0), 0.0
            cue_pos = self._cue_spawn
        elif setup == "behind_cue":
            robot_pos, robot_angle, cue_pos = self._sample_behind_cue_robot_cue(rng, aim)
        else:
            raise ValueError(f"Unknown setup: {setup}")
        self.last_setup = {} if setup == "default" else self.last_setup

        if layout is None:
            avoid = lambda pt: self._overlaps_robot(pt, robot_pos, robot_angle, BALL_RADIUS + 0.005)
            layout = random_layout(n, rng, cue_pos=cue_pos, pocket_radius=self._pocket_radius,
                                   avoid=avoid)
        else:
            layout = {int(k): (float(v[0]), float(v[1])) for k, v in layout.items()}
            expected = set(range(0, n + 1))
            if set(layout) != expected:
                raise ValueError(f"layout ids {sorted(layout)} != expected {sorted(expected)}")

        self._add_robot(self._space, robot_pos, robot_angle)
        self._add_ball(self._space, CUE_ID, layout[CUE_ID], BALL_COLORS["cue"])
        for i in range(1, n + 1):
            self._add_ball(self._space, i, layout[i], OBJECT_COLORS[(i - 1) % len(OBJECT_COLORS)])

    @staticmethod
    def _robot_corners(pos, angle):
        c, s = math.cos(angle), math.sin(angle)
        out = []
        for lx, ly in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            px, py = lx * ROBOT_LENGTH / 2, ly * ROBOT_WIDTH / 2
            out.append((pos[0] + c * px - s * py, pos[1] + s * px + c * py))
        return out

    @staticmethod
    def _inside_table(x, y, margin):
        lim_x = TABLE_LENGTH / 2 - CUSHION_THICKNESS - margin
        lim_y = TABLE_WIDTH / 2 - CUSHION_THICKNESS - margin
        return abs(x) <= lim_x and abs(y) <= lim_y

    @staticmethod
    def _overlaps_robot(pt, pos, angle, clearance):
        """True if pt lies inside the robot box expanded by clearance."""
        c, s = math.cos(-angle), math.sin(-angle)
        dx, dy = pt[0] - pos[0], pt[1] - pos[1]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= ROBOT_LENGTH / 2 + clearance and abs(ly) <= ROBOT_WIDTH / 2 + clearance

    def _sample_behind_cue_setup(self):
        """Legacy behind_cue sampler: places the cue plus red/blue/yellow. Unchanged."""
        ball_margin = BALL_RADIUS + 0.005
        for _ in range(SETUP_MAX_TRIES):
            cue = self._random_table_pos(margin=CUSHION_THICKNESS + ball_margin)
            aim = random.uniform(-math.pi, math.pi)          # direction robot -> cue
            dist = random.uniform(*SETUP_DISTANCE_RANGE)     # robot front -> cue center
            offset = random.uniform(*SETUP_AIM_OFFSET_RANGE)
            heading = aim + offset

            front = (cue[0] - dist * math.cos(aim), cue[1] - dist * math.sin(aim))
            pos = (front[0] - ROBOT_LENGTH / 2 * math.cos(heading),
                   front[1] - ROBOT_LENGTH / 2 * math.sin(heading))

            if not all(self._inside_table(x, y, 0.005) for x, y in self._robot_corners(pos, heading)):
                continue
            if self._overlaps_robot(cue, pos, heading, BALL_RADIUS):
                continue  # large aim offsets at short range can put the cue ball inside the robot

            balls = {"cue": cue}
            ok = True
            for name in ("red", "blue", "yellow"):
                for _ in range(100):
                    q = self._random_table_pos(margin=CUSHION_THICKNESS + ball_margin)
                    if self._overlaps_robot(q, pos, heading, BALL_RADIUS + 0.005):
                        continue
                    if any(math.hypot(q[0] - b[0], q[1] - b[1]) < 2 * BALL_RADIUS + 0.005
                           for b in balls.values()):
                        continue
                    balls[name] = q
                    break
                else:
                    ok = False
                    break
            if not ok:
                continue

            self.last_setup = {"distance": dist, "aim_offset": offset}
            return pos, heading, balls
        raise RuntimeError("Could not sample a valid behind_cue setup")

    def _sample_behind_cue_robot_cue(self, rng, aim_override=None):
        """Multiball behind_cue: sample only the robot pose and cue position (object balls are
        placed afterwards by random_layout). Same geometry as the legacy sampler. If aim_override
        is given it replaces the sampled aim (the draw is still consumed)."""
        ball_margin = BALL_RADIUS + 0.005
        hl = TABLE_LENGTH / 2 - (CUSHION_THICKNESS + ball_margin)
        hw = TABLE_WIDTH / 2 - (CUSHION_THICKNESS + ball_margin)
        for _ in range(SETUP_MAX_TRIES):
            cue = (rng.uniform(-hl, hl), rng.uniform(-hw, hw))
            if not _clear_of_pockets(cue[0], cue[1], self._pocket_radius):
                continue
            aim = rng.uniform(-math.pi, math.pi)
            if aim_override is not None:
                aim = float(aim_override)
            dist = rng.uniform(*SETUP_DISTANCE_RANGE)
            offset = rng.uniform(*SETUP_AIM_OFFSET_RANGE)
            heading = aim + offset
            front = (cue[0] - dist * math.cos(aim), cue[1] - dist * math.sin(aim))
            pos = (front[0] - ROBOT_LENGTH / 2 * math.cos(heading),
                   front[1] - ROBOT_LENGTH / 2 * math.sin(heading))
            if not all(self._inside_table(x, y, 0.005) for x, y in self._robot_corners(pos, heading)):
                continue
            if self._overlaps_robot(cue, pos, heading, BALL_RADIUS):
                continue
            self.last_setup = {"distance": dist, "aim_offset": offset}
            return pos, heading, cue
        raise RuntimeError("Could not sample a valid behind_cue setup")

    # ---- stepping -------------------------------------------------------------

    def step(self, action: Action, render: bool = True) -> Observation:
        """Advance one control step (DT). render=False skips drawing the frame (obs["frame"]
        is None), for fast physics-only rollouts such as letting a shot settle."""
        left_pwm = max(-1.0, min(1.0, action["left_pwm"]))
        right_pwm = max(-1.0, min(1.0, action["right_pwm"]))

        # Differential-drive kinematics: wheel speeds set the body's forward speed and yaw
        # rate directly (no sideways slip, no coasting; zero PWM = stopped). The velocity
        # is re-imposed every substep, so the robot pushes balls but is not pushed back.
        vl = left_pwm * MAX_WHEEL_SPEED
        vr = right_pwm * MAX_WHEEL_SPEED
        v_linear = (vl + vr) / 2.0
        omega = (vr - vl) / WHEEL_SEPARATION

        if action.get("fire", False):
            force = max(0.0, min(1.0, action.get("fire_force", 0.5)))
            impulse_mag = force * MAX_STRIKE_IMPULSE
            heading = self._robot_body.angle
            impulse = (impulse_mag * math.cos(heading), impulse_mag * math.sin(heading))

            robot_front = self._robot_body.local_to_world((ROBOT_LENGTH / 2, 0))
            cue_pos = self._ball_bodies[CUE_ID].position
            dist = math.hypot(cue_pos.x - robot_front.x, cue_pos.y - robot_front.y)

            self.last_strike_connected = dist < ROBOT_LENGTH / 2 + BALL_RADIUS + 0.02
            if self.last_strike_connected:
                self._ball_bodies[CUE_ID].apply_impulse_at_local_point(impulse, (0, 0))
        else:
            self.last_strike_connected = False

        sub_dt = DT / SUBSTEPS
        rb = self._robot_body
        for _ in range(SUBSTEPS):
            rb.velocity = (v_linear * math.cos(rb.angle), v_linear * math.sin(rb.angle))
            rb.angular_velocity = omega
            self._space.step(sub_dt)
            self._apply_rolling_resistance(sub_dt)
            if not self._legacy:
                self._check_pockets()

        for body in self._ball_bodies.values():
            body.angular_velocity *= ANGULAR_DAMPING

        return self._get_observation(render)

    def _apply_rolling_resistance(self, dt: float):
        dv = ROLLING_DECEL * dt
        for body in self._ball_bodies.values():
            speed = body.velocity.length
            body.velocity = (0.0, 0.0) if speed <= dv else body.velocity * (1.0 - dv / speed)

    def _check_pockets(self):
        """Pot any ball whose centre is within pocket_radius of a pocket. Object balls are
        removed and recorded (in order); the cue ball is a scratch and is respawned. Called
        every substep so a fast ball is caught at its closest approach to a pocket."""
        r2 = self._pocket_r2
        nx, ny, pockets = self._near_x, self._near_y, self._pockets
        potted_now = None
        for ball_id, body in self._ball_bodies.items():
            pos = body.position
            x, y = pos.x, pos.y
            if -nx < x < nx and -ny < y < ny:
                continue  # interior: too far from every pocket
            for k, (px, py) in enumerate(pockets):
                dx, dy = x - px, y - py
                if dx * dx + dy * dy <= r2:
                    if potted_now is None:
                        potted_now = []
                    potted_now.append((ball_id, k))
                    break
        if potted_now:
            self._pot_events.extend(potted_now)
            for ball_id, _ in potted_now:
                if ball_id == CUE_ID:
                    self._scratch = True
                    self._respawn_cue()
                else:
                    self._remove_ball(ball_id)
                    self._potted.append(ball_id)

    def _remove_ball(self, ball_id: int):
        self._space.remove(self._ball_bodies[ball_id], self._ball_shapes[ball_id])
        del self._ball_bodies[ball_id]
        del self._ball_shapes[ball_id]
        del self._ball_colors[ball_id]

    def _spawn_point_legal(self, pt) -> bool:
        """True if a cue ball centred at pt overlaps no other ball, the robot, a pocket or a cushion."""
        x, y = pt
        if not _center_in_play(x, y, BALL_RADIUS + 0.003):
            return False
        if not _clear_of_pockets(x, y, self._pocket_radius):
            return False
        rb = self._robot_body
        if self._overlaps_robot(pt, (rb.position.x, rb.position.y), rb.angle, BALL_RADIUS + 0.002):
            return False
        min_sep = 2 * BALL_RADIUS + 0.002
        for ball_id, body in self._ball_bodies.items():
            if ball_id != CUE_ID and math.hypot(x - body.position.x, y - body.position.y) < min_sep:
                return False
        return True

    def _find_cue_spawn(self) -> tuple[float, float]:
        """cue_spawn if legal, else the closest legal point found by searching outward in 1 mm
        rings (72 directions per ring). Deterministic."""
        sx, sy = self._cue_spawn
        if self._spawn_point_legal((sx, sy)):
            return (sx, sy)
        for k in range(1, int(max(TABLE_LENGTH, TABLE_WIDTH) / 0.001)):
            r = k * 0.001
            for j in range(72):
                a = 2 * math.pi * j / 72
                pt = (sx + r * math.cos(a), sy + r * math.sin(a))
                if self._spawn_point_legal(pt):
                    return pt
        raise RuntimeError("no legal cue respawn point on the table")

    def _respawn_cue(self):
        body = self._ball_bodies[CUE_ID]
        body.position = self._find_cue_spawn()
        body.velocity = (0.0, 0.0)
        body.angular_velocity = 0.0

    def settle(self, max_steps: int = 600) -> tuple[Observation, int]:
        """Step with the robot idle (no rendering) until all balls stop or max_steps pass.
        Returns the final observation (rendered) and the number of steps taken. A shot is only
        finished once every ball on the table is below the settle velocity threshold."""
        idle = Action(left_pwm=0.0, right_pwm=0.0, fire=False, fire_force=0.0)
        n = 0
        for n in range(1, max_steps + 1):
            self.step(idle, render=False)
            if self._balls_settled():
                break
        return self._get_observation(), n

    def _balls_settled(self) -> bool:
        for body in self._ball_bodies.values():
            speed = body.velocity.length
            if speed > SETTLE_VELOCITY_THRESHOLD:
                return False
        return True

    # ---- observation ----------------------------------------------------------

    def ball_slots(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed-slot view of the balls (multiball mode only).

        Returns (positions, mask): positions is float64 (8, 2) in metres indexed by ball id
        (0 = cue, 1..7); mask is bool (8,), True if that ball is on the table. Absent or potted
        balls have position (0, 0) and mask False."""
        if self._legacy:
            raise RuntimeError("ball_slots() is only available in multiball mode (n_object_balls=...)")
        positions = np.zeros((MAX_OBJECT_BALLS + 1, 2))
        mask = np.zeros(MAX_OBJECT_BALLS + 1, dtype=bool)
        for ball_id, body in self._ball_bodies.items():
            positions[ball_id] = (body.position.x, body.position.y)
            mask[ball_id] = True
        return positions, mask

    def _get_observation(self, render: bool = True) -> Observation:
        rb = self._robot_body
        ball_positions = [
            (float(self._ball_bodies[i].position.x), float(self._ball_bodies[i].position.y))
            for i in sorted(self._ball_bodies)
        ]
        obs = Observation(
            frame=self.render() if render else None,
            robot_pose=(float(rb.position.x), float(rb.position.y), float(rb.angle)),
            ball_positions=ball_positions,
            ball_settled=self._balls_settled(),
        )
        if not self._legacy:
            obs["ball_positions_cm"] = {
                i: (float(b.position.x) * 100.0, float(b.position.y) * 100.0)
                for i, b in sorted(self._ball_bodies.items())
            }
            obs["potted"] = list(self._potted)
            # (ball_id, pocket index into pocket_centers()) per pot since reset, in order. The
            # cue (id 0) appears here on a scratch; `potted` stays object balls only.
            obs["potted_pockets"] = list(self._pot_events)
            obs["scratch"] = self._scratch
            obs["n_balls_on_table"] = len(self._ball_bodies)
        return obs

    # ---- rendering ------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Return a (RENDER_HEIGHT, RENDER_WIDTH, 3) uint8 frame at a uniform px/m scale."""
        dpi = 100
        fig = plt.figure(figsize=(RENDER_WIDTH / dpi, RENDER_HEIGHT / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        half_w = TABLE_LENGTH / 2 + RENDER_MARGIN
        half_h = half_w * RENDER_HEIGHT / RENDER_WIDTH
        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_h, half_h)
        ax.set_facecolor("#1a5c2a")
        ax.axis("off")
        fig.patch.set_facecolor("#1a5c2a")

        if not self._legacy:
            self._draw_multiball_table(ax)
        else:
            self._draw_legacy_table(ax)

        # Balls, drawn in id order (cue first). The cue ball is rendered exactly as before.
        for ball_id in sorted(self._ball_bodies):
            body = self._ball_bodies[ball_id]
            circle = plt.Circle(
                (body.position.x, body.position.y),
                BALL_RADIUS,
                color=self._ball_colors[ball_id],
                ec="black",
                linewidth=0.5,
            )
            ax.add_patch(circle)

        # Robot
        rb = self._robot_body
        cos_a = math.cos(rb.angle)
        sin_a = math.sin(rb.angle)

        corners_local = [
            (-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
            (ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2),
            (ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
            (-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2),
        ]
        corners_world = []
        for lx, ly in corners_local:
            wx = rb.position.x + lx * cos_a - ly * sin_a
            wy = rb.position.y + lx * sin_a + ly * cos_a
            corners_world.append((wx, wy))

        robot_patch = plt.Polygon(corners_world, color="#555555", ec="black", linewidth=0.5)
        ax.add_patch(robot_patch)

        # Heading indicator
        front_x = rb.position.x + (ROBOT_LENGTH / 2) * cos_a
        front_y = rb.position.y + (ROBOT_LENGTH / 2) * sin_a
        ax.plot(
            [rb.position.x, front_x],
            [rb.position.y, front_y],
            color="orange",
            linewidth=1.5,
        )

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)
        assert img.shape == (RENDER_HEIGHT, RENDER_WIDTH, 3), img.shape
        return img

    def _draw_multiball_table(self, ax):
        """Multiball table drawn at the physical geometry: the felt ends exactly where a ball
        surface meets the rail (+-hl - cushion radius), so a ball at rest against a cushion is
        drawn touching it. Rail is a filled band out to the old outer outline; pockets on top."""
        hl, hw = TABLE_LENGTH / 2, TABLE_WIDTH / 2
        t = self._cushion_radius
        ax.add_patch(patches.Rectangle(
            (-hl - CUSHION_THICKNESS, -hw - CUSHION_THICKNESS),
            TABLE_LENGTH + 2 * CUSHION_THICKNESS, TABLE_WIDTH + 2 * CUSHION_THICKNESS,
            linewidth=0, facecolor=RAIL_COLOR, zorder=0))
        ax.add_patch(patches.Rectangle(
            (-hl + t, -hw + t), TABLE_LENGTH - 2 * t, TABLE_WIDTH - 2 * t,
            linewidth=0, facecolor=FELT_COLOR, zorder=0))
        for px, py in pocket_centers():
            ax.add_patch(plt.Circle((px, py), self._pocket_radius, color="#0a0a0a", zorder=1))

    def _draw_legacy_table(self, ax):
        """Legacy table drawing, unchanged (render hashes in tests/baseline_legacy.json)."""
        # Cushions
        cushion = patches.Rectangle(
            (-TABLE_LENGTH / 2 - CUSHION_THICKNESS, -TABLE_WIDTH / 2 - CUSHION_THICKNESS),
            TABLE_LENGTH + 2 * CUSHION_THICKNESS,
            TABLE_WIDTH + 2 * CUSHION_THICKNESS,
            linewidth=2,
            edgecolor="#5c3a1a",
            facecolor="none",
        )
        ax.add_patch(cushion)

        inner = patches.Rectangle(
            (-TABLE_LENGTH / 2, -TABLE_WIDTH / 2),
            TABLE_LENGTH,
            TABLE_WIDTH,
            linewidth=1,
            edgecolor="#2a7a3a",
            facecolor="#1a6c2a",
        )
        ax.add_patch(inner)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    env = SimEnv()
    obs = env.reset()
    print(f"Table: {TABLE_LENGTH:.4f}m x {TABLE_WIDTH:.4f}m")
    print(f"Robot pose: {obs['robot_pose']}")
    print(f"Ball positions: {obs['ball_positions']}")

    for i in range(100):
        action = Action(
            left_pwm=random.uniform(-1, 1),
            right_pwm=random.uniform(-1, 1),
            fire=(i == 50),
            fire_force=0.8,
        )
        obs = env.step(action)

    print(f"\nAfter 100 steps:")
    print(f"Robot pose: {obs['robot_pose']}")
    print(f"Ball positions: {obs['ball_positions']}")
    print(f"Balls settled: {obs['ball_settled']}")

    frame = obs["frame"]
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    plt.figure(figsize=(5, 5))
    plt.imshow(frame)
    plt.title("SimEnv — after 100 random steps")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
