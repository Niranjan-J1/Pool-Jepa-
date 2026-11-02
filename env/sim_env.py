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

# Ball properties
BALL_RADIUS = 0.01
BALL_MASS = 0.02

# Physics tuning
BALL_ELASTICITY = 0.85
BALL_FRICTION = 0.4
CUSHION_ELASTICITY = 0.7
CUSHION_FRICTION = 0.3
LINEAR_DAMPING = 0.97
ANGULAR_DAMPING = 0.95

# Solenoid
MAX_STRIKE_IMPULSE = 0.05

# "behind_cue" reset: robot front placed behind the cue ball, roughly aimed at it
SETUP_DISTANCE_RANGE = (0.03, 0.07)       # m, robot front -> cue ball center
SETUP_AIM_OFFSET_RANGE = (-math.pi / 4, math.pi / 4)  # heading offset from the aim line
SETUP_MAX_TRIES = 1000

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


class SimEnv(BaseEnv):
    def __init__(self):
        self._space: pymunk.Space | None = None
        self._robot_body: pymunk.Body | None = None
        self._robot_shape: pymunk.Shape | None = None
        self._ball_bodies: dict[str, pymunk.Body] = {}
        self._ball_shapes: dict[str, pymunk.Shape] = {}
        self._fig = None
        self._ax = None
        self.last_setup: dict = {}
        self.last_strike_connected = False

    def _create_space(self) -> pymunk.Space:
        space = pymunk.Space()
        space.gravity = (0, 0)
        space.damping = LINEAR_DAMPING
        return space

    def _add_walls(self, space: pymunk.Space):
        hw = TABLE_WIDTH / 2
        hl = TABLE_LENGTH / 2
        t = CUSHION_THICKNESS

        wall_segments = [
            ((-hl, -hw), (hl, -hw)),   # bottom
            ((-hl, hw), (hl, hw)),     # top
            ((-hl, -hw), (-hl, hw)),   # left
            ((hl, -hw), (hl, hw)),     # right
        ]
        for a, b in wall_segments:
            seg = pymunk.Segment(space.static_body, a, b, t)
            seg.elasticity = CUSHION_ELASTICITY
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

    def _add_ball(self, space: pymunk.Space, name: str, pos: tuple[float, float]):
        moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        body = pymunk.Body(BALL_MASS, moment)
        body.position = pos
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = BALL_ELASTICITY
        shape.friction = BALL_FRICTION
        shape.collision_type = 2
        space.add(body, shape)
        self._ball_bodies[name] = body
        self._ball_shapes[name] = shape

    def _random_table_pos(self, margin: float = 0.05) -> tuple[float, float]:
        hl = TABLE_LENGTH / 2 - margin
        hw = TABLE_WIDTH / 2 - margin
        return (random.uniform(-hl, hl), random.uniform(-hw, hw))

    def reset(self, setup: str = "default") -> Observation:
        """Reset the table.

        setup="default":    robot at a fixed spot, cue ball at the center, targets random.
        setup="behind_cue": cue ball at a random spot; robot front 3-7 cm behind it along a
                            random aim line, heading within +/-45 deg of that line; targets
                            random and non-overlapping.
        """
        self._space = self._create_space()
        self._ball_bodies.clear()
        self._ball_shapes.clear()

        self._add_walls(self._space)

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
        for name in ("cue", "red", "blue", "yellow"):
            self._add_ball(self._space, name, ball_pos[name])

        self.last_strike_connected = False
        return self._get_observation()

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

    def step(self, action: Action) -> Observation:
        left_pwm = max(-1.0, min(1.0, action["left_pwm"]))
        right_pwm = max(-1.0, min(1.0, action["right_pwm"]))

        max_wheel_speed = 0.3  # m/s
        vl = left_pwm * max_wheel_speed
        vr = right_pwm * max_wheel_speed

        v_linear = (vl + vr) / 2.0
        omega = (vr - vl) / WHEEL_SEPARATION

        angle = self._robot_body.angle
        fx = v_linear * math.cos(angle) * ROBOT_MASS * 10
        fy = v_linear * math.sin(angle) * ROBOT_MASS * 10
        self._robot_body.apply_force_at_local_point((fx, 0), (0, 0))
        self._robot_body.torque = omega * self._robot_body.moment * 10

        if action.get("fire", False):
            force = max(0.0, min(1.0, action.get("fire_force", 0.5)))
            impulse_mag = force * MAX_STRIKE_IMPULSE
            heading = self._robot_body.angle
            impulse = (impulse_mag * math.cos(heading), impulse_mag * math.sin(heading))

            robot_front = self._robot_body.local_to_world((ROBOT_LENGTH / 2, 0))
            cue_pos = self._ball_bodies["cue"].position
            dist = math.hypot(cue_pos.x - robot_front.x, cue_pos.y - robot_front.y)

            self.last_strike_connected = dist < ROBOT_LENGTH / 2 + BALL_RADIUS + 0.02
            if self.last_strike_connected:
                self._ball_bodies["cue"].apply_impulse_at_local_point(impulse, (0, 0))
        else:
            self.last_strike_connected = False

        for _ in range(SUBSTEPS):
            self._space.step(DT / SUBSTEPS)

        for body in self._ball_bodies.values():
            body.angular_velocity *= ANGULAR_DAMPING

        return self._get_observation()

    def _balls_settled(self) -> bool:
        for body in self._ball_bodies.values():
            speed = body.velocity.length
            if speed > SETTLE_VELOCITY_THRESHOLD:
                return False
        return True

    def _get_observation(self) -> Observation:
        rb = self._robot_body
        ball_positions = [
            (float(b.position.x), float(b.position.y))
            for b in self._ball_bodies.values()
        ]
        return Observation(
            frame=self.render(),
            robot_pose=(float(rb.position.x), float(rb.position.y), float(rb.angle)),
            ball_positions=ball_positions,
            ball_settled=self._balls_settled(),
        )

    def render(self) -> np.ndarray:
        fig, ax = plt.subplots(1, 1, figsize=(2, 2), dpi=64)
        ax.set_xlim(-TABLE_LENGTH / 2 - 0.03, TABLE_LENGTH / 2 + 0.03)
        ax.set_ylim(-TABLE_WIDTH / 2 - 0.03, TABLE_WIDTH / 2 + 0.03)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a5c2a")
        ax.axis("off")
        fig.patch.set_facecolor("#1a5c2a")

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

        # Balls
        ball_names = ["cue", "red", "blue", "yellow"]
        for name in ball_names:
            if name not in self._ball_bodies:
                continue
            body = self._ball_bodies[name]
            color = BALL_COLORS[name]
            circle = plt.Circle(
                (body.position.x, body.position.y),
                BALL_RADIUS,
                color=color,
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

        fig.tight_layout(pad=0)
        fig.canvas.draw()

        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        plt.close(fig)

        from PIL import Image
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((128, 128), Image.LANCZOS)
        return np.array(pil_img, dtype=np.uint8)


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
