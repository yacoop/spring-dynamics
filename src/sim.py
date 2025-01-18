import numpy as np
import numba as nb
import pygame as pg
from pygame import gfxdraw
import time


class Simulation:
    def __init__(self):
        pg.init()

        self.WIDTH = 1280
        self.HEIGHT = 720

        self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pg.time.Clock()

        self.NUM_NODES = 50
        self.FIXED_COUNT = 2
        self.SPRING_LENGTH = 300

        self.FIXED_TIMESTEP = 1e-4
        self.SIMULATION_SPEED = 2

        # self.LINEAR_CONSTANT = 10000.0
        self.LINEAR_CONSTANT = 114230.0
        self.LINEAR_ASYMYMPTOTIC_CONSTANT = 1.0

        self.ANGLE_CONSTANT = 1000000.0
        self.ANGLE_ASYMPTOTIC_CONSTANT = 1.0

        self.EQUILIBRIUM_LENGTH = 20
        self.EQUILIBRIUM_ANGLE = 30
        self.MINIMUM_LENGTH = 0.8 * self.EQUILIBRIUM_LENGTH

        self.DAMPING = 0.01

        self.SLEEVE_WIDTH = 40

        self.positions = self.setup_poses()
        self.velocities = np.zeros_like(self.positions)

        self.forces = np.zeros_like(self.positions)
        self.linear_forces = np.zeros_like(self.positions)
        self.angular_forces = np.zeros_like(self.positions)

        self.angles = np.zeros(len(self.positions) - 2)
        self.chains = np.diff(self.positions)

        self.running = False

        self.t = 0
        self.previous_time = 0.0
        self.current_time = time.perf_counter()
        self.delta_time = 0.0
        self.accumulator = 0.0

        self.font = pg.font.SysFont("Arial", 20)

    def setup_poses(self):
        x_coords = 10 + np.arange(self.NUM_NODES) / self.NUM_NODES * self.SPRING_LENGTH
        y_coords = self.HEIGHT // 2 + (
            (np.arange(self.NUM_NODES) % 2 - 0.5) * self.EQUILIBRIUM_LENGTH
        )
        positions = np.column_stack((x_coords, y_coords))

        return positions

    def run(self):
        self.precompile()

        self.running = True
        self.previous_time = time.perf_counter()
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

            self.current_time = time.perf_counter()
            self.delta_time = self.current_time - self.previous_time
            self.previous_time = self.current_time
            self.accumulator += self.delta_time

            while self.accumulator >= self.FIXED_TIMESTEP:
                self.update()
                self.accumulator -= self.FIXED_TIMESTEP
                self.t += self.FIXED_TIMESTEP

            # self.draw_with_aa(debug=True)
            self.draw(debug=False)

            self.clock.tick(300)

    def draw(self, debug):
        BG_COLOR = (210, 210, 210)
        self.screen.fill(BG_COLOR)
        # connections
        for pos1, pos2 in zip(self.positions, self.positions[1:]):
            pg.draw.line(self.screen, (0, 0, 0), pos1, pos2, 2)

        for i in range(len(self.positions)):
            # nodes
            if i < self.FIXED_COUNT or i > self.NUM_NODES - 1 - self.FIXED_COUNT:
                pg.draw.circle(
                    self.screen,
                    (255, 0, 0),
                    self.positions[i],
                    2,
                )

            else:
                pg.draw.circle(
                    self.screen,
                    (255, 255, 255),
                    self.positions[i],
                    2,
                )

        pg.draw.line(
            self.screen,
            (100, 100, 100),
            np.array([0, self.SLEEVE_WIDTH + self.HEIGHT // 2]),
            np.array([self.WIDTH, self.SLEEVE_WIDTH + self.HEIGHT // 2]),
            2,
        )

        pg.draw.line(
            self.screen,
            (100, 100, 100),
            np.array([0, -self.SLEEVE_WIDTH + self.HEIGHT // 2]),
            np.array([self.WIDTH, -self.SLEEVE_WIDTH + self.HEIGHT // 2]),
            2,
        )

        if debug:
            total_force = self.font.render("TOTAL FORCE", True, (255, 0, 0))
            self.screen.blit(total_force, [20, 60])

            linear_force = self.font.render("LINEAR FORCE", True, (0, 255, 0))
            self.screen.blit(linear_force, [20, 80])

            angular_force = self.font.render("ANGULAR FORCE", True, (0, 0, 255))
            self.screen.blit(angular_force, [20, 100])
            for i in range(len(self.positions)):
                # total force vector (red)
                pg.draw.line(
                    self.screen,
                    (255, 0, 0),
                    self.positions[i],
                    self.positions[i] + self.forces[i],
                    2,
                )

                # linear force vector (green)
                pg.draw.line(
                    self.screen,
                    (0, 255, 0),
                    self.positions[i],
                    self.positions[i] + self.linear_forces[i],
                    2,
                )

                # angular force vector (green)
                pg.draw.line(
                    self.screen,
                    (0, 0, 255),
                    self.positions[i],
                    self.positions[i] + self.angular_forces[i],
                    2,
                )

            for position, angle in zip(self.positions[1:-1], self.angles):
                angle_text = self.font.render(f"{angle:.0f}", True, (255, 255, 255))
                self.screen.blit(angle_text, position)

        fps_counter = self.font.render(
            f"{self.clock.get_fps():.0f}", True, (255, 255, 255)
        )
        self.screen.blit(fps_counter, [20, 20])

        timer = self.font.render(f"{self.t:.0f}", True, (255, 255, 255))
        self.screen.blit(timer, [20, 40])

        pg.display.flip()

    def draw_with_aa(self, debug):
        BG_COLOR = (210, 210, 210)
        self.screen.fill(BG_COLOR)
        # connections
        for pos1, pos2 in zip(self.positions, self.positions[1:]):
            gfxdraw.aapolygon(self.screen, (pos1, pos1, pos2, pos2), (0, 0, 0))

        for i in range(len(self.positions)):
            # nodes
            gfxdraw.aacircle(
                self.screen,
                int(self.positions[i][0]),
                int(self.positions[i][1]),
                5,
                (255, 255, 255),
            )
            gfxdraw.filled_circle(
                self.screen,
                int(self.positions[i][0]),
                int(self.positions[i][1]),
                5,
                (255, 255, 255),
            )

        if debug:
            total_force = self.font.render("TOTAL FORCE", True, (255, 0, 0))
            self.screen.blit(total_force, [20, 60])

            linear_force = self.font.render("LINEAR FORCE", True, (0, 255, 0))
            self.screen.blit(linear_force, [20, 80])

            angular_force = self.font.render("ANGULAR FORCE", True, (0, 0, 255))
            self.screen.blit(angular_force, [20, 100])
            for i in range(len(self.positions)):
                # total force vector (red)
                gfxdraw.aapolygon(
                    self.screen,
                    (
                        self.positions[i],
                        self.positions[i],
                        self.positions[i] + self.forces[i],
                        self.positions[i] + self.forces[i],
                    ),
                    (255, 0, 0),
                )

                # linear force vector (green)
                gfxdraw.aapolygon(
                    self.screen,
                    (
                        self.positions[i],
                        self.positions[i],
                        self.positions[i] + self.linear_forces[i],
                        self.positions[i] + self.linear_forces[i],
                    ),
                    (0, 255, 0),
                )

                # angular force vector (green)
                gfxdraw.aapolygon(
                    self.screen,
                    (
                        self.positions[i],
                        self.positions[i],
                        self.positions[i] + self.angular_forces[i],
                        self.positions[i] + self.angular_forces[i],
                    ),
                    (0, 0, 255),
                )

            fps_counter = self.font.render(
                f"{self.clock.get_fps():.0f}", True, (255, 255, 255)
            )
            self.screen.blit(fps_counter, [20, 20])

            timer = self.font.render(f"{self.t:.0f}", True, (255, 255, 255))
            self.screen.blit(timer, [20, 40])

            for position, angle in zip(self.positions[1:-1], self.angles):
                angle_text = self.font.render(f"{angle:.0f}", True, (255, 255, 255))
                self.screen.blit(angle_text, position)
        pg.display.flip()

    def get_physics_args(self):
        return (
            self.positions,
            self.velocities,
            self.forces,
            self.FIXED_TIMESTEP,
            self.LINEAR_CONSTANT,
            self.EQUILIBRIUM_LENGTH,
            self.ANGLE_CONSTANT,
            self.EQUILIBRIUM_ANGLE,
            self.DAMPING,
            self.ANGLE_ASYMPTOTIC_CONSTANT,
            self.LINEAR_ASYMYMPTOTIC_CONSTANT,
            self.MINIMUM_LENGTH,
            self.SLEEVE_WIDTH,
            self.FIXED_COUNT,
            self.t,
        )

    def precompile(self):
        physics_process(*self.get_physics_args())

    def update(self):
        (
            self.positions,
            self.velocities,
            self.forces,
            self.linear_forces,
            self.angular_forces,
            self.angles,
        ) = physics_process(*self.get_physics_args())


@nb.njit(cache=True)
def physics_process(
    positions,
    velocities,
    forces,
    FIXED_TIMESTEP,
    LINEAR_CONSTANT,
    EQUILIBRIUM_LENGTH,
    ANGLE_CONSTANT,
    EQUILIBRIUM_ANGLE,
    DAMPING,
    ANGLE_ASYMPTOTIC_CONSTANT,
    LINEAR_ASYMPTOTIC_CONSTANT,
    MINIMUM_LENGTH,
    SLEEVE_WIDTH,
    FIXED_COUNT,
    t,
):
    vel_temp = np.zeros_like(velocities)
    # fixed_velocity = np.array([30*np.sin(t/60*np.pi), 0])
    fixed_velocity = np.array([30 * np.sign(25 - t), 0])
    # fixed_velocity = np.array([30, 0])
    for i in range(FIXED_COUNT, len(positions) - FIXED_COUNT):
        vel_temp[i] = velocities[i] + forces[i] * 0.5 * FIXED_TIMESTEP
        positions[i] += vel_temp[i] * FIXED_TIMESTEP

    positions[:, 1] = np.clip(positions[:, 1], -SLEEVE_WIDTH + 360, SLEEVE_WIDTH + 360)

    positions[-FIXED_COUNT:] += fixed_velocity * FIXED_TIMESTEP

    linear_forces, angular_forces, angles = calc_forces(
        positions,
        LINEAR_CONSTANT,
        EQUILIBRIUM_LENGTH,
        ANGLE_CONSTANT,
        EQUILIBRIUM_ANGLE,
        ANGLE_ASYMPTOTIC_CONSTANT,
        LINEAR_ASYMPTOTIC_CONSTANT,
        MINIMUM_LENGTH,
    )
    forces = linear_forces + angular_forces

    for i in range(FIXED_COUNT, len(positions) - FIXED_COUNT):
        velocities[i] = vel_temp[i] + 0.5 * forces[i] * FIXED_TIMESTEP
        velocities[i] *= 1 - DAMPING

    return positions, velocities, forces, linear_forces, angular_forces, angles


@nb.njit(cache=True)
def calc_forces(
    positions,
    LINEAR_CONSTANT,
    EQUILIBRIUM_LENGTH,
    ANGLE_CONSTANT,
    EQUILIBRIUM_ANGLE,
    ANGLE_ASYMPTOTIC_CONSTANT,
    LINEAR_ASYMPTOTIC_CONSTANT,
    MINIMUM_LENGTH,
):
    ds, unit_vectors = get_ds_and_unit_vectors(positions)
    linear_forces = np.zeros_like(positions)
    angular_forces = np.zeros_like(positions)
    angles = np.zeros(len(positions) - 1)
    for i, (d, uv) in enumerate(zip(ds, unit_vectors)):
        delta_d = (
            d - EQUILIBRIUM_LENGTH - LINEAR_ASYMPTOTIC_CONSTANT / (d - MINIMUM_LENGTH)
        )
        linear_force = LINEAR_CONSTANT * -delta_d * uv
        linear_forces[i] -= linear_force
        linear_forces[i + 1] += linear_force

    for i, ((uv1, uv2), (d1, d2)) in enumerate(
        zip(zip(unit_vectors[:-1], unit_vectors[1:]), zip(ds[:-1], ds[1:]))
    ):
        # NumbaPerformanceWarning: np.dot() is faster on contiguous arrays

        # reversed so it creates an angle
        uv1 = np.ascontiguousarray(-uv1)
        uv2 = np.ascontiguousarray(uv2)

        angles[i] = get_angle(uv1, uv2)
        direction1, direction2 = get_angular_force_directions(uv1, uv2)
        delta_theta = (
            angles[i] - EQUILIBRIUM_ANGLE - ANGLE_ASYMPTOTIC_CONSTANT / angles[i]
        )
        torque = -ANGLE_CONSTANT * delta_theta
        angular_forces[i] -= torque / d1 * direction1
        angular_forces[i + 1] += torque / d1 * direction1 + torque / d2 * direction2
        angular_forces[i + 2] -= torque / d2 * direction2

    return linear_forces, angular_forces, angles


@nb.njit(cache=True)
def get_ds_and_unit_vectors(positions):
    vectors = positions[1:] - positions[:-1]
    d = norm(vectors)
    unit_vectors = vectors / d[:, np.newaxis]

    return d, unit_vectors


@nb.njit(cache=True)
def norm(a):
    norms = np.empty(a.shape[0], dtype=a.dtype)
    for i in nb.prange(a.shape[0]):
        norms[i] = np.sqrt(a[i, 0] * a[i, 0] + a[i, 1] * a[i, 1])

    return norms


@nb.njit(cache=True)
def get_angle(v1, v2):
    sin_theta = np.dot(v1, v2)
    sin_theta = max(-1.0, min(1.0, sin_theta))
    angle = np.arccos(sin_theta)

    return np.rad2deg(angle)


@nb.jit(cache=True)
def get_angular_force_directions(v1, v2):
    plane = v2 - v1
    dir1 = np.array([v1[1], -v1[0]])
    dir2 = np.array([v2[1], -v2[0]])
    s1 = np.sign(np.dot(dir1, plane))
    s2 = -np.sign(np.dot(dir2, plane))
    dir1 *= s1
    dir2 *= s2

    return dir1, dir2
