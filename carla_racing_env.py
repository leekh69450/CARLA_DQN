import glob
import sys
import os

carla_root = r"C:\Users\Kangh\Downloads\newCARLA_0.9.10.1\WindowsNoEditor\PythonAPI\carla\dist"

try:
    sys.path.append(
        glob.glob(
            os.path.join(
                carla_root,
                "carla-*%d.%d-%s.egg" % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    "win-amd64"
                )
            )
        )[0]
    )
except IndexError:
    raise RuntimeError("CARLA egg not found. Check your path and Python version.")

import carla
import numpy as np
import random
import pygame
import weakref

from reward import reward_function          # your reward function
from action import get_action_set           # discrete action triplets


class CarlaRacingEnv:
    def __init__(self, host="localhost", port=2000, seed=0,
                 image_width=84, image_height=84,
                 seconds_per_step=0.05, render=False):

        self.host = host
        self.port = port
        self.seed(seed)

        self.image_width = image_width
        self.image_height = image_height
        self.seconds_per_step = seconds_per_step
        self.render_enabled = render

        # --- pygame window for visualization (optional) ---
        self.display = None
        if self.render_enabled:
            pygame.init()
            self.display = pygame.display.set_mode((self.image_width, self.image_height))
            pygame.display.set_caption("CARLA DQN Racing")

        # Connect to server
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(5.0)

        # World & map
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.seconds_per_step
        self.world.apply_settings(settings)

        self.blueprints = self.world.get_blueprint_library()
        self.action_set = get_action_set()

        # Actors
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        # State
        self.image = None
        self.has_collision = False
        self.has_lane_invasion = False
        self.previous_distance = 0.0  # distance to next waypoint at previous step

        # Spectator (third-person view)
        self.spectator = self.world.get_spectator()
        self.follow_spectator = True   # set False if you ever want to disable

        #fix spawning location
        self.fixed_spawn = True
        self.spawn_index = 0   # change this to pick another spawn
        



    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)


    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self):
        """Reset the CARLA world, spawn the vehicle, restart sensors."""
        self._cleanup()

        # Spawn car
        vehicle_bp = self.blueprints.find('vehicle.tesla.model3')
        spawn_points = self.map.get_spawn_points()

        if self.fixed_spawn:
            spawn_point = spawn_points[self.spawn_index % len(spawn_points)]
        else:
            spawn_point = random.choice(spawn_points)

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        while self.vehicle is None:
            # If spawn failed due to collision/occupancy, retry the same spawn
            self.world.tick()
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        # RGB camera
        cam_bp = self.blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.image_width))
        cam_bp.set_attribute("image_size_y", str(self.image_height))
        cam_bp.set_attribute("fov", "110")

        cam_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7)  # front hood position
        )

        self.camera = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle)

        # Camera callback
        self.image = None
        weak_self = weakref.ref(self)
        self.camera.listen(lambda img: CarlaRacingEnv._on_camera(weak_self, img))

        # Collision sensor
        col_bp = self.blueprints.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle)

        weak_self_col = weakref.ref(self)
        self.collision_sensor.listen(
            lambda event: CarlaRacingEnv._on_collision(weak_self_col, event))

        # Lane invasion sensor
        li_bp = self.blueprints.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(
            li_bp, carla.Transform(), attach_to=self.vehicle)

        weak_self_li = weakref.ref(self)
        self.lane_invasion_sensor.listen(
            lambda event: CarlaRacingEnv._on_lane_invasion(weak_self_li, event))

        # Reset flags
        self.has_collision = False
        self.has_lane_invasion = False

        # Step world until first image arrives
        while self.image is None:
            self.world.tick()
            self._update_spectator()

        # Initialize previous_distance for reward function
        loc = self.vehicle.get_transform().location
        current_position = (loc.x, loc.y)
        next_waypoint_loc = self._get_next_waypoint_location(loc)
        next_waypoint = (next_waypoint_loc.x, next_waypoint_loc.y)
        curr_pos_arr = np.array(current_position, dtype=np.float32)
        next_wp_arr = np.array(next_waypoint, dtype=np.float32)
        self.previous_distance = float(np.linalg.norm(curr_pos_arr - next_wp_arr))

        return self._get_state()


    @staticmethod
    def _on_camera(weak_self, img):
        self = weak_self()
        if self is None:
            return

        raw = np.frombuffer(img.raw_data, dtype=np.uint8)
        raw = raw.reshape((img.height, img.width, 4))[:, :, :3]  # drop alpha
        self.image = raw


    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return
        self.has_collision = True


    @staticmethod
    def _on_lane_invasion(weak_self, event):
        self = weak_self()
        if self is None:
            return
        self.has_lane_invasion = True


    def _get_next_waypoint_location(self, location, distance_ahead=2.0):
        """
        Get the location of a waypoint a certain distance in front of the vehicle.
        """
        waypoint = self.map.get_waypoint(location, project_to_road=True)
        next_waypoints = waypoint.next(distance_ahead)
        if len(next_waypoints) == 0:
            # If no next waypoint (end of road), just use current waypoint
            return waypoint.transform.location
        return next_waypoints[0].transform.location
    



    def _update_spectator(self):
        """Attach spectator camera to follow the vehicle (chase cam)."""
        if (not self.follow_spectator) or (self.vehicle is None) or (self.spectator is None):
            return

        t = self.vehicle.get_transform()
        forward = t.get_forward_vector()

        # Tune these for your preferred chase camera
        dist_back = 8.0
        height = 3.0

        loc = t.location + carla.Location(
            x=-forward.x * dist_back,
            y=-forward.y * dist_back,
            z=height
        )

        rot = carla.Rotation(
            pitch=-15.0,
            yaw=t.rotation.yaw,
            roll=0.0
        )

        self.spectator.set_transform(carla.Transform(loc, rot))




    # -------------------------------------------------------------------------
    # STEP
    # -------------------------------------------------------------------------
    def step(self, action):
        # Unpack the action triplet
        steer, throttle, brake = action

        control = carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake)
        )

        self.vehicle.apply_control(control)

        # Advance simulation
        self.world.tick()
        self._update_spectator()

        # Fetch image state
        obs = self._get_state()

        # Compute speed (m/s)
        vel = self.vehicle.get_velocity()
        speed = float(np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2))

        # Current position (x, y)
        loc = self.vehicle.get_transform().location
        current_position = (float(loc.x), float(loc.y))

        # Next waypoint position ahead of vehicle
        next_loc = self._get_next_waypoint_location(loc)
        next_waypoint = (float(next_loc.x), float(next_loc.y))

        # Distance to next waypoint (for updating previous_distance)
        curr_pos_arr = np.array(current_position, dtype=np.float32)
        next_wp_arr = np.array(next_waypoint, dtype=np.float32)
        distance_to_waypoint = float(np.linalg.norm(curr_pos_arr - next_wp_arr))

        # Lane invasion event flag (use once per step)
        lane_invasion = self.has_lane_invasion
        # Reset for next step so we only penalize on the step where it happened
        self.has_lane_invasion = False

        # Compute reward using your reward_function
        reward = reward_function(
            collision=self.has_collision,
            speed=speed,
            lane_invasion=lane_invasion,
            current_position=current_position,
            next_waypoint=next_waypoint,
            previous_distance=self.previous_distance
        )

        # Update previous_distance for next step
        self.previous_distance = distance_to_waypoint

        # Episode done if collision occurred
        done = bool(self.has_collision)

        info = {}

        if self.render_enabled:
            self.render()

        return obs, reward, done, info


    def _get_state(self):
        """Returns image as (C, H, W) float32 normalized."""
        img = self.image  # shape: (H, W, 3), RGB

        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0

        # Convert RGB → grayscale using luma coefficients
        # gray = 0.299 R + 0.587 G + 0.114 B
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        # gray shape: (H, W)

        # Add channel dimension and transpose to (C, H, W)
        gray = np.expand_dims(gray, axis=0)   # (1, H, W)

        return gray



    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    def _cleanup(self):
        actors = [
            self.camera,
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.vehicle
        ]
        for a in actors:
            if a is not None:
                a.destroy()

        self.image = None
        self.has_collision = False
        self.has_lane_invasion = False
        self.previous_distance = 0.0


    def render(self):
        """Optional visualization using pygame."""
        if not self.render_enabled or self.image is None:
            return

        img = self.image
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.display.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # User closed window; you could set a flag if needed
                pass


    def close(self):
        self._cleanup()

        # Disable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

        if self.display is not None:
            pygame.quit()
