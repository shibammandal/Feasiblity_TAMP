"""
Tabletop manipulation environment for TAMP feasibility learning.

Provides a PyBullet simulation with randomized objects and camera views.
"""

import pybullet as p
import pybullet_data
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import random

from .robot import FrankaPanda


class TabletopEnv:
    """
    PyBullet tabletop environment for pick-and-place manipulation.
    
    Features:
    - Franka Panda robot arm
    - Randomized object spawning
    - RGB camera rendering
    - State vector extraction
    """
    
    def __init__(
        self,
        render: bool = False,
        camera_width: int = 128,
        camera_height: int = 128,
        num_objects: Tuple[int, int] = (2, 6),
        object_types: List[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the tabletop environment.
        
        Args:
            render: Whether to render the GUI
            camera_width: Width of captured images
            camera_height: Height of captured images
            num_objects: (min, max) number of objects to spawn
            object_types: List of object types to spawn ("cube", "cylinder")
            seed: Random seed for reproducibility
        """
        self.render = render
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_objects_range = num_objects
        self.object_types = object_types or ["cube", "cylinder", "lego", "duck", "teddy"]
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Connect to PyBullet
        self.physics_client = p.connect(
            p.GUI if render else p.DIRECT
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Physics parameters
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # Environment state
        self.robot: Optional[FrankaPanda] = None
        self.table_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.object_ids: List[int] = []
        self.object_info: List[Dict[str, Any]] = []
        
        # Camera parameters
        self._setup_camera()
        
        
        # Connection status
        self.connected = True
        
        # Initialize environment
        self.reset()
        
    def _setup_camera(self):
        """Configure the camera for image capture."""
        # Camera positioned above and in front of the table
        self.camera_target = [0.5, 0, 0.4]  # Look at center of table
        self.camera_distance = 1.2
        self.camera_yaw = 0
        self.camera_pitch = -45
        
        # Camera intrinsic parameters
        self.camera_fov = 60
        self.camera_aspect = self.camera_width / self.camera_height
        self.camera_near = 0.1
        self.camera_far = 2.0
        
        # Compute view and projection matrices
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=self.camera_aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
    def reset(self, num_objects: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment with new random objects.
        
        Args:
            num_objects: Specific number of objects, or random if None
            
        Returns:
            Initial observation dict with state vector and image
        """
        # Clear existing objects
        self._clear_objects()
        
        # Load ground plane if not exists
        if self.plane_id is None:
            self.plane_id = p.loadURDF(
                "plane.urdf",
                physicsClientId=self.physics_client
            )
            
        # Load table if not exists
        if self.table_id is None:
            self.table_id = p.loadURDF(
                "table/table.urdf",
                basePosition=[0.5, 0, 0],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.physics_client
            )
            
        # Initialize robot if not exists
        if self.robot is None:
            self.robot = FrankaPanda(
                physics_client=self.physics_client,
                base_position=(0, 0, 0)
            )
        else:
            self.robot.reset()
            
        # Spawn random objects
        if num_objects is None:
            num_objects = random.randint(*self.num_objects_range)
        self._spawn_objects(num_objects)
        
        # Let physics settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client)
            
        return self.get_observation()
    
    def _clear_objects(self):
        """Remove all spawned objects."""
        for obj_id in self.object_ids:
            p.removeBody(obj_id, physicsClientId=self.physics_client)
        self.object_ids = []
        self.object_info = []
        
    def _spawn_objects(self, num_objects: int):
        """
        Spawn random objects on the table.
        
        Args:
            num_objects: Number of objects to spawn
        """
        # Table surface parameters
        table_x_range = (0.3, 0.7)
        table_y_range = (-0.25, 0.25)
        table_z = 0.625  # Table surface height
        
        # Random colors
        colors = [
            [1, 0, 0, 1],    # Red
            [0, 1, 0, 1],    # Green
            [0, 0, 1, 1],    # Blue
            [1, 1, 0, 1],    # Yellow
            [1, 0, 1, 1],    # Magenta
            [0, 1, 1, 1],    # Cyan
        ]
        
        for i in range(num_objects):
            # Random type, size, position
            obj_type = random.choice(self.object_types)
            size = random.uniform(0.03, 0.06)
            
            # Try to find non-overlapping position
            for attempt in range(20):
                x = random.uniform(*table_x_range)
                y = random.uniform(*table_y_range)
                z = table_z + size / 2
                
                # Check distance from other objects
                valid = True
                for info in self.object_info:
                    dist = np.sqrt((x - info["position"][0])**2 + 
                                   (y - info["position"][1])**2)
                    if dist < (size + info["size"]) * 1.5:
                        valid = False
                        break
                        
                if valid:
                    break
            else:
                continue  # Skip if can't find valid position
                
            # Create collision and visual shapes
            # Create body based on type
            if obj_type == "cube":
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3, physicsClientId=self.physics_client)
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=colors[i%len(colors)], physicsClientId=self.physics_client)
                obj_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=[x, y, z], physicsClientId=self.physics_client)
            elif obj_type == "cylinder":
                collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=size/2, height=size, physicsClientId=self.physics_client)
                visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=size/2, length=size, rgbaColor=colors[i%len(colors)], physicsClientId=self.physics_client)
                obj_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=[x, y, z], physicsClientId=self.physics_client)
            elif obj_type == "lego":
                obj_id = p.loadURDF("lego/lego.urdf", basePosition=[x, y, z], globalScaling=0.5, physicsClientId=self.physics_client)
            elif obj_type == "duck":
                obj_id = p.loadURDF("duck_vhacd.urdf", basePosition=[x, y, z], globalScaling=0.8, physicsClientId=self.physics_client)
            elif obj_type == "teddy":
                obj_id = p.loadURDF("teddy_vhacd.urdf", basePosition=[x, y, z], globalScaling=0.5, physicsClientId=self.physics_client)
            else: # Fallback cube
                collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3, physicsClientId=self.physics_client)
                visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=colors[i%len(colors)], physicsClientId=self.physics_client)
                obj_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=[x, y, z], physicsClientId=self.physics_client)
            
            # Skip createMultiBody for loaded URDFs since loadURDF does it
            if obj_type in ["cube", "cylinder", "fallback"]:
                 pass # Already created above
            else: 
                 # For URDFs, we might want to color them?
                 p.changeVisualShape(obj_id, -1, rgbaColor=colors[i%len(colors)], physicsClientId=self.physics_client)

            self.object_ids.append(obj_id)
            self.object_info.append({
                "id": obj_id,
                "type": obj_type,
                "size": size,
                "position": [x, y, z],
                "color": colors[i % len(colors)]
            })
            continue # Skip the loop's shared createMultiBody call

                

            
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation.
        
        Returns:
            Dict with 'state_vector' and 'image' keys
        """
        return {
            "state_vector": self.get_state(),
            "image": self.get_image()
        }
        
    def get_state(self) -> np.ndarray:
        """
        Get environment state as a vector.
        
        State includes:
        - Robot state (15 dims): joints, gripper, ee pose
        - Object states (7 dims each): position (3), orientation (4)
        - Padded to fixed size for variable number of objects
        
        Returns:
            State vector of fixed size
        """
        # Robot state
        robot_state = self.robot.get_state_vector()  # 15 dims
        
        # Object states (max 6 objects, 7 dims each = 42 dims)
        max_objects = 6
        object_state = np.zeros(max_objects * 7)
        
        for i, obj_id in enumerate(self.object_ids[:max_objects]):
            pos, orn = p.getBasePositionAndOrientation(
                obj_id,
                physicsClientId=self.physics_client
            )
            object_state[i*7:(i+1)*7] = np.concatenate([pos, orn])
            
        # Combine: 15 + 42 = 57 dims
        return np.concatenate([robot_state, object_state]).astype(np.float32)
    
    def get_image(self) -> np.ndarray:
        """
        Capture RGB image from camera.
        
        Returns:
            RGB image array of shape (H, W, 3)
        """
        _, _, rgba, _, _ = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.render else p.ER_TINY_RENDERER,
            physicsClientId=self.physics_client
        )
        
        # Convert RGBA to RGB
        rgb = np.array(rgba, dtype=np.uint8).reshape(
            self.camera_height, self.camera_width, 4
        )[:, :, :3]
        
        return rgb
    
    def get_object_poses(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get poses of all objects.
        
        Returns:
            List of (position, orientation) tuples
        """
        poses = []
        for obj_id in self.object_ids:
            pos, orn = p.getBasePositionAndOrientation(
                obj_id,
                physicsClientId=self.physics_client
            )
            poses.append((np.array(pos), np.array(orn)))
        return poses
    
    def step_simulation(self, num_steps: int = 1):
        """
        Step the physics simulation.
        
        Args:
            num_steps: Number of simulation steps
        """
        for _ in range(num_steps):
            p.stepSimulation(physicsClientId=self.physics_client)
            
    def check_collision(self, exclude_gripper: bool = True) -> bool:
        """
        Check for any collisions in the environment.
        
        Args:
            exclude_gripper: Exclude gripper-object contact
            
        Returns:
            True if collision detected
        """
        return self.robot.is_collision()
    
    def sample_grasp_target(self) -> Optional[Tuple[np.ndarray, int]]:
        """
        Sample a valid grasp target position for a random object.
        
        Returns:
            (target_position, object_id) or None if no valid target
        """
        if not self.object_ids:
            return None
            
        obj_id = random.choice(self.object_ids)
        pos, _ = p.getBasePositionAndOrientation(
            obj_id,
            physicsClientId=self.physics_client
        )
        
        # Grasp from above with small offset
        grasp_height = 0.1  # Approach height above object
        target_pos = np.array([pos[0], pos[1], pos[2] + grasp_height])
        
        return target_pos, obj_id
    
    def sample_place_target(self) -> np.ndarray:
        """
        Sample a random place target on the table.
        
        Returns:
            Target position (x, y, z)
        """
        table_x_range = (0.3, 0.7)
        table_y_range = (-0.25, 0.25)
        table_z = 0.65  # Slightly above table surface
        
        x = random.uniform(*table_x_range)
        y = random.uniform(*table_y_range)
        
        return np.array([x, y, table_z])
    
    def get_action(self, action_type: str, target_position: np.ndarray) -> np.ndarray:
        """
        Create action vector for ML input.
        
        Args:
            action_type: "pick" or "place"
            target_position: (x, y, z) target position
            
        Returns:
            Action vector (8 dims): [action_type_onehot (2), target_pos (3), 
                                    target_orientation (3, euler)]
        """
        action_onehot = [1, 0] if action_type == "pick" else [0, 1]
        # Default grasp orientation: gripper pointing down
        target_euler = [np.pi, 0, 0]
        
        return np.concatenate([
            action_onehot,
            target_position,
            target_euler
        ]).astype(np.float32)
    
    def close(self):
        """Close the PyBullet connection."""
        if self.connected:
            p.disconnect(physicsClientId=self.physics_client)
            self.connected = False
        
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
