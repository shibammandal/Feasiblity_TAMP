"""
Franka Panda robot arm control for PyBullet simulation.

Provides inverse kinematics, joint control, and gripper operations.
"""

import pybullet as p
import numpy as np
from typing import Optional, Tuple, List
import os


class FrankaPanda:
    """Franka Panda robot arm controller for PyBullet."""
    
    # Panda arm has 7 DOF + 2 gripper fingers
    NUM_ARM_JOINTS = 7
    NUM_GRIPPER_JOINTS = 2
    
    # Joint indices
    ARM_JOINT_INDICES = list(range(7))
    GRIPPER_JOINT_INDICES = [9, 10]
    END_EFFECTOR_LINK = 11
    
    # Default home position
    HOME_POSITION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    
    # Joint limits
    JOINT_LIMITS_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    # Gripper limits
    GRIPPER_OPEN = 0.04
    GRIPPER_CLOSED = 0.0
    
    def __init__(
        self,
        physics_client: int,
        base_position: Tuple[float, float, float] = (0, 0, 0),
        base_orientation: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Initialize the Franka Panda robot.
        
        Args:
            physics_client: PyBullet physics client ID
            base_position: (x, y, z) position of robot base
            base_orientation: (x, y, z, w) quaternion orientation
        """
        self.physics_client = physics_client
        self.base_position = base_position
        self.base_orientation = base_orientation or p.getQuaternionFromEuler([0, 0, 0])
        
        # Load robot URDF
        self.robot_id = self._load_robot()
        
        # Initialize to home position
        self.reset()
        
    def _load_robot(self) -> int:
        """Load the Panda URDF from PyBullet's data directory."""
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            useFixedBase=True,
            physicsClientId=self.physics_client
        )
        
        # Set joint damping for stability
        for joint_idx in self.ARM_JOINT_INDICES:
            p.changeDynamics(
                robot_id,
                joint_idx,
                linearDamping=0.0,
                angularDamping=0.0,
                physicsClientId=self.physics_client
            )
            
        return robot_id
    
    def reset(self, joint_positions: Optional[List[float]] = None):
        """
        Reset robot to specified or home position.
        
        Args:
            joint_positions: Target joint angles (7 values), or None for home
        """
        if joint_positions is None:
            joint_positions = self.HOME_POSITION
            
        # Set arm joints
        for i, joint_idx in enumerate(self.ARM_JOINT_INDICES):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                joint_positions[i],
                physicsClientId=self.physics_client
            )
            
        # Open gripper
        self.open_gripper()
        
    def get_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions."""
        joint_states = p.getJointStates(
            self.robot_id,
            self.ARM_JOINT_INDICES,
            physicsClientId=self.physics_client
        )
        return np.array([state[0] for state in joint_states])
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current arm joint velocities."""
        joint_states = p.getJointStates(
            self.robot_id,
            self.ARM_JOINT_INDICES,
            physicsClientId=self.physics_client
        )
        return np.array([state[1] for state in joint_states])
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector position and orientation.
        
        Returns:
            position: (x, y, z) position
            orientation: (x, y, z, w) quaternion
        """
        link_state = p.getLinkState(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            physicsClientId=self.physics_client
        )
        position = np.array(link_state[0])
        orientation = np.array(link_state[1])
        return position, orientation
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        max_iterations: int = 100,
        residual_threshold: float = 1e-4
    ) -> Optional[np.ndarray]:
        """
        Compute joint angles for target end-effector pose.
        
        Args:
            target_position: (x, y, z) target position
            target_orientation: (x, y, z, w) quaternion, or None for position-only IK
            max_iterations: Maximum IK solver iterations
            residual_threshold: Convergence threshold
            
        Returns:
            Joint angles if solution found, None otherwise
        """
        if target_orientation is not None:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.END_EFFECTOR_LINK,
                target_position,
                target_orientation,
                lowerLimits=self.JOINT_LIMITS_LOWER.tolist(),
                upperLimits=self.JOINT_LIMITS_UPPER.tolist(),
                jointRanges=(self.JOINT_LIMITS_UPPER - self.JOINT_LIMITS_LOWER).tolist(),
                restPoses=self.HOME_POSITION,
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
                physicsClientId=self.physics_client
            )
        else:
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.END_EFFECTOR_LINK,
                target_position,
                lowerLimits=self.JOINT_LIMITS_LOWER.tolist(),
                upperLimits=self.JOINT_LIMITS_UPPER.tolist(),
                jointRanges=(self.JOINT_LIMITS_UPPER - self.JOINT_LIMITS_LOWER).tolist(),
                restPoses=self.HOME_POSITION,
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
                physicsClientId=self.physics_client
            )
            
        # Return only arm joint positions (first 7)
        joint_positions = np.array(joint_positions[:self.NUM_ARM_JOINTS])
        
        # Validate solution is within limits (relaxed for demo/data generation)
        tol = 0.2
        if np.all(joint_positions >= self.JOINT_LIMITS_LOWER - tol) and \
           np.all(joint_positions <= self.JOINT_LIMITS_UPPER + tol):
            return joint_positions
        return None
    
    def set_joint_positions(self, joint_positions: np.ndarray, velocity: float = 1.0):
        """
        Set target joint positions using position control.
        
        Args:
            joint_positions: Target joint angles (7 values)
            velocity: Maximum joint velocity
        """
        p.setJointMotorControlArray(
            self.robot_id,
            self.ARM_JOINT_INDICES,
            p.POSITION_CONTROL,
            targetPositions=joint_positions.tolist(),
            forces=[240.0] * self.NUM_ARM_JOINTS,
            positionGains=[0.03] * self.NUM_ARM_JOINTS,
            velocityGains=[1.0] * self.NUM_ARM_JOINTS,
            physicsClientId=self.physics_client
        )
        
    def open_gripper(self):
        """Open the gripper fingers."""
        for joint_idx in self.GRIPPER_JOINT_INDICES:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=self.GRIPPER_OPEN,
                force=20.0,
                physicsClientId=self.physics_client
            )
            
    def close_gripper(self):
        """Close the gripper fingers."""
        for joint_idx in self.GRIPPER_JOINT_INDICES:
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=self.GRIPPER_CLOSED,
                force=20.0,
                physicsClientId=self.physics_client
            )
            
    def get_gripper_state(self) -> float:
        """Get current gripper opening (0 = closed, 0.04 = open)."""
        joint_states = p.getJointStates(
            self.robot_id,
            self.GRIPPER_JOINT_INDICES,
            physicsClientId=self.physics_client
        )
        return (joint_states[0][0] + joint_states[1][0]) / 2
    
    def is_collision(self) -> bool:
        """Check if robot is in self-collision or collision with env."""
        # Get all contact points for the robot
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            physicsClientId=self.physics_client
        )
        
        # Filter out gripper-object contact (expected during grasping)
        for contact in contact_points:
            link_a = contact[3]
            link_b = contact[4]
            # Check for arm self-collision or collision with other bodies
            if link_a not in self.GRIPPER_JOINT_INDICES and \
               link_b not in self.GRIPPER_JOINT_INDICES:
                return True
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get robot state as a vector for ML input.
        
        Returns:
            State vector: [joint_positions (7), gripper_state (1), 
                          ee_position (3), ee_orientation (4)] = 15 dims
        """
        joint_positions = self.get_joint_positions()
        gripper_state = np.array([self.get_gripper_state()])
        ee_position, ee_orientation = self.get_end_effector_pose()
        
        return np.concatenate([
            joint_positions,
            gripper_state,
            ee_position,
            ee_orientation
        ])
