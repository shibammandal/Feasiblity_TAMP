"""
Baseline TAMP planner without ML guidance.

Uses traditional motion planning for all feasibility checks.
"""

import numpy as np
import pybullet as p
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque

from environments.tabletop_env import TabletopEnv
from environments.robot import FrankaPanda
from data_generation.motion_planner import RRTStarPlanner


class ActionType(Enum):
    """Types of primitive actions."""
    PICK = "pick"
    PLACE = "place"


@dataclass
class Action:
    """A primitive action in the TAMP plan."""
    action_type: ActionType
    target_object: Optional[int]  # Object ID for pick
    target_position: np.ndarray   # Target pose
    trajectory: Optional[List[np.ndarray]] = None


@dataclass
class PlanResult:
    """Result of a planning attempt."""
    success: bool
    plan: Optional[List[Action]]
    planning_time: float
    motion_planning_calls: int
    nodes_expanded: int


class BaselineTAMPPlanner:
    """
    Baseline TAMP planner using BFS over symbolic states.
    
    Checks motion feasibility for every candidate action.
    """
    
    def __init__(
        self,
        env: TabletopEnv,
        motion_planning_timeout: float = 2.0,
        max_plan_length: int = 10
    ):
        """
        Initialize the baseline planner.
        
        Args:
            env: The tabletop environment
            motion_planning_timeout: Timeout for each motion planning call
            max_plan_length: Maximum number of actions in a plan
        """
        self.env = env
        self.max_plan_length = max_plan_length
        
        # Initialize motion planner
        self.motion_planner = RRTStarPlanner(
            physics_client=env.physics_client,
            robot_id=env.robot.robot_id,
            joint_indices=FrankaPanda.ARM_JOINT_INDICES,
            joint_limits_lower=FrankaPanda.JOINT_LIMITS_LOWER,
            joint_limits_upper=FrankaPanda.JOINT_LIMITS_UPPER,
            timeout=motion_planning_timeout
        )
        
        # Statistics
        self.motion_planning_calls = 0
        self.nodes_expanded = 0
        
    def plan(
        self,
        goal_positions: Dict[int, np.ndarray],
        timeout: float = 60.0
    ) -> PlanResult:
        """
        Plan to achieve goal object positions.
        
        Args:
            goal_positions: Dict mapping object_id -> target position
            timeout: Overall planning timeout
            
        Returns:
            PlanResult with success status and plan
        """
        start_time = time.time()
        self.motion_planning_calls = 0
        self.nodes_expanded = 0
        
        # BFS over action sequences
        initial_state = self._get_symbolic_state()
        queue = deque([(initial_state, [])])  # (state, action_sequence)
        visited = set()
        
        while queue:
            # Check timeout
            if time.time() - start_time > timeout:
                return PlanResult(
                    success=False,
                    plan=None,
                    planning_time=time.time() - start_time,
                    motion_planning_calls=self.motion_planning_calls,
                    nodes_expanded=self.nodes_expanded
                )
                
            current_state, actions = queue.popleft()
            state_key = self._state_to_key(current_state)
            
            if state_key in visited:
                continue
            visited.add(state_key)
            self.nodes_expanded += 1
            
            # Check if goal reached
            if self._check_goal(current_state, goal_positions):
                return PlanResult(
                    success=True,
                    plan=actions,
                    planning_time=time.time() - start_time,
                    motion_planning_calls=self.motion_planning_calls,
                    nodes_expanded=self.nodes_expanded
                )
                
            # Check plan length limit
            if len(actions) >= self.max_plan_length:
                continue
                
            # Generate candidate actions
            candidates = self._generate_candidates(current_state, goal_positions)
            
            for action in candidates:
                # Check feasibility via motion planning
                trajectory = self._check_feasibility(action)
                
                if trajectory is not None:
                    # Action is feasible
                    action.trajectory = trajectory
                    new_state = self._apply_action(current_state, action)
                    queue.append((new_state, actions + [action]))
                    
        # No plan found
        return PlanResult(
            success=False,
            plan=None,
            planning_time=time.time() - start_time,
            motion_planning_calls=self.motion_planning_calls,
            nodes_expanded=self.nodes_expanded
        )
    
    def _get_symbolic_state(self) -> Dict:
        """Get current symbolic state."""
        state = {
            'robot_holding': None,  # Object ID or None
            'robot_config': self.env.robot.get_joint_positions(),
            'objects': {}
        }
        
        for i, obj_id in enumerate(self.env.object_ids):
            pos, orn = p.getBasePositionAndOrientation(
                obj_id,
                physicsClientId=self.env.physics_client
            )
            state['objects'][obj_id] = {
                'position': np.array(pos),
                'on_table': pos[2] < 0.7  # Rough check
            }
            
        return state
    
    def _state_to_key(self, state: Dict) -> tuple:
        """Convert state to hashable key for visited set."""
        obj_positions = tuple(
            tuple(np.round(info['position'], 2))
            for info in state['objects'].values()
        )
        return (state['robot_holding'], obj_positions)
    
    def _check_goal(self, state: Dict, goal_positions: Dict[int, np.ndarray]) -> bool:
        """Check if goal is satisfied."""
        for obj_id, target_pos in goal_positions.items():
            if obj_id not in state['objects']:
                return False
            current_pos = state['objects'][obj_id]['position']
            if np.linalg.norm(current_pos[:2] - target_pos[:2]) > 0.05:  # 5cm tolerance
                return False
        return True
    
    def _generate_candidates(
        self,
        state: Dict,
        goal_positions: Dict[int, np.ndarray]
    ) -> List[Action]:
        """Generate candidate actions from current state."""
        candidates = []
        
        if state['robot_holding'] is None:
            # Can pick any object
            for obj_id in state['objects']:
                pick_pos = state['objects'][obj_id]['position'].copy()
                pick_pos[2] += 0.1  # Approach from above
                candidates.append(Action(
                    action_type=ActionType.PICK,
                    target_object=obj_id,
                    target_position=pick_pos
                ))
        else:
            # Can place the held object
            obj_id = state['robot_holding']
            if obj_id in goal_positions:
                # Place at goal
                candidates.append(Action(
                    action_type=ActionType.PLACE,
                    target_object=obj_id,
                    target_position=goal_positions[obj_id]
                ))
            else:
                # Place at random location
                place_pos = self.env.sample_place_target()
                candidates.append(Action(
                    action_type=ActionType.PLACE,
                    target_object=obj_id,
                    target_position=place_pos
                ))
                
        return candidates
    
    def _check_feasibility(self, action: Action) -> Optional[List[np.ndarray]]:
        """
        Check if action is feasible via motion planning.
        
        This is the expensive operation we want to replace with ML.
        """
        self.motion_planning_calls += 1
        
        # Try multiple grasp orientations (yaw)
        target_orb_base = p.getQuaternionFromEuler([np.pi, 0, 0])
        possible_yaws = [0, np.pi/2, np.pi, -np.pi/2]
        
        target_config = None
        for yaw in possible_yaws:
            # Rotate gripper around Z axis
            orn = p.getQuaternionFromEuler([np.pi, 0, yaw])
            config = self.env.robot.inverse_kinematics(
                action.target_position,
                np.array(orn)
            )
            if config is not None:
                target_config = config
                break
        
        if target_config is None:
            print("  IK Failed")
            return None  # IK failed
            
        # Plan motion
        current_config = self.env.robot.get_joint_positions()
        trajectory = self.motion_planner.plan(current_config, target_config)
        
        if trajectory is None:
            print("  RRT Failed")
        else:
            print("  RRT Success")
        
        return trajectory
    
    def _apply_action(self, state: Dict, action: Action) -> Dict:
        """Apply action to get new state (symbolic, not actual execution)."""
        new_state = {
            'robot_holding': state['robot_holding'],
            'robot_config': state['robot_config'].copy(),
            'objects': {k: v.copy() for k, v in state['objects'].items()}
        }
        
        if action.action_type == ActionType.PICK:
            new_state['robot_holding'] = action.target_object
        else:  # PLACE
            new_state['robot_holding'] = None
            new_state['objects'][action.target_object]['position'] = action.target_position.copy()
            
        return new_state
    
    def execute_plan(self, plan: List[Action], visualize: bool = True):
        """
        Execute a plan in the environment.
        
        Args:
            plan: List of actions to execute
            visualize: Whether to step simulation for visualization
        """
        for action in plan:
            if action.trajectory is None:
                print(f"Warning: Action has no trajectory, skipping")
                continue
                
            # Execute trajectory
            for waypoint in action.trajectory:
                self.env.robot.set_joint_positions(waypoint)
                if visualize:
                    for _ in range(20):
                        self.env.step_simulation()
                        
            # Gripper action
            if action.action_type == ActionType.PICK:
                self.env.robot.close_gripper()
            else:
                self.env.robot.open_gripper()
                
            if visualize:
                for _ in range(50):
                    self.env.step_simulation()
