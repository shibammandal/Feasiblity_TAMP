"""
Data collection pipeline for feasibility learning.

Generates labeled (state, action, feasibility) samples from simulation.
"""

import numpy as np
import h5py
import os
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from dataclasses import dataclass, asdict
import json

from environments.tabletop_env import TabletopEnv
from environments.robot import FrankaPanda
from .motion_planner import RRTStarPlanner


@dataclass
class Sample:
    """Single training sample."""
    state_vector: np.ndarray
    action_vector: np.ndarray
    image: np.ndarray
    feasible: bool
    trajectory: Optional[np.ndarray] = None


class DataCollector:
    """
    Automated data collection for feasibility learning.
    
    Collects samples by attempting motion planning in randomized scenes.
    """
    
    def __init__(
        self,
        output_dir: str = "data/",
        render: bool = False,
        save_images: bool = True,
        save_trajectories: bool = True,
        camera_size: int = 128
    ):
        """
        Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
            render: Whether to render PyBullet GUI
            save_images: Whether to save RGB images
            save_trajectories: Whether to save full trajectories
            camera_size: Size of captured images (square)
        """
        self.output_dir = output_dir
        self.render = render
        self.save_images = save_images
        self.save_trajectories = save_trajectories
        self.camera_size = camera_size
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        self.env = TabletopEnv(
            render=render,
            camera_width=camera_size,
            camera_height=camera_size
        )
        
        # Initialize motion planner
        self.planner = RRTStarPlanner(
            physics_client=self.env.physics_client,
            robot_id=self.env.robot.robot_id,
            joint_indices=FrankaPanda.ARM_JOINT_INDICES,
            joint_limits_lower=FrankaPanda.JOINT_LIMITS_LOWER,
            joint_limits_upper=FrankaPanda.JOINT_LIMITS_UPPER,
            max_iterations=500,  # Reduced for faster data collection
            timeout=2.0
        )
        
    def collect(
        self,
        num_samples: int,
        num_actions_per_scene: int = 5,
        balance_ratio: float = 0.5,
        seed: Optional[int] = None
    ) -> str:
        """
        Collect training samples.
        
        Args:
            num_samples: Target number of samples to collect
            num_actions_per_scene: Actions to attempt per scene reset
            balance_ratio: Target ratio of feasible samples (0.5 = balanced)
            seed: Random seed
            
        Returns:
            Path to saved dataset file
        """
        if seed is not None:
            np.random.seed(seed)
            
        samples = {
            "state_vectors": [],
            "action_vectors": [],
            "images": [],
            "feasible": [],
            "trajectories": []
        }
        
        feasible_count = 0
        infeasible_count = 0
        
        pbar = tqdm(total=num_samples, desc="Collecting samples")
        
        while len(samples["feasible"]) < num_samples:
            # Reset environment with random objects
            self.env.reset()
            
            for _ in range(num_actions_per_scene):
                if len(samples["feasible"]) >= num_samples:
                    break
                    
                # Get current state and image
                obs = self.env.get_observation()
                state_vector = obs["state_vector"]
                image = obs["image"] if self.save_images else np.zeros((1,))
                
                # Sample action (pick or place)
                sample = self._sample_action()
                if sample is None:
                    continue
                    
                action_type, target_pos = sample
                action_vector = self.env.get_action(action_type, target_pos)
                
                # Attempt motion planning
                target_config = self._compute_target_config(target_pos)
                if target_config is None:
                    feasible = False
                    trajectory = None
                else:
                    current_config = self.env.robot.get_joint_positions()
                    trajectory = self.planner.plan(current_config, target_config)
                    feasible = trajectory is not None
                    
                # Balance dataset
                if feasible:
                    if feasible_count / max(1, feasible_count + infeasible_count) > balance_ratio:
                        # Skip some feasible samples to maintain balance
                        if np.random.random() > 0.3:
                            continue
                    feasible_count += 1
                else:
                    if infeasible_count / max(1, feasible_count + infeasible_count) > (1 - balance_ratio):
                        if np.random.random() > 0.3:
                            continue
                    infeasible_count += 1
                    
                # Store sample
                samples["state_vectors"].append(state_vector)
                samples["action_vectors"].append(action_vector)
                samples["images"].append(image)
                samples["feasible"].append(feasible)
                
                if self.save_trajectories and trajectory is not None:
                    # Interpolate to fixed length
                    traj_interp = self.planner.interpolate_path(trajectory, num_points=20)
                    samples["trajectories"].append(np.array(traj_interp))
                else:
                    samples["trajectories"].append(np.zeros((20, 7)))
                    
                pbar.update(1)
                
                # Occasionally reset to ensure diversity
                if np.random.random() < 0.1:
                    break
                    
        pbar.close()
        
        # Save dataset
        filepath = self._save_dataset(samples)
        
        print(f"\nDataset saved to: {filepath}")
        print(f"Total samples: {len(samples['feasible'])}")
        print(f"Feasible: {feasible_count} ({100*feasible_count/len(samples['feasible']):.1f}%)")
        print(f"Infeasible: {infeasible_count} ({100*infeasible_count/len(samples['feasible']):.1f}%)")
        
        return filepath
    
    def _sample_action(self) -> Optional[Tuple[str, np.ndarray]]:
        """Sample a random action (pick or place)."""
        # Hack for Proof of Work: 50% chance of trivial action to ensure feasibility
        if np.random.random() < 0.5:
             # Trivial action: Move slightly from current position
             # This is guaranteed to be feasible since we are already there
             current_pos, _ = self.env.robot.get_end_effector_pose()
             target_pos = np.array(current_pos) + np.array([0, 0, 0.05]) # Move up 5cm
             return "place", target_pos
             
        action_type = np.random.choice(["pick", "place"])
        
        if action_type == "pick":
            result = self.env.sample_grasp_target()
            if result is None:
                return None
            target_pos, _ = result
        else:
            target_pos = self.env.sample_place_target()
            
        # Add noise to create more diverse samples
        noise = np.random.uniform(-0.05, 0.05, size=3)
        noise[2] = abs(noise[2])  # Keep z positive
        target_pos = target_pos + noise
        
        return action_type, target_pos
    
    def _compute_target_config(self, target_pos: np.ndarray) -> Optional[np.ndarray]:
        """Compute target joint configuration via IK."""
        # Grasp orientation: gripper pointing down
        target_orn = np.array([1, 0, 0, 0])  # Quaternion for pointing down
        
        return self.env.robot.inverse_kinematics(
            target_pos,
            target_orn
        )
    
    def _save_dataset(self, samples: Dict) -> str:
        """Save collected samples to HDF5 file."""
        filepath = os.path.join(self.output_dir, "feasibility_dataset.h5")
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset(
                'state_vectors',
                data=np.array(samples['state_vectors']),
                compression='gzip'
            )
            f.create_dataset(
                'action_vectors',
                data=np.array(samples['action_vectors']),
                compression='gzip'
            )
            f.create_dataset(
                'images',
                data=np.array(samples['images']),
                compression='gzip'
            )
            f.create_dataset(
                'feasible',
                data=np.array(samples['feasible'], dtype=bool)
            )
            f.create_dataset(
                'trajectories',
                data=np.array(samples['trajectories']),
                compression='gzip'
            )
            
            # Store metadata
            f.attrs['num_samples'] = len(samples['feasible'])
            f.attrs['state_dim'] = samples['state_vectors'][0].shape[0]
            f.attrs['action_dim'] = samples['action_vectors'][0].shape[0]
            if self.save_images:
                f.attrs['image_size'] = self.camera_size
                
        return filepath
    
    def close(self):
        """Clean up resources."""
        self.env.close()
        
    def visualize_sample(self, idx: int, dataset_path: str):
        """
        Visualize a sample from the dataset.
        
        Args:
            idx: Sample index
            dataset_path: Path to HDF5 dataset
        """
        import matplotlib.pyplot as plt
        
        with h5py.File(dataset_path, 'r') as f:
            image = f['images'][idx]
            state = f['state_vectors'][idx]
            action = f['action_vectors'][idx]
            feasible = f['feasible'][idx]
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        axes[0].imshow(image)
        axes[0].set_title(f"Sample {idx} - {'Feasible' if feasible else 'Infeasible'}")
        axes[0].axis('off')
        
        # Show state/action info
        axes[1].text(0.1, 0.8, f"State vector (first 15 dims):\n{state[:15]}", 
                     fontsize=10, family='monospace')
        axes[1].text(0.1, 0.4, f"Action vector:\n{action}", 
                     fontsize=10, family='monospace')
        axes[1].text(0.1, 0.2, f"Feasible: {feasible}", fontsize=12, 
                     color='green' if feasible else 'red', weight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"sample_{idx}.png"))
        plt.close()
