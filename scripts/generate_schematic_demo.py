import os
import sys
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.robot import FrankaPanda

def interpolate_path(start, end, steps):
    path = []
    for t in np.linspace(0, 1, steps):
        path.append(start * (1 - t) + end * t)
    return path

def generate_schematic_gif(output_path: str = "demo_schematic.gif"):
    print(f"Generating schematic animation to {output_path}...")
    
    # Connect to PyBullet (DIRECT mode = no GUI, faster)
    physics_client = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)
    
    # Load robot
    robot = FrankaPanda(physics_client)
    
    # Define a simple kinematic path (Triangle movement)
    # 1. Home
    # 2. Reach proper target
    # 3. Move sideways
    # 4. Return
    
    home_joints = np.array(robot.HOME_POSITION)
    
    # Define Cartesian waypoints
    # Note: These are approximate valid configurations
    targets = [
        [0.5, 0.0, 0.5],   # Center
        [0.5, 0.2, 0.5],   # Left
        [0.5, -0.2, 0.5],  # Right
        [0.5, 0.0, 0.5]    # Center
    ]
    
    # Compute Joint Waypoints via IK
    joint_waypoints = [home_joints]
    
    print("Computing kinematics...")
    for target_pos in targets:
        # Orientation: Gripper pointing down
        orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        # Simple IK
        joint_pos = robot.inverse_kinematics(
            target_position=target_pos,
            target_orientation=orn
        )
        
        if joint_pos is not None:
            joint_waypoints.append(joint_pos)
        else:
            print(f"Warning: IK failed for {target_pos}")
            
    # Interpolate for smoothness
    full_trajectory = []
    steps_per_segment = 15
    for i in range(len(joint_waypoints) - 1):
        segment = interpolate_path(joint_waypoints[i], joint_waypoints[i+1], steps_per_segment)
        full_trajectory.extend(segment)
        
    print(f"Generated trajectory with {len(full_trajectory)} frames.")

    # Setup Matplotlib Figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Franka Panda Trajectory (Time = 0.00s)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    
    # Set fixed limits to avoid camera jumping
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(0, 1.0)
    
    # Line objects for links
    line, = ax.plot([], [], [], 'o-', lw=2, markersize=5)
    
    # Link indices to plot (Base -> ... -> EE)
    # Panda link indices roughly: 0, 1, 2, ..., 11(EE)
    # We will grab positions of all joints
    num_joints = p.getNumJoints(robot.robot_id)
    # We want to trace the spine of the robot
    link_indices = list(range(num_joints))

    def update(frame_idx):
        if frame_idx >= len(full_trajectory):
            return line,
            
        # Set robot state
        q = full_trajectory[frame_idx]
        robot.reset(q[:7]) # Reset sets the state immediately
        
        # Get link positions
        xs, ys, zs = [], [], []
        
        # Base position
        xs.append(0)
        ys.append(0)
        zs.append(0)
        
        for i in link_indices:
            state = p.getLinkState(robot.robot_id, i)
            # Link world position
            pos = state[0] 
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        ax.set_title(f"Franka Panda Trajectory (Time = {frame_idx * 0.1:.2f}s)")
        return line,

    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(full_trajectory), 
        interval=50, 
        blit=False # blit=False often more stable for 3D
    )
    
    ani.save(output_path, writer='pillow', fps=20)
    print("Done!")
    p.disconnect()

if __name__ == "__main__":
    generate_schematic_gif()
