import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# Mock BT classes to avoid dependency
class Status:
    IDLE = 0
    RUNNING = 1
    SUCCESS = 2
    FAILURE = 3

STATUS_COLORS = {
    Status.IDLE: 'lightgray',
    Status.RUNNING: 'yellow',
    Status.SUCCESS: 'lightgreen',
    Status.FAILURE: 'salmon'
}

STATUS_NAMES = {
    Status.IDLE: "IDLE",
    Status.RUNNING: "RUNNING",
    Status.SUCCESS: "SUCCESS",
    Status.FAILURE: "FAILURE"
}

class Node:
    def __init__(self, name):
        self.name = name
        self.status = Status.IDLE
        self.children = []
        self.x = 0
        self.y = 0

    def add_child(self, child):
        self.children.append(child)

    def tick(self, context):
        raise NotImplementedError

    def reset(self):
        self.status = Status.IDLE
        for c in self.children:
            c.reset()

class Sequence(Node):
    def tick(self, context):
        self.status = Status.RUNNING
        for child in self.children:
            result = child.tick(context)
            if result == Status.RUNNING:
                return Status.RUNNING
            if result == Status.FAILURE:
                self.status = Status.FAILURE
                return Status.FAILURE
        self.status = Status.SUCCESS
        return Status.SUCCESS

class Selector(Node):
    def tick(self, context):
        self.status = Status.RUNNING
        for child in self.children:
            result = child.tick(context)
            if result == Status.RUNNING:
                return Status.RUNNING
            if result == Status.SUCCESS:
                self.status = Status.SUCCESS
                return Status.SUCCESS
        self.status = Status.FAILURE
        return Status.FAILURE

class Action(Node):
    def __init__(self, name, func):
        super().__init__(name)
        self.func = func

    def tick(self, context):
        self.status = Status.RUNNING
        # We assume immediate return for this visual demo, but could be async
        result = self.func(context)
        self.status = result
        return result

# --- Data Loading ---
def get_data_samples():
    """Get a feasible and infeasible sample from the dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_test', 'feasibility_dataset.h5')
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, utilizing mock data.")
        return None, None

    with h5py.File(data_path, 'r') as f:
        feasible = f['feasible'][:]
        
        # Robustly get indices
        inf_indices = np.where(feasible == 0)[0]
        fea_indices = np.where(feasible == 1)[0]
        
        infeasible_idx = inf_indices[0] if len(inf_indices) > 0 else 0
        feasible_idx = fea_indices[0] if len(fea_indices) > 0 else 0
        
        return infeasible_idx, feasible_idx

# --- Layout ---
# --- Layout ---
def get_subtree_width(node, spacing=1.5):
    """Calculate the total width of a subtree."""
    if not node.children:
        return 1.2  # Approximate width of a node box
        
    children_width = sum(get_subtree_width(c, spacing) for c in node.children)
    # Add spacing between children
    total_gap = (len(node.children) - 1) * spacing
    return max(1.2, children_width + total_gap)

def layout_tree(node, x=0, y=0, level_height=1.0, spacing=0.5):
    """Calculate node positions to avoid overlap."""
    node.x = x
    node.y = y
    
    if not node.children:
        return
        
    # Calculate widths of all children
    child_widths = [get_subtree_width(c, spacing) for c in node.children]
    total_children_width = sum(child_widths) + (len(node.children) - 1) * spacing
    
    # Start position for the first child (far left)
    current_x = x - total_children_width / 2
    
    for i, child in enumerate(node.children):
        # We want to place the child such that its center is at current_x + child_width/2
        child_w = child_widths[i]
        child_center_x = current_x + child_w / 2
        
        layout_tree(child, child_center_x, y - level_height, level_height, spacing)
        
        # Advance current_x
        current_x += child_w + spacing

# --- Visualization ---
def draw_tree(ax, node):
    # Draw logic lines first
    for child in node.children:
        ax.plot([node.x, child.x], [node.y, child.y], 'k-', zorder=1)
        draw_tree(ax, child)
        
    # Draw node
    rect = patches.FancyBboxPatch(
        (node.x - 0.6, node.y - 0.3), 1.2, 0.6,
        boxstyle="round,pad=0.1",
        facecolor=STATUS_COLORS[node.status],
        edgecolor='black',
        zorder=2
    )
    ax.add_patch(rect)
    ax.text(node.x, node.y, node.name, ha='center', va='center', fontsize=9, zorder=3)

# --- BT Logic Functions ---
def plan_action(context):
    print("  Planning action...")
    # Simulate checking the infeasible sample
    # In a real app, this would check the model model(state, action)
    if context.get('use_repair'):
        return Status.SUCCESS # Repaired plan works
    else:
        # Initial attempt fails (simulated by infeasible data sample)
        return Status.FAILURE

def execute_action(context):
    if context.get('use_repair'):
        print("  Executing repaired plan...")
        return Status.SUCCESS
    print("  Executing invalid plan (should not happen)...")
    return Status.FAILURE

def analyze_failure(context):
    print("  Analyzing failure...")
    # Simulate diagnosis
    context['diagnosis'] = "Kinematic Limit"
    return Status.SUCCESS

def repair_plan(context):
    print("  Repairing plan...")
    # Simulate finding the feasible sample
    context['use_repair'] = True
    return Status.SUCCESS

def generate_bt_demo():
    print("Initializing BT Demo...")
    
    # Load real data indices
    infeasible_idx, feasible_idx = get_data_samples()
    if infeasible_idx is None:
        infeasible_idx, feasible_idx = 7, 42 # Fallback
        
    # Context state
    context = {
        'use_repair': False,
        'infeasible_idx': infeasible_idx,
        'feasible_idx': feasible_idx,
        'sample_idx': infeasible_idx # Start with infeasible
    }
    
    # Build Tree
    # Root: Fallback (Standard Execution OR Repair Execution)
    root = Selector("TAMP Root")
    
    # Branch 1: Standard TAMP
    seq_std = Sequence("Standard TAMP")
    act_plan = Action("Plan Motion", plan_action)
    act_exec = Action("Execute", execute_action)
    seq_std.add_child(act_plan)
    seq_std.add_child(act_exec)
    
    # Branch 2: Repair
    seq_repair = Sequence("Repair Routine")
    act_diag = Action("Diagnose", analyze_failure)
    act_fix = Action("Repair Plan", repair_plan)
    act_exec_rep = Action("Exec Repaired", execute_action)
    seq_repair.add_child(act_diag)
    seq_repair.add_child(act_fix)
    seq_repair.add_child(act_exec_rep)
    
    root.add_child(seq_std)
    root.add_child(seq_repair)
    
    layout_tree(root, y=2)
    
    # Simulation Frames
    frames = []
    
    # Helper to capture state
    def capture_frame(ax):
        ax.clear()
        ax.set_xlim(-7, 7)
        ax.set_ylim(-3, 3)
        ax.axis('off')
        draw_tree(ax, root)
        
        # Draw "Data Log" using real variables
        log_text = "Data Log:\n"
        if context.get('diagnosis'):
             log_text += f"Error: {context['diagnosis']}\n"
             
        sample_idx = context.get('sample_idx', None)
        
        if sample_idx is not None:
             # Check if we are in the repair phase (feasible sample) or failure phase (infeasible)
             if context.get('use_repair'):
                 log_text += f"Retrieving Sample #{sample_idx}\n"
                 log_text += "Feasibility: High (0.92)\n" # Simulated score based on label 1
             else:
                 log_text += f"Retrieving Sample #{sample_idx}\n"
                 log_text += "Feasibility: Low (0.04)\n"  # Simulated score based on label 0
             
        ax.text(-6.5, 2.5, log_text, fontsize=8, family='monospace')
    
    # Run the tick loop manually to generate frames
    # We want to show the traversal step-by-step
    
    # Step 1: Initial State
    fig, ax = plt.subplots(figsize=(10, 6))
    capture_frame(ax)
    # plt.savefig('frame_0.png')
    # Use simple list for animation frames? No, we need to redraw
    
    # We will define a generator that yields the tree state at each step
    # But since our classes are synchronous, we'll fake the intermediate steps
    # by manually toggling statuses for the demo video.
    
    demo_steps = [
        # 1. Start Std Sequence
        (root, Status.RUNNING), (seq_std, Status.RUNNING),
        # 2. Try Plan -> Fail
        (act_plan, Status.RUNNING), (act_plan, Status.FAILURE),
        (seq_std, Status.FAILURE),
        # 3. Switch to Repair
        (seq_repair, Status.RUNNING),
        # 4. Diagnose
        (act_diag, Status.RUNNING), (act_diag, Status.SUCCESS),
        # 5. Repair
        (act_fix, Status.RUNNING),
        # Side effect: context changed
        (act_fix, Status.SUCCESS),
        # 6. Execute Repaired
        (act_exec_rep, Status.RUNNING), (act_exec_rep, Status.SUCCESS),
        # 7. Propagate Success
        (seq_repair, Status.SUCCESS), (root, Status.SUCCESS)
    ]
    
    def update(frame_idx):
        if frame_idx >= len(demo_steps):
            return []
            
        node, status = demo_steps[frame_idx]
        node.status = status
        
        # Apply side effects for visuals
        if node.name == "Repair Plan" and status == Status.SUCCESS:
            context['use_repair'] = True
            context['sample_idx'] = context['feasible_idx']
        if node.name == "Diagnose" and status == Status.SUCCESS:
            context['diagnosis'] = "Infeasible Trajectory"
            
        capture_frame(ax)
        return ax.patches + ax.texts + ax.lines

    ani = animation.FuncAnimation(fig, update, frames=len(demo_steps), interval=800, blit=False)
    ani.save("demo_bt.gif", writer='pillow', fps=1.5)
    print("Saved demo_bt.gif")

if __name__ == "__main__":
    generate_bt_demo()
