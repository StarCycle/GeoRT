import urdfpy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from collections import defaultdict

def visualize_urdf_skeleton(urdf_path, axis_length=0.05, show_base_link=True):
    """
    Parse a URDF file and visualize its joint axes and rotation axis directions in 3D.

    Parameters:
    urdf_path (str): Path to the URDF file.
    axis_length (float): Display length of the axis arrows.
    show_base_link (bool): Whether to show the base link (palm root) coordinate axes.
    """
    try:
        # 1. Load the URDF file
        robot = urdfpy.URDF.load(urdf_path)
        print(f"Successfully loaded URDF file: {urdf_path}")
    except Exception as e:
        print(f"Failed to load URDF file: {e}")
        return

    # 2. Compute forward kinematics (FK) for all links at the default joint configuration
    fk = robot.link_fk()

    # Print all link information
    print(f"\nURDF contains {len(robot.links)} links:")
    for i, link in enumerate(robot.links):
        joint_name = "None"
        for j in robot.joints:
            if j.child == link.name:
                joint_name = j.name
                break
        print(f"  {i}: {link.name} (parent joint: {joint_name})")

    # 3. Set up 3D plotting environment
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'URDF Joint Axes Visualization:\n{os.path.basename(urdf_path)}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    # Used to store all coordinate points to automatically adjust the view range
    all_points = []

    # Record the number of joints at each position
    position_joint_count = defaultdict(int)

    # 4. First draw the base link(s) (palm root)
    if show_base_link:
        # Find base links (links without a parent joint)
        base_links = []
        for link in robot.links:
            is_child = False
            for joint in robot.joints:
                if joint.child == link.name:
                    is_child = True
                    break
            if not is_child:  # A link without a parent joint is a base link
                base_links.append(link)

        print(f"\nFound {len(base_links)} base links:")
        for link in base_links:
            print(f"  - {link.name}")

        for base_link in base_links:
            if base_link in fk:
                T_base = fk[base_link]
                base_pos = T_base[:3, 3]
                all_points.append(base_pos)

                rotation_matrix = T_base[:3, :3]
                x_axis, y_axis, z_axis = rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]

                # Draw base link axes with thicker lines and different colors
                ax.quiver(base_pos[0], base_pos[1], base_pos[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          length=axis_length*1.5, color='r', linewidth=2, normalize=True, label='Base X' if base_link == base_links[0] else "")
                ax.quiver(base_pos[0], base_pos[1], base_pos[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          length=axis_length*1.5, color='g', linewidth=2, normalize=True, label='Base Y' if base_link == base_links[0] else "")
                ax.quiver(base_pos[0], base_pos[1], base_pos[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          length=axis_length*1.5, color='b', linewidth=2, normalize=True, label='Base Z' if base_link == base_links[0] else "")

                # Add a marker at the base position
                ax.scatter(base_pos[0], base_pos[1], base_pos[2],
                           color='purple', marker='s', s=100, label='Base Link' if base_link == base_links[0] else "")

                print(f"  Base link '{base_link.name}' position: {base_pos}")
            else:
                print(f"Warning: Base link '{base_link.name}' is not in FK results.")

    # 5. Iterate over all joints, draw skeleton lines and joint coordinate axes
    for joint in robot.joints:
        parent_link_name = joint.parent
        child_link_name = joint.child

        try:
            # Get Link objects
            parent_link = robot.link_map[parent_link_name]
            child_link = robot.link_map[child_link_name]

            # Use Link objects as keys to get transform matrices
            T_parent = fk[parent_link]
            T_child = fk[child_link]
        except KeyError:
            print(f"Warning: Could not find pose for link '{parent_link_name}' or '{child_link_name}', skipping joint '{joint.name}'.")
            continue

        parent_pos = T_parent[:3, 3]
        child_pos = T_child[:3, 3]
        all_points.extend([parent_pos, child_pos])

        # Draw skeleton line
        ax.plot(
            [parent_pos[0], child_pos[0]],
            [parent_pos[1], child_pos[1]],
            [parent_pos[2], child_pos[2]],
            'k-', linewidth=1.5, alpha=0.7, marker='o', markersize=3
        )

        rotation_matrix = T_child[:3, :3]
        x_axis, y_axis, z_axis = rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]

        # Record the number of joints at the current position
        pos_key = tuple(np.round(child_pos, decimals=4))  # Round to 4 decimals to avoid floating point issues
        position_joint_count[pos_key] += 1

        # Get the joint count at the current position
        joint_count = position_joint_count[pos_key]

        # For the first joint at the same position, draw all three axes
        if joint_count == 1:
            ax.quiver(child_pos[0], child_pos[1], child_pos[2],
                      x_axis[0], x_axis[1], x_axis[2],
                      length=axis_length, color='r', alpha=0.7, normalize=True)
            ax.quiver(child_pos[0], child_pos[1], child_pos[2],
                      y_axis[0], y_axis[1], y_axis[2],
                      length=axis_length, color='g', alpha=0.7, normalize=True)
            ax.quiver(child_pos[0], child_pos[1], child_pos[2],
                      z_axis[0], z_axis[1], z_axis[2],
                      length=axis_length, color='b', alpha=0.7, normalize=True)

        # Check whether joint is movable (non-fixed joint)
        if joint.joint_type != 'fixed' and joint.axis is not None:
            # Transform joint axis from local frame to world frame
            axis_local = np.array(joint.axis)
            axis_world = rotation_matrix @ axis_local

            # Draw rotation axis direction
            ax.quiver(child_pos[0], child_pos[1], child_pos[2],
                      axis_world[0], axis_world[1], axis_world[2],
                      length=axis_length * 1.5,
                      color='y', linestyle='--', linewidth=1.5, normalize=True)

            # Add a marker at the arrow tip to distinguish different joints
            if joint_count > 1:
                end_pos = child_pos + axis_world * axis_length * 1.5
                ax.scatter(end_pos[0], end_pos[1], end_pos[2],
                           color='y', marker='o', s=20, alpha=0.8)

    # 6. Automatically adjust plot range to include all points
    if all_points:
        points = np.array(all_points)
        max_range = (points.max(axis=0) - points.min(axis=0)).max()
        if max_range < 1e-6: max_range = 1.0 # Avoid division by zero

        center = points.mean(axis=0)

        ax.set_xlim(center[0] - max_range * 0.6, center[0] + max_range * 0.6)
        ax.set_ylim(center[1] - max_range * 0.6, center[1] + max_range * 0.6)
        ax.set_zlim(center[2] - max_range * 0.6, center[2] + max_range * 0.6)

        ax.set_aspect('equal', adjustable='box')

    # 7. Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='r', lw=2, label='Base X-axis'),
        Line2D([0], [0], color='g', lw=2, label='Base Y-axis'),
        Line2D([0], [0], color='b', lw=2, label='Base Z-axis'),
        Line2D([0], [0], color='r', lw=1, alpha=0.7, label='Joint X-axis'),
        Line2D([0], [0], color='g', lw=1, alpha=0.7, label='Joint Y-axis'),
        Line2D([0], [0], color='b', lw=1, alpha=0.7, label='Joint Z-axis'),
        Line2D([0], [0], color='y', linestyle='--', lw=1.5, label='Joint Rotation Axis'),
        Line2D([0], [0], color='k', marker='o', linestyle='-', lw=1.5, label='Skeleton Links'),
        Line2D([0], [0], color='purple', marker='s', linestyle='', markersize=8, label='Base Link'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    # 8. Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Point to your URDF file ---
    my_urdf_path = "/home/starcycle/GeoRT/assets/hand_v1/urdf/hand.urdf"

    if os.path.exists(my_urdf_path):
        visualize_urdf_skeleton(my_urdf_path, axis_length=0.01, show_base_link=True)
    else:
        print(f"Error: File '{my_urdf_path}' does not exist. Please check the path.")