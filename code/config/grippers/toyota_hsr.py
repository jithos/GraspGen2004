import torch
import numpy as np
import trimesh
import os
from grasp_gen.robot import load_control_points_core, load_default_gripper_config
from pathlib import Path
import trimesh.transformations as tra
class GripperModel():
    def __init__(self, data_root_dir=None):
        if data_root_dir is None:
            data_root_dir = f'{Path(__file__).parent.parent.parent}/assets/toyota_hsr_gripper'
        
        # Load the specific HSR meshes for palm, wrist, fingers
        self.palm = trimesh.load(os.path.join(data_root_dir, "palm.stl"))
        self.wrist = trimesh.load(os.path.join(data_root_dir, "wrist_roll.stl"))
        
        # Left finger parts
        self.l_prox = trimesh.load(os.path.join(data_root_dir, "l_proximal.stl"))
        self.l_dist = trimesh.load(os.path.join(data_root_dir, "l_distal.stl"))
        
        # Right finger parts
        self.r_prox = trimesh.load(os.path.join(data_root_dir, "r_proximal.stl"))
        self.r_dist = trimesh.load(os.path.join(data_root_dir, "r_distal.stl"))


        # Origin of finger joints
        self.l_joint_origin = np.array([-0.01675, -0.0245, -0.0175])
        self.r_joint_origin = np.array([-0.01675, 0.0245, -0.0175])

        # Apply initial translations to position parts correctly
        self.wrist.apply_translation([-0.012, 0.0, -0.1405])
        self.l_prox.apply_translation([-0.01675, -0.0245, -0.0175])
        self.l_dist.apply_translation([-0.01675, -0.0245, 0.0525]) 
        self.r_prox.apply_translation([-0.01675, 0.0245, -0.0175])
        self.r_dist.apply_translation([-0.01675, 0.0245, 0.0525])

        # Constants for HSR gripper logic
        self.MAX_WIDTH = 0.13603458
        # Offsets relative to a GraspGen/Robotiq model depth of 0.195m
        # HSR Depth Closed: ~0.0904m -> Offset = 0.195 - 0.0890 = 0.106
        self.OFFSET_CLOSED = 0.106
        # HSR Depth Open: ~0.1518m -> Offset = 0.195 - 0.0415 = 0.1535
        self.OFFSET_OPEN = 0.1535

        # Default to closed offset
        self.grasp_offset = np.array([0.0, 0.0, self.OFFSET_CLOSED])

    def calculate_z_offset(self, aperture):
        """
        Calculates the Z-offset (retraction) needed based on aperture.
        Aperture: 0.0 (closed) to 1.0 (open)
        """
        # Linear interpolation of the offset based on aperture
        z_offset = self.OFFSET_CLOSED + aperture * (self.OFFSET_OPEN - self.OFFSET_CLOSED)
        return z_offset

    def get_z_offset_for_width(self, width):
        aperture = np.clip(width / self.MAX_WIDTH, 0.0, 1.0)
        return self.calculate_z_offset(aperture)

    def get_dynamic_mesh(self, aperture=0, z_offset=None):
        """
        Generates the gripper mesh based on the given aperture and optional Z-offset.
        aperture: 0.0 (closed) to 1.0 (fully open)
        z_offset: Optional manual Z-offset. If None, calculated from aperture.
        """
        angle = aperture

        # Calculate offset if not provided
        if z_offset is None:
            z_offset = self.calculate_z_offset(aperture)
        
        # Update self.grasp_offset for consistency if anyone reads it
        self.grasp_offset = np.array([0.0, 0.0, z_offset])

        # --- Left Finger ---
        # Proximal: Rotation around the X-axis at the proximal joint
        l_prox_mat = tra.translation_matrix(self.l_joint_origin) @ \
                     tra.rotation_matrix(angle, [1, 0, 0]) @ \
                     tra.translation_matrix(-self.l_joint_origin)
        
        # Distal: Parent transform * Local transform (Mimic)
        # Distal joint is at l_prox_offset + [0, 0, 0.07]
        l_dist_origin = self.l_joint_origin + np.array([0, 0, 0.07])
        
        l_dist_mat = l_prox_mat @ \
                     tra.translation_matrix(l_dist_origin) @ \
                     tra.rotation_matrix(-angle, [1, 0, 0]) @ \
                     tra.translation_matrix(-l_dist_origin)
                     
        # --- Right Finger ---
        r_prox_mat = tra.translation_matrix(self.r_joint_origin) @ \
                     tra.rotation_matrix(-angle, [1, 0, 0]) @ \
                     tra.translation_matrix(-self.r_joint_origin)
        
        r_dist_origin = self.r_joint_origin + np.array([0, 0, 0.07])
        
        r_dist_mat = r_prox_mat @ \
                     tra.translation_matrix(r_dist_origin) @ \
                     tra.rotation_matrix(angle, [1, 0, 0]) @ \
                     tra.translation_matrix(-r_dist_origin)

        # Transform copies of the meshes
        l_p = self.l_prox.copy().apply_transform(l_prox_mat)
        l_d = self.l_dist.copy().apply_transform(l_dist_mat)
        r_p = self.r_prox.copy().apply_transform(r_prox_mat)
        r_d = self.r_dist.copy().apply_transform(r_dist_mat)
        wrist_t = self.wrist.copy() 
        palm_t = self.palm.copy()
        
        # Combine everything and center on TCP
        full_mesh = trimesh.util.concatenate([palm_t, wrist_t, l_p, l_d, r_p, r_d])
        full_mesh.apply_translation(self.grasp_offset)
        
        return full_mesh
    
    def get_gripper_collision_mesh(self):
        # Collision mesh at fully open state (max retraction)
        mesh =self.get_dynamic_mesh(aperture=0.5)

        # rotate mesh um Z -90 degrees to match HSR TCP orientation
        rot_z_neg_90 = tra.rotation_matrix(-np.pi / 2, [0, 0, 1])
        mesh.apply_transform(rot_z_neg_90)

        mesh.apply_transform(tra.rotation_matrix(np.pi, [0, 0, 1]))

        return mesh

    def get_gripper_visual_mesh(self, aperture=0.5, z_offset=None):
        return self.get_dynamic_mesh(aperture=aperture, z_offset=z_offset)

def get_gripper_offset_bins():
    # For M2T2-only

    offset_bins = [
        0, 0.00794435329, 0.0158887021, 0.0238330509,
        0.0317773996, 0.0397217484, 0.0476660972,
        0.055610446, 0.0635547948, 0.0714991435, 0.08
    ]

    offset_bin_weights = [
        0.16652107, 0.21488856, 0.37031708, 0.55618503, 0.75124664,
        0.93943357, 1.07824539, 1.19423112, 1.55731375, 3.17161779
    ]
    return offset_bins, offset_bin_weights


def load_control_points() -> torch.Tensor:
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():
    """
    Load the control points for the gripper, used for visualization.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)

    control_points = load_control_points_core(gripper_config)

    mid_point = (control_points[0] + control_points[1]) / 2

    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]