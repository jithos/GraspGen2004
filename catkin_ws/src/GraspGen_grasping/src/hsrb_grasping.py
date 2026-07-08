#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import ros_numpy
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from tf import TransformListener
from tf.transformations import (
    quaternion_about_axis, 
    quaternion_multiply, 
    unit_vector, 
    quaternion_from_matrix
)
from pathlib import Path
import os
import time
import scipy

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import (
    depth_and_segmentation_to_point_clouds, 
    point_cloud_outlier_removal, 
    filter_colliding_grasps
)
from grasp_gen.utils.meshcat_utils import (
    create_visualizer, 
    visualize_pointcloud, 
    visualize_grasp, 
    get_color_from_score,
    make_frame
)
from grasp_gen.robot import get_gripper_info, import_module_from_path

from robokudo_msgs.msg import GenericImgProcAnnotatorAction, GenericImgProcAnnotatorResult

VISUALIZER_ENABLED = True

GRASP_FILTERING_THRESHOLD = 0.2
X_MIN_ANGLE = -45
X_MAX_ANGLE = 0
Y_MIN_ANGLE = -30
Y_MAX_ANGLE = 30
Z_MIN_ANGLE = -180
Z_MAX_ANGLE = 180

class GraspGenWrapper():

    def __init__(self):
        rospy.init_node("hsrb_graspgen_wrapper")
        self.server = actionlib.SimpleActionServer(
            "/pose_estimator/find_grasppose_hsrb_graspgen",
            GenericImgProcAnnotatorAction,
            self.execute_cb,
            auto_start=False
        )
        rospy.loginfo("HSRB GraspGenWrapper initializing...")
        
        # Load GraspGen model config
        gripper_cfg_path = rospy.get_param("~gripper_config", "/GraspGen2004/code/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml")
        self.grasp_cfg = load_grasp_cfg(gripper_cfg_path)
        self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
        
        # HSR specific gripper info
        self.gripper_name = "toyota_hsr"
        self.gripper_info = get_gripper_info(self.gripper_name)
        self.gripper_collision_mesh = self.gripper_info.collision_mesh
        
        if VISUALIZER_ENABLED:
            rospy.logwarn("Make sure your already started MeshCat visualizer before running the GraspGen2004 ROS node")
            self.visualizer = create_visualizer()

        # Depth scale
        self.depth_scale = rospy.get_param("~depth_scale", 1000.0)
        self.cam_info = None

        self.server.start()
        rospy.loginfo("HSRB GraspGenWrapper server started")

    @staticmethod
    def add_symmetry_hull(pc, thickness_offset=0.005):
        """Adds a mirrored hull of points to the point cloud to help grasp inference on objects."""
        center = np.mean(pc, axis=0)
        mirrored_pc = center - (pc - center)
        ray_dirs = pc / np.linalg.norm(pc, axis=1, keepdims=True)
        
        dists = np.linalg.norm(pc - center, axis=1, keepdims=True)
        smart_mirrored_pc = mirrored_pc + ray_dirs * (np.mean(dists) + thickness_offset)
        
        return np.vstack([pc, smart_mirrored_pc])

    @staticmethod
    def transform_and_crop(pc, T, y=0.05, z=0.211, x=0.14):
        """Transforms the point cloud to the grasp frame and crops to a box around the origin. Used for thickness estimation."""
        points = np.asarray(pc)
        points_t = tra.transform_points(points, T)
        mask = (
            (np.abs(points_t[:,1]) < y / 2) &    # height (Y)
            (np.abs(points_t[:,2]) < z ) &       # search depth (Z)
            (np.abs(points_t[:,0]) < x / 2)      # Gripper width
        )
        return points_t[mask]

    @staticmethod
    def compute_thickness(points):
        if len(points) < 10:
            return None
        projections_x = points[:,0]  
        thickness_x = projections_x.max() - projections_x.min()
        return thickness_x

    def grasp_thickness(self, pc, T_points_to_grasp_frame, T_center=None, vis=None):
        local_pts = self.transform_and_crop(pc, T_points_to_grasp_frame)
        if vis is not None and T_center is not None:
            local_pts_viz = tra.transform_points(local_pts, np.linalg.inv(T_points_to_grasp_frame))
            local_pts_viz = tra.transform_points(local_pts_viz, T_center)
            visualize_pointcloud(vis, "local_pts_thickness", local_pts_viz, size=0.005, color=[0, 255, 255])
        
        thickness = self.compute_thickness(local_pts)
        return thickness

    def get_hsr_z_offset(self, object_width):
        # Path logic to find toyota_hsr.py config
        script_dir = Path(__file__).parent
        # Go up from catkin_ws/src/GraspGen_grasping/src/ to catkin_ws/src/
        # Then to workspace root
        # Then to code/config/grippers/toyota_hsr.py
        # Based on structure: 
        # GraspGenforHSR/
        #   catkin_ws/src/GraspGen_grasping/src/hsrb_grasping.py
        #   code/config/grippers/toyota_hsr.py
        
        gripper_module_path = script_dir.parents[3] / "code" / "config" / "grippers" / "toyota_hsr.py"
        
        gripper_module = import_module_from_path(str(gripper_module_path))
        if hasattr(gripper_module, "GripperModel"):
            gripper_model = gripper_module.GripperModel()
            max_w = gripper_model.MAX_WIDTH
            if object_width is None:
                aperture = 0.5 # Default to mid-range if width is unknown
            else:
                aperture = np.clip(object_width / max_w, 0.0, 1.0)
            collision_offset = gripper_model.calculate_z_offset(aperture)
            return collision_offset
        rospy.logwarn("Could not load GripperModel, returning 0")
        return 0.135 # default offset if we can't compute it

    @staticmethod
    def align_hsr_camera(pose):
        approach = pose[1, 0]
        if approach > 0:
            rospy.loginfo("Rotating HSR pose around Z by 180 degrees for camera alignment")
            rot_z_180 = tra.rotation_matrix(np.pi, [0, 0, 1])
            pose = pose @ rot_z_180
        return pose

    @staticmethod
    def filter_grasps_by_pose(grasps, scores):
        filtered_grasps = []
        filtered_scores = []
        for i, (grasp, score) in enumerate(zip(grasps, scores)):
            # New scoring apporach with euler angles
            R = grasp[:3, :3]
            R_eul = scipy.spatial.transform.Rotation.from_matrix(R).as_euler("XYZ",degrees=True)
            if ( R_eul[0] > X_MIN_ANGLE
                and R_eul[0] < X_MAX_ANGLE
                and R_eul[1] > Y_MIN_ANGLE
                and R_eul[1] < Y_MAX_ANGLE
                and R_eul[2] > Z_MIN_ANGLE
                and R_eul[2] < Z_MAX_ANGLE
            ):
                filtered_grasps.append(grasp)
                filtered_scores.append(score)

            # Old scoring approach with single element in rotation matrix
            # grasp_approach = grasp[:3, 2]
            # score_front = np.dot(grasp_approach, [0, 0, 1])
            # if score_front >= GRASP_FILTERING_THRESHOLD: # 0.08
            #      filtered_grasps.append(grasp)
            #      filtered_scores.append(score)

        return np.array(filtered_grasps), np.array(filtered_scores)

    def get_cam_info(self):
        cam_info_topic = rospy.get_param("~camera_info_topic", '/hsrb/head_rgbd_sensor/depth_registered/camera_info')
        try:
            self.cam_info = rospy.wait_for_message(cam_info_topic, CameraInfo, timeout=2.0)
        except rospy.ROSException:
            rospy.logwarn(f"Could not get CameraInfo from {cam_info_topic}, using default HSR intrinsics")
            self.cam_info = CameraInfo()
            self.cam_info.K = [538.391033533567, 0.0, 315.3074696331638, 0.0, 538.085452058436, 233.0483557773859, 0.0, 0.0, 1.0]

    def execute_cb(self, goal):
        result = GenericImgProcAnnotatorResult()
        try:
            self.get_cam_info()
            intrinsics = np.array(self.cam_info.K).reshape(3, 3)
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]

            depth_image = ros_numpy.numpify(goal.depth)
            depth = depth_image.astype(np.float32) / self.depth_scale
            segmentation_mask = ros_numpy.numpify(goal.mask_detections[0])
            rgb = ros_numpy.numpify(goal.rgb)

            pc_scene, pc_object, pc_colors_scene, pc_colors_object = depth_and_segmentation_to_point_clouds(
                depth_image=depth,
                segmentation_mask=segmentation_mask,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rgb_image=rgb,
                target_object_id=1,
                remove_object_from_scene=True
            )

            pc_object_torch = torch.from_numpy(pc_object)
            pc_filtered, pc_removed = point_cloud_outlier_removal(pc_object_torch, threshold=0.02, K=100)
            pc_filtered = pc_filtered.numpy()

            # Augment PC for better inference
            pc_sym= self.add_symmetry_hull(pc_filtered)
            pc_sym_sampled = pc_sym[np.random.choice(len(pc_sym), min(15000, len(pc_sym)), replace=False)]

            if VISUALIZER_ENABLED: visualize_pointcloud(self.visualizer, "obj-pc-augmented", pc_sym_sampled)

            # Grasp inference on augmented PC
            grasps_inferred, grasp_conf_inferred = GraspGenSampler.run_inference(
                pc_sym_sampled,
                self.grasp_sampler,
                grasp_threshold=0.80,
                num_grasps=4000,
                topk_num_grasps=2000
            )

            if len(grasps_inferred) == 0:
                rospy.logwarn("No grasps found from inference!")
                result.success = False
                return

            grasp_conf_inferred = grasp_conf_inferred.cpu().numpy()
            grasps_inferred = grasps_inferred.cpu().numpy()
            grasps_inferred[:, 3, 3] = 1

            # Collision filtering
            if len(pc_scene) > 8192:
                scene_pc_downsampled = pc_scene[np.random.choice(len(pc_scene), 8192, replace=False)]
            else:
                scene_pc_downsampled = pc_scene

            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_pc_downsampled,
                grasp_poses=grasps_inferred,
                gripper_collision_mesh=self.gripper_collision_mesh,
                collision_threshold=0.01,
                num_collision_samples=2000
            )

            collision_free_grasps = grasps_inferred[collision_free_mask]
            collision_free_scores = grasp_conf_inferred[collision_free_mask]

            if len(collision_free_grasps) == 0:
                rospy.logwarn("No collision-free grasps found!")
                result.success = False
                return

            # Pose filtering (HSR specific)
            filtered_grasps, filtered_scores = self.filter_grasps_by_pose(collision_free_grasps, collision_free_scores)
            
            if len(filtered_grasps) > 0:
                best_idx = np.argmax(filtered_scores)
                best_grasp_pose = filtered_grasps[best_idx]
                best_score = filtered_scores[best_idx]
            else:
                rospy.logwarn("No grasps remain after pose filtering. Using best collision-free grasp.")
                best_idx = np.argmax(collision_free_scores)
                best_grasp_pose = collision_free_grasps[best_idx]
                best_score = collision_free_scores[best_idx]

            rospy.loginfo("HSR specific post-processing")
            # Post-processing for HSR
            # 1. Thickness / Z-Offset
            # Use visualizer centering for debug only if needed
            object_center = pc_filtered.mean(axis=0)
            T_center_viz = tra.translation_matrix(-object_center)

            thickness = self.grasp_thickness(pc_sym_sampled, np.linalg.inv(best_grasp_pose), T_center=T_center_viz, vis=(self.visualizer if VISUALIZER_ENABLED else None))
            if thickness is None:
                thickness = 0.07 # Default
            
            z_offset = self.get_hsr_z_offset(thickness)
            
            # 2. Final transformations
            # best_grasp_pose is in camera frame
            hsr_pose = best_grasp_pose @ tra.translation_matrix([0, 0, z_offset]) @ tra.rotation_matrix(-np.pi / 2, [0, 0, 1]) @ tra.rotation_matrix(np.pi, [0, 0, 1])
            
            hsr_pose = self.align_hsr_camera(hsr_pose)

            # Convert to ROS pose
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = hsr_pose[:3, 3]
            quat = tra.quaternion_from_matrix(hsr_pose) # [w, x, y, z]
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat[1], quat[2], quat[3], quat[0]

            result.pose_results = [pose]
            result.class_confidences = [float(best_score)]
            result.class_names = ['Unknown Object']
            result.success = True
            
            rospy.loginfo("Visualizing grasp poses")
            if VISUALIZER_ENABLED:
                # Visualization
                visualize_pointcloud(self.visualizer, "scene", pc_scene, pc_colors_scene)
                visualize_pointcloud(self.visualizer, "object", pc_object, pc_colors_object)
                            # Visualize all grasps
                for i, grasp in enumerate(grasps_inferred[:100]):
                    visualize_grasp(
                        self.visualizer,
                        f"grasps/{i:03d}/grasp",
                        T_center_viz @ grasp,
                        color=[200,200,200],
                        gripper_name=self.gripper_name,
                        linewidth=0.8,
                    )

                # Visualize collision-free grasps
                for i, grasp in enumerate(collision_free_grasps[:100]):
                    visualize_grasp(
                        self.visualizer,
                        f"collision_free_grasps/{i:03d}/grasp",
                        T_center_viz @ grasp,
                        color=[0,255,0],
                        gripper_name=self.gripper_name,
                        linewidth=0.8,
                    )

                # Visualize colliding grasps
                colliding_grasps = grasps_inferred[~collision_free_mask]
                colliding_scores = grasp_conf_inferred[~collision_free_mask]
                for i, grasp in enumerate(colliding_grasps[:20]):
                    visualize_grasp(
                        self.visualizer,
                        f"colliding_grasps/{i:03d}/grasp",
                        T_center_viz @ grasp,
                        color=[255,0,0],
                        gripper_name=self.gripper_name,
                        linewidth=0.8,
                    )

                # Visualize filtered grasps
                for i, grasp in enumerate(filtered_grasps[:100]):
                    visualize_grasp(
                        self.visualizer,
                        f"filtered_grasps/{i:03d}/grasp",
                        T_center_viz @ grasp,
                        color=[0,0,255],
                        gripper_name=self.gripper_name,
                        linewidth=0.8,
                    )
                visualize_grasp(self.visualizer, "chosen_grasp", hsr_pose, color=[255, 255, 0], gripper_name=self.gripper_name)
            
            rospy.loginfo('GraspGen (HSR): Grasp generation successful')

        except Exception as e:
            rospy.logerr(f"Error in HSRB GraspGenWrapper: {e}")
            result.success = False
        finally:
            self.server.set_succeeded(result)

if __name__ == "__main__":
    wrapper = GraspGenWrapper()
    rospy.spin()
