import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
# def initialize_camera_pose(params, curr_time_idx, forward_prop, curr_data):
#     with torch.no_grad():
#         if curr_time_idx > 1 and forward_prop:
#             # Initialize the camera pose for the current frame based on a constant velocity model
#             # Rotation
#             prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
#             prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
#             new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
#             params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
#             # Translation
#             prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
#             prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
#             new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
#             params['cam_trans'][..., curr_time_idx] = new_tran.detach()
#         else:
#             # Initialize the camera pose for the current frame
#             params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
#             params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
#     return params

def depth_to_pointcloud(depth, intrinsics):
    """Convert a depth image into a point cloud using NumPy."""
    # Ensure depth is a numpy array (if it comes as a tensor)
    # if not isinstance(depth, np.ndarray):
    #     depth = depth.cpu().numpy()

    depth = depth.squeeze(0)  

    height, width = depth.shape
    intrinsics = (intrinsics).cpu().numpy()
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Generate meshgrid for pixel coordinates
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    # Calculate normalized coordinates
    x = (x_indices - cx) / fx
    y = (y_indices - cy) / fy
    z = depth

    # Filter out invalid depth values (depth <= 0 are invalid)
    valid = (z > 0)
    x = x[valid]
    y = y[valid]
    z = z[valid]

    # Combine x, y, z into a single [N, 3] array, where N is the number of valid points
    points = np.stack((x * z, y * z, z), axis=-1)

    return points

def apply_icp_to_depths(last_depth, curr_depth, intrinsics):
    """Apply ICP to align two depth images given camera intrinsics."""
    # Convert numpy depth arrays to Open3D point clouds
    source_points = depth_to_pointcloud(last_depth.cpu().numpy(), intrinsics)
    target_points = depth_to_pointcloud(curr_depth.cpu().numpy(), intrinsics)

    # Create Open3D point cloud objects
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
    
    # Configure ICP
    threshold = 0.05  # Maximum distance points can be from each other to be considered a match
    trans_init = np.eye(4)  # Initial transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Convert the transformation matrix to a Torch tensor
    transformation_matrix = torch.from_numpy(reg_p2p.transformation).float()
    rotation_matrix = transformation_matrix[:3, :3]
    translation_vector = transformation_matrix[:3, 3]

    return rotation_matrix, translation_vector




def quaternion_multiply(q1, q2):
    # Extract components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Compute quaternion product components
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1) 


def initialize_camera_pose(params, curr_time_idx, forward_prop, curr_data):
    with torch.no_grad():
        if curr_time_idx >= 1 and forward_prop:
            prev_quat = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_tran = params['cam_trans'][..., curr_time_idx-1].detach()

            # ICP
            curr_depth = curr_data['depth']
            last_depth = curr_data['last_d']
            intrinsics = curr_data['intrinsics']
            relative_rotation, relative_translation = apply_icp_to_depths(last_depth, curr_depth, intrinsics)

            # rotation matrix to quaternion
            relative_quat = matrix_to_quaternion(relative_rotation)
            new_quat = quaternion_multiply(prev_quat, relative_quat)
            new_quat = F.normalize(new_quat, p=2, dim=-1)

            # translation
            rotated_translation = torch.matmul(relative_rotation.to(device="cuda"), \
                                               relative_translation.unsqueeze(-1)).squeeze(-1)
            new_tran = prev_tran.to(device="cuda") + rotated_translation

            # Store the updated pose
            params['cam_unnorm_rots'][..., curr_time_idx] = new_quat
            params['cam_trans'][..., curr_time_idx] = new_tran
        else:
            # If not the first frame and no forward propagation, just carry over the previous pose
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()

    return params
