import open3d as o3d
import numpy as np
import os

def create_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def vis(pcd_path):
    # ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(pcd)
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    down_pcd, indices, inverse_indices = pcd.voxel_down_sample_and_trace(voxel_size=0.5, 
                                                                        min_bound=min_bound, 
                                                                        max_bound=max_bound)

    extracted_ints = [int_vector[0] for int_vector in inverse_indices]
    print(indices)
    print(extracted_ints)
    o3d.visualization.draw_geometries([down_pcd])

    select_pcd = pcd.select_by_index(extracted_ints)
    o3d.visualization.draw_geometries([select_pcd])

pcd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test', 'room_scan1.pcd')
vis(pcd_path)
# pcd_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tutorials', 'room_scan2.pcd')
# vis(pcd_path)
