import numpy as np
import math
import open3d as o3d
import copy


def read_pcd(file_path):
    return o3d.io.read_point_cloud(file_path)


def write_pcd(save_path, pcd):
    o3d.io.write_point_cloud(save_path, pcd)


def show_pcd(pcd_array, window_name="Open3D"):
    o3d.visualization.draw_geometries(pcd_array, window_name=window_name)


def kdtree(radius, max_nn):
    return o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)


def mesh_frame(size):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])


def draw_registration_result(source, target, addition=None, transformation=np.identity(4), window_name=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    source_temp.transform(transformation)
    data = [source_temp, target_temp]
    if addition:
        data = data + addition
    if window_name:
        o3d.visualization.draw_geometries(data, window_name=window_name)
    else:
        o3d.visualization.draw_geometries(data)


def process(pcd, rx, ry, rz, dx, dy, dz):
    translation = np.array([[1, 0, 0, dx],
                            [0, 1, 0, dy],
                            [0, 0, 1, dz],
                            [0, 0, 0, 1]])
    x_rot = np.array([[1, 0, 0, 0],
                      [0, math.cos(np.radians(rx)), -(math.sin(np.radians(rx))), 0],
                      [0, math.sin(np.radians(rx)), math.cos(np.radians(rx)), 0],
                      [0, 0, 0, 1]])
    y_rot = np.array([[math.cos(np.radians(ry)), 0, math.sin(np.radians(ry)), 0],
                      [0, 1, 0, 0],
                      [-(math.sin(np.radians(ry))), 0, math.cos(np.radians(ry)), 0],
                      [0, 0, 0, 1]])
    z_rot = np.array([[math.cos(np.radians(rz)), math.sin(np.radians(rz)), 0, 0],
                      [-(math.sin(np.radians(rz))), math.cos(np.radians(rz)), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    pcd.transform(x_rot)
    pcd.transform(y_rot)
    pcd.transform(z_rot)
    pcd.transform(translation)
    return pcd


def crop_volume(pcd, min_xyz, max_xyz):
    """
    :param pcd: open3d point cloud
    :param min_xyz: [min_x, min_y, min_z]
    :param max_xyz: [max_x, max_y, max_z]
    :return: cropped point cloud
    """
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=np.array(min_xyz), max_bound=np.array(max_xyz)))
    return pcd


def move_to_origin(pcd):
    dx, dy, dz = o3d.geometry.PointCloud.get_center(pcd)
    return process(pcd, 0, 0, 0, -dx, -dy, -dz)
