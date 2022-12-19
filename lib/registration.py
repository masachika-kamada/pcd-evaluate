import numpy as np
import open3d as o3d


def icp_registration(source, target, result_ransac, distance_threshold):
    if result_ransac is None:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result_icp


def evaluate_registration(source, target):
    return o3d.pipelines.registration.evaluate_registration(source, target, 100, np.identity(4))
