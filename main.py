import open3d as o3d
from lib.utils import *
from lib.registration import *


# データの読み込み
pcd_s = read_pcd("data/slam.pcd")
pcd_t = read_pcd("data/cad.pcd")

# 陰影をつける
pcd_s.paint_uniform_color([0, 0.651, 0.929])
pcd_s.estimate_normals(kdtree(radius=1, max_nn=30))
pcd_t.paint_uniform_color([1, 0.706, 0])
pcd_t.estimate_normals(kdtree(radius=1, max_nn=30))

# icp registration
process(pcd_s, 0, 0, -85, 3, 1, 0)
result_icp = icp_registration(pcd_s, pcd_t, None, 100)
pcd_s.transform(result_icp.transformation)

# 点群の可視化
pcd_t = crop_volume(pcd_t, [-100, -100, -0], [100, 100, 0.3])
o3d.visualization.draw_geometries([pcd_s, pcd_t], window_name="ICP")

# 評価
res = evaluate_registration(pcd_s, pcd_t)
print(res)
# --> RegistrationResult with fitness=1.000000e+00, inlier_rmse=2.353392e-01, and correspondence_set size of 983
