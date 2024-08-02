from collections import deque
import os, argparse, time
import cv2, torch
from tqdm import tqdm
import pdb
import numpy as np
from rellis_utils.lidar2img import load_from_bin, get_cam_mtx, get_mtx_from_yaml, points_filter, parse_poses_slam, DIST_COEFF
from universal_utils.universal_utils import write_geoatt_vtk3_prob, write_geoatt_vtk3

# binning_num = 30 # FOR EXPERIMENTS
# binning_num = 1  # FOR VISUALIZATION
def accumulate(points_queue, first_pose):
    # input: points_queue: { 'points', 'labels', 'pose', 'idx' }
    # output: accumulated points with label in the global frame. [N*4 numpy array]

    global_points, global_labels = None, []
    # accumulate frames.
    for i in range(len(points_queue)):
        data = points_queue[i]

        points_now = data['points']
        label_now = data['labels']
        pose_now = data['pose']

        # Euclidean to Homogeneous
        hom_points = np.ones((points_now.shape[0], 4))
        hom_points[:, :3] = points_now
        # points_now_to_global = np.matmul(pose_now, hom_points.T).T		
        points_now_to_pose0 = np.matmul(np.matmul(np.linalg.inv(first_pose), pose_now), hom_points.T).T		
        
        if global_points is not None:
            global_points = np.concatenate((global_points, points_now_to_pose0[:, :3]), axis=0)
        else:
            global_points = points_now_to_pose0[:, :3]
        global_labels.extend(label_now)
    return global_points, global_labels

def process_each_frame(seq, lbl_path, os1_path, distCoeff, P, RT_os1_to_cam, os1_r, os1_t, save_vtk = False, ds=None):
    label =  np.load(lbl_path, allow_pickle=True) # (13, 1200, 1920)

    if ds is not None:
        label = torch.nn.functional.interpolate(torch.tensor(label).to(torch.float32).unsqueeze(0), (IMG_HEIGHT, IMG_WIDTH), mode='bilinear').squeeze(0)
        label = np.array(label)

    os1_points = load_from_bin(os1_path)

    # 1. Project, Find Correspondence
    os1_filtered_pts, _ = points_filter(os1_points, label.shape[1], label.shape[0], P, RT_os1_to_cam)
    try:
        os1_projected, _ = cv2.projectPoints(os1_filtered_pts[:, :], os1_r, os1_t, P, distCoeff)
    except:
        return None, None
    projected_points = os1_projected.squeeze(1)
    ### os1_points: (131072, 3) => os1_filtered_pts: (11630, 3) => os1_projected: (11630, 2)
    
    # 3. Get Label Information
    valid_points = []
    valid_semantics = []
    for idx, (projected_pt, org_pt) in enumerate(zip(projected_points, os1_filtered_pts)):
        if 0 <= projected_pt[1] < IMG_HEIGHT and 0 <= projected_pt[0] < IMG_WIDTH:
            pt_label = label[:, int(projected_pt[1]), int(projected_pt[0])] # ORIGIN OF UNCERTAINTY!

            valid_points.append(np.array(org_pt))
            valid_semantics.append(pt_label)
        # else: # For Projection Visualization
        #     valid_points.append(np.array(org_pt))
        #     valid_semantics.append(300)
    
    # 4. Make VTK visualization
    if save_vtk:
        write_geoatt_vtk3(os.path.join(VTK_OUT_DIR, seq, os.path.basename(lbl_path)[:-4]), np.array(valid_points), label=valid_semantics)
    return np.array(valid_points), valid_semantics


def process_each_sequence(seq, distCoeff, binning_num=1, save_vtk = True, ds = None):
    camera_info_file = os.path.join(RELLIS_CAMERA_INFO, seq, 'camera_info.txt')
    transforms_yaml_file = os.path.join(RELLIS_TRANSFORM, seq, 'transforms.yaml')
    pose_file = os.path.join(RELLIS_ROOT, seq, 'poses.txt')

    # 1. Parse Camera Info File & Poses
    P = get_cam_mtx(camera_info_file)
    poses = parse_poses_slam(pose_file)

    # 2. Parse Ouster LiDAR => Camera Transformation
    RT_os1_to_cam = get_mtx_from_yaml(transforms_yaml_file)
    RT_R, RT_T = RT_os1_to_cam[:3,:3], RT_os1_to_cam[:3,3].reshape(3, 1)
    os1_r, _ = cv2.Rodrigues(RT_R)
    os1_t = RT_T

    # 4. Construct the correspondence file relationship!
    seq_prob_dir = os.path.join(PROB_IN_DIR, seq)
    os1_pts_dir = os.path.join(RELLIS_ROOT, seq, OS1_LIDAR_POSTFIX)

    lbl_files = sorted([os.path.join(seq_prob_dir, f) for f in np.sort(os.listdir(seq_prob_dir)) if f.endswith(".npy")])

    curr_acc = 0
    saved_num = 1
    accum_points = deque()

    first_pose = None

    for lbl in tqdm(lbl_files):
        lbl_key = os.path.basename(lbl)[5:11]
        os1_path = os.path.join(os1_pts_dir, f"{lbl_key}.bin")
        tf = poses[int(lbl_key)]

        if first_pose is None:
            first_pose = tf

        if os.path.isfile(lbl) and os.path.isfile(os1_path):
            # 5. Process Each Files!
            p, l = process_each_frame(seq, lbl, os1_path, distCoeff, P, RT_os1_to_cam, os1_r, os1_t, ds=ds)
            
            if p is not None and l is not None:
                pass
            else:
                continue

            curr_acc += 1
            accum_points.append({ 'points': p, 'labels': l, 'pose': tf, 'idx': int(lbl_key)}) # p, tf numpy array

            if curr_acc == binning_num:
                # 6. Accumulate!
                global_accumulated_points, accumulated_label = accumulate(accum_points, first_pose)

                if save_vtk:
                    write_geoatt_vtk3_prob(os.path.join(VTK_OUT_DIR, seq, f"merged{saved_num}_frame{curr_acc}"), global_accumulated_points, accumulated_label)
                
                concat_map = np.concatenate((global_accumulated_points, np.array(accumulated_label)), axis=1)
                np.save(os.path.join(NPY_OUT_DIR, seq, f"merged{saved_num}"), concat_map)
                
                saved_num += 1
                curr_acc, accum_points = 0, deque()
        else:
            pass
            # print(f"\nERROR - {lbl} | {os1_path} might not exist")
    
    # Last Few Frames
    if curr_acc != 0:
        # 6. Accumulate!
        global_accumulated_points, accumulated_label = accumulate(accum_points, first_pose)
        if save_vtk:
            write_geoatt_vtk3_prob(os.path.join(VTK_OUT_DIR, seq, f"merged{saved_num}_frame{curr_acc}"), global_accumulated_points, accumulated_label)
        
        concat_map = np.concatenate((global_accumulated_points, np.array(accumulated_label)), axis=1)
        np.save(os.path.join(NPY_OUT_DIR, seq, f"merged{saved_num}"), concat_map)
        
    return

# 1. Dataset Location
# Where is Image, LiDAR, Pose, ...
RELLIS_ROOT = '/data/Rellis-3D'
RELLIS_CAMERA_INFO = '/data/Rellis_3D_cam_intrinsic/Rellis-3D'
RELLIS_TRANSFORM = '/data/Rellis_3D_cam2lidar_20210224/Rellis_3D'
OS1_LIDAR_POSTFIX = 'os1_cloud_node_kitti_bin'

if __name__ == '__main__':
    start = time.time()
    IMG_WIDTH, IMG_HEIGHT = 1920, 1200
     # 1. Parameter Parsing
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('target_remark', help='the remark')
    parser.add_argument('sequence', help='sequence number: Only Support `00000`, `00001`, `00002`, `00003`, `00004`.')
    parser.add_argument('binning_num', type=int, help='NUM_BIN_point_cloud: How many frames do you want to merge in one output file.')
    parser.add_argument('--ds_factor', nargs='?', type=float, default=2.0, help='# of the epochs')
    args = parser.parse_args()

    target_remark, seq, binning_num = args.target_remark, args.sequence, args.binning_num
    ds_factor = args.ds_factor if args.ds_factor > 0.0 else None

    if seq not in ['00000', '00001', '00002', '00003', '00004']:
        raise NotImplementedError("Not Defined Sequence")
    
    # 0. Process Parameters

    # 1. Directory Settings! You may modify this! ##########################################################
    MY_DIR_ROOT = '/kjyoung/convertedRellis'
    OUT_REMARK = os.path.join(MY_DIR_ROOT, target_remark)
    # in OUT_REMARK directory
    # 01_inferenced_npy: 2D Seg Model Inferenced (C * H * W) Numpy Files
    # 02_inferenced_vis: 2D Seg Model Inferenced Visualizations
    # 03_nby3pK_npy: 3D Points Projected & Merged (N * (3 + C)) Numpy Files
    # 04_vtk: 3D Point Projected & Merged VTK Visualizations
    # 05_pVec_pcd: PCD converted files - Each point has 13 double fields for describing class probabilities!

    PROB_IN_DIR = os.path.join(OUT_REMARK, f'01_inferenced_npy')
    NPY_OUT_DIR = os.path.join(OUT_REMARK, f'03_nby3pK_npy')
    VTK_OUT_DIR = os.path.join(OUT_REMARK, f'04_vtk')
    PVEC_OUT_DIR = os.path.join(OUT_REMARK, f'05_pVec_pcd')

    if not os.path.isdir(PROB_IN_DIR):
        raise Exception(f"There is No Inferenced Data! {PROB_IN_DIR}")

    os.makedirs(NPY_OUT_DIR, exist_ok=True)
    os.makedirs(VTK_OUT_DIR, exist_ok=True)
    os.makedirs(PVEC_OUT_DIR, exist_ok=True) # For next step. Python is much easier to deal with makedirs.

    seq_dirs = [os.path.join(PROB_IN_DIR, seq), os.path.join(NPY_OUT_DIR, seq), os.path.join(VTK_OUT_DIR, seq), os.path.join(PVEC_OUT_DIR, seq)]
    for seq_dir in seq_dirs:
        os.makedirs(seq_dir, exist_ok=True)
            
    ########################################################################################################
    print(f"\n---- Sequence[{seq}] Process Started ----")
    process_each_sequence(seq, DIST_COEFF, binning_num=args.binning_num, ds=ds_factor)

    end = time.time()
    print(f"\n[{seq}] done:: {end - start:.5f} sec")

# IDMap + LiDAR => Frame-wise Projected Annotated PointCloud1
# rellis_acc_label.py:: IDMap .npy + Ouster LiDAR => Accumulate Annotated PointCloud.
# python rellis_acc_label.py === python rellis_acc_inference.py 231130_IntegratedFramework1_GT 00000 30 --ds_factor -1
# => process_each_sequence(seq, DIST_COEFF, binning_num=30, ds=None)