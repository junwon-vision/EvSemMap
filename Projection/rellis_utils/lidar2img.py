# FROM https://github.com/unmannedlab/RELLIS-3D/blob/main/utils/lidar2img.ipynb
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import yaml

DIST_COEFF = np.array([-0.134313,-0.025905,0.002181,0.00084,0]).reshape((5,1))

def load_from_bin(bin_path):
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:,:3]

def print_projection_plt(points, color, image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def depth_color(val, min_d=0, max_d=120):
    np.clip(val, 0, max_d, out=val) 
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

def points_filter(points,img_width,img_height,P,RT):
    ctl = RT
    ctl = np.array(ctl)
    fov_x = 2*np.arctan2(img_width, 2*P[0,0])*180/3.1415926+10
    fov_y = 2*np.arctan2(img_height, 2*P[1,1])*180/3.1415926+10
    R= np.eye(4)
    p_l = np.ones((points.shape[0],points.shape[1]+1))
    p_l[:,:3] = points
    p_c = np.matmul(ctl,p_l.T)
    p_c = p_c.T
    x = p_c[:,0]
    y = p_c[:,1]
    z = p_c[:,2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    xangle = np.arctan2(x, z)*180/np.pi;
    yangle = np.arctan2(y, z)*180/np.pi;
    flag2 = (xangle > -fov_x/2) & (xangle < fov_x/2)
    flag3 = (yangle > -fov_y/2) & (yangle < fov_y/2)
    res = p_l[flag2&flag3,:3]
    res = np.array(res)
    x = res[:, 0]
    y = res[:, 1]
    z = res[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    color = depth_color(dist, 0, 70)
    return res,color

def get_cam_mtx(filepath):
    data = np.loadtxt(filepath)
    P = np.zeros((3,3))
    P[0,0] = data[0]
    P[1,1] = data[1]
    P[2,2] = 1
    P[0,2] = data[2]
    P[1,2] = data[3]
    return P

def get_mtx_from_yaml(filepath, key='os1_cloud_node-pylon_camera_node'):
    with open(filepath,'r') as f:
        data = yaml.load(f,Loader= yaml.Loader)
    q = data[key]['q']
    q = np.array([q['x'],q['y'],q['z'],q['w']])
    t = data[key]['t']
    t = np.array([t['x'],t['y'],t['z']])
    R_vc = Rotation.from_quat(q)
    R_vc = R_vc.as_matrix()

    RT = np.eye(4,4)
    RT[:3,:3] = R_vc
    RT[:3,-1] = t
    RT = np.linalg.inv(RT)
    return RT

def parse_poses_slam(filename):
    """ read poses file with per-scan poses from given filename

        IMU -> Global World.

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)
    poses = []

    for line in file:
        values = [float(v) for v in line.strip().split()]
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1

        poses.append(pose)

    return poses

if __name__ == '__main__':
    RELLIS_ROOT = '/data/Rellis-3D'
    RELLIS_CAMERA_INFO = '/data/Rellis_3D_cam_intrinsic/Rellis-3D'
    RELLIS_TRANSFORM = '/data/Rellis_3D_cam2lidar_20210224/Rellis_3D'
    RELLIS_VEL2OUS = '/kjyoung/Rellis_vel2ous/Rellis-3D'

    Sequence = ['00000', '00001', '00002', '00003', '00004']
    IMG_POSTFIX = 'pylon_camera_node'
    OS1_LIDAR_POSTFIX = 'os1_cloud_node_kitti_bin'
    VEL_LIDAR_POSTFIX = 'vel_cloud_node_kitti_bin'
    img_file = os.path.join(RELLIS_ROOT, Sequence[0], IMG_POSTFIX, 'frame000104-1581624663_149.jpg')
    os1_lidar_file = os.path.join(RELLIS_ROOT, Sequence[0], OS1_LIDAR_POSTFIX, '000104.bin')
    vel_lidar_file = os.path.join(RELLIS_ROOT, Sequence[0], VEL_LIDAR_POSTFIX, '000104.bin')
    camera_info_file = os.path.join(RELLIS_CAMERA_INFO, Sequence[0], 'camera_info.txt')
    transforms_yaml_file = os.path.join(RELLIS_TRANSFORM, Sequence[0], 'transforms.yaml')
    vel2ous_transforms_yaml_file = os.path.join(RELLIS_VEL2OUS, Sequence[0], 'vel2os1.yaml')

    print(img_file)
    print(os1_lidar_file)
    print(vel_lidar_file)
    print(camera_info_file)
    print(transforms_yaml_file)

    image = cv2.imread(img_file)
    img_height, img_width, channels = image.shape

    os1_points = load_from_bin(os1_lidar_file)
    vel_points = load_from_bin(vel_lidar_file)

    distCoeff = np.array([-0.134313,-0.025905,0.002181,0.00084,0])
    distCoeff = distCoeff.reshape((5,1))

    P = get_cam_mtx(camera_info_file)
    RT= get_mtx_from_yaml(transforms_yaml_file)
    R_vc = RT[:3,:3]
    T_vc = RT[:3,3]
    T_vc = T_vc.reshape(3, 1)
    rvec,_ = cv2.Rodrigues(R_vc)
    tvec = T_vc

    RT_vel2os1 = get_mtx_from_yaml(vel2ous_transforms_yaml_file, key="vel2os1")
    velpcd_ = np.ones((vel_points.shape[0], 4))
    velpcd_[:, :3] = vel_points
    vel_points = RT_vel2os1 @ velpcd_.T
    vel_points = vel_points.T[:, :3]

    xyz_v, c_ = points_filter(os1_points, img_width, img_height, P, RT)
    imgpoints, _ = cv2.projectPoints(xyz_v[:,:], rvec, tvec, P, distCoeff)
    imgpoints = np.squeeze(imgpoints,1)
    imgpoints = imgpoints.T

    vel_xyz_v, vel_c_ = points_filter(vel_points, img_width, img_height, P, RT)
    vel_imgpoints, _ = cv2.projectPoints(vel_xyz_v[:,:], rvec, tvec, P, distCoeff)
    vel_imgpoints = np.squeeze(vel_imgpoints, 1)
    vel_imgpoints = vel_imgpoints.T


    res = print_projection_plt(points=imgpoints, color=c_, image=image)
    plt.subplots(1,1, figsize = (20,20) )
    plt.title("Ouster points to camera image Result")
    plt.imshow(res)
    plt.savefig('./231121-os1.png')

    res = print_projection_plt(points=vel_imgpoints, color=vel_c_, image=image)
    plt.subplots(1,1, figsize = (20,20) )
    plt.title("Velodyne points to camera image Result")
    plt.imshow(res)
    plt.savefig('./231121-vel.png')

    res = print_projection_plt(points=np.concatenate((imgpoints, vel_imgpoints), axis=1),
                            color=np.concatenate((c_, vel_c_)),
                            image=image)
    plt.subplots(1, 1, figsize=(20, 20))
    plt.title("Ouster + Velodyne Points to Camera Image Result!!")
    plt.imshow(res)
    plt.savefig('./231121-ouster+velodyne.png')

    print("Jobs Done~")