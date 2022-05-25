import pickle
import os
import numpy as np
from guage.calibrate_gauge import calib
from guage.calibrate_gauge import get_Ex
from guage.calibrate_gauge import pixel2world

# mat_inter_matrix = pickle.load(load_file)
# coff_dis_matrix = pickle.load(load_file)
# print(mat_inter_matrix,coff_dis_matrix)

inter_corner_shape = (8, 5)  #角点形状,横向8个，纵向5个
size_per_grid = 0.026   #每个格子边长物理尺寸，26mm
img_dir = "./guage/calibration_data/inter_img"    #标定照片文件夹
img_type = "jpg"           #标定照片格式
save_corner_path = "./guage/calibration_data/save_coner"  #内参标定识别的角点识别
if (not os.path.exists(save_corner_path)):
    os.makedirs(save_corner_path)

pavement_H = 1.339      #棋盘格距离地面的高度
ex_img_path = "./guage/calibration_data/ex_img/1560.jpg"   #外参标定图片路径

#内参标定
mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir, img_type, save_corner_path)

#外参标定
R_matrix,T_vec,success = get_Ex(ex_img_path,size_per_grid,inter_corner_shape[0],inter_corner_shape[1],mat_inter,coff_dis)


def location(u, v):  # 便于测速代码引用
    inter_corner_shape = (8, 5)  # 角点形状,横向8个，纵向5个
    size_per_grid = 0.026  # 每个格子边长物理尺寸，26mm
    img_dir = "E:/Internet Downloads/yolov5_deepsort_speed-master\
    yolov5_deepsort_speed-master/guage/calibration_data/inter_img"  # 标定照片文件夹
    img_type = "jpg"  # 标定照片格式
    save_corner_path = "E:/Internet Downloads/yolov5_deepsort_speed-master\
    yolov5_deepsort_speed-master/guage/calibration_data/save_coner"  # 内参标定识别的角点识别
    if (not os.path.exists(save_corner_path)):
        os.makedirs(save_corner_path)
    pavement_H = 1.339  # 棋盘格距离地面的高度
    ex_img_path = "E:/Internet Downloads/yolov5_deepsort_speed-master\
    yolov5_deepsort_speed-master/guage/calibration_data/ex_img/1560.jpg"  # 外参标定图片路径
    # 内参标定
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir, img_type, save_corner_path)
    # 外参标定
    R_matrix, T_vec, success = get_Ex(ex_img_path, size_per_grid, inter_corner_shape[0], inter_corner_shape[1],
                                      mat_inter, coff_dis)
    loc = pixel2world(R_matrix, T_vec, mat_inter, coff_dis, pavement_H, u, v)
    return loc


#测距
pixes_pave = np.zeros((6,3),np.float32)
pixes_pave[0,:] = np.array([700,975,1])  # 构造像素坐标
pixes_pave[1,:] = np.array([1425,1016,1])  # 构造像素坐标
pixes_pave[2,:] = np.array([1003,992,1])  # 构造像素坐标
pixes_pave[3,:]= np.array([1328,1012,1])  # 构造像素坐标
pixes_pave[4,:]= np.array([1005,809,1])  # 构造像素坐标
pixes_pave[5,:]= np.array([1252,822,1])  # 构造像素坐标


pave_coor = np.zeros((6,3),np.float32) #存储点的世界坐标
for i in range(6):
    # 计算点的世界坐标
    pixes_pave[i,:] = pixel2world(R_matrix, T_vec, mat_inter,coff_dis,pavement_H,pixes_pave[i,0],pixes_pave[i,1])




point_distance = np.zeros((6,6),np.float32) #存储点与点之间的距离
for i in range(6):
    for j in range(6):
        point_distance[i,j] = np.linalg.norm(pixes_pave[i,:]-pixes_pave[j,:])


print(point_distance)







