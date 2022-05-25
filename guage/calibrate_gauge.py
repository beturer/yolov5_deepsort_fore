import os
import numpy as np
import cv2
import glob
import pickle

# 相机参数标定得到相机的内参矩阵，畸变矩阵
def calib(inter_corner_shape, size_per_grid, img_dir, img_type,save_corner_path):
    '''
    :param inter_corner_shape:  棋盘格角点形状
    :param size_per_grid: 每个格子的物理尺寸，m为单位
    :param img_dir:  标定图片存储文件夹
    :param img_type: 标定图片格式
    :param save_corner_path: 保存识别出的角点图片路径
    :return: 内参矩阵 3*3矩阵 mat_inter, 畸变系数 1*5矩阵 coff_dis
    '''

    # 标准：仅用于subpix校准，此处不使用。
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w, h = inter_corner_shape
    # cpp_int：int形式的角点，以“ int”形式保存世界空间中角点的坐标
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w * h, 3), np.float32)
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    # cp_world：世界空间中的角点，保存世界空间中角点的坐标。
    cp_world = cp_int * size_per_grid

    obj_points = []  # 世界空间中的点
    img_points = []  # 图像空间中的点（与obj_points相关）
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    # print(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 找到角点cp_img：像素空间中的角点。
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
        # if ret is True, save.
        if ret == True:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            #画出每张图的角点并保存
            cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
            cv2.imwrite(save_corner_path + os.sep + img_name, img)
            # 显示角点
            # cv2.imshow('FoundCorners', img)
            # cv2.waitKey(1)

    cv2.destroyAllWindows()
    # 校准相机
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None,None)
    # print(("ret:"), ret)
    # print(("internal matrix:\n"), mat_inter)
    # # in the form of (k_1,k_2,p_1,p_2,k_3)
    # print(("distortion cofficients:\n"), coff_dis)
    # print(("rotation vectors:\n"), v_rot)
    # print(("translation vectors:\n"), v_trans)
    # 计算重新投影的误差
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        total_error += error

    #保存标定的内参和畸变系数
    save_file = open("camera_internal_parametes.bin", "wb")
    pickle.dump(mat_inter, save_file)  # 顺序存入变量
    pickle.dump(coff_dis, save_file)
    save_file.close()

    return mat_inter, coff_dis


#外参获取函数
def get_Ex(photo_path,size_per_grid,x_nums,y_nums,mtx,dist):
    '''
    :param photo_path: 标定图片路径
    :param size_per_grid: 棋盘格的尺寸
    :param x_nums: 棋盘格横向角点个数
    :param y_nums: 棋盘格纵向角点个数
    :param mtx:    内参矩阵
    :param dist:   畸变参数
    :return:  R_matrix:旋转矩阵, tvec：偏移量,标定成功标志：sucess
    '''

    # 设置(生成)标定图在世界坐标中的坐标
    # 生成x_nums*y_nums个坐标，每个坐标包含x,y,z三个元素，z=0
    world_point = np.zeros((x_nums * y_nums, 3), np.float32)
    # mgrid[]生成包含两个二维矩阵的矩阵，每个矩阵都有x_nums列,y_nums行,设置世界坐标的坐标
    world_point[:, :2] = size_per_grid*np.mgrid[:x_nums, :y_nums].T.reshape(-1, 2)

    # 设置角点查找限制
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image = cv2.imread(photo_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 查找角点
    ok, corners = cv2.findChessboardCorners(gray, (x_nums, y_nums),None)

    if ok:
        # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
        cv2.drawChessboardCorners(image, (x_nums,y_nums), corners, ok)
    # print(ok)
    if ok:
        # 获取更精确的角点位置
        exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 获取外参
        _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)

        #根据旋转矢量rvec和平移向量tvec获得旋转矩阵R和偏移T
        #R_matrix = np.zeros((3,3),np.float32)
        R_matrix , _ = cv2.Rodrigues(rvec)

    return R_matrix,tvec,ok


#测距函数，室外统一为棋盘格平行于地面
def pixel2world(R_matrix,T_vec,inter_mtx,dist,pavement_H,u,v):
    '''
    :param R_matrix: 外参旋转矩阵
    :param T_vec:    外参偏移矩阵
    :param inter_mtx: 相机内参矩阵
    :param dist:     相机畸变系数
    :param pavement_H:  棋盘格距离地面的高度
    :param u:   像素横向坐标,左上角为原点
    :param v:   像素纵向坐标
    :return:    Xw,Yw为像素对应地面的坐标,
    :return:    error为分辨率误差
    '''

    camera_vec = -1 * np.dot(R_matrix.T, T_vec)  #相机在世界坐标系的坐标
    K = np.linalg.inv(inter_mtx)  # 内参矩阵的逆
    T = np.dot(R_matrix.T, K)  # 旋转矩阵的逆左乘内参矩阵的逆

    Zc = 0.1*pavement_H
    pixes = np.array([u, v, 1])
    Zc_point = np.dot(T, pixes) * Zc + camera_vec.T
    Zc_point = Zc_point.T  # 转置一下,方便索引

    # 根据透视原理得到像素对应地面坐标,三点共线建立方程，地面为Zw = pavement_H
    #平行的情况
    k = (camera_vec[2] - Zc_point[2]) / (Zc_point[2] - pavement_H)
    Xw = Zc_point[0] - (camera_vec[0] - Zc_point[0]) / k
    Yw = Zc_point[1] - (camera_vec[1] - Zc_point[1]) / k

    #垂直的情况
    # k  = (camera_vec[1] - Zc_point[1]) / (Zc_point[1] - pavement_H)
    # Xw = Zc_point[0] - (camera_vec[0] - Zc_point[0]) / k
    # Yw = Zc_point[2] - (camera_vec[2] - Zc_point[2]) / k

    #计算分辨率带来的误差
    error = 0

    return np.array([Xw,Yw,error],dtype=object)






