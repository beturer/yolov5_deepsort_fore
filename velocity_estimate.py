import numpy as np
import numpy.fft as nf  # 用于滤波时的卷积运算
# from guage.get_camera_parameters import location  # 改进的测距方法
import math

'''原本的测距方法'''
def location(height, width, X, Y):
    '''
    height :image's height
    width :image's width
    X,Y: image's centerpoint
    '''
    # 相机参数俯仰角α(度)、垂直视场角θ(度)、水平视场角β(度)、摄像头高度H(m)、摄像画面底边到摄像头的距离D_min(m)、顶边到摄像头的距离D_max(m)。
    Alpha = 90.0
    Theta = 55.3
    Beta = 87.0
    H = 7.2
    D_min = H * math.tan((Alpha - Theta / 2) / 180 * math.pi)
    W = 2 * D_min * math.tan((Beta / 2) / 180 * math.pi)
    L1 = W * (width + 1 - (float(X) - 1 / 2)) / width  # 中间变量
    L2 = H * (float(height + 1 - Y) - 1 / 2) / height  # 中间变量
    Xm = (L1 - W / 2) * L2 / (H - L2) + L1  # 世界坐标
    Ym = D_min * L2 / (H - L2)  # 世界坐标
    loc = np.array([Xm, Ym])
    return loc


'''
velocity_estimate函数输入
id代表被检测到的车辆的id，frame_idx代表视频在第几帧,(x,y)是车辆像素坐标,(height,width)是图像的高和宽;
np_store是所存储的数据,sheet用于导出速度序列到excel表格.
'''


def velocity_estimate(id, frame_idx, x, y, height, width, np_store, sheet):
    how_many_frame_will_detect = 45
    video_frame_rate = 60  # 视频帧数
    point = 7   # 存储point个时间点的数据
    distance_point = np.zeros(point - 1)
    velocity_point = np.zeros(point - 1)
    loc = np.zeros((2, point))
    hn = np.zeros(17)  # 用于滤波
    # for i in range(17):  # 设置滤波器响应的系数
    #     if i == 8:
    #         hn[i] = 9 / 16
    #     else:
    #         hn[i] = math.sin(math.pi * 9 / 16 * (i - 8)) / (math.pi * (i - 8)) * (
    #                     0.54 - 0.46 * (math.cos(math.pi * i / 8)))
    # 如果用于存储数据的变量原本大小不够,扩充变量的大小(开始)
    if id > np_store.shape[0]:
        np_store = np.vstack((np_store, np.zeros((500, 2 * point + 2))))
    if id + 1 > sheet.shape[1]:
        sheet = np.hstack((sheet, np.zeros((sheet.shape[0], 10))))
    if int(frame_idx / how_many_frame_will_detect) > sheet.shape[0]:
        sheet = np.vstack((sheet, np.zeros((10, sheet.shape[1]))))
    # 如果用于存储数据的变量原本大小不够,扩充变量的大小(结束)
    if frame_idx % how_many_frame_will_detect == 2:
        for j in range(2 * point - 2):  # point为7的时候,循环12次,把后面store的X、Y坐标逐个赋给前面的store
            np_store[id][j + 2] = np_store[id][j + 4]  # 最前面两列数据不做改变，最后面两列数据在循环外改变
        np_store[id][2 * point] = x  # point为7的时候,该项为np_store[i][14]
        np_store[id][2 * point + 1] = y  # # point为7的时候,该项为np_store[i][15]
        print('fi', frame_idx)

    # 视频帧数小于特定值，所存储的时间段不到point-1，求平均需要除于当前存储段数
    if (frame_idx >= how_many_frame_will_detect + 2) and (frame_idx < (point - 1) * how_many_frame_will_detect + 2):
        sheet[int(frame_idx / how_many_frame_will_detect) - 1][0] = \
            frame_idx * (how_many_frame_will_detect / video_frame_rate) / how_many_frame_will_detect
        for j in range(int(frame_idx / how_many_frame_will_detect) + 1):
            loc[:, int(point - (j + 1))] = \
                location(height, width, np_store[id][int(2 * point) - 2 * j], np_store[id][int(2 * point) + 1 - 2 * j])
        for k in range(int(frame_idx / how_many_frame_will_detect)):
            distance_point[k] = math.sqrt(sum(np.square(loc[:, point - (k + 1)] - loc[:, point - (k + 2)])))
            velocity_point[k] = distance_point[k] / (how_many_frame_will_detect / video_frame_rate)
        xn = velocity_point[0:int(frame_idx / how_many_frame_will_detect)]
        # N1 = len(xn)  #卷积运算进行滤波(开始)
        # N2 = len(hn)
        # N = N1 + N2
        # Yk = np.zeros(N)
        # Yk = nf.fft(Yk)
        # xnn = np.hstack((xn, np.zeros(N - N1)))
        # hnn = np.hstack((hn, np.zeros(N - N2)))
        # Xk = nf.fft(xnn)
        # Hk = nf.fft(hnn)
        # for j in range(N):
        #     Yk[j] = Xk[j] * Hk[j]
        # yn = nf.ifft(Yk)
        # yn = np.abs(yn)  #卷积运算进行滤波(结束)
        # yn = yn.tolist()
        # yn0 = yn[int(N2 / 2):int(N2 / 2) + N1]
        # yn_mean = np.mean(yn0)
        yn_mean = np.mean(xn)
        velocity = yn_mean / (how_many_frame_will_detect / video_frame_rate)
        sheet[int(frame_idx / how_many_frame_will_detect) - 1][id + 1] = velocity * 3.6
        if (velocity > 40) or (velocity < 0.2):  # 判断速度是否可信
            try:
                velocity = np_store[id][1]
            except:
                velocity = 0
        else:
            np_store[id][1] = velocity  # 存储可信速度
    elif frame_idx >= (point - 1) * how_many_frame_will_detect + 2:  # 视频帧数大于指定值除于point-1来求平均
        sheet[int(frame_idx / how_many_frame_will_detect) - 1][0] = frame_idx / how_many_frame_will_detect
        for j in range(0, 2 * point, 2):  # 以2为步长,point为7的时候,loc有7个元素
            loc[:, int(j / 2)] = location(height, width, np_store[id][j + 2], np_store[id][j + 3])
        for k in range(point - 1):  # 当point为7,distance_point有6个元素
            distance_point[k] = math.sqrt(sum(np.square(loc[:, k + 1] - loc[:, k])))
            velocity_point[k] = distance_point[k] / (how_many_frame_will_detect / video_frame_rate)
        xn = velocity_point
        # N1 = len(xn)
        # N2 = len(hn)
        # N = N1 + N2
        # Yk = np.zeros(N)
        # Yk = nf.fft(Yk)
        # xnn = np.hstack((xn, np.zeros(N - N1)))
        # hnn = np.hstack((hn, np.zeros(N - N2)))
        # Xk = nf.fft(xnn)
        # Hk = nf.fft(hnn)
        # for j in range(N):
        #     Yk[j] = Xk[j] * Hk[j]
        # yn = nf.ifft(Yk)
        # yn = np.abs(yn)
        # yn = yn.tolist()
        # yn0 = yn[int(N2/2):int(N2/2)+N1]
        # yn0 = sorted(yn0)
        # yn0.remove(yn0[0])
        # yn0.remove(yn0[-1])
        xn = xn.tolist()
        xn = sorted(xn)
        xn.remove(xn[0])
        xn.remove(xn[-1])
        yn_mean = np.mean(xn)
        velocity = yn_mean / (how_many_frame_will_detect / video_frame_rate)
        sheet[int(frame_idx / how_many_frame_will_detect) - 1][id + 1] = velocity * 3.6
        if (velocity > 40) or (velocity < 0.2):  # 判断速度是否可信
            try:
                velocity = np_store[id][1]
            except:
                velocity = 0
        else:
            np_store[id][1] = velocity
    else:
        velocity = 0

    return velocity, np_store, sheet
