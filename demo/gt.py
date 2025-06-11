import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show3Dpose(vals, ax):
    # 视角
    ax.view_init(elev=15., azim=70)

    # 关节连线，Human3.6M常见骨架连接（可根据你的数据微调）
    I = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 8, 12, 13, 8, 10]
    J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    # 颜色分左右和中心，示意用
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]  # 红 蓝 绿
    LR = [1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 3, 1, 1, 2, 2, 3, 3]  # 随便分左右中心

    for i in range(len(I)):
        x = [vals[I[i], 0], vals[J[i], 0]]
        y = [vals[I[i], 1], vals[J[i], 1]]
        z = [vals[I[i], 2], vals[J[i], 2]]
        ax.plot(x, y, z, lw=2, color=colors[LR[i] - 1])

    # 设置坐标轴范围和比例
    RADIUS = 1
    root = vals[0]
    ax.set_xlim3d([-RADIUS + root[0], RADIUS + root[0]])
    ax.set_ylim3d([-RADIUS + root[1], RADIUS + root[1]])
    ax.set_zlim3d([-RADIUS + root[2], RADIUS + root[2]])

    ax.set_box_aspect([1,1,1])
    ax.axis('off')


def visualize_h36m_3d(npz_path, frame_idx=0):
    data = np.load(npz_path, allow_pickle=True)

    print("All keys in npz:", data.files)
    for k in data.files:
        print(f"Key: {k}, shape: {data[k].shape}")
    # 读取3D关键点，key根据文件内容不同可能不同，常见'reconstruction'或'positions_3d'等
    if 'reconstruction' in data:
        keypoints3d = data['reconstruction']
    elif 'positions_3d' in data:
        keypoints3d = data['positions_3d']
    else:
        raise KeyError("No known 3D keypoint array found in npz file.")

    # keypoints3d形状假设是 (num_frames, num_joints, 3)
    print(f"Loaded 3D keypoints with shape: {keypoints3d.shape}")

    # 选一个帧数可视化
    if frame_idx >= keypoints3d.shape[0]:
        print("Frame index out of range, using 0.")
        frame_idx = 0

    pose = keypoints3d[frame_idx]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    show3Dpose(pose, ax)
    plt.title(f'H36M 3D Pose Frame {frame_idx}')
    plt.show()


if __name__ == "__main__":
    npz_path = 'D:\Code\HoT\dataset\h36m\data_3d_h36m.npz'  # 改成你的文件路径
    visualize_h36m_3d(npz_path, frame_idx=0)
