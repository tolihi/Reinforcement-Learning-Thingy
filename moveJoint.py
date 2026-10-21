import mujoco
import mujoco.viewer
import time
import numpy as np
import ikpy.chain #作逆运动学求解
import transform3d as tf
from scipy.spatial.transform import Rotation as R, Slerp

"""
def car():
    model = mujoco.MjModel.from_xml_path('D:\桌面\mujoco-3.11.0-windows-x86_64\model\cards\cards.xml')
    data = mujoco.MjData(model)

    #data.ctrl[0] = 0.0
    #data.ctrl[1] = 0.001
    with mujoco.viewer.launch_passive(model, data) as viewer:
        #仿真循环
        while viewer.is_running():
            mujoco.mj_step(model, data) #update
            viewer.sync() #sync
"""
def ur5e():
    model = mujoco.MjModel.from_xml_path("D:\CS\code\.venv\\training\mujoco_menagerie-main\\universal_robots_ur5e\scene.xml")
    data = mujoco.MjData(model)

    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
    data.qpos[:6] = start_joints
    data.ctrl[:6] = [-1.57, -1.34, 2.65, -1.3, 1.55, 0]

    #xml文件中，motor驱动器到达数值是直接到达，general驱动器则为缓慢到达这个值
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)

def JointSpaceTrajectory():
def JointSpaceControl():

    model = mujoco.MjModel.from_xml_path("D:\\CS\\code\\.venv\\training\\mujoco_menagerie-main\\universal_robots_ur5e\\scene.xml")
    data = mujoco.MjData(model)

    mychain = ikpy.chain.Chain.from_mjcf_file("D:\\CS\\code\\.venv\\training\\mujoco_menagerie-main\\universal_robots_ur5e\\ur5e.xml",
                                              active_links_mask=[False, False] + [True] * 6)

    start_joint = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
    data.qpos[:6] = start_joint

    # 目标点情况
    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)

    # 根据目标位姿求解关节角
    joint_angle = mychain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position = ref_pos)
    end_joint = joint_angle[2:]

def arm():
    model = mujoco.MjModel.from_xml_path()

if __name__ == "__main__":
    ur5e()