import mujoco
import mujoco.viewer
import time
import numpy as np
import ikpy.chain #作逆运动学求解
from scipy.spatial.transform import Rotation as R


r"""
def car():
    model = mujoco.MjModel.from_xml_path(r'D:\桌面\mujoco-3.11.0-windows-x86_64\model\cards\cards.xml')
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
    model = mujoco.MjModel.from_xml_path(r"D:\CS\code\.venv\training\mujoco_menagerie-main\universal_robots_ur5e\scene.xml")
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

class JointSpaceTrajectory:
    def __init__(self, start_joints, end_joints, steps, duration=4.0):
        self.start_joints = np.array(start_joints, dtype=float)
        self.end_joints = np.array(end_joints, dtype=float)
        self.steps = steps
        self.duration = duration
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.i = 0

    def get_next_waypoint(self, sim_time):
        target_i = int(min(sim_time / self.duration, 1.0) * self.steps)
        if target_i > self.i:      # 只增不减，避免时间回退时抖动
            self.i = target_i
        return self.start_joints + self.step * self.i


def JointSpaceControl():

    model = mujoco.MjModel.from_xml_path(r"D:\CS\code\.venv\training\mujoco_menagerie-main\universal_robots_ur5e\scene.xml")
    data = mujoco.MjData(model)

    mychain = ikpy.chain.Chain.from_mjcf_file(r"D:\CS\code\.venv\training\mujoco_menagerie-main\universal_robots_ur5e\ur5e.xml",
                                              active_links_mask=[False, False] + [True] * 6)

    start_joint = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])
    data.qpos[:6] = start_joint

    # 目标点情况
    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0]
    ee_orientation = R.from_euler('xyz', ee_euler).as_matrix()
    print("ee_orientation =\n", np.round(ee_orientation, 4))

    # 根据目标位姿求解关节角
    joint_angle = mychain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position = ref_pos)
    end_joint = joint_angle[2:]
    print("ik joints =", np.round(end_joint, 4))
    traj = JointSpaceTrajectory(start_joint, end_joint, steps=200, duration=4.0)
    data.ctrl[:6] = start_joint
    mujoco.mj_forward(model, data)   # 让首帧姿态就是 start_joint

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.ctrl[:6] = traj.get_next_waypoint(data.time)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.002)


if __name__ == "__main__":
    JointSpaceControl()