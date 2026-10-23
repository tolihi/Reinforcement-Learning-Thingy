# Shadow Hand MuJoCo PPO 入门示例

这个例子使用 Menagerie 的 `shadow_hand/scene_right.xml`，让右手通过 PPO 学习一个简化的“抓住物体并抬起”任务。原模型的 20 个位置执行器被组合成 5 个可学习的手指协同控制量：拇指、食指、中指、无名指、小指。手腕保持在中立位置，便于先理解连续动作 PPO。

## 环境

需要 Python 3.10+，以及 `mujoco`、`numpy`、`torch`。在你准备用来运行的 Python 环境里安装：

```powershell
python -m pip install -r requirements.txt
```

也可以直接用已有的 Python 3.12 环境运行。如果使用 `D:\CS\code\.venv\Scripts\python.exe`，该环境已经有 PyTorch 和 NumPy，但需要先安装 MuJoCo：

```powershell
D:\CS\code\.venv\Scripts\python.exe -m pip install mujoco
```

脚本默认加载：

```text
D:\CS\code\.venv\training\mujoco_menagerie-main\shadow_hand\scene_right.xml
```

可通过 `--model` 指向另一份场景 XML。场景引用 `right_hand.xml` 和 `assets`，因此不要单独移动 Menagerie 模型文件。

## 训练

```powershell
cd D:\CS\code\.venv\training\shadow_hand_ppo
python train_shadow_hand_ppo.py train --steps 100000
```

刚开始学习时可以用较短的运行检查日志和保存流程：

```powershell
python train_shadow_hand_ppo.py train --steps 4096 --log-every 1024
```

默认每隔 10,000 个环境步保存检查点，训练结束再保存一次。产物放在 `runs` 目录：

- `shadow_hand_ppo.pt`：Actor-Critic 网络和训练步数。
- `training_log.csv`：每回合回报、长度、是否抬起物体和物体高度。

## 查看策略

训练有检查点后运行：

```powershell
python train_shadow_hand_ppo.py play
```

MuJoCo Viewer 会打开并实时显示确定性策略。若使用其他模型路径或检查点：

```powershell
python train_shadow_hand_ppo.py play --model "D:\path\scene_right.xml" --checkpoint "D:\path\shadow_hand_ppo.pt"
```

## PPO 与环境对应关系

- **观测（60 维）**：手部 24 个关节角度和速度、物体相对出生点的位置和速度、五个指尖与物体的距离、手物接触数。
- **动作（5 维）**：每个手指从张开到预设弯曲目标的连续比例。策略输出经 `tanh` 限幅到 `[-1, 1]`，环境再映射到张合比例 `[0, 1]`。
- **奖励**：指尖靠近物体、产生接触、抬高物体会得到奖励；物体掉到地面以下会受罚；连续抬高一段时间视为成功。
- **PPO 更新**：Actor-Critic、广义优势估计（GAE）、概率比裁剪、随机小批量更新。

关键参数可以从命令行调整，例如 `--rollout-steps`、`--epochs`、`--lr`、`--horizon`。主要实现都在 `train_shadow_hand_ppo.py`，适合从 `ShadowHandGraspEnv.step()` 和 `PPO.update()` 开始阅读。

## 说明

这是用于学习和实验的基础示例，不保证短时间内就学会稳定抓取；可以先观察 `training_log.csv`，再增加训练步数或调整奖励权重。它不会修改 Menagerie 的 XML 文件。
