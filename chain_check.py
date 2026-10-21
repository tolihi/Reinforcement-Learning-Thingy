import ikpy.chain

# 第一步：加载时不指定 mask，让 ikpy 使用默认行为
chain = ikpy.chain.Chain.from_mjcf_file(
    "D:\\CS\\code\\.venv\\training\\mujoco_menagerie-main\\universal_robots_ur5e\\ur5e.xml"
)

# 第二步：打印链条结构，观察每个 link 的名称和类型
print(f"总链接数: {len(chain.links)}")
for i, link in enumerate(chain.links):
    print(f"[{i}] name={link.name!r:30s} type={link.joint_type!r:12s} "
          f"current_mask={chain.active_links_mask[i]}")