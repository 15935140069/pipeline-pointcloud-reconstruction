
from pathlib import Path
import numpy as np
import pye57
from pye57 import libe57
import open3d as o3d

file_path = Path("C:/3. 智能制造/深水-深技服-扫描/已配准.e57")

with pye57.E57(str(file_path)) as e57:

    print("扫描站数量:", e57.scan_count)

    # 读取第一个扫描站的信息
    header = e57.get_header(0)

    print("该站总点数:", header.point_count)
    print("字段列表:", header.point_fields)

    print("旋转矩阵:")
    print(header.rotation_matrix)

    print("平移向量:")
    print(header.translation)

    # 检查坐标及颜色字段
    required = {
        "cartesianX", "cartesianY", "cartesianZ",
        "colorRed", "colorGreen", "colorBlue"
    }

    if not required.issubset(set(header.point_fields)):
        raise ValueError("该扫描站缺少XYZ或RGB字段")

    # 最多读取10万个点
    capacity = min(10000000, header.point_count)

    if capacity == 0:
        raise ValueError("该扫描站没有点记录")

    # 创建坐标缓冲区
    x = np.empty(capacity, dtype=np.float64)
    y = np.empty(capacity, dtype=np.float64)
    z = np.empty(capacity, dtype=np.float64)

    # 创建颜色缓冲区
    r = np.empty(capacity, dtype=np.float64)
    g = np.empty(capacity, dtype=np.float64)
    b = np.empty(capacity, dtype=np.float64)

    imf = e57.image_file
    root = imf.root()

    points = root["data3D"][0]["points"]

    # 创建E57读取缓冲区
    buffers = libe57.VectorSourceDestBuffer()

    for name, array in [
        ("cartesianX", x),
        ("cartesianY", y),
        ("cartesianZ", z),
        ("colorRed", r),
        ("colorGreen", g),
        ("colorBlue", b)
    ]:
        buffers.append(
            libe57.SourceDestBuffer(
                imf, name, array,
                capacity, True, True
            )
        )

    # 读取点云
    reader = points.reader(buffers)

    try:
        count = reader.read()
    finally:
        reader.close()

    if count == 0:
        raise ValueError("未读取到点")

    print("\n实际读取点数:", count)

    # 拼接XYZ坐标
    xyz_local = np.column_stack([
        x[:count],
        y[:count],
        z[:count]
    ])

    # 拼接RGB颜色
    rgb = np.column_stack([
        r[:count],
        g[:count],
        b[:count]
    ])

    print("\n原始局部坐标形状:", xyz_local.shape)
    print("前5个点:")
    print(xyz_local[:5])

    # 转换到统一坐标系
    R = np.asarray(header.rotation_matrix)
    t = np.asarray(header.translation)

    xyz = xyz_local @ R.T + t

    # XYZ和RGB同时过滤
    mask = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(rgb).all(axis=1)
    )

    xyz = xyz[mask]
    rgb = rgb[mask]

    if len(xyz) == 0:
        raise ValueError("过滤后没有有效点")

    print("\n有效点数:", len(xyz))
    print("坐标最小值:", xyz.min(axis=0))
    print("坐标最大值:", xyz.max(axis=0))

    print("\n原始RGB最小值:", rgb.min(axis=0))
    print("原始RGB最大值:", rgb.max(axis=0))

    # 将RGB转换到Open3D要求的[0,1]范围
    # 这里暂按常见的8位或16位颜色处理
    scale = 255.0 if rgb.max() <= 255 else 65535.0

    rgb = np.clip(rgb / scale, 0, 1)

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    print("\nOpen3D对象:")
    print(pcd)

    print("是否包含颜色:", pcd.has_colors())

    # # 显示点云
    # o3d.visualization.draw_geometries(
    #     [pcd],
    #     window_name="E57 RGB Point Cloud Preview",
    #     width=1280,
    #     height=800
    # )

    print("\n=========点云裁剪=============")
    print("鼠标左键: 旋转视角; \n鼠标滚轮: 缩放点云; \n按住K框选; 按住C裁剪点云; "
          "\n按住F返回自由观察")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="E57点云裁剪窗口",
        width=1280,
        height=800
    )
    vis.add_geometry(pcd)
    vis.run()
    cropped_pcd = vis.get_cropped_geometry()
    vis.destroy_window()
    if cropped_pcd is None or len(cropped_pcd) == 0:
        print("点云为空")
    else:
        print("裁剪完成")
        output_dir=Path(__file__).resolve().parent.parent / "outputs"
        output_dir.mkdir(parents=True,exist_ok=True)
        output_path =output_dir / 'roi_preview.ply'
        success = o3d.io.write_point_cloud(str(output_path),cropped_pcd)

