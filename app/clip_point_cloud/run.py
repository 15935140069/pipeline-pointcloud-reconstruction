"""Run from the PyCharm project root: python -m app.clip_point_cloud.run"""

from pathlib import Path

from crop_viewer import select_boxes
from e57_reader import build_preview
from e57_writer import export_crop

from time import perf_counter


# 只需修改下面两个路径
SOURCE_E57 = Path(r"C:\3. 智能制造\深水-深技服-扫描\e57_point_cloud\BZ251A5mm.e57")
OUTPUT_E57 = Path(r"C:\3. 智能制造\深水-深技服-扫描\e57_point_cloud\BZ251A5mm_clip_result.e57")


def main():
    print("正在读取全部扫描站并生成降采样预览")
    started = perf_counter()
    xyz, rgb = build_preview(SOURCE_E57, max_points=1_000_000)
    print(f"预览点数：{len(xyz):,},"
          f"准备耗时：{perf_counter() - started:.1f}秒")

    boxes = select_boxes(xyz, rgb)
    if boxes is None:
        print("取消裁剪，未保存文件")
        return

    print("正在从原始 E57 裁剪全部扫描站……")
    count = export_crop(SOURCE_E57, OUTPUT_E57, boxes)
    print(f"完成！输出 {count:,} 个原始点：{OUTPUT_E57}")


if __name__ == "__main__":
    main()
