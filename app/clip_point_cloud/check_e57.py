from pathlib import Path
import pye57

source = Path(r"C:\3. 智能制造\深水-深技服-扫描\已配准.e57")
target = Path(r"C:\3. 智能制造\深水-深技服-扫描\clip_result.e57")

def check(path):
    with pye57.E57(str(path)) as e57:
        counts = [
            e57.get_header(i).point_count
            for i in range(e57.scan_count)
        ]
    total = sum(counts)
    size = path.stat().st_size
    print(f"\n文件：{path.name}")
    print(f"扫描站数量：{len(counts)}")

    for i, count in enumerate(counts):
        print(f"扫描站 {i + 1}：{count:,} 点")

    print(f"总点数：{total:,}")
    print(f"文件大小：{size / 1024 ** 3:.2f} GB")
    print(f"平均每点：{size / total:.2f} 字节")

    return total


original = check(source)
cropped = check(target)

print(f"\n裁剪后保留比例：{cropped / original * 100:.2f}%")