"""Read large E57 scans in chunks and build a small registered preview."""

import numpy as np
from pathlib import Path
from time import perf_counter


def world_xyz(chunk, header):
    local = np.column_stack([chunk[k] for k in ("cartesianX", "cartesianY", "cartesianZ")])
    return local @ np.asarray(header.rotation_matrix).T + np.asarray(header.translation)


def sampled_indices(start, count, step):
    """Chunk-relative indices on a regular scan-wide sampling grid."""
    return np.flatnonzero((start + np.arange(count)) % step == 0)


def iter_chunks(e57, scan_index, fields=None, chunk_size=100_000):
    from pye57 import libe57

    fields = list(fields or e57.get_header(scan_index).point_fields)
    image = e57.image_file
    points = image.root()["data3D"][scan_index]["points"]
    arrays = {name: np.empty(chunk_size, dtype=np.float64) for name in fields}
    buffers = libe57.VectorSourceDestBuffer()

    for name, array in arrays.items():
        buffers.append(libe57.SourceDestBuffer(image, name, array, chunk_size, True, True))

    reader = points.reader(buffers)
    try:
        while count := reader.read():
            yield {name: array[:count] for name, array in arrays.items()}
    finally:
        reader.close()


def build_preview(path, max_points=1_000_000):
    import pye57

    # 1. 检查预览缓存
    source = Path(path)
    stat = source.stat()
    signature = np.array(
        [stat.st_size, stat.st_mtime_ns], dtype=np.int64
    )

    cache = (
        source.parent
        / ".clip_cache"
        / f"{source.stem}_{max_points}.npz"
    )

    if cache.exists():
        with np.load(cache) as saved:
            if np.array_equal(saved["signature"], signature):
                print(f"使用预览缓存：{cache}")
                return saved["xyz"], saved["rgb"]

    # 2. 首次读取 E57
    xyz_parts, rgb_parts = [], []

    with pye57.E57(str(path)) as e57:
        per_scan = max(1, max_points // e57.scan_count)

        for scan_index in range(e57.scan_count):
            started = perf_counter()
            header = e57.get_header(scan_index)

            print(
                f"预览扫描站 {scan_index + 1}/{e57.scan_count}: "
                f"{header.point_count:,} 点"
            )

            # 只读取显示必需的字段
            needed = (
                "cartesianX", "cartesianY", "cartesianZ",
                "cartesianInvalidState",
                "colorRed", "colorGreen", "colorBlue"
            )

            fields = [
                name for name in needed
                if name in header.point_fields
            ]

            step = max(
                1,
                (header.point_count + per_scan - 1) // per_scan
            )

            start = 0

            # 3. 分块读取，先抽样，再坐标变换
            for chunk in iter_chunks(
                e57, scan_index, fields, chunk_size=500_000
            ):
                count = len(chunk["cartesianX"])
                idx = sampled_indices(start, count, step)
                start += count

                if not len(idx):
                    continue

                sampled = {
                    name: chunk[name][idx]
                    for name in (
                        "cartesianX", "cartesianY", "cartesianZ"
                    )
                }

                xyz = world_xyz(sampled, header)

                valid = np.isfinite(xyz).all(axis=1)

                if "cartesianInvalidState" in chunk:
                    valid &= (
                        chunk["cartesianInvalidState"][idx] == 0
                    )

                xyz_parts.append(xyz[valid])

                if {
                    "colorRed", "colorGreen", "colorBlue"
                } <= chunk.keys():
                    rgb_parts.append(np.column_stack([
                        chunk[k][idx][valid]
                        for k in (
                            "colorRed", "colorGreen", "colorBlue"
                        )
                    ]))
                else:
                    rgb_parts.append(
                        np.full((valid.sum(), 3), np.nan)
                    )

            print(
                f"扫描站 {scan_index + 1} 预览耗时："
                f"{perf_counter() - started:.1f} 秒"
            )

    # 4. 合并预览点云
    xyz = np.concatenate(xyz_parts)
    raw_rgb = np.concatenate(rgb_parts)

    colored = np.isfinite(raw_rgb).all(axis=1)

    scale = (
        np.max(raw_rgb[colored])
        if np.any(colored)
        else 255.0
    )

    scale = (
        65535.0 if scale > 255
        else (255.0 if scale > 1 else 1.0)
    )

    rgb = np.clip(
        np.nan_to_num(raw_rgb, nan=0) / scale * 255,
        0, 255
    ).astype(np.uint8)

    rgb[~colored] = [180, 180, 180]

    # 5. 保存缓存，供下次打开使用
    cache.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        cache,
        signature=signature,
        xyz=xyz,
        rgb=rgb
    )

    print(f"已保存预览缓存：{cache}")

    return xyz, rgb
