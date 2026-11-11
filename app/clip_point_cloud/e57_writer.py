"""Export all registered scans within the chosen 3D crop as one E57."""

from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crop_viewer import inside_boxes
from e57_reader import iter_chunks, world_xyz


def select_chunk(chunk, header, boxes):
    xyz = world_xyz(chunk, header)
    mask = np.isfinite(xyz).all(axis=1) & inside_boxes(xyz, boxes)
    if "cartesianInvalidState" in chunk:
        mask &= chunk["cartesianInvalidState"] == 0
    return {name: values[mask] for name, values in chunk.items()}


def export_crop(source, target, boxes):
    """Keep original scan-local coordinates and poses; use disk staging to cap RAM."""
    import pye57

    source, target = Path(source), Path(target)
    if not boxes:
        raise ValueError("未确认裁剪区域")
    if target.exists() or source.resolve() == target.resolve():
        raise FileExistsError(f"输出路径不可覆盖: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with TemporaryDirectory(dir=target.parent) as temporary:
        with pye57.E57(str(source)) as original, pye57.E57(str(target), mode="w") as output:
            for scan_index in range(original.scan_count):
                header = original.get_header(scan_index)

                fields = [
                    name for name in header.point_fields
                    if not name.startswith("nor:")
                ]

                paths = {name: Path(temporary) / f"{scan_index}_{name}.bin" for name in fields}
                kept = 0

                with ExitStack() as stack:
                    files = {name: stack.enter_context(open(path, "wb"))
                             for name, path in paths.items()}
                    for chunk in iter_chunks(original, scan_index, fields):
                        selected = select_chunk(chunk, header, boxes)
                        count = len(selected["cartesianX"])
                        if count:
                            for name, values in selected.items():
                                np.asarray(values, dtype=np.float64).tofile(files[name])
                            kept += count

                if kept:
                    data = {name: np.memmap(path, dtype=np.float64, mode="r", shape=(kept,))
                            for name, path in paths.items()}
                    try:
                        output.write_scan_raw(
                            data,
                            rotation=header.rotation,
                            translation=header.translation
                        )
                    finally:
                        for array in data.values():
                            array._mmap.close()
                        del data
                total += kept
                print(f"扫描站 {scan_index + 1}/{original.scan_count}：保留 {kept:,} 点")

    if not total:
        raise ValueError("裁剪区域内没有原始点")
    with pye57.E57(str(target)) as check:
        written = sum(check.get_header(i).point_count for i in range(check.scan_count))
        if written != total:
            raise RuntimeError(f"E57 验证失败：预计 {total:,} 点，实际 {written:,} 点")
    return total
