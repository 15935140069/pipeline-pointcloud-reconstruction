"""Interactive 3D clipping boxes with one-step undo."""

import numpy as np
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper


def inside_boxes(xyz, boxes):
    """Points inside every committed oriented box (plane normals point outward)."""
    if not boxes:
        raise ValueError("没有已确认的裁剪区域")
    inside = np.ones(len(xyz), dtype=bool)
    for planes in boxes:
        for nx, ny, nz, offset in planes:
            inside &= xyz @ np.array([nx, ny, nz]) + offset <= 1e-7
    return inside


def align_box_to_world(widget):
    """让黄色包围盒重新恢复标准正交状态"""
    from vtkmodules.vtkCommonDataModel import vtkPolyData,vtkPlanes
    polydata = vtkPolyData()
    widget.GetPolyData(polydata)

    center=np.array(polydata.GetPoint(14))
    corner = np.array(polydata.GetPoint(0))
    sizes = [
        np.linalg.norm(np.array(polydata.GetPoint(i))-corner)
        for i in (1, 3, 4)
    ]
    bounds=[
        value
        for c, size in zip(corner, sizes)
        for value in (c-size/2,c+size/2)
    ]
    widget.SetPlaceFactor(1.0)
    widget.PlaceWidget(
        float(center[0]-sizes[0]/2),
        float(center[0]+sizes[0]/2),
        float(center[1] - sizes[1] / 2),
        float(center[1] + sizes[1] / 2),
        float(center[2] - sizes[2] / 2),
        float(center[2] + sizes[2] / 2),
    )
    planes=vtkPlanes()
    widget.GetPlanes(planes)
    return planes

class ClipHistory:
    def __init__(self):
        self.boxes = []

    def commit(self, planes):
        self.boxes.append(np.asarray(planes, dtype=float).copy())

    def undo(self):
        if self.boxes:
            self.boxes.pop()


def select_boxes(xyz, rgb):
    import pyvista as pv

    history = ClipHistory()
    current_box = None
    saved = False
    plotter = pv.Plotter(window_size=(1400, 850), title="E57 三维点云裁剪")

    def message(text):
        plotter.add_text(
            f"Drag yellow box | C: crop | Ctrl+Z: undo | S: export\n{text}",
            name="instructions", position="upper_left", font_size=11,
        )

    def refresh():
        mask = inside_boxes(xyz, history.boxes) if history.boxes else np.ones(len(xyz), bool)
        plotter.add_points(xyz[mask], scalars=rgb[mask], rgb=True,
                           name="cloud", point_size=2, render_points_as_spheres=False,
                           reset_camera=False)
        message(f"Crops: {len(history.boxes)} | Preview points: {mask.sum():,}")

    def on_box_change(vtk_planes):
        nonlocal current_box
        current_box = np.array([
            [*vtk_planes.GetPlane(i).GetNormal(),
             -np.dot(vtk_planes.GetPlane(i).GetNormal(), vtk_planes.GetPlane(i).GetOrigin())]
            for i in range(vtk_planes.GetNumberOfPlanes())
        ])

    def apply_crop():
        if current_box is None:
            message("Adjust the box first")
            return
        candidate = history.boxes + [current_box]
        if not inside_boxes(xyz, candidate).any():
            message("No preview points inside this box")
            return
        history.commit(current_box)
        refresh()

    def undo():
        if plotter.iren.interactor.GetControlKey():
            history.undo()
            refresh()

    def reset_rotation():
        on_box_change(align_box_to_world(box_widget))
        plotter.render()
        message("Reset rotation")

    def finish():
        nonlocal saved
        if not history.boxes:
            message("Press C to commit a crop first")
            return
        saved = True
        plotter.iren.terminate_app()

    plotter.add_points(xyz, scalars=rgb, rgb=True, name="cloud", point_size=2)
    box_widget = plotter.add_box_widget(on_box_change, bounds=plotter.bounds, factor=0.5,
                           use_planes=True, rotation_enabled=True, color="yellow")
    plotter.add_key_event("c", apply_crop)
    plotter.add_key_event("z", undo)
    plotter.add_key_event("s", finish)
    # 按下g可以让包围盒重新恢复标准正交的状态
    plotter.add_key_event("g", reset_rotation)
    message("Adjust the box, then press C")
    plotter.show()
    plotter.close()
    return history.boxes if saved else None
