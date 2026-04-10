import argparse
import json
import os
import os.path as osp
import pickle
import textwrap

import cv2
import numpy as np
from pyquaternion import Quaternion


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize GT/pred trajectories on a camera image without the official nuScenes SDK."
    )
    parser.add_argument("--instruction-json", required=True, help="Path to one instruction_jsons/*.json file.")
    parser.add_argument("--record-index", type=int, default=0, help="Record index inside the instruction json.")
    parser.add_argument("--sample-token", default=None, help="Select one record by sample token instead of index.")
    parser.add_argument("--window-id", type=int, default=None, help="Optional filter with sample token.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Optional filter with sample token.")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera channel, default CAM_FRONT.")
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="If set, render a 2x3 panel with all six surround-view cameras.",
    )
    parser.add_argument(
        "--visualization-only",
        action="store_true",
        help="Render a front-view illustrative trajectory instead of strict geometric projection.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=800,
        help="Tile width used for six-camera mosaic rendering.",
    )
    parser.add_argument(
        "--infos-pkl",
        default="data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl",
        help="Path to infos pkl. Relative paths are resolved from repo root.",
    )
    parser.add_argument(
        "--z-value",
        type=float,
        default=0.0,
        help="Height used when lifting xy trajectory points to xyz in ego frame.",
    )
    parser.add_argument(
        "--output",
        default="visualization/output/front_camera_example.jpg",
        help="Output image path.",
    )
    return parser.parse_args()


def resolve_repo_path(path):
    if osp.isabs(path):
        return path
    return osp.join(REPO_ROOT, path)


def load_instruction_record(path, record_index=0, sample_token=None, window_id=None, frame_idx=None):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    if sample_token is None:
        if not records:
            raise ValueError(f"No records found in {path}")
        return payload, records[record_index]

    for record in records:
        if record.get("sample_token") != sample_token:
            continue
        if window_id is not None and int(record.get("window_id", -1)) != int(window_id):
            continue
        if frame_idx is not None and int(record.get("frame_idx", -1)) != int(frame_idx):
            continue
        return payload, record
    raise ValueError("No matching record found in instruction json.")


def load_info_index(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    infos = data.get("infos", [])
    return {info["token"]: info for info in infos}


def ego_points_to_cam_pixels(cam_info, points_xyz_ego):
    """
    points_xyz_ego: [N, 3], points in ego frame of this sample/camera timestamp
    return: projected pixels and a valid mask over original points
    """
    pts = np.asarray(points_xyz_ego, dtype=float).copy()

    translation = np.asarray(cam_info["sensor2ego_translation"], dtype=float)
    rotation = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
    intrinsic = np.asarray(cam_info["cam_intrinsic"], dtype=float)

    # ego -> camera
    pts = pts - translation[None, :]
    pts = (rotation.T @ pts.T).T

    valid = pts[:, 2] > 1e-3
    uv = np.zeros((len(pts), 2), dtype=float)
    if valid.any():
        proj = (intrinsic @ pts[valid].T).T
        uv[valid] = proj[:, :2] / proj[:, 2:3]
    return uv, valid


def trajectory_to_visual_pixels(
    img_shape,
    points_xy,
    anchor=None,
    meters_per_pixel_y=0.08,
    meters_per_pixel_x=0.12,
):
    h, w = img_shape[:2]
    if anchor is None:
        anchor = (int(w * 0.5), int(h * 0.9))
    anchor = np.asarray(anchor, dtype=np.float32)

    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool), anchor

    uv = np.zeros((len(points_xy), 2), dtype=np.float32)
    # x forward -> image y up, y left -> image x left
    uv[:, 0] = anchor[0] - points_xy[:, 1] / meters_per_pixel_x
    uv[:, 1] = anchor[1] - points_xy[:, 0] / meters_per_pixel_y

    valid = np.ones((len(points_xy),), dtype=bool)
    return uv, valid, anchor


def prepend_origin(points_xy, z_value):
    points_xy = np.asarray(points_xy, dtype=float)
    origin_xy = np.array([[0.0, 0.0]], dtype=float)
    if points_xy.size == 0:
        xy = origin_xy
    else:
        xy = np.concatenate([origin_xy, points_xy], axis=0)
    xyz = np.concatenate([xy, np.full((len(xy), 1), z_value)], axis=1)
    return xyz


def prepend_origin_xy(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    origin_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    if points_xy.size == 0:
        return origin_xy
    return np.concatenate([origin_xy, points_xy], axis=0)


def draw_polyline(img, uv, valid, color, label, thickness=3):
    uv = np.asarray(uv, dtype=np.int32)
    valid = np.asarray(valid, dtype=bool)
    h, w = img.shape[:2]

    if len(uv) != len(valid):
        raise ValueError("uv and valid must have the same length")

    keep = []
    for p, ok in zip(uv, valid):
        if not ok:
            continue
        if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h:
            continue
        keep.append(p)

    if len(keep) >= 2:
        keep_arr = np.asarray(keep, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [keep_arr], False, color, thickness, lineType=cv2.LINE_AA)

    for i, p in enumerate(keep):
        r = 6 if i == 0 else 4
        cv2.circle(img, tuple(int(x) for x in p), r, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            "t0" if i == 0 else str(i),
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if keep:
        anchor = keep[0]
        cv2.putText(
            img,
            label,
            (int(anchor[0]) + 8, int(anchor[1]) + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_visual_polyline(img, points_xy, color, label, meters_per_pixel_y=0.08, meters_per_pixel_x=0.12):
    points_xy = prepend_origin_xy(points_xy)
    uv, valid, anchor = trajectory_to_visual_pixels(
        img.shape,
        points_xy,
        meters_per_pixel_y=meters_per_pixel_y,
        meters_per_pixel_x=meters_per_pixel_x,
    )
    uv = np.asarray(uv, dtype=np.int32)

    keep = []
    h, w = img.shape[:2]
    for p, ok in zip(uv, valid):
        if not ok:
            continue
        if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h:
            continue
        keep.append(p)

    if len(keep) >= 2:
        cv2.line(img, tuple(keep[0]), tuple(keep[1]), (255, 255, 255), 2, cv2.LINE_AA)
        keep_arr = np.asarray(keep[1:], dtype=np.int32).reshape(-1, 1, 2)
        if len(keep_arr) >= 2:
            cv2.polylines(img, [keep_arr], False, color, 3, lineType=cv2.LINE_AA)

    cv2.circle(img, tuple(anchor.astype(int)), 7, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, tuple(anchor.astype(int)), 4, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    cv2.putText(
        img,
        "ego",
        (int(anchor[0]) + 8, int(anchor[1]) - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for i, p in enumerate(keep):
        if i == 0:
            continue
        r = 5 if i == 1 else 4
        cv2.circle(img, tuple(int(x) for x in p), r, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            str(i),
            (int(p[0]) + 6, int(p[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if len(keep) >= 2:
        anchor_p = keep[1]
        cv2.putText(
            img,
            label,
            (int(anchor_p[0]) + 8, int(anchor_p[1]) + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA,
        )


def add_footer_panel(image, record, instruction):
    panel_h = 190
    canvas = np.full((image.shape[0] + panel_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[: image.shape[0]] = image
    cv2.rectangle(canvas, (0, image.shape[0]), (canvas.shape[1], canvas.shape[0]), (245, 245, 245), -1)

    meta_lines = [
        f"scene={record.get('scene_name')}  frame={record.get('frame_idx')}  token={record.get('sample_token')}",
        f"instruction_id={record.get('instruction_id')}  window_id={record.get('window_id')}  error={record.get('prediction_error')}",
    ]
    y = image.shape[0] + 30
    for line in meta_lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2, cv2.LINE_AA)
        y += 32

    cv2.putText(canvas, "Instruction:", (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (30, 30, 30), 2, cv2.LINE_AA)
    y += 34
    for line in textwrap.wrap(instruction or "", width=95)[:4]:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (40, 40, 40), 2, cv2.LINE_AA)
        y += 28
    return canvas


def render_single_camera(info, record, cam_name="CAM_FRONT", z_value=0.0, show_legend=True, visualization_only=False):
    cam_info = info["cams"][cam_name]
    img_path = resolve_repo_path(cam_info["data_path"])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    pred_xy = np.asarray(record.get("pred_traj_xy", []), dtype=float)
    gt_xy = np.asarray(record.get("gt_traj_xy", []), dtype=float)

    if visualization_only:
        if len(gt_xy) > 0:
            draw_visual_polyline(img, gt_xy, (0, 255, 0), "GT")
        if len(pred_xy) > 0:
            draw_visual_polyline(img, pred_xy, (0, 0, 255), "Pred")
    else:
        if len(gt_xy) > 0:
            gt_xyz = prepend_origin(gt_xy, z_value)
            gt_uv, gt_valid = ego_points_to_cam_pixels(cam_info, gt_xyz)
            draw_polyline(img, gt_uv, gt_valid, (0, 255, 0), "GT", thickness=3)

        if len(pred_xy) > 0:
            pred_xyz = prepend_origin(pred_xy, z_value)
            pred_uv, pred_valid = ego_points_to_cam_pixels(cam_info, pred_xyz)
            draw_polyline(img, pred_uv, pred_valid, (0, 0, 255), "Pred", thickness=3)

    cv2.putText(img, cam_name, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (20, 20, 20), 2, cv2.LINE_AA)
    if show_legend:
        cv2.putText(
            img,
            (
                "GT=green  Pred=red  visualization only"
                if visualization_only else f"GT=green  Pred=red  z={z_value:.2f}"
            ),
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return img


def make_camera_mosaic(images_by_name, tile_width=800):
    layout = [
        ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
        ["CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"],
    ]
    rows = []
    tile_h = None
    for row_names in layout:
        row_tiles = []
        for name in row_names:
            img = images_by_name[name]
            scale = tile_width / float(img.shape[1])
            resized = cv2.resize(img, (tile_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
            if tile_h is None:
                tile_h = resized.shape[0]
            if resized.shape[0] != tile_h:
                resized = cv2.resize(resized, (tile_width, tile_h), interpolation=cv2.INTER_LINEAR)
            row_tiles.append(resized)
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def visualize_ego_traj_on_camera(
    info,
    record,
    instruction,
    out_path,
    cam_name="CAM_FRONT",
    z_value=0.0,
    visualization_only=False,
):
    img = render_single_camera(
        info,
        record,
        cam_name=cam_name,
        z_value=z_value,
        show_legend=True,
        visualization_only=visualization_only,
    )

    canvas = add_footer_panel(img, record, instruction)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"saved to {out_path}")


def visualize_ego_traj_on_all_cameras(
    info,
    record,
    instruction,
    out_path,
    z_value=0.0,
    tile_width=800,
    visualization_only=False,
):
    camera_names = [
        "CAM_FRONT_LEFT",
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
    ]
    images_by_name = {
        name: render_single_camera(
            info,
            record,
            cam_name=name,
            z_value=z_value,
            show_legend=False,
            visualization_only=visualization_only,
        )
        for name in camera_names
    }
    mosaic = make_camera_mosaic(images_by_name, tile_width=tile_width)
    cv2.putText(
        mosaic,
        (
            "All Cameras  GT=green  Pred=red  visualization only"
            if visualization_only else f"All Cameras  GT=green  Pred=red  z={z_value:.2f}"
        ),
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    canvas = add_footer_panel(mosaic, record, instruction)
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"saved to {out_path}")


def main():
    args = parse_args()
    payload, record = load_instruction_record(
        args.instruction_json,
        record_index=args.record_index,
        sample_token=args.sample_token,
        window_id=args.window_id,
        frame_idx=args.frame_idx,
    )
    instruction = record.get("instruction", payload.get("instruction", ""))

    info_index = load_info_index(resolve_repo_path(args.infos_pkl))
    sample_token = record["sample_token"]
    if sample_token not in info_index:
        raise KeyError(f"sample token {sample_token} not found in infos pkl")

    if args.all_cameras:
        visualize_ego_traj_on_all_cameras(
            info=info_index[sample_token],
            record=record,
            instruction=instruction,
            out_path=resolve_repo_path(args.output),
            z_value=args.z_value,
            tile_width=args.tile_width,
            visualization_only=args.visualization_only,
        )
    else:
        visualize_ego_traj_on_camera(
            info=info_index[sample_token],
            record=record,
            instruction=instruction,
            out_path=resolve_repo_path(args.output),
            cam_name=args.camera,
            z_value=args.z_value,
            visualization_only=args.visualization_only,
        )


if __name__ == "__main__":
    main()
