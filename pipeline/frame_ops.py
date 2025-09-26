"""Per-frame processing helpers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from runtime import lazy_modules as lazy

from .context import RunContext, Caches

RVM_IN_H, RVM_IN_W = 384, 256


def rotate_if_needed(frame_bgr, rot: int):
    cv2 = lazy.cv2
    rot = (rot or 0) % 360
    if rot == 90:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame_bgr, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


def run_pose_every(ctx: RunContext, caches: Caches, frame_bgr):
    np = lazy.np
    pose_worker = getattr(ctx, "pose_worker", None)

    if pose_worker is not None:
        if (caches.frames % max(1, ctx.args.pose_every)) == 0:
            pose_worker.submit(frame_bgr)

        latest = pose_worker.latest()
        if latest is not None:
            pose_id, xy_all_src, conf_all = latest
            if pose_id != caches.last_pose_result_id:
                caches.last_pose_result_id = pose_id
                caches.last_xy_all_src = xy_all_src
                caches.last_conf_all = conf_all

        xy_all_src = caches.last_xy_all_src
        conf_all = caches.last_conf_all
    else:
        xy_all_src = conf_all = None
        if (caches.frames % max(1, ctx.args.pose_every)) == 0:
            xy_all_src, conf_all = ctx.pose.predict_downscaled(frame_bgr, scale=float(ctx.args.pose_scale))
            if xy_all_src is not None:
                caches.last_xy_all_src = xy_all_src
                caches.last_conf_all = conf_all
        else:
            xy_all_src = caches.last_xy_all_src
            conf_all = caches.last_conf_all

    bbox_src_debug = None
    if xy_all_src is not None:
        if ctx.use_nearest and xy_all_src.shape[0] > 1:
            bboxes = [
                lazy.kps_to_bbox(xy_all_src[i], conf_all[i] if conf_all is not None else None, thr=ctx.KP_THR)
                for i in range(xy_all_src.shape[0])
            ]
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
            idx = int(np.argmax(areas))
            xy_use_src = xy_all_src[idx : idx + 1]
            conf_use = conf_all[idx : idx + 1] if conf_all is not None else None
            bbox_src_debug = bboxes[idx].astype(np.float32)
        else:
            xy_use_src = xy_all_src
            conf_use = conf_all
            if xy_all_src.shape[0] == 1:
                bbox_src_debug = lazy.kps_to_bbox(
                    xy_all_src[0], conf_all[0] if conf_all is not None else None, thr=ctx.KP_THR
                ).astype(np.float32)
    else:
        xy_use_src = None
        conf_use = None

    if bbox_src_debug is not None:
        caches.last_bbox_src_debug = bbox_src_debug
    else:
        bbox_src_debug = caches.last_bbox_src_debug

    return xy_use_src, conf_use, bbox_src_debug


def run_hands_every(ctx: RunContext, caches: Caches, frame_bgr, bbox_src_debug: Optional[Tuple[int, int, int, int]]):
    if not ctx.args.hands:
        return []
    if ((caches.frames % max(1, ctx.args.hands_every)) == 0) or (caches.last_hands_src is None):
        roi_xyxy = None
        if ctx.args.hands_roi and (bbox_src_debug is not None):
            x1, y1, x2, y2 = bbox_src_debug.tolist()
            pad_x = (x2 - x1) * float(ctx.args.roi_pad_x)
            pad_y = (y2 - y1) * float(ctx.args.roi_pad_y)
            roi_xyxy = (int(x1 - pad_x), int(y1 - pad_y), int(x2 + pad_x), int(y2 + pad_y))
        hands_src = ctx.hands.detect(frame_bgr, roi_xyxy=roi_xyxy)
        caches.last_hands_src = hands_src
        return hands_src
    return caches.last_hands_src


def run_matting_every(ctx: RunContext, caches: Caches, frame_bgr):
    if not (ctx.args.matte and (ctx.rvm is not None)):
        return None
    if ((caches.frames % max(1, ctx.args.rvm_every)) == 0) or (caches.last_alpha_src is None):
        frame_h, frame_w = frame_bgr.shape[:2]
        if (caches.last_rvm_mode != "full") or (caches.last_rvm_input_hw != (frame_h, frame_w)):
            ctx.rvm.reset_states()
        alpha_src = ctx.rvm.alpha(frame_bgr, downsample=float(ctx.args.rvm_down))
        caches.last_alpha_src = alpha_src
        caches.last_rvm_mode = "full"
        caches.last_rvm_input_hw = (frame_h, frame_w)
        return alpha_src
    return caches.last_alpha_src


def snap_rect_to_grid(rect, grid, fw, fh):
    x1, y1, x2, y2 = rect
    x1 = max(0, (x1 // grid) * grid)
    y1 = max(0, (y1 // grid) * grid)
    x2 = min(fw, ((x2 + grid - 1) // grid) * grid)
    y2 = min(fh, ((y2 + grid - 1) // grid) * grid)
    if x2 <= x1:
        x2 = min(fw, x1 + grid)
    if y2 <= y1:
        y2 = min(fh, y1 + grid)
    return (x1, y1, x2, y2)


def run_rvm_on_roi_or_full(ctx: RunContext, caches: Caches, frame_bgr, bbox_src, fw, fh):
    np = lazy.np
    cv2 = lazy.cv2
    use_cutout = bool(getattr(ctx.args, "bbox_cutout", False))
    matte_on = bool(getattr(ctx.args, "matte", False))
    if not matte_on:
        return None
    if bbox_src is None or not use_cutout:
        return run_matting_every(ctx, caches, frame_bgr)

    rect = lazy.padded_bbox_src(
        bbox_src,
        fw,
        fh,
        pad_x=getattr(ctx.args, "bbox_pad_x", 0.15),
        pad_y=getattr(ctx.args, "bbox_pad_y", 0.20),
    )
    if rect is None:
        return run_matting_every(ctx, caches, frame_bgr)

    last_rect = getattr(caches, "last_roi_rect", None)
    rect = lazy.ema_rect(last_rect, rect, alpha=0.7)
    rect = snap_rect_to_grid(rect, grid=8, fw=fw, fh=fh)
    caches.last_roi_rect = rect

    x1, y1, x2, y2 = rect
    roi_bgr = frame_bgr[y1:y2, x1:x2]
    roi_h, roi_w = roi_bgr.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return np.zeros((fh, fw), np.uint8)

    roi_bgr_in = cv2.resize(roi_bgr, (RVM_IN_W, RVM_IN_H), interpolation=cv2.INTER_LINEAR)
    roi_input_hw = (RVM_IN_H, RVM_IN_W)
    if (caches.last_rvm_mode != "roi") or (caches.last_rvm_input_hw != roi_input_hw):
        ctx.rvm.reset_states()
    roi_alpha_in = ctx.rvm.alpha(
        roi_bgr_in,
        downsample=float(ctx.args.rvm_down),
        enforce_stride=False,
        reset_on_resize=False,
    )

    roi_alpha = cv2.resize(roi_alpha_in, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
    alpha_src = np.zeros((fh, fw), dtype=np.uint8)
    alpha_src[y1:y2, x1:x2] = roi_alpha
    caches.last_rvm_mode = "roi"
    caches.last_rvm_input_hw = roi_input_hw
    return alpha_src


def compute_center_and_transform(ctx: RunContext, xy_use_src, conf_use, fw: int, fh: int):
    np = lazy.np
    M_base = lazy.make_letterbox_affine(fw, fh, ctx.OUT_W, ctx.OUT_H)
    if ctx.args.center and (xy_use_src is not None) and (xy_use_src.shape[0] >= 1):
        xy0_canvas = lazy.apply_affine_xy(xy_use_src[0], M_base)
        conf0 = conf_use[0] if (conf_use is not None) else None
        cx, cy = lazy.person_center(xy0_canvas, conf0, kp_thr=ctx.KP_THR, method=ctx.args.center_method)
        ctx.center.update(cx, cy)
    ndx, ndy = ctx.center.offset() if ctx.args.center else (0.0, 0.0)
    M = M_base.copy()
    M[0, 2] += ndx
    M[1, 2] += ndy
    return M_base, M, ndx, ndy


def compute_bbox_canvas(ctx: RunContext, M, bbox_src_debug, fw: int, fh: int):
    use_cutout = bool(getattr(ctx.args, "bbox_cutout", False))
    rect_canvas_for_cut = None
    bbox_canvas = None
    if bbox_src_debug is None:
        return None, None

    if use_cutout:
        rect_src_for_cut = lazy.padded_bbox_src(
            bbox_src_debug,
            fw,
            fh,
            pad_x=getattr(ctx.args, "bbox_pad_x", 0.10),
            pad_y=getattr(ctx.args, "bbox_pad_y", 0.10),
        )
        if rect_src_for_cut is not None:
            rect_canvas_for_cut = lazy.bbox_apply_affine(rect_src_for_cut, M)
            bbox_canvas = rect_canvas_for_cut

    if bbox_canvas is None:
        bbox_canvas = lazy.bbox_apply_affine(bbox_src_debug, M)

    return bbox_canvas, rect_canvas_for_cut


def apply_matte_and_grade(ctx: RunContext, frame_bgr, M, bbox_canvas, alpha_src):
    cv2 = lazy.cv2
    np = lazy.np
    frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
    frame_rgba = cv2.warpAffine(
        frame_rgba,
        M,
        (ctx.OUT_W, ctx.OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0 if ctx.args.matte else 255),
    )

    if ctx.args.matte and alpha_src is not None:
        alpha_canvas = cv2.warpAffine(
            alpha_src,
            M,
            (ctx.OUT_W, ctx.OUT_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if ctx.args.bbox_clip and (bbox_canvas is not None):
            x1, y1, x2, y2 = bbox_canvas.astype(int)
            mx = int(0.10 * (x2 - x1 + 1))
            my = int(0.15 * (y2 - y1 + 1))
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(ctx.OUT_W - 1, x2 + mx)
            y2 = min(ctx.OUT_H - 1, y2 + my)
            clip = np.zeros((ctx.OUT_H, ctx.OUT_W), dtype=np.uint8)
            clip[y1 : y2 + 1, x1 : x2 + 1] = 255
            alpha_canvas = cv2.min(alpha_canvas, clip)
        kernel = np.ones((3, 3), dtype=np.uint8)
        alpha_canvas = cv2.morphologyEx(alpha_canvas, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha_canvas = cv2.morphologyEx(alpha_canvas, cv2.MORPH_DILATE, kernel, iterations=1)
        frame_rgba[..., 3] = alpha_canvas
    else:
        frame_rgba[..., 3] = 255

    if ctx.args.grade and (ctx.stats is not None) and (ctx.lut_gamma_dark is not None):
        alpha = frame_rgba[..., 3]
        if lazy.np.any(alpha > 8):
            rgb = frame_rgba[..., :3]
            graded = cv2.LUT(rgb, ctx.lut_gamma_dark)
            contrast = float(ctx.stats["contrast"])
            if abs(contrast - 1.0) > 1e-3:
                graded = cv2.addWeighted(
                    graded,
                    contrast,
                    lazy.np.full_like(graded, 128),
                    1.0 - contrast,
                    128 * (1.0 - contrast),
                )
            tb, tg, tr = [int(x) for x in ctx.stats["tint_bgr"]]
            tint_strength = float(ctx.stats["tint_strength"])
            if tint_strength > 1e-4:
                tinted = cv2.add(graded, lazy.np.array([tb, tg, tr], dtype=lazy.np.uint8))
                graded = cv2.addWeighted(graded, 1.0 - tint_strength, tinted, tint_strength, 0.0)
            mask = alpha.astype(lazy.np.uint8)
            fg = cv2.bitwise_and(graded, graded, mask=mask)
            inv = cv2.bitwise_not(mask)
            bg = cv2.bitwise_and(rgb, rgb, mask=inv)
            rgb[:] = cv2.add(bg, fg)
    return frame_rgba


def apply_cutout_if_requested(ctx: RunContext, frame_rgba, rect_canvas_for_cut) -> None:
    if bool(getattr(ctx.args, "bbox_cutout", False)) and rect_canvas_for_cut is not None:
        lazy.cutout_alpha_inplace(frame_rgba[..., 3], rect_canvas_for_cut)


def prepare_coordinates_and_hands(ctx: RunContext, M, xy_use_src, conf_use, hands_src, bbox_canvas):
    xy_send = None
    conf_send = None
    if xy_use_src is not None:
        xy_send = lazy.apply_affine_xy(xy_use_src, M)
        conf_send = conf_use

    hands_send: List[Dict] = []
    if hands_src:
        def inside_expanded_bbox_canvas(centroid_canvas: Iterable[float], bbox, mx_ratio=0.10, my_ratio=0.15) -> bool:
            if bbox is None:
                return True
            x1, y1, x2, y2 = bbox
            mx = int(mx_ratio * (x2 - x1 + 1))
            my = int(my_ratio * (y2 - y1 + 1))
            x1 = max(0, int(x1) - mx)
            y1 = max(0, int(y1) - my)
            x2 = min(ctx.OUT_W - 1, int(x2) + mx)
            y2 = min(ctx.OUT_H - 1, int(y2) + my)
            cx, cy = centroid_canvas
            return (x1 <= cx <= x2) and (y1 <= cy <= y2)

        for item in hands_src:
            xy_src = item["xy_src"]
            centroid_src = lazy.np.array(item["centroid_src"], lazy.np.float32)
            xy_canvas = lazy.apply_affine_xy(xy_src, M)
            centroid_canvas = lazy.apply_affine_xy(centroid_src, M)
            if ctx.use_nearest and (bbox_canvas is not None) and not inside_expanded_bbox_canvas(centroid_canvas, bbox_canvas):
                continue
            hands_send.append({"xy": xy_canvas, "score": float(item["score"]), "handed": int(item["handed"])})
    return xy_send, conf_send, hands_send


def _get_flip_flags(ctx: RunContext):
    flip_h = bool(getattr(ctx.args, "horizontal_flip", False))
    flip_v = bool(getattr(ctx.args, "flip_vertical", False))
    return flip_h, flip_v


def _get_flip_code(flip_h: bool, flip_v: bool) -> int:
    if flip_h and flip_v:
        return -1
    if flip_h:
        return 1
    return 0


def apply_input_flips_if_needed(ctx: RunContext, frame_bgr):
    cv2 = lazy.cv2
    flip_h, flip_v = _get_flip_flags(ctx)
    if not (flip_h or flip_v):
        return frame_bgr, False
    flip_code = _get_flip_code(flip_h, flip_v)
    frame_bgr = cv2.flip(frame_bgr, flip_code)
    return frame_bgr, True


def apply_output_flips_if_needed(ctx: RunContext, frame_rgba, xy_send, hands_send, bbox_canvas, *, skip: bool = False):
    if skip:
        return frame_rgba, xy_send, hands_send, bbox_canvas
    cv2 = lazy.cv2
    flip_h, flip_v = _get_flip_flags(ctx)
    if not (flip_h or flip_v):
        return frame_rgba, xy_send, hands_send, bbox_canvas
    flip_code = _get_flip_code(flip_h, flip_v)
    frame_rgba = cv2.flip(frame_rgba, flip_code)
    if xy_send is not None:
        if flip_h:
            xy_send[..., 0] = (ctx.OUT_W - 1) - xy_send[..., 0]
        if flip_v:
            xy_send[..., 1] = (ctx.OUT_H - 1) - xy_send[..., 1]
    for hand in hands_send:
        if flip_h:
            hand["xy"][..., 0] = (ctx.OUT_W - 1) - hand["xy"][..., 0]
            if hand.get("handed") in (0, 1):
                hand["handed"] = 1 - int(hand["handed"])
        if flip_v:
            hand["xy"][..., 1] = (ctx.OUT_H - 1) - hand["xy"][..., 1]
    if bbox_canvas is not None:
        x1, y1, x2, y2 = bbox_canvas
        if flip_h:
            x1, x2 = (ctx.OUT_W - 1) - x2, (ctx.OUT_W - 1) - x1
        if flip_v:
            y1, y2 = (ctx.OUT_H - 1) - y2, (ctx.OUT_H - 1) - y1
        bbox_canvas = lazy.np.array([x1, y1, x2, y2], lazy.np.float32)
    return frame_rgba, xy_send, hands_send, bbox_canvas


def show_preview(ctx: RunContext, frame_rgba, xy_send, conf_send, bbox_canvas) -> bool:
    if ctx.args.no_preview:
        return False
    cv2 = lazy.cv2
    preview_bgra = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGRA)
    alpha_ch = preview_bgra[..., 3]
    fg_bgr = preview_bgra[..., 0:3]
    bg = lazy.np.full((ctx.OUT_H, ctx.OUT_W, 3), (0, 255, 0), dtype=lazy.np.uint8)
    composite = bg.copy()
    cv2.copyTo(fg_bgr, alpha_ch, composite)

    if ctx.args.show_skel and xy_send is not None:
        lazy.draw_pose(composite, xy_send, conf_send, ctx.KP_THR, draw_names=False)
        if bbox_canvas is not None:
            x1, y1, x2, y2 = bbox_canvas.astype(int)
            x1 = lazy.np.clip(x1, 0, ctx.OUT_W - 1)
            x2 = lazy.np.clip(x2, 0, ctx.OUT_W - 1)
            y1 = lazy.np.clip(y1, 0, ctx.OUT_H - 1)
            y2 = lazy.np.clip(y2, 0, ctx.OUT_H - 1)
            cv2.rectangle(composite, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(composite, "NEAREST", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 2)

    cv2.imshow("Preview", composite)
    return (cv2.waitKey(1) & 0xFF) == 27


def log_fps(caches: Caches) -> None:
    import time

    now = time.time()
    if caches.last_log == 0.0:
        caches.last_log = now
        caches.frames = 0
        return
    if now - caches.last_log >= 1.0:
        print(f"[FPS] {caches.frames} fps")
        caches.frames = 0
        caches.last_log = now
