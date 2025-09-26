"""Context construction and main processing loop."""
from __future__ import annotations

import argparse
import threading
import time
from typing import Callable, Optional

from runtime import Profiler
from runtime import lazy_modules as lazy

from .context import Caches, RunContext
from .frame_ops import (
    apply_cutout_if_requested,
    apply_matte_and_grade,
    compute_bbox_canvas,
    compute_center_and_transform,
    log_fps,
    apply_output_flips_if_needed,
    prepare_coordinates_and_hands,
    rotate_if_needed,
    run_hands_every,
    run_pose_every,
    run_rvm_on_roi_or_full,
    show_preview,
)
from .pose_async import PoseWorker
from .setup import (
    build_models,
    build_sender_or_exit,
    init_ndi_or_exit,
    open_capture,
    prepare_grading,
    prepare_preview,
    probe_source_size_or_exit,
    warmup_cuda_kernels,
)


ProgressCallback = Callable[[str], None]


def _notify_progress(progress: Optional[ProgressCallback], message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        pass


def build_context(
    args: argparse.Namespace,
    stop_event: Optional[threading.Event] = None,
    progress: Optional[ProgressCallback] = None,
) -> RunContext:
    width, height = args.w, args.h
    kp_thr = float(lazy.np.clip(args.kp_thr, 0.0, 1.0))
    use_nearest = args.nearest_only or lazy.ONLY_NEAREST

    _notify_progress(progress, "Initializing NDI runtime…")
    init_ndi_or_exit()
    _notify_progress(progress, "Opening capture source…")
    cap = open_capture(args, width, height)
    cam_w, cam_h = probe_source_size_or_exit(cap)

    out_w, out_h = (cam_w, cam_h) if args.ndi_follow_camera else (width, height)
    center = lazy.CenterTracker(out_w, out_h, args.center_smooth, args.center_deadzone)

    _notify_progress(progress, "Loading neural models (pose/hands/matting)…")
    device, pose, rvm, hands = build_models(args)
    if getattr(device, "type", None) == "cuda":
        _notify_progress(progress, "Warming up CUDA kernels…")
    warmup_cuda_kernels(args, device, pose, rvm, cam_w, cam_h)
    _notify_progress(progress, "Starting pose worker…")
    pose_worker = PoseWorker(pose, scale=float(args.pose_scale))
    pose_worker.start()
    _notify_progress(progress, "Creating NDI sender…")
    sender = build_sender_or_exit(args, out_w, out_h)
    _notify_progress(progress, "Preparing preview window…")
    prepare_preview(args, out_w, out_h)

    _notify_progress(progress, "Starting camera capture…")
    cam_mgr = lazy.CameraCapture(args, cap)
    cam_mgr.start()
    _notify_progress(progress, "Preparing UDP output and grading…")
    udp = lazy.UDPPoseSender(lazy.UDP_IP, lazy.UDP_PORT)
    stats, lut = prepare_grading(args, out_w, out_h)

    _notify_progress(progress, "Initialization complete")
    return RunContext(
        args=args,
        OUT_W=out_w,
        OUT_H=out_h,
        KP_THR=kp_thr,
        use_nearest=use_nearest,
        cap=cap,
        cam_mgr=cam_mgr,
        center=center,
        device=device,
        pose=pose,
        rvm=rvm,
        hands=hands,
        sender=sender,
        udp=udp,
        pose_worker=pose_worker,
        stats=stats,
        lut_gamma_dark=lut,
        stop_event=stop_event or threading.Event(),
    )


def run_loop(ctx: RunContext) -> None:
    caches = Caches(last_log=time.time())
    prof = Profiler(enabled=True)

    cv2 = lazy.cv2
    np = lazy.np

    try:
        while not ctx.stop_event.is_set():
            prof.tick()
            frame_read = ctx.cam_mgr.read()
            if isinstance(frame_read, tuple):
                ok, frame_bgr = frame_read
            else:
                ok, frame_bgr = frame_read is not None, frame_read
            if not ok or frame_bgr is None:
                time.sleep(0.005)
                continue
            prof.mark("capture")

            frame_bgr = rotate_if_needed(frame_bgr, ctx.args.rotate)
            prof.mark("rotate")

            caches.frames += 1

            xy_use_src, conf_use, bbox_src_debug = run_pose_every(ctx, caches, frame_bgr)
            prof.mark("pose")

            hands_src = run_hands_every(ctx, caches, frame_bgr, bbox_src_debug)
            prof.mark("hands")

            fh, fw = frame_bgr.shape[:2]
            alpha_src = run_rvm_on_roi_or_full(ctx, caches, frame_bgr, bbox_src_debug, fw, fh)
            prof.mark("rvm")

            M_base, M, _, _ = compute_center_and_transform(ctx, xy_use_src, conf_use, fw, fh)
            prof.mark("center")

            bbox_canvas = None
            rect_canvas_for_cut = None
            if bbox_src_debug is not None:
                bbox_canvas, rect_canvas_for_cut = compute_bbox_canvas(ctx, M, bbox_src_debug, fw, fh)

            frame_rgba = apply_matte_and_grade(ctx, frame_bgr, M, bbox_canvas, alpha_src)
            prof.mark("matte")

            apply_cutout_if_requested(ctx, frame_rgba, rect_canvas_for_cut)
            prof.mark("cutout")

            xy_send, conf_send, hands_send = prepare_coordinates_and_hands(ctx, M, xy_use_src, conf_use, hands_src, bbox_canvas)
            prof.mark("prep")

            frame_rgba, xy_send, hands_send, bbox_canvas = apply_output_flips_if_needed(
                ctx, frame_rgba, xy_send, hands_send, bbox_canvas
            )
            prof.mark("flip")

            frame_to_send = frame_rgba if frame_rgba.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame_rgba)
            ctx.udp.send(xy_send, conf_send, hands_send)
            ctx.sender.send_rgba(frame_to_send)
            prof.mark("io")

            if ctx.stop_event.is_set():
                break
            if show_preview(ctx, frame_rgba, xy_send, conf_send, bbox_canvas):
                break
            prof.mark("preview")

            log_fps(caches)
            prof.end_frame(caches.frames)

    except KeyboardInterrupt:
        pass


def cleanup(ctx: RunContext) -> None:
    ctx.stop_event.set()
    ctx.cam_mgr.stop()
    ctx.cap.release()
    if not ctx.args.no_preview:
        lazy.cv2.destroyAllWindows()
    ctx.sender.close()
    lazy.ndi.destroy()
    ctx.hands.close()
    if getattr(ctx, "pose_worker", None) is not None:
        ctx.pose_worker.stop()
    print("[INFO] Finished.")


def run_pipeline(args: argparse.Namespace) -> None:
    stop_event = threading.Event()
    ctx = build_context(args, stop_event=stop_event, progress=lambda msg: print(f"[INFO] {msg}"))
    try:
        run_loop(ctx)
    finally:
        cleanup(ctx)
