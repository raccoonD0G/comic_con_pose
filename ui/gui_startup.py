# ui/gui_startup.py
from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Any

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None
    ttk = None


@dataclass
class FieldSpec:
    key: str
    label: str
    type: str  # 'str'|'int'|'float'|'bool'|'choice'|'file'
    default: object
    choices: List[object] = field(default_factory=list)
    help: str = ""


# 각 탭 상단에 간단한 설명을 보여줍니다.
TAB_DESCRIPTIONS: Dict[str, str] = {
    "Source": "입력 소스(웹캠/파일)와 캡처 백엔드/프레임레이트 같은 캡처 관련 옵션입니다.",
    "Output": "NDI 출력 크기/프레임레이트, 미리보기, 디버그용 박스 클리핑 등을 설정합니다.",
    "Pose / Center": "사람 포즈 탐지(신뢰도/이미지 크기)와 캔버스 중심 추적 옵션을 조절합니다.",
    "Hands / Pace": "MediaPipe Hands 사용 여부/세부옵션과 웹캠 프레임 페이싱(드랍/슬립)을 설정합니다.",
    "Matte / Grade": "배경 분리(RVM)와 전경 그레이딩(오토/매뉴얼 컬러 보정)을 켭니다.",
    "Performance": "성능 관련 주기/스케일/FP16 사용 여부를 설정합니다.",
}

# 기능 토글에 대한 더 자세한 문구(토글 바로 아래 줄에 표시)
FEATURE_HELP: Dict[str, str] = {
    "center": "인물의 어깨/엉덩이/코 등을 기준으로 캔버스 중앙을 자동 보정합니다.",
    "hands": "MediaPipe Hands로 손 21개 랜드마크를 추적합니다. ROI를 켜면 인물 박스 주변만 검사하여 속도가 좋아집니다.",
    "matte": "Robust Video Matting으로 배경 알파 마스크를 생성합니다. 다운샘플 비율을 높일수록 빠르지만 경계 품질이 낮아질 수 있습니다.",
    "grade": "전경(사람)만 대상으로 감마/밝기/대비/틴트를 적용합니다. Auto는 배경 참조 이미지로 적정값을 추정합니다.",
    "ndi_follow_camera": "입력 카메라(혹은 소스)의 원본 해상도를 그대로 NDI로 보냅니다. 켜면 너비/높이 설정은 무시됩니다.",
}

# 스키마
SCHEMA: Dict[str, List[FieldSpec]] = {
    "Source": [
        FieldSpec("src", "Video path (empty=webcam)", "file", "", help="동영상 파일을 선택하거나 비워두면 웹캠을 사용합니다."),
        FieldSpec("cam", "Camera index", "int", 1),
        FieldSpec("backend", "OpenCV backend", "choice", "ds", choices=["ds", "ms", "auto"]),
        FieldSpec("cap_fps", "Capture FPS (0=auto)", "float", 0.0),
        FieldSpec("mjpg", "Use MJPG", "bool", False),
        FieldSpec("rotate", "Rotate CW", "choice", 270, choices=[0, 90, 180, 270]),
        FieldSpec("mirror", "Mirror output", "bool", True),
        FieldSpec("ndi_follow_camera", "NDI follow camera size", "bool", False, help=FEATURE_HELP["ndi_follow_camera"]),
    ],
    "Output": [
        FieldSpec("w", "NDI Width", "int", 1080),
        FieldSpec("h", "NDI Height", "int", 1920),
        FieldSpec("ndi_name", "NDI Sender Name", "str", "WebcamCutout"),
        FieldSpec("fpsN", "FrameRate Numerator", "int", 60000),
        FieldSpec("fpsD", "FrameRate Denominator", "int", 1001),
        FieldSpec("no_preview", "Disable preview window", "bool", True),
        FieldSpec("show_skel", "Draw skeleton (preview)", "bool", True),
        FieldSpec("bbox_clip", "Clip alpha to bbox (expanded)", "bool", False,
                  help="Matting이 켜져 있을 때 확장된 bbox 밖의 알파를 잘라냅니다."),
    ],
    "Pose / Center": [
        FieldSpec("conf", "YOLO conf", "float", 0.35),
        FieldSpec("imgsz", "YOLO imgsz", "int", 640),
        FieldSpec("kp_thr", "Keypoint thr", "float", 0.25),
        FieldSpec("nearest_only", "Pick only nearest person", "bool", False),
        FieldSpec("center", "Enable Centering", "bool", False, help=FEATURE_HELP["center"]),
        FieldSpec("center_method", "Center method", "choice", "shoulders",
                  choices=["bbox", "hips", "nose", "shoulders"]),
        FieldSpec("center_smooth", "Center smooth (0~1)", "float", 0.8),
        FieldSpec("center_deadzone", "Center deadzone(px)", "int", 12),
    ],
    "Hands / Pace": [
        FieldSpec("hands", "Enable MediaPipe Hands", "bool", False, help=FEATURE_HELP["hands"]),
        FieldSpec("hands_max", "Hands max", "int", 2),
        FieldSpec("hands_det_conf", "Hands det conf", "float", 0.5),
        FieldSpec("hands_track_conf", "Hands track conf", "float", 0.5),
        FieldSpec("hands_complexity", "Hands complexity", "choice", 0, choices=[0, 1]),
        FieldSpec("hands_roi", "Hands ROI by person bbox", "bool", True,
                  help="인물 bbox 주변만 손을 탐지해 속도를 개선합니다."),
        FieldSpec("roi_pad_x", "ROI pad X ratio", "float", 0.10),
        FieldSpec("roi_pad_y", "ROI pad Y ratio", "float", 0.15),

        FieldSpec("target_fps", "Target FPS", "float", 60.0),
        FieldSpec("pace", "Pacing mode", "choice", "drop", choices=["sleep", "drop"]),
        FieldSpec("queue_len", "Capture queue length", "int", 1),
    ],
    "Matte / Grade": [
        FieldSpec("matte", "Enable RVM matting", "bool", True, help=FEATURE_HELP["matte"]),
        FieldSpec("rvm_every", "RVM every N frames", "int", 1),
        FieldSpec("rvm_down", "RVM downsample ratio", "float", 0.25),

        FieldSpec("grade", "Enable grading", "bool", False, help=FEATURE_HELP["grade"]),
        FieldSpec("grade_mode", "Grade mode", "choice", "auto", choices=["auto", "manual"]),
        FieldSpec("bg_ref", "BG ref image", "file", "Background.png"),
        FieldSpec("grade_dark", "Dark (brightness scale)", "float", 0.82),
        FieldSpec("grade_gamma", "Gamma (<1 darker)", "float", 0.95),
        FieldSpec("grade_contrast", "Contrast", "float", 1.08),
        FieldSpec("tint_b", "Tint B", "int", 18),
        FieldSpec("tint_g", "Tint G", "int", 5),
        FieldSpec("tint_r", "Tint R", "int", 12),
        FieldSpec("tint_strength", "Tint strength (0~1)", "float", 0.45),
    ],
    "Performance": [
        FieldSpec("pose_scale", "Pose inference scale", "float", 0.6),
        FieldSpec("pose_every", "Pose every N frames", "int", 1),
        FieldSpec("hands_every", "Hands every N frames", "int", 3),
        FieldSpec("half", "Use FP16 on CUDA", "bool", True),
    ],
    "Cutout / BBox": [
        FieldSpec("bbox_cutout", "Use YOLO bbox as hard cutout", "bool", True,
                  help="사람 bbox 밖을 완전 투명 처리합니다(RVM 사용 여부와 무관)."),
        FieldSpec("bbox_pad_x", "BBox pad X ratio", "float", 0.25,
                  help="가로 패딩 비율(예: 0.10 = 10%)"),
        FieldSpec("bbox_pad_y", "BBox pad Y ratio", "float", 0.25,
                  help="세로 패딩 비율(예: 0.10 = 10%)"),

        FieldSpec("bbox_person_mode", "Person selection", "choice", "nearest",
                  choices=["nearest", "largest", "all"],
                  help="여러 명일 때 어느 박스를 쓸지 선택"),
        FieldSpec("bbox_min_area", "Min bbox area ratio", "float", 0.02,
                  help="프레임 대비 최소 박스 면적 비율(너무 작은 검출 무시)"),
        FieldSpec("bbox_conf_min", "Min bbox conf", "float", 0.25,
                  help="사람 박스 최소 confidence"),
        FieldSpec("bbox_debug", "Debug draw bbox(mask)", "bool", True,
                  help="프리뷰에 박스/마스크를 그려 디버깅"),
    ],
}


def defaults_from_schema() -> dict:
    d = {}
    for fields in SCHEMA.values():
        for spec in fields:
            d[spec.key] = spec.default
    return d


def namespace_from_dict(d: dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in d.items():
        setattr(ns, k, v)
    # 호환 보정(명시적으로 키가 없었어도 기본 채워 넣기)
    extras = {
        "ndi_follow_camera": d.get("ndi_follow_camera", False),
        "bbox_clip": d.get("bbox_clip", False),
        "no_preview": d.get("no_preview", True),
        "show_skel": d.get("show_skel", False),
        "half": d.get("half", True),
    }
    for k, v in extras.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def _build_gui_and_get_values(defaults: dict) -> dict:
    if tk is None:
        return defaults

    root = tk.Tk()
    root.title("WebcamCutout — Startup")
    root.geometry("900x680")

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)

    # key -> {"var": tk.Variable, "widgets": [widget,...], "type": "bool"/...}
    registry: Dict[str, Dict[str, Any]] = {}

    def add_help_label(parent, text: str):
        if not text:
            return
        hl = ttk.Label(parent, text=text, foreground="#666", wraplength=820, justify="left")
        hl.pack(fill="x", padx=6, pady=(6, 2))

    def add_field(frame, spec: FieldSpec):
        row = ttk.Frame(frame)
        row.pack(fill="x", padx=6, pady=4)

        # 라벨
        lbl = ttk.Label(row, text=spec.label, width=28, anchor="w")
        lbl.pack(side="left")

        widgets = [lbl]
        val = defaults.get(spec.key, spec.default)

        if spec.type == "bool":
            var = tk.BooleanVar(value=bool(val))
            cb = ttk.Checkbutton(row, variable=var)
            cb.pack(side="left")
            widgets.append(cb)

        elif spec.type == "choice":
            var = tk.StringVar(value=str(val))
            om = ttk.OptionMenu(row, var, str(val), *[str(c) for c in spec.choices])
            om.pack(side="left", fill="x", expand=True)
            widgets.append(om)

        elif spec.type == "file":
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            btn = ttk.Button(row, text="Browse", command=lambda v=var: v.set(filedialog.askopenfilename() or v.get()))
            btn.pack(side="left", padx=6)
            widgets.extend([ent, btn])

        else:
            # str/int/float
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            widgets.append(ent)

        # help 텍스트(필드 단위)
        if spec.help and spec.type != "bool":
            ttk.Label(row, text=spec.help, foreground="#777").pack(side="left", padx=8)

        # bool 토글류는 다음 줄에 안내문을 더 크게 배치
        if spec.type == "bool" and spec.help:
            sub = ttk.Label(frame, text=spec.help, foreground="#666", wraplength=820, justify="left")
            sub.pack(fill="x", padx=6, pady=(2, 6))
            widgets.append(sub)

        registry[spec.key] = {"var": var, "widgets": widgets, "type": spec.type}

    # 탭 구성
    for tab_name, fields in SCHEMA.items():
        frame = ttk.Frame(nb)
        nb.add(frame, text=tab_name)
        add_help_label(frame, TAB_DESCRIPTIONS.get(tab_name, ""))
        for spec in fields:
            add_field(frame, spec)

    # --- 의존성/상태 업데이트 ---
    def set_enabled(keys: List[str], enabled: bool):
        state = "normal" if enabled else "disabled"
        for k in keys:
            info = registry.get(k)
            if not info:
                continue
            for w in info["widgets"]:
                try:
                    w.configure(state=state)
                except tk.TclError:
                    # 일반 Label 등은 state 옵션이 없을 수 있음
                    pass

    def update_state(*_):
        # center -> method/smooth/deadzone
        set_enabled(["center_method", "center_smooth", "center_deadzone"],
                    registry["center"]["var"].get())

        # hands -> all hands_* (자기 자신 제외) + roi pads
        hands_on = registry["hands"]["var"].get()
        set_enabled(
            ["hands_max", "hands_det_conf", "hands_track_conf", "hands_complexity",
             "hands_roi", "hands_every"],
            hands_on
        )
        # hands_roi -> roi pads
        set_enabled(["roi_pad_x", "roi_pad_y"], hands_on and registry["hands_roi"]["var"].get())

        # matte -> rvm_every, rvm_down, bbox_clip(Output에 있지만 matting과 강결합)
        set_enabled(["rvm_every", "rvm_down", "bbox_clip"], registry["matte"]["var"].get())

        # grade -> grade_* 전부 (grade_mode/bg_ref 포함)
        grade_on = registry["grade"]["var"].get()
        set_enabled(["grade_mode", "bg_ref", "grade_dark", "grade_gamma",
                     "grade_contrast", "tint_b", "tint_g", "tint_r", "tint_strength"],
                    grade_on)

        # ndi_follow_camera -> w/h 비활성화
        set_enabled(["w", "h"], not registry["ndi_follow_camera"]["var"].get())

        # matte -> rvm_every, rvm_down, bbox_clip
        set_enabled(["rvm_every", "rvm_down", "bbox_clip"], registry["matte"]["var"].get())

        # grade -> grade_* 전부
        grade_on = registry["grade"]["var"].get()
        set_enabled(["grade_mode", "bg_ref", "grade_dark", "grade_gamma",
                     "grade_contrast", "tint_b", "tint_g", "tint_r", "tint_strength"],
                    grade_on)

        # ndi_follow_camera -> w/h 비활성화
        set_enabled(["w", "h"], not registry["ndi_follow_camera"]["var"].get())

        # === NEW: bbox_cutout 의존성 ===
        if "bbox_cutout" in registry:
            cutout_on = registry["bbox_cutout"]["var"].get()
            set_enabled([
                "bbox_pad_x", "bbox_pad_y", "bbox_feather",
                "bbox_person_mode", "bbox_min_area", "bbox_conf_min", "bbox_debug"
            ], cutout_on)

    # 토글 변수에 trace 걸기
    for k in ["center", "hands", "hands_roi", "matte", "grade", "ndi_follow_camera", "bbox_cutout"]:
        if k in registry and isinstance(registry[k]["var"], tk.Variable):
            registry[k]["var"].trace_add("write", update_state)

    # 초기 상태 반영
    update_state()

    # 버튼
    btns = ttk.Frame(root)
    btns.pack(fill="x", padx=8, pady=8)
    result: Dict[str, Any] = {}

    def on_start():
        try:
            for _, fields in SCHEMA.items():
                for spec in fields:
                    info = registry[spec.key]
                    var = info["var"]
                    if spec.type == "bool":
                        result[spec.key] = bool(var.get())
                    elif spec.type == "choice":
                        val = var.get()
                        if isinstance(spec.default, int):
                            result[spec.key] = int(val)
                        elif isinstance(spec.default, float):
                            result[spec.key] = float(val)
                        else:
                            result[spec.key] = val
                    else:
                        s = var.get()
                        if spec.type == "int":
                            result[spec.key] = int(s) if s != "" else int(spec.default)
                        elif spec.type == "float":
                            result[spec.key] = float(s) if s != "" else float(spec.default)
                        else:
                            result[spec.key] = s
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
            return
        root.destroy()

    def on_cancel():
        result.clear()
        result.update(defaults)
        root.destroy()

    ttk.Button(btns, text="Start", command=on_start).pack(side="right")
    ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="right", padx=6)

    root.mainloop()
    return result or defaults


def get_args_with_gui_fallback(ap: argparse.ArgumentParser) -> argparse.Namespace:
    """
    - CLI 인자를 주면: ap.parse_args() 그대로 사용
    - 인자가 없으면: Tkinter GUI로 값 입력받아 argparse.Namespace 생성
    """
    if len(sys.argv) > 1:
        return ap.parse_args()
    defaults = defaults_from_schema()
    gui_vals = _build_gui_and_get_values(defaults)
    return namespace_from_dict(gui_vals)
