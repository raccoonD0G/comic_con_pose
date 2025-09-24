# ui/gui_startup.py
from __future__ import annotations

import argparse
import contextlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    type: str  # 'str'|'int'|'float'|'bool'|'choice'|'file'|'camera'
    default: object
    choices: List[object] = field(default_factory=list)
    help: str = ""


def _enumerate_cameras_linux() -> List[Tuple[int, str]]:
    devices: List[Tuple[int, str]] = []
    sys_class = Path("/sys/class/video4linux")
    if sys_class.exists():
        def sort_key(path: Path) -> int:
            match = re.search(r"(\d+)$", path.name)
            return int(match.group(1)) if match else sys.maxsize

        for entry in sorted(sys_class.glob("video*"), key=sort_key):
            match = re.search(r"(\d+)$", entry.name)
            if not match:
                continue
            idx = int(match.group(1))
            name_path = entry / "name"
            try:
                name = name_path.read_text(encoding="utf-8").strip()
            except OSError:
                name = entry.name
            devices.append((idx, name))
    if devices:
        return devices

    dev_dir = Path("/dev")
    if dev_dir.exists():
        def dev_sort_key(path: Path) -> int:
            match = re.search(r"(\d+)$", path.name)
            return int(match.group(1)) if match else sys.maxsize

        for entry in sorted(dev_dir.glob("video*"), key=dev_sort_key):
            match = re.search(r"(\d+)$", entry.name)
            if not match:
                continue
            idx = int(match.group(1))
            devices.append((idx, entry.name))
    return devices


def _extract_windows_pnp_id(display_name: str) -> str | None:
    prefix = "@device:pnp:"
    if not display_name.lower().startswith(prefix):
        return None

    remainder = display_name[len(prefix):]
    remainder = remainder.split("#{", 1)[0]
    remainder = remainder.rstrip("\\")
    if remainder.startswith("\\\\?\\"):
        remainder = remainder[4:]
    remainder = remainder.replace("#", "\\")
    remainder = remainder.replace("\\\\", "\\")
    remainder = remainder.strip("\\")
    if not remainder:
        return None
    return remainder.upper()


def _win32_directshow_display_names() -> List[str]:
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return []

    HRESULT = ctypes.c_long

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

        def __init__(self, value: str):  # type: ignore[override]
            super().__init__()
            value = value.strip("{}")
            parts = value.split("-")
            if len(parts) != 5:
                raise ValueError(f"Invalid GUID string: {value}")
            self.Data1 = int(parts[0], 16)
            self.Data2 = int(parts[1], 16)
            self.Data3 = int(parts[2], 16)
            data4 = bytes.fromhex(parts[3] + parts[4])
            for i in range(8):
                self.Data4[i] = data4[i]

    CLSID_SystemDeviceEnum = GUID("{62BE5D10-60EB-11d0-BD3B-00A0C911CE86}")
    CLSID_VideoInputDeviceCategory = GUID("{860BB310-5D01-11d0-BD3B-00A0C911CE86}")
    IID_ICreateDevEnum = GUID("{29840822-5B84-11D0-BD3B-00A0C911CE86}")

    class ICreateDevEnum(ctypes.Structure):
        pass

    class IEnumMoniker(ctypes.Structure):
        pass

    class IMoniker(ctypes.Structure):
        pass

    class ICreateDevEnumVtbl(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            ("CreateClassEnumerator", ctypes.c_void_p),
        ]

    class IEnumMonikerVtbl(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            ("Next", ctypes.c_void_p),
            ("Skip", ctypes.c_void_p),
            ("Reset", ctypes.c_void_p),
            ("Clone", ctypes.c_void_p),
        ]

    class IMonikerVtbl(ctypes.Structure):
        _fields_ = [
            ("QueryInterface", ctypes.c_void_p),
            ("AddRef", ctypes.c_void_p),
            ("Release", ctypes.c_void_p),
            ("GetClassID", ctypes.c_void_p),
            ("IsDirty", ctypes.c_void_p),
            ("Load", ctypes.c_void_p),
            ("Save", ctypes.c_void_p),
            ("GetSizeMax", ctypes.c_void_p),
            ("BindToObject", ctypes.c_void_p),
            ("BindToStorage", ctypes.c_void_p),
            ("Reduce", ctypes.c_void_p),
            ("ComposeWith", ctypes.c_void_p),
            ("Enum", ctypes.c_void_p),
            ("IsEqual", ctypes.c_void_p),
            ("Hash", ctypes.c_void_p),
            ("IsRunning", ctypes.c_void_p),
            ("GetTimeOfLastChange", ctypes.c_void_p),
            ("Inverse", ctypes.c_void_p),
            ("CommonPrefixWith", ctypes.c_void_p),
            ("RelativePathTo", ctypes.c_void_p),
            ("GetDisplayName", ctypes.c_void_p),
            ("ParseDisplayName", ctypes.c_void_p),
            ("IsSystemMoniker", ctypes.c_void_p),
        ]

    ICreateDevEnum._fields_ = [("lpVtbl", ctypes.POINTER(ICreateDevEnumVtbl))]
    IEnumMoniker._fields_ = [("lpVtbl", ctypes.POINTER(IEnumMonikerVtbl))]
    IMoniker._fields_ = [("lpVtbl", ctypes.POINTER(IMonikerVtbl))]

    ole32 = ctypes.OleDLL("ole32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole32.CoInitializeEx.restype = HRESULT
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(GUID),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    ole32.CoCreateInstance.restype = HRESULT
    ole32.CreateBindCtx.argtypes = [wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    ole32.CreateBindCtx.restype = HRESULT
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole32.CoTaskMemFree.restype = None

    COINIT_APARTMENTTHREADED = 0x2
    CLSCTX_INPROC_SERVER = 0x1

    init_hr = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    initialized = init_hr in (0, 1)
    if init_hr < 0:
        return []

    devices: List[str] = []
    create_dev_enum = ctypes.POINTER(ICreateDevEnum)()
    try:
        dev_enum_void = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(CLSID_SystemDeviceEnum),
            None,
            CLSCTX_INPROC_SERVER,
            ctypes.byref(IID_ICreateDevEnum),
            ctypes.byref(dev_enum_void),
        )
        if hr < 0:
            return []

        create_dev_enum = ctypes.cast(dev_enum_void, ctypes.POINTER(ICreateDevEnum))
        create_enum_fn = ctypes.WINFUNCTYPE(
            HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.POINTER(IEnumMoniker)),
            wintypes.DWORD,
        )(create_dev_enum.contents.lpVtbl.contents.CreateClassEnumerator)

        enum_moniker = ctypes.POINTER(IEnumMoniker)()
        hr = create_enum_fn(create_dev_enum, ctypes.byref(CLSID_VideoInputDeviceCategory), ctypes.byref(enum_moniker), 0)
        if hr != 0:
            return []

        try:
            release_enum = None
            next_fn = ctypes.WINFUNCTYPE(
                HRESULT,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.POINTER(IMoniker)),
                ctypes.POINTER(ctypes.c_ulong),
            )(enum_moniker.contents.lpVtbl.contents.Next)
            release_enum = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(enum_moniker.contents.lpVtbl.contents.Release)

            bind_ctx = ctypes.c_void_p()
            hr = ole32.CreateBindCtx(0, ctypes.byref(bind_ctx))
            if hr < 0:
                return []

            try:
                release_bind = None
                if bind_ctx:
                    vtbl = ctypes.cast(bind_ctx, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
                    release_bind = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl.contents[2])

                get_display = None
                release_moniker = None

                while True:
                    fetched = ctypes.c_ulong()
                    moniker = ctypes.POINTER(IMoniker)()
                    hr = next_fn(enum_moniker, 1, ctypes.byref(moniker), ctypes.byref(fetched))
                    if hr != 0 or fetched.value == 0:
                        break

                    try:
                        vtbl = moniker.contents.lpVtbl.contents
                        if get_display is None:
                            get_display = ctypes.WINFUNCTYPE(
                                HRESULT,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_void_p),
                            )(vtbl.GetDisplayName)
                        if release_moniker is None:
                            release_moniker = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl.Release)

                        name_ptr = ctypes.c_void_p()
                        hr = get_display(moniker, bind_ctx, None, ctypes.byref(name_ptr))
                        if hr == 0 and name_ptr.value:
                            try:
                                devices.append(ctypes.wstring_at(name_ptr.value))
                            finally:
                                ole32.CoTaskMemFree(name_ptr)
                    finally:
                        if release_moniker is not None and moniker:
                            release_moniker(moniker)

                if release_bind is not None and bind_ctx:
                    release_bind(bind_ctx)
            finally:
                if enum_moniker and release_enum is not None:
                    release_enum(enum_moniker)
        finally:
            if create_dev_enum:
                release_create = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(create_dev_enum.contents.lpVtbl.contents.Release)
                release_create(create_dev_enum)
    finally:
        if initialized:
            ole32.CoUninitialize()

    return devices


def _windows_pnp_name_map() -> Dict[str, str]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_PnPEntity) | Where-Object { $_.PNPClass -eq 'Camera' -or $_.Service -eq 'usbvideo' -or $_.ClassGuid -eq '{e5323777-f976-4f5b-9b55-b94699c46e44}' } | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return {}

    output = proc.stdout.strip()
    if not output:
        return {}

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {}

    mapping: Dict[str, str] = {}
    entries = data if isinstance(data, list) else [data]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("Name")
        pnp_id = entry.get("PNPDeviceID")
        if not name or not pnp_id:
            continue
        mapping[str(pnp_id).upper()] = str(name)
    return mapping


def _enumerate_cameras_windows() -> List[Tuple[int, str]]:
    display_names = _win32_directshow_display_names()
    pnp_map = _windows_pnp_name_map()

    devices: List[Tuple[int, str]] = []
    for idx, display_name in enumerate(display_names):
        pnp_id = _extract_windows_pnp_id(display_name)
        friendly = pnp_map.get(pnp_id or "") if pnp_id else None
        if not friendly:
            friendly = pnp_map.get(display_name.upper())
        if not friendly:
            friendly = display_name
        devices.append((idx, friendly))

    if devices:
        return devices

    if pnp_map:
        return [(i, name) for i, name in enumerate(pnp_map.values())]

    return []


def _enumerate_cameras_macos() -> List[Tuple[int, str]]:
    cmd = ["system_profiler", "SPCameraDataType", "-json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return []

    cameras = data.get("SPCameraDataType", [])
    names: List[str] = []
    for cam in cameras:
        name = cam.get("_name") or cam.get("spcamera_model_id")
        if name:
            names.append(str(name))
    return [(i, name) for i, name in enumerate(names)]


def _probe_opencv_indices(max_probes: int) -> List[int]:
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    found: List[int] = []
    for idx in range(max_probes):
        cap = cv2.VideoCapture(idx)
        if cap is None:
            continue
        try:
            if cap.isOpened():
                found.append(idx)
        finally:
            cap.release()
    return found


def _merge_with_probed_indices(
    devices: List[Tuple[int, str]],
    probed: Iterable[int],
) -> List[Tuple[int, str]]:
    probed_list = list(dict.fromkeys(int(i) for i in probed if isinstance(i, int)))
    if not probed_list:
        return devices

    used: set[int] = set()
    merged: List[Tuple[int, str]] = []

    sequential = all(idx == i for i, (idx, _) in enumerate(devices))
    assigned_from_sequential = 0
    if sequential:
        for actual_idx, (_, name) in zip(probed_list, devices):
            used.add(actual_idx)
            merged.append((actual_idx, name))
            assigned_from_sequential += 1

    remaining_devices = devices[assigned_from_sequential:]
    remaining_probed = [idx for idx in probed_list if idx not in used]

    fallback_iter = (idx for idx in remaining_probed if idx not in used)

    for idx, name in remaining_devices:
        assigned = int(idx)
        if assigned not in probed_list:
            assigned = next(fallback_iter, assigned)
        used.add(assigned)
        merged.append((assigned, name))

    for idx in probed_list:
        if idx not in used:
            merged.append((idx, f"Camera {idx}"))

    merged.sort(key=lambda item: item[0])
    return merged


def enumerate_cameras() -> List[Tuple[int, str]]:
    try:
        system = platform.system().lower()
    except Exception:
        system = sys.platform.lower()

    if system.startswith("linux"):
        devices = _enumerate_cameras_linux()
        merge_with_probes = True
    elif system.startswith("windows"):
        devices = _enumerate_cameras_windows()
        merge_with_probes = True
    elif system.startswith("darwin") or system.startswith("mac"):
        devices = _enumerate_cameras_macos()
        merge_with_probes = False
    else:
        devices = []
        merge_with_probes = False

    devices = [(int(idx), str(name)) for idx, name in devices]

    # Deduplicate while preserving order
    seen = set()
    unique_devices: List[Tuple[int, str]] = []
    for idx, name in devices:
        key = (idx, name)
        if key in seen:
            continue
        seen.add(key)
        unique_devices.append((idx, name))

    max_probe = max((idx for idx, _ in unique_devices), default=-1)
    max_probe = max(8, max_probe + 4)
    probed = _probe_opencv_indices(max_probe)
    if merge_with_probes:
        unique_devices = _merge_with_probed_indices(unique_devices, probed)
    else:
        existing_indices = {idx for idx, _ in unique_devices}
        for idx in probed:
            idx = int(idx)
            if idx in existing_indices:
                continue
            unique_devices.append((idx, f"Camera {idx}"))
            existing_indices.add(idx)
    return unique_devices


CAMERA_CHOICES: List[Tuple[int, str]] = enumerate_cameras()


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
        FieldSpec("cam", "Camera device", "camera", 0),
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

        elif spec.type == "camera":
            choices = CAMERA_CHOICES
            if choices:
                var = tk.StringVar()
                display: List[str] = []
                mapping: Dict[str, int] = {}
                for idx, name in choices:
                    label = f"[{idx}] {name}" if name else f"Camera {idx}"
                    display.append(label)
                    mapping[label] = idx

                default_idx = int(val) if isinstance(val, int) else int(spec.default)
                default_label = None
                for label, idx in mapping.items():
                    if idx == default_idx:
                        default_label = label
                        break
                if default_label is None and display:
                    default_label = display[0]
                if default_label:
                    var.set(default_label)

                cb = ttk.Combobox(row, textvariable=var, values=display)
                cb.pack(side="left", fill="x", expand=True)
                widgets.append(cb)
                registry[spec.key] = {"var": var, "widgets": widgets, "type": spec.type, "choices_map": mapping}
                return

            # fallback: manual entry
            var = tk.StringVar(value=str(val))
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side="left", fill="x", expand=True)
            widgets.append(ent)

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
                    elif spec.type == "camera":
                        mapping = info.get("choices_map")
                        if mapping:
                            sel = var.get()
                            idx = mapping.get(sel)
                            if idx is None:
                                with contextlib.suppress(ValueError):
                                    idx = int(sel)
                            if idx is None:
                                idx = int(spec.default)
                            result[spec.key] = int(idx)
                        else:
                            s = var.get()
                            result[spec.key] = int(s) if s != "" else int(spec.default)
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
