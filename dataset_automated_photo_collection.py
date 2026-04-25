from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2

DEFAULT_CLASSES = ["left", "right", "stop", "spd1", "spd2", "spd3", "spd4", "none"]


def parse_classes(s: str) -> list[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else list(DEFAULT_CLASSES)


def ensure_dirs(root: Path, classes: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (root / c).mkdir(parents=True, exist_ok=True)


def next_filename(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return folder / f"{ts}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Save webcam frames into gesture class folders."
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./newdata/raw"),
        help="Raw capture root: each class is a subfolder (default: ./newdata/raw).",
    )
    ap.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_CLASSES),
        help=f"Comma-separated folder names (default: {','.join(DEFAULT_CLASSES)}).",
    )
    ap.add_argument(
        "--camera", type=int, default=0, help="OpenCV camera index (default 0)."
    )
    ap.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable horizontal flip of the preview (saved files still unflipped unless --save-mirrored).",
    )
    ap.add_argument(
        "--save-mirrored",
        action="store_true",
        help="Also flip saved images horizontally (usually leave OFF so labels match real-world left/right).",
    )
    ap.add_argument(
        "--auto-sec",
        type=float,
        default=0.0,
        help="If >0, key 'a' toggles saving every this many seconds using the live frame.",
    )
    args = ap.parse_args()
    mirror_preview = not args.no_mirror_preview

    classes = parse_classes(args.classes)
    out_root: Path = args.output_dir.resolve()
    ensure_dirs(out_root, classes)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    idx = 0
    counts = {c: len(list((out_root / c).glob("*.jpg"))) for c in classes}
    auto_mode = False
    last_auto_save = 0.0

    help_lines = [
        "1-8: pick class | n/p: next/prev | SPACE: save | a: auto | q: quit",
        f"Saving to: {out_root}",
    ]

    print("\n".join(help_lines))
    print("Classes:", classes)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        to_show = frame
        if mirror_preview:
            to_show = cv2.flip(to_show, 1)

        vis = to_show.copy()
        cls_name = classes[idx]
        y = 28
        cv2.putText(
            vis,
            f"class [{idx + 1}/{len(classes)}]: {cls_name}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )
        y += 28
        cv2.putText(
            vis,
            f"saved this folder (jpg): {counts[cls_name]}",
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 0),
            2,
        )
        y += 26
        for line in help_lines:
            cv2.putText(
                vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1
            )
            y += 22
        if auto_mode and args.auto_sec > 0:
            cv2.putText(
                vis,
                f"AUTO every {args.auto_sec:.1f}s ON",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
            )

        cv2.imshow("capture_webcam_dataset (focus this window for keys)", vis)

        now = time.time()
        key = cv2.waitKey(1) & 0xFF

        def save_current_frame() -> None:
            nonlocal counts
            img_to_save = frame
            if args.save_mirrored:
                img_to_save = cv2.flip(img_to_save, 1)
            path = next_filename(out_root / cls_name)
            if not cv2.imwrite(str(path), img_to_save):
                print(f"FAILED to write: {path}")
            else:
                counts[cls_name] += 1
                print(f"saved -> {path}")

        if key == ord("q") or key == 27:
            break
        if key == ord(" "):
            save_current_frame()
        if key == ord("n"):
            idx = (idx + 1) % len(classes)
        if key == ord("p"):
            idx = (idx - 1) % len(classes)
        if key == ord("a") and args.auto_sec > 0:
            auto_mode = not auto_mode
            last_auto_save = now
            print("AUTO mode:", "ON" if auto_mode else "OFF")

        if ord("1") <= key <= ord("9"):
            digit = key - ord("1")
            if 0 <= digit < len(classes):
                idx = digit

        if auto_mode and args.auto_sec > 0 and (now - last_auto_save) >= args.auto_sec:
            save_current_frame()
            last_auto_save = now

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    main()
