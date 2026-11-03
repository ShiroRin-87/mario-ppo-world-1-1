"""Show ten independent World 1-1 instances in one window or MP4."""

import argparse
import json
import shutil
import subprocess
import sys
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.route_robustness import HELD_OUT_CASES, MIDCOURSE_CASES, OPENING_CASES, iter_case
from scripts.revision import source_revision

SIZE = (1920, 1080)
FPS = 15
CASES = (
    ("标准开局", {}),
    ("开局向右", OPENING_CASES[0]),
    ("开局冲刺", OPENING_CASES[1]),
    ("开局跳跃", OPENING_CASES[2]),
    ("开局反向", OPENING_CASES[3]),
    ("中途松手", MIDCOURSE_CASES[0]),
    ("中途反向", MIDCOURSE_CASES[3]),
    ("中途跳跃", MIDCOURSE_CASES[5]),
    ("多次覆盖", HELD_OUT_CASES[-2]),
    ("死亡后恢复", {0: (3,) * 120}),
)
PHASES = {
    "ready": ("待开始", "#a9b8ce"),
    "playing": ("运行中", "#7cbdf6"),
    "override": ("注入扰动", "#ffd17b"),
    "restored": ("检查点已恢复", "#7fe1d3"),
    "clear": ("到达旗杆", "#9ce5ad"),
    "failed": ("本局结束", "#f59b9b"),
}


@dataclass
class Lane:
    title: str
    stream: object
    delay: int
    frame: object
    result: dict | None = None
    restored_until: int = 0

    def advance(self, tick):
        if tick < self.delay or self.result is not None:
            return
        try:
            self.frame = next(self.stream)
        except StopIteration as finished:
            self.result = finished.value
        else:
            if self.frame.phase == "restored":
                self.restored_until = tick + FPS

    def phase(self, tick):
        if self.result is not None:
            return "clear" if self.result["clear"] else "failed"
        if self.frame.phase == "playing" and tick < self.restored_until:
            return "restored"
        return self.frame.phase


class Grid:
    def __init__(self, lanes, font_path):
        self.lanes = lanes
        self.small = ImageFont.truetype(str(font_path), 22)
        self.title = ImageFont.truetype(str(font_path), 29)
        self.counter = ImageFont.truetype(str(font_path), 43)
        self.background = Image.new("RGB", SIZE, "#0d1627")
        draw = ImageDraw.Draw(self.background)
        draw.text((59, 24), "超级马里奥  World 1-1", font=ImageFont.truetype(str(font_path), 48), fill="#f2f7ff")
        draw.text((61, 88), "10 个独立实例   ·   同一实时策略   ·   开局与中途按键扰动", font=self.small, fill="#a9bbd3")
        draw.text((60, 1024), "橙色  注入扰动     青色  恢复检查点     绿色  到达旗杆", font=self.small, fill="#b5c7de")
        draw.text((1858, 1024), "原速 15 决策步 / 秒", anchor="ra", font=self.small, fill="#b5c7de")
        self.origins = [(60 + 364 * (i % 5), 145 + 423 * (i // 5)) for i in range(len(lanes))]
        for (x, y), lane in zip(self.origins, lanes):
            draw.rounded_rectangle((x - 2, y - 2, x + 346, y + 402), radius=12, fill="#1c2b42")
            draw.text((x + 12, y + 5), lane.title, font=self.title, fill="#f2f7ff")

    @property
    def finished(self):
        return all(lane.result is not None for lane in self.lanes)

    def render(self, tick):
        image = self.background.copy()
        draw = ImageDraw.Draw(image)
        for (x, y), lane in zip(self.origins, self.lanes):
            lane.advance(tick)
            game = Image.fromarray(lane.frame.pixels)
            image.paste(game.resize((344, 322), Image.Resampling.NEAREST), (x, y + 47))
            state, color = PHASES[lane.phase(tick)]
            draw.ellipse((x + 14, y + 380, x + 24, y + 390), fill=color)
            draw.text((x + 34, y + 371), state, font=self.small, fill=color)
            draw.text((x + 331, y + 371), f"x = {lane.frame.x}", anchor="ra", font=self.small, fill="#e3ecfa")
        cleared = sum(lane.phase(tick) == "clear" for lane in self.lanes)
        draw.text((1858, 37), f"已通关  {cleared:02d} / 10", anchor="ra", font=self.counter, fill="#a9e9b9")
        return image


def record(grid, target):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("FFmpeg is required to export MP4")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid4().hex}.tmp.mp4")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1920x1080",
        "-framerate", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "24", "-maxrate", "2200k", "-bufsize", "4400k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    tick = 0
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            while not grid.finished:
                image = grid.render(tick)
                encoder.stdin.write(image.tobytes())
                tick += 1
            final_frame = image.tobytes()
            for _ in range(FPS * 2):
                encoder.stdin.write(final_frame)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed to encode the grid")
        if not all(lane.result["clear"] for lane in grid.lanes):
            raise RuntimeError("At least one game did not reach the flag")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return tick + FPS * 2


def preview(grid):
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("World 1-1 十路同屏演示")
    width = min(root.winfo_screenwidth() - 60, 1600)
    height = min(root.winfo_screenheight() - 100, int(width * 9 / 16))
    root.geometry(f"{width}x{height}")
    display = tk.Label(root, bg="#0d1627", borderwidth=0)
    display.pack(fill="both", expand=True)
    root.bind("<Escape>", lambda _event: root.destroy())
    tick = 0

    def update():
        nonlocal tick
        started = perf_counter()
        image = grid.render(tick)
        size = (max(1, display.winfo_width()), max(1, display.winfo_height()))
        display.image = ImageTk.PhotoImage(ImageOps.contain(image, size))
        display.configure(image=display.image)
        tick += 1
        if not grid.finished:
            delay = max(1, round((1 / FPS - (perf_counter() - started)) * 1000))
            root.after(delay, update)

    root.after(0, update)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-model", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="Export a single MP4")
    target.add_argument("--preview", action="store_true", help="Show the live ten-instance grid")
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Use --font to supply a font with Chinese glyphs")
    if args.output and (args.output.exists() or args.output.with_suffix(".json").exists()):
        parser.error("Choose an unused output filename")
    with np.load(args.route_model, allow_pickle=False) as model:
        features, actions = model["features"], model["actions"]
    with ExitStack() as stack:
        lanes = []
        for i, (title, overrides) in enumerate(CASES):
            stream = stack.enter_context(closing(iter_case(
                features, actions, overrides, recover=True, initial_frame=True,
            )))
            lanes.append(Lane(f"{i + 1:02d}  {title}", stream, i * 4, next(stream)))
        grid = Grid(lanes, args.font)
        if args.preview:
            preview(grid)
            return
        started = perf_counter()
        frames = record(grid, args.output)
        report = {
            "source_commit": source_revision(ROOT),
            "grid": [5, 2], "resolution": list(SIZE), "fps": FPS,
            "frames": frames, "duration_seconds": frames / FPS,
            "render_seconds": perf_counter() - started,
            "instances": [{"title": lane.title, "start_delay_frames": lane.delay, **lane.result} for lane in lanes],
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PASS {len(lanes)}/{len(lanes)} reached the flag, {frames / FPS:.1f} seconds")


if __name__ == "__main__":
    main()
