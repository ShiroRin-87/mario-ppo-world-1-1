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
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.route_robustness import HELD_OUT_CASES, MIDCOURSE_CASES, OPENING_CASES, iter_case
from scripts.revision import source_revision

SIZE = (1920, 1080)
FPS = 15
CASES = (
    ("标准开局", {}),
    ("开局跳跃", OPENING_CASES[2]),
    ("中途反向", MIDCOURSE_CASES[3]),
    ("死亡后恢复", {0: (3,) * 120}),
    ("开局向右", OPENING_CASES[0]),
    ("开局冲刺", OPENING_CASES[1]),
    ("开局反向", OPENING_CASES[3]),
    ("中途松手", MIDCOURSE_CASES[0]),
    ("中途跳跃", MIDCOURSE_CASES[5]),
    ("多次覆盖", HELD_OUT_CASES[-2]),
)
PANELS = (
    (0, 0, 576, 540), (576, 0, 576, 540),
    (0, 540, 576, 540), (576, 540, 576, 540),
    (1152, 0, 384, 360), (1536, 0, 384, 360),
    (1152, 360, 384, 360), (1536, 360, 384, 360),
    (1152, 720, 384, 360), (1536, 720, 384, 360),
)
PHASES = {
    "ready": ("待开始", "#a9b8ce"),
    "playing": ("运行中", "#7cbdf6"),
    "override": ("注入扰动", "#ffd17b"),
    "restored": ("已恢复", "#7fe1d3"),
    "clear": ("到达旗杆", "#9ce5ad"),
    "failed": ("本局结束", "#f59b9b"),
}
BUTTON_NAMES = {"NOOP": "松开", "right": "→", "left": "←"}
ACTION_LABELS = tuple(
    "+".join(BUTTON_NAMES.get(button, button) for button in buttons)
    for buttons in SIMPLE_MOVEMENT
)


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
        self.small = ImageFont.truetype(str(font_path), 18)
        self.large = ImageFont.truetype(str(font_path), 23)

    @property
    def finished(self):
        return all(lane.result is not None for lane in self.lanes)

    def render(self, tick):
        image = Image.new("RGB", SIZE)
        draw = ImageDraw.Draw(image)
        for (x, y, width, height), lane in zip(PANELS, self.lanes):
            lane.advance(tick)
            game = Image.fromarray(lane.frame.pixels)
            image.paste(game.resize((width, height), Image.Resampling.NEAREST), (x, y))
            state, color = PHASES[lane.phase(tick)]
            big = width == 576
            font = self.large if big else self.small
            bar = y + height - (36 if big else 28)
            draw.rectangle((x, bar, x + width - 1, y + height - 1), fill="#101a2b")
            draw.text((x + 10, bar + 3), lane.title, font=font, fill="#eff5ff")
            draw.text((x + (228 if big else 136), bar + 3), state, font=font, fill=color)
            action = lane.frame.action if lane.result is None else None
            buttons = "—" if action is None else ACTION_LABELS[action]
            action_color = PHASES["override"][1] if lane.frame.phase == "override" else "#eff5ff"
            draw.text((x + (360 if big else 212), bar + 3), f"键 {buttons}", font=font, fill=action_color)
            draw.text((x + width - 10, bar + 3), f"x={lane.frame.x}", anchor="ra", font=font, fill="#eff5ff")
        for x in (576, 1152, 1536):
            draw.line((x, 0, x, 1079), fill="#101a2b", width=2)
        draw.line((0, 540, 1151, 540), fill="#101a2b", width=2)
        for y in (360, 720):
            draw.line((1152, y, 1919, y), fill="#101a2b", width=2)
        cleared = sum(lane.phase(tick) == "clear" for lane in self.lanes)
        draw.rectangle((0, 0, 574, 32), fill="#101a2b")
        draw.text((10, 2), "World 1-1  ·  十路同屏", font=self.large, fill="#eff5ff")
        draw.text((565, 2), f"已通关 {cleared:02d}/10", anchor="ra", font=self.large, fill="#a9e9b9")
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
    root.bind("<F11>", lambda _event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
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
            "layout": "four-large-six-small", "panels": PANELS, "resolution": list(SIZE), "fps": FPS,
            "frames": frames, "duration_seconds": frames / FPS,
            "render_seconds": perf_counter() - started,
            "instances": [{"title": lane.title, "start_delay_frames": lane.delay, **lane.result} for lane in lanes],
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PASS {len(lanes)}/{len(lanes)} reached the flag, {frames / FPS:.1f} seconds")


if __name__ == "__main__":
    main()
