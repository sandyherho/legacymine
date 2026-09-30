"""Rebuild both MP4 videos from scratch.

Usage: python videos/make_videos.py [reef] [bubble]

With no argument both videos are built.  Steps for each video:

reef    1. reef_sim.py re-runs the Indonesian reef case of the package
           with dense frame output (same physics and settings as the
           repository run; about two minutes on one core);
        2. reef_render.py draws 1920x1080 frames;
        3. ffmpeg encodes one loop and a 5x looped copy.
bubble  1. bubble_km.py solves the Keller-Miksis bubble (seconds);
        2. bubble_render.py draws 1920x1080 frames;
        3. ffmpeg encodes one loop and a 4x looped copy.

Frames are rendered by parallel worker processes, one per CPU.  Every
loop starts with a 0.3 s fade-in and ends with a 0.8 s hold and a 0.5 s
fade-out, so repeats join without a jump.  Intermediate files go to
outputs/cache/video (ignored by git); finished videos to outputs/videos.
Requires ffmpeg with libx264 on the PATH, or the imageio-ffmpeg package.
"""

import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(ROOT, "outputs", "cache", "video")
VIDDIR = os.path.join(ROOT, "outputs", "videos")
FPS = 30
BG = "0x05070d"          # background colour of the dark style
HOLD, FADE_IN, FADE_OUT = 0.8, 0.3, 0.5


def ffmpeg_exe():
    """Path to an ffmpeg executable."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install ffmpeg or "
                         "`pip install imageio-ffmpeg`")


def run(args):
    """Run a command, stopping on failure."""
    t0 = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"failed: {' '.join(args[:3])}")
    out = r.stdout.strip()
    if out:
        print(out)
    return time.time() - t0


def render_parallel(script, n_frames):
    """Render frames 0..n_frames-1 with one worker per CPU."""
    workers = max(os.cpu_count() or 1, 1)
    edges = np.linspace(0, n_frames, workers + 1).astype(int)
    procs = [subprocess.Popen([sys.executable, script, str(a), str(b)],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
             for a, b in zip(edges[:-1], edges[1:]) if b > a]
    for p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            sys.stderr.write(err)
            raise SystemExit(f"rendering failed in {script}")
        print(out.strip())


def encode(png_dir, n_frames, stem, loops):
    """Encode one loop and a looped copy; return their paths."""
    ff = ffmpeg_exe()
    os.makedirs(VIDDIR, exist_ok=True)
    dur = n_frames / FPS + HOLD
    vf = (f"tpad=stop_mode=clone:stop_duration={HOLD},"
          f"fade=t=in:st=0:d={FADE_IN}:color={BG},"
          f"fade=t=out:st={dur - FADE_OUT:.4f}:d={FADE_OUT}:color={BG},"
          "format=yuv420p")
    one = os.path.join(VIDDIR, f"{stem}_single_loop.mp4")
    many = os.path.join(VIDDIR, f"{stem}_loop{loops}x.mp4")
    run([ff, "-hide_banner", "-loglevel", "error", "-y", "-framerate",
         str(FPS), "-i", os.path.join(png_dir, "f%04d.png"), "-vf", vf,
         "-c:v", "libx264", "-preset", "slow", "-tune", "animation",
         "-crf", "21", "-movflags", "+faststart", one])
    run([ff, "-hide_banner", "-loglevel", "error", "-y", "-stream_loop",
         str(loops - 1), "-i", one, "-c", "copy", "-movflags",
         "+faststart", many])
    return one, many


def fresh(path):
    """Empty a frame directory so stale frames never leak into a video."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def reef():
    """Build the reef explosion video."""
    print("--- reef explosion: simulation", flush=True)
    run([sys.executable, os.path.join(HERE, "reef_sim.py")])
    n = int(np.load(os.path.join(WORK, "reef_frames.npz"))["times"].size)
    png = os.path.join(WORK, "reef_png")
    fresh(png)
    print(f"--- reef explosion: rendering {n} frames", flush=True)
    render_parallel(os.path.join(HERE, "reef_render.py"), n)
    for p in encode(png, n, "reef_explosion", 5):
        print(p)


def bubble():
    """Build the breathing-bubble video."""
    print("--- bubble: Keller-Miksis solution", flush=True)
    run([sys.executable, os.path.join(HERE, "bubble_km.py")])
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "bubble_render.py"), "--count"],
                       capture_output=True, text=True, check=True)
    n = int(r.stdout.split()[0])
    png = os.path.join(WORK, "bubble_png")
    fresh(png)
    print(f"--- bubble: rendering {n} frames", flush=True)
    render_parallel(os.path.join(HERE, "bubble_render.py"), n)
    for p in encode(png, n, "reef_bubble", 4):
        print(p)


if __name__ == "__main__":
    t0 = time.time()
    todo = sys.argv[1:] or ["reef", "bubble"]
    for name in todo:
        {"reef": reef, "bubble": bubble}[name]()
    print(f"--- done in {time.time() - t0:.0f} s")
