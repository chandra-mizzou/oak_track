"""Command-line interface."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _add_table_args(
    p: argparse.ArgumentParser,
    required_height: bool,
    optical_default: float | None = 0.03,
) -> None:
    kwargs: dict = {"type": float, "help": "Table height h in metres"}
    if required_height:
        kwargs["required"] = True
    else:
        kwargs["default"] = None
    p.add_argument("--table-height", **kwargs)
    p.add_argument(
        "--optical-height",
        type=float,
        default=optical_default,
        help="Camera optical center above the table (m). CSV z is still h.",
    )


def _add_record_args(p: argparse.ArgumentParser, out_required: bool) -> None:
    p.add_argument(
        "--out",
        type=Path,
        required=out_required,
        default=None,
        help="Run folder. If omitted on run.py, uses runs/slide_YYYYMMDD_HHMMSS",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--mono-resolution", default="800p", choices=["400p", "800p"])
    p.add_argument("--imu-rate", type=int, default=200)
    p.add_argument("--no-ir", action="store_true")
    p.add_argument("--no-depth", action="store_true")
    p.add_argument("--duration", type=float, default=None, help="Seconds; default is until Ctrl+C")
    p.add_argument(
        "--orientation",
        choices=["auto", "inverted", "upright"],
        default="auto",
        help="auto: detect Y-up (inverted) from gravity; inverted: always rotate 180°",
    )
    p.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV undistort alpha (0=crop valid pixels, 1=keep full FOV)",
    )


def _add_process_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--detector", choices=["hsv", "depth_blob", "aruco"], default="hsv")
    p.add_argument("--still-time", type=float, default=1.0)
    p.add_argument("--aruco-id", type=int, default=0)
    p.add_argument(
        "--slide-distance",
        type=float,
        default=None,
        help="Optional measured slide distance (m) to scale IMU translation",
    )
    p.add_argument("--no-preview", action="store_true")


def _default_out_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"slide_{stamp}"


def _print_summary(summary: dict) -> None:
    print("Part A camera IMU CSV:", summary["camera_imu_csv"])
    print("Part B object CSV:    ", summary["object_csv"])
    print("Fused object (x,y,z): ", tuple(summary["object_fused_xyz"]))
    print("Detections used:      ", summary["n_object_detections"])
    print("RMS reprojection px:  ", summary["object_rms_reproj_px"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="oak-track",
        description="Log OAK-D Pro W video+IMU, write camera IMU CSV, localize a table object.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record color, depth, IMU, and timestamps")
    _add_table_args(rec, required_height=True)
    _add_record_args(rec, out_required=True)

    proc = sub.add_parser("process", help="Build camera_imu.csv and object.csv from a run folder")
    proc.add_argument("--run", type=Path, required=True)
    _add_table_args(proc, required_height=False, optical_default=None)
    _add_process_args(proc)

    run = sub.add_parser(
        "run",
        help="Record a slide (video + timestamps + IMU) then process it in one step",
    )
    _add_table_args(run, required_height=True)
    _add_record_args(run, out_required=False)
    _add_process_args(run)
    run.add_argument(
        "--simulate",
        action="store_true",
        help="Use a synthetic slide instead of the live camera",
    )

    sim = sub.add_parser("simulate", help="Write a synthetic run folder (no hardware)")
    _add_table_args(sim, required_height=True)
    sim.add_argument("--out", type=Path, required=True)
    sim.add_argument("--object-x", type=float, default=0.05)
    sim.add_argument("--object-y", type=float, default=0.90)
    sim.add_argument("--slide", type=float, default=0.40)
    sim.add_argument("--still", type=float, default=1.0)
    sim.add_argument("--slide-time", type=float, default=2.5)

    args = parser.parse_args(argv)

    if args.cmd == "record":
        from oak_track.capture import record_oak

        record_oak(
            out_dir=args.out,
            table_height=args.table_height,
            fps=args.fps,
            color_size=(args.width, args.height),
            mono_resolution=args.mono_resolution,
            imu_rate_hz=args.imu_rate,
            ir_dot_projector=not args.no_ir,
            save_depth=not args.no_depth,
            duration_s=args.duration,
            camera_optical_height_m=args.optical_height,
            orientation=args.orientation,
            undistort_alpha=args.undistort_alpha,
        )
        print(f"Timestamps: {args.out / 'frames.csv'}")
        print(f"IMU log:    {args.out / 'imu.csv'}")
        return 0

    if args.cmd == "simulate":
        from oak_track.simulate import simulate_run

        simulate_run(
            out_dir=args.out,
            table_height=args.table_height,
            optical_height=args.optical_height,
            object_xy=(args.object_x, args.object_y),
            slide_m=args.slide,
            still_s=args.still,
            slide_s=args.slide_time,
        )
        print(f"Simulated run written to {args.out}")
        return 0

    if args.cmd == "process":
        from oak_track.pipeline import process_run

        summary = process_run(
            run_dir=args.run,
            table_height=args.table_height,
            optical_height=args.optical_height,
            detector=args.detector,
            still_time_s=args.still_time,
            write_preview=not args.no_preview,
            aruco_id=args.aruco_id,
            slide_distance=args.slide_distance,
        )
        _print_summary(summary)
        return 0

    if args.cmd == "run":
        from oak_track.pipeline import capture_then_process

        out_dir = args.out or _default_out_dir()
        print(f"Run folder: {out_dir}")
        if not args.simulate:
            print("Hold still ~1s, slide along the table width, then Ctrl+C (or wait for --duration).")
        summary = capture_then_process(
            out_dir=out_dir,
            table_height=args.table_height,
            optical_height=args.optical_height,
            fps=args.fps,
            color_size=(args.width, args.height),
            mono_resolution=args.mono_resolution,
            imu_rate_hz=args.imu_rate,
            ir_dot_projector=not args.no_ir,
            save_depth=not args.no_depth,
            duration_s=args.duration,
            detector=args.detector,
            still_time_s=args.still_time,
            write_preview=not args.no_preview,
            slide_distance=args.slide_distance,
            simulate=args.simulate,
            aruco_id=args.aruco_id,
            orientation=args.orientation,
            undistort_alpha=args.undistort_alpha,
        )
        print(f"Timestamps: {out_dir / 'frames.csv'}")
        print(f"IMU log:    {out_dir / 'imu.csv'}")
        print(f"Video:      {out_dir / 'color.mp4'}")
        _print_summary(summary)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
