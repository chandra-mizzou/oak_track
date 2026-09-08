"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--table-height", type=float, required=True, help="Table height h in metres")
    p.add_argument(
        "--optical-height",
        type=float,
        default=0.03,
        help="Camera optical center above the table (m). CSV z is still h.",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="oak-track",
        description="Log OAK-D Pro W video+IMU, write camera IMU CSV, localize a table object.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="Record color, depth, and IMU from a live OAK-D Pro W")
    _add_common(rec)
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--fps", type=int, default=30)
    rec.add_argument("--width", type=int, default=1280)
    rec.add_argument("--height", type=int, default=720)
    rec.add_argument("--mono-resolution", default="800p", choices=["400p", "800p"])
    rec.add_argument("--imu-rate", type=int, default=200)
    rec.add_argument("--no-ir", action="store_true")
    rec.add_argument("--no-depth", action="store_true")
    rec.add_argument("--duration", type=float, default=None, help="Seconds; default is until Ctrl+C")

    proc = sub.add_parser("process", help="Build camera_imu.csv and object.csv from a run folder")
    proc.add_argument("--run", type=Path, required=True)
    proc.add_argument("--table-height", type=float, default=None)
    proc.add_argument("--optical-height", type=float, default=None)
    proc.add_argument("--detector", choices=["hsv", "depth_blob", "aruco"], default="hsv")
    proc.add_argument("--still-time", type=float, default=1.0)
    proc.add_argument("--aruco-id", type=int, default=0)
    proc.add_argument(
        "--slide-distance",
        type=float,
        default=None,
        help="Optional measured slide distance (m) to scale IMU translation",
    )
    proc.add_argument("--no-preview", action="store_true")

    sim = sub.add_parser("simulate", help="Write a synthetic run folder (no hardware)")
    _add_common(sim)
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
        )
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
        print("Part A camera IMU CSV:", summary["camera_imu_csv"])
        print("Part B object CSV:    ", summary["object_csv"])
        print("Fused object (x,y,z): ", tuple(summary["object_fused_xyz"]))
        print("Detections used:      ", summary["n_object_detections"])
        print("RMS reprojection px:  ", summary["object_rms_reproj_px"])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
