# oak_track

Record an **OAK-D Pro W** while it slides along a table, then:

1. **Part A** — log video + IMU and write a two-column CSV: `frame_number`, `(x, y, z)` of the **camera**, with frame 0 at `(0, 0, h)`.
2. **Part B** — estimate the **object** on the table as `(x, y, z)` in the same frame, as precisely as this hardware allows.

`h` is the table height. The camera and the object both sit on the table, so both are reported with `z = h`.

**Usual command** — records video, timestamps, and IMU, then writes both CSVs:

```bash
python run.py --table-height 0.75 --optical-height 0.03
```

Hold still ~1 second, slide along the table width, then Ctrl+C. To stop after a fixed time instead:

```bash
python run.py --out runs/slide1 --table-height 0.75 --duration 8
```

---

## Why IMU-only position is not enough

The OAK-D Pro W IMU is either a **BNO085/BNO086** (9-axis, on-chip fusion) or a **BMI270** (6-axis accel+gyro). Accelerometer double integration drifts by centimetres to metres within a few seconds. That is still what Part A asks for, so the pipeline writes it — but **object coordinates must not rely on IMU translation**.

The accurate object estimate uses, in order of importance:

| Cue | Why it is strong in this setup |
| --- | --- |
| Onboard stereo depth (7.5 cm baseline, IR dots) | Object is at 0.8–1.0 m, inside the Pro W’s useful depth band (~0.7–8 m). |
| Table plane `z = h` | Reduces the object to `(x, y)` on a known plane. |
| Multi-view while sliding | The slide itself is a **large baseline** (tens of cm). Bearings of the object from many camera X positions triangulate much better than a single 7.5 cm stereo pair. |
| RGB-D visual odometry | Recovers camera X when the IMU path explodes or stays near zero. |
| Optional tape measure of the slide | Scales IMU translation (`--slide-distance`) if you want Part A to match the real travel. |

Ray–plane intersection with `z = h` is **not** the primary method: if the optical center is also at height `h`, those rays are parallel to the table. The code therefore uses stereo depth + multi-view, and treats the optical center as `h + optical_height` internally (default 3 cm, the camera body sitting on the table). CSV `z` values stay at `h` as specified.

---

## Coordinate system

Right-handed metres:

- **Origin:** camera at the first video frame, reported as `(0, 0, h)`.
- **+X:** table width / slide direction (camera-right at t = 0).
- **+Y:** along the table toward the object (camera-forward at t = 0).
- **+Z:** up. Table plane is `z = h`.

Camera optical axes follow OpenCV / Luxonis RDF: `x` right, `y` down, `z` forward.

---

## Physical setup (do this carefully)

1. Measure **table height `h`** from the floor to the table top with a tape (mm).
2. Measure **optical-center height** above the table (typically 2–4 cm for an OAK-D sitting on its bottom). Pass it as `--optical-height`.
3. Place a **high-contrast object** 80–100 cm in front of the start pose. Best options, in order:
   - 4×4 ArUco marker (`--detector aruco`) — most precise 2D centroid.
   - Saturated red / green object on a dull table (`--detector hsv`, default is red).
4. Put the OAK-D Pro W on the table, USB cable out of the slide path, lens aimed at the object, **not tilted**.
5. Enable the **IR laser dot projector** (default). It gives the stereo matcher texture on a bare table. Use a **powered USB3 Y-cable** on Pro models; the projector is power-hungry.
6. Mark the start and end of the slide on the table edge and measure that distance. Optional but useful: `--slide-distance`.
7. Lighting: no strobe, no strong sunlight on the IR projector. A still second at the beginning is required for IMU bias.

Recommended object distance **≥ 70 cm** so 800p stereo is valid (`MinZ` is ~70 cm at 800p, ~35–40 cm at 400p with extended disparity).

---

## Install

Run these from the **oak_track repo root** (the folder that contains `record.py` and `pyproject.toml`), not from a different OAK project directory.

Prefer `python -m pip` over `pip`. If a venv was moved, recreated, or only half-installed, bash still hashes the old `pip` path and you get `bin/pip: No such file or directory`.

**Repair an existing venv** (for example `venv_oak`):

```bash
deactivate 2>/dev/null || true
hash -r
source venv_oak/bin/activate          # or: source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip
python -m pip install -r requirements.txt
```

**Or create a fresh venv:**

```bash
deactivate 2>/dev/null || true
hash -r
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`record.py` / `process.py` add `src/` to `PYTHONPATH` themselves, so you do **not** need `pip install -e ".[device]"`. That editable install is optional.

Runnable files at the repo root:

| File | What it does |
| --- | --- |
| `run.py` | **Record + process in one step** (video, timestamps, IMU, then CSVs) |
| `record.py` | Capture only |
| `process.py` | Process an existing run folder |
| `simulate.py` | Synthetic run when no camera is plugged in |

---

## Capture + process (one command)

Hold the camera **still for ~1 second**, then slide it smoothly along the table width. Do not lift it or yaw it. Stop with Ctrl+C (or pass `--duration`).

```bash
python run.py \
  --out runs/slide1 \
  --table-height 0.75 \
  --optical-height 0.03 \
  --fps 30 \
  --mono-resolution 800p
```

That writes `color.mp4`, `frames.csv` (frame number + timestamps), `imu.csv`, then `camera_imu.csv` and `object.csv` in the same folder.

The camera is detected automatically: plug the OAK-D into USB, then run `run.py` / `record.py`. DepthAI opens the first OAK it sees. You do not pass a port or device ID. If several OAKs are plugged in, the first one in the USB list is used.

**Linux USB permissions:** if you see `X_LINK_UNBOOTED` / `Insufficient permissions` / `No available devices`, install udev rules once:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then unplug the camera, plug it back in, wait a few seconds, and rerun. Do not use `sudo python`.

To split the steps:

```bash
python record.py --out runs/slide1 --table-height 0.75 --optical-height 0.03
python process.py --run runs/slide1 --table-height 0.75
```

On USB2 hosts drop to `--mono-resolution 400p`.

What is stored:

| File | Contents |
| --- | --- |
| `color.mp4` | RGB preview stream |
| `depth/000000.png` … | 16-bit depth in millimetres, aligned to RGB |
| `imu.csv` | Device timestamps, accel, gyro, optional quaternion / linear accel |
| `frames.csv` | Frame index ↔ color/depth timestamps |
| `calibration.json` | Intrinsics, distortion, IMU–camera extrinsics from the EEPROM |

---

## Process

```bash
python process.py \
  --run runs/slide1 \
  --table-height 0.75 \
  --optical-height 0.03 \
  --detector hsv \
  --slide-distance 0.42
```

Use `--detector aruco` if you put a marker on the object.

Outputs (two columns, metres):

```text
frame_number,xyz
0,"(0.000000, 0.000000, 0.750000)"
1,"(0.012431, 0.000184, 0.750000)"
```

| File | Meaning |
| --- | --- |
| `camera_imu.csv` | **Part A** — IMU-predicted camera `(x, y, z)`, frame 0 = `(0, 0, h)`, `z = h` |
| `camera_vo.csv` | RGB-D odometry camera path (used for Part B when it is sane) |
| `object.csv` | **Part B** — per-frame object `(x, y, h)` |
| `object_fused.csv` | Single static object position (best overall estimate) |
| `preview.mp4` | Detections overlaid |
| `summary.json` | Fused object, detection count, RMS reprojection |

---

## How Part B is computed (precision path)

For each frame with a detection `(u, v)` and median stereo depth `Z` in a small window:

1. Back-project to a camera-frame point with the RGB intrinsics.
2. Transform into the world using the camera pose (VO if the recovered slide is 5 cm–2 m, otherwise IMU).
3. Snap `z` to `h`.
4. Jointly refine **camera X(t), Y(t)** and a **static object (X, Y, h)** with Huber-weighted reprojection + depth residuals, a smoothness prior, and `Y_camera ≈ 0` (straight slide along the table width).

That last step is what uses the slide as a wide-baseline stereo rig. Expect roughly:

- **Y (range):** ~1–3 cm with 800p subpixel depth + IR dots at 1 m, better after multi-frame fusion.
- **X (along slide):** limited by camera-path error. VO + `--slide-distance` typically keeps this in the low centimetres. IMU-only X can be much worse.
- **Z:** exactly `h` by construction in the CSV. Internally the optical center is `h + optical_height`.

Put an ArUco marker on the object if you need the tightest `(u, v)` and a known size for a PnP check.

---

## Dry run without hardware

```bash
python run.py --simulate --out runs/sim --table-height 0.75
```

---

## Practical checklist for best object accuracy

1. Measure `h` and optical height; do not guess.
2. Hold still 1 s, then slide slowly (1–3 s over ≥ 20 cm). Faster slides starve the integrator and blur the rolling-shutter RGB camera (IMX378 variant).
3. Keep pitch/roll fixed; only translate along X.
4. Use IR dots + 800p + USB3.
5. High-contrast object or ArUco; avoid objects the same colour as the table.
6. Pass `--slide-distance` from a tape.
7. Trust `object_fused.csv` more than any single row of `object.csv`.
8. Treat `camera_imu.csv` as the requested IMU log, not as ground truth.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```
