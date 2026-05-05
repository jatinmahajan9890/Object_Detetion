# Smart-Inspect: Algorithmic Anomaly Subtraction

Smart-Inspect is a real-time, edge-capable Computer Vision pipeline designed for Structural Health Monitoring (SHM). It abandons traditional, computationally heavy Deep Learning classification in favor of **Zero-Shot spatial isolation** and **deterministic matrix mathematics** (Algorithmic Anomaly Subtraction).

## Core Architecture
1. **The Traffic Cop Router:** Utilizes `YOLO-World` to dynamically isolate architectural targets (walls, doors, floors) while actively ignoring everyday environmental clutter (chairs, clocks, people).
2. **Deterministic Defect Engine:** Extracts targets using advanced OpenCV geometry (GrabCut, Canny Edges, Adaptive Thresholding) and quantifies damage using non-linear spatial filtering and Absolute Matrix Difference calculations.
3. **True-Async Multithreading:** Decouples the physical camera sensor I/O from the AI inference engine, allowing the system to process high-resolution frames locally on consumer hardware without inducing CPU thermal throttling or UI latency.

## Usage

**Live Camera Feed:**
To run the live inspection tool with the True-Async threaded camera:
`python3 -m app.test_camera`

**The Data Flywheel (Offline Analyst):**
To analyze unknown objects caught by the live camera:
`python3 -m app.offline_analyst`
