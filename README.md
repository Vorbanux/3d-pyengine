# 3d-pyengine
# 3D Software Render Engine

A lightweight, low-level 3D software rendering engine built from scratch using Python, Pygame-CE, and NumPy. This engine calculates all 3D transformations, matrix projections, clipping, and polygon rasterization strictly on the CPU, showcasing deep understanding of computer graphics fundamentals.

---

## 📺 Gameplay & Engine Demo

![3D Engine Animation](https://github.com/Vorbanux/3d-pyengine/blob/main/My%203D%20Engine%202026-06-07%2018-59-50.gif)

*The engine demonstrates fluid wireframe and solid polygon rendering with real-time camera transformations and proper depth sorting.*

---

## 🛠️ Tech Stack & Libraries

*   **Language:** Python 3.14+ (Optimized logic loops)
*   **Math & Matrices:** NumPy (High-speed vectorized matrix multiplications for vertices)
*   **Window & Input:** Pygame-CE (Community Edition) for pixel buffer blitting and keyboard/mouse input handling

---

## 📐 Implementation & Graphics Pipeline

This engine does not rely on modern GPU graphics APIs (like OpenGL or DirectX). Instead, it implements a classic hardware-like pipeline strictly on the CPU (**Software Rendering**):

1.  **Coordinate Spaces:** Leverages a **Left-Handed Coordinate System** with standard Euler Angles (Yaw/Pitch) for responsive FPS camera tracking.
2.  **MVP Transformation Matrix:** Model-View-Projection matrix operations are vectorized via NumPy (`clip_points = (mvp @ vertices.T).T`).
3.  **Near Plane Clipping:** Implements the **Sutherland-Hodgman Algorithm** to clip geometry cleanly against the `W >= NEAR_PLANE` boundary, avoiding division-by-zero errors.
4.  **Perspective Division:** Translates Clip Space coordinates to Normalized Device Coordinates (NDC) and maps them to Screen Space pixels.
5.  **Depth Buffering:** Dynamically calculates average polygon depth based on the `W` component to sort drawing queues, preventing layers from overlapping incorrectly.
6.  **Procedural Topology:** Features optimized structural index loops (**DRY Principle**) for automated 3D mesh building (Cubes, Prisms, and customizable N-gon Cylinders).

---

## 🚀 How To Run (Installation)

Follow these simple steps to launch the engine locally on your machine:

### 1. Clone the repository
```bash
git clone https://github.com
cd 3d-pyengine
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```bash
pip install pygame-ce
pip install numpy
```

### 3. Run the engine
```bash
python main.py
```

---

## 🎮 Controls

*   **`W` / `S`** — Move Camera Forward / Backward
*   **`A` / `D`** — Strafe Camera Left / Right
*   **`Mouse Move`** — Look Around (First-Person Perspective)
*   **`ESC`** — Close the engine application safely

---

## 🔥 For devilopers

If you decide to use my engine for development, you need to know this:

*  You'll need two files: engine.py and grafics.py. You should import grafics.py. You can delete main.py. It's needed for familiarizing yourself with the engine's capabilities.
*  Currently there are shapes: plane (face), pyramid (pyramid), cube (cube), line (edge) (the line is not working, will not be displayed for unknown reasons).
*  Now I will provide the necessary data types and explain each argument from the required classes (remember: the figures are static, their movement is not implemented, although they already have the necessary functions for this):
  ```python
import grafics as g

g.cube(line_color=[red: int, green: int, blue: int, alpha: int], line_width: int, color=[red: int, green: int, blue: int, alpha: int], primary=[x: float, y: float, z: float], world_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None, local_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None)
g.prism(line_color=[red: int, green: int, blue: int, alpha: int], line_width: int, color=[red: int, green: int, blue: int, alpha: int], primary=[x: float, y: float, z: float], world_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None, local_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None)
g.face(line_color=[red: int, green: int, blue: int, alpha: int], color=[red: int, green: int, blue: int, alpha: int], line_width: int, primary=[x: float, y: float, z: float], world_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None, local_poses=[[x: float, y: float, z: float], [x: float, y: float, z: float], ...]=None)

g.start()

# primary - A point in space that is the primary point of a figure; other points are measured from it if local coordinates are entered. The point itself is always given in world coordinates. Edge is not working
```
*  To ensure looping and correct rendering, you need to call the start() function from the grafics.py file at the end of your code. No arguments are required.
```python
import grafics as g

grafics.start()
```
