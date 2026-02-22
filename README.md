# Hydrogenic Orbital Visualizer

This repo contains multiple OpenGL orbital visualizers (`atom`, `wave_atom_2d`, `atom_realtime`, `atom_raytracer`) and a Python orbital sampler (`src/schrodinger.py`).

## What Changed

- Added a cross-platform `CMakeLists.txt`.
- Updated OpenGL setup so the legacy 2D renderers work on macOS.
- Added Dirac radial mode + higher nuclear charge (`Z`) support in `src/schrodinger.py`.
- Fixed orbital sampling cache bugs in the realtime/raytracer samplers when `n/l/m` changes.

## Build

### macOS (Homebrew)

```bash
brew install cmake glfw glew glm pkg-config
cmake -S . -B build
cmake --build build -j
```

Notes:
- `atom_raytracer` is skipped on macOS because it needs OpenGL 4.3 SSBOs, while macOS supports up to OpenGL 4.1.

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config \
  libglfw3-dev libglew-dev libglm-dev libgl1-mesa-dev
cmake -S . -B build
cmake --build build -j
```

### Windows (vcpkg example)

```bash
vcpkg install glfw3 glew glm
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Dirac + Higher-Z Orbital Generation

The sampler now supports:
- `--mode schrodinger` (hydrogenic Schrodinger)
- `--mode dirac` (Dirac-Coulomb radial large-component scaling)
- arbitrary nuclear charge `--z` for hydrogen-like ions

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Example (iron-like hydrogenic ion, Dirac radial mode):

```bash
python3 src/schrodinger.py \
  --mode dirac --z 26 --n 3 --l 1 --m 0 --j-branch plus \
  --samples 80000
```

Example (interactive mode):

```bash
python3 src/schrodinger.py
```

Generated JSON files are written to `orbitals/` by default.

## Source Layout

- `src/atom.cpp`: 2D legacy atom model
- `src/wave_atom_2d.cpp`: 2D wave/orbit visualizer
- `src/atom_realtime.cpp`: realtime 3D point/sphere orbital renderer
- `src/atom_raytracer.cpp`: SSBO-based raytracer (non-macOS)
- `src/schrodinger.py`: orbital sample generator (Schrodinger + Dirac radial mode)
