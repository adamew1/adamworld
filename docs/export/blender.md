# Importing an AdamWorld into Blender

A world exports in two complementary forms:

- **Gaussian splats** — `splats.ply` (or `splats.spz`): the photoreal radiance field.
- **Mesh** — `mesh.glb`: a coarse colored environment mesh (collision / blockout / scale reference).

## Get the files

```bash
adamworld export worlds/my_world --format ply --out my_world.ply
adamworld export worlds/my_world --format glb --out my_world.glb   # builds the mesh if needed
```

## Mesh (`.glb`) — works in stock Blender

1. **File → Import → glTF 2.0 (.glb/.gltf)** and pick `my_world.glb`.
2. The mesh imports with **vertex colors** (COLOR_0). To see them in the viewport:
   switch to **Material Preview**, add a material, and plug a **Color Attribute**
   node into Base Color (the attribute is named `Col`).
3. The world is captured around the origin with **+Y up** and roughly metric scale
   (1 unit ≈ 1 m). If your scene uses +Z up, rotate the object −90° about X.

## Gaussian splats (`.ply`)

Blender has no native 3DGS renderer. Use a free add-on, e.g. **KIRI Engine's 3DGS
Render** or **3DGS Render Blender Add-on**:

1. Install the add-on (Edit → Preferences → Add-ons → Install…).
2. Use its **Import Gaussian Splatting** entry and select `splats.ply`.
3. It converts the splats to a point/particle representation you can render with EEVEE/Cycles.

Tip: the mesh and the splats share the same coordinate frame and origin, so import
both and they line up — block out with the mesh, present with the splats.
