# Importing an AdamWorld into Unity

Two forms are exported:

- **Mesh** — `mesh.glb`: a coarse colored environment mesh (great for collision & layout).
- **Gaussian splats** — `splats.ply` / `splats.spz`: the photoreal radiance field.

## Mesh (`.glb`)

Unity doesn't import glTF out of the box — add one importer:

1. Install **glTFast** (Window → Package Manager → Add package by name →
   `com.unity.cloud.gltfast`), or **UnityGLTF**.
2. Drag `mesh.glb` into the Project window (or use glTFast runtime import). It
   imports as a mesh with **vertex colors**; use a shader that reads vertex color
   (e.g. a Shader Graph with a *Vertex Color* node into Base Color), since URP/HDRP
   lit shaders ignore vertex color by default.
3. Coordinate frame: AdamWorld is **+Y up**, right-handed, ~metric, centered at the
   origin. glTF importers handle the handedness flip; verify scale against a 1 m cube.
4. Add a **Mesh Collider** for walk-through/physics.

## Gaussian splats (`.ply` / `.spz`)

Use a Unity 3DGS renderer package, e.g. **UnityGaussianSplatting** (aras-p) or a
splat asset from the Asset Store:

1. Install the package and run its **import** on `splats.ply` to bake a
   `GaussianSplatAsset`.
2. Add the provided **Gaussian Splat Renderer** component to a GameObject and assign
   the asset.
3. The splats share the mesh's origin/scale, so the collider mesh and the splats
   align — collide against the mesh, render the splats.

> `.spz` is the compact form (≈10× smaller). If your importer only accepts `.ply`,
> export with `--format ply`.
