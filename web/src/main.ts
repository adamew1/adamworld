import * as THREE from "three";
import { VRButton } from "three/examples/jsm/webxr/VRButton.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

import { FreeLookControls } from "./controls";

// M2: load a generated world's Gaussian splats with Spark and navigate it in the
// browser — WASD/mouse + touch + WebXR. No GPU required; only generation needs one.

const app = document.getElementById("app") as HTMLDivElement;
const progressEl = document.getElementById("progress") as HTMLDivElement;
const hud = document.getElementById("hud") as HTMLDivElement;

function setProgress(frac: number): void {
  progressEl.style.width = `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  if (frac >= 1) setTimeout(() => (progressEl.style.width = "0"), 600);
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.xr.enabled = true;
app.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
const spark = new SparkRenderer({ renderer });
scene.add(spark);

const camera = new THREE.PerspectiveCamera(70, 1, 0.05, 2000);
camera.position.set(0, 0, 0.01);
const controls = new FreeLookControls(camera, renderer.domElement);

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  controls.update(clock.getDelta(), renderer.xr.isPresenting);
  renderer.render(scene, camera);
});

interface SplatLike extends THREE.Object3D {
  initialized?: Promise<unknown>;
}

async function loadSplatUrl(url: string, name: string): Promise<void> {
  setProgress(0.15);
  const mesh = new SplatMesh({ url }) as unknown as SplatLike;
  mesh.quaternion.set(1, 0, 0, 0); // Spark uses a flipped convention vs. our capture frame
  scene.add(mesh);
  if (mesh.initialized) await mesh.initialized;
  hud.textContent = `AdamWorld — ${name}`;
  setProgress(1);
}

/** Load a world by id via the API (tries .spz then .ply). */
async function loadWorld(id: string): Promise<void> {
  const res = await fetch(`/worlds/${id}`);
  if (!res.ok) throw new Error(`world ${id} not found`);
  const data = (await res.json()) as { meta: { name: string }; assets: Record<string, string> };
  const url = data.assets["splats.spz"] ?? data.assets["splats.ply"];
  if (!url) throw new Error("world has no splats yet");
  await loadSplatUrl(url, data.meta.name);
}

const params = new URLSearchParams(window.location.search);
const src = params.get("src");
const worldId = params.get("world");
if (src) {
  void loadSplatUrl(src, src.split("/").pop() ?? "splats").catch((e) => (hud.textContent = String(e)));
} else if (worldId) {
  void loadWorld(worldId).catch((e) => (hud.textContent = String(e)));
} else {
  hud.textContent = "AdamWorld — pass ?world=<id> or ?src=<url-to-splats>";
}
