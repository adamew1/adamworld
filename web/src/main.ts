import * as THREE from "three";

// M0 scaffold: a minimal three.js scene that proves the toolchain builds and
// renders with no GPU. M2 replaces the placeholder grid with a Spark SplatMesh
// loaded from a generated world, plus WASD/mouse/touch navigation and WebXR.

const app = document.getElementById("app") as HTMLDivElement;
const progressEl = document.getElementById("progress") as HTMLDivElement;
const hud = document.getElementById("hud") as HTMLDivElement;

export function setProgress(frac: number): void {
  progressEl.style.width = `${Math.round(frac * 100)}%`;
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
scene.add(new THREE.GridHelper(20, 20, 0x334155, 0x1e293b));
scene.add(new THREE.AmbientLight(0xffffff, 0.8));

const camera = new THREE.PerspectiveCamera(70, 1, 0.1, 1000);
camera.position.set(0, 1.6, 4);
camera.lookAt(0, 1, 0);

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

let t = 0;
function animate(): void {
  requestAnimationFrame(animate);
  t += 0.003;
  camera.position.x = Math.sin(t) * 4;
  camera.position.z = Math.cos(t) * 4;
  camera.lookAt(0, 1, 0);
  renderer.render(scene, camera);
}
animate();

/** Fetch a generated world's metadata (splat loading via Spark lands in M2). */
export async function loadWorld(id: string): Promise<void> {
  setProgress(0.1);
  const res = await fetch(`/worlds/${id}`);
  if (!res.ok) throw new Error(`world ${id} not found`);
  const data = (await res.json()) as { meta: { name: string }; assets: Record<string, string> };
  hud.textContent = `AdamWorld — ${data.meta.name}`;
  setProgress(1);
}

const params = new URLSearchParams(window.location.search);
const worldId = params.get("world");
if (worldId) void loadWorld(worldId);
