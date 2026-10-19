import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";

// M5: a block-out layout editor. Place/move/scale boxes, planes, and cylinders,
// label each, pick a style prompt, then "Generate world" runs stage 8. The block
// transforms are exported in the exact column-major form the server rasterizer
// expects (three.js Matrix4.elements).

interface Block {
  mesh: THREE.Mesh;
  shape: "box" | "plane" | "cylinder";
  label: string;
}

const app = document.getElementById("app") as HTMLDivElement;
const statusEl = document.getElementById("status") as HTMLDivElement;
const labelInput = document.getElementById("label") as HTMLInputElement;
const styleInput = document.getElementById("style") as HTMLInputElement;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
scene.add(new THREE.GridHelper(40, 40, 0x334155, 0x1e293b));
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const sun = new THREE.DirectionalLight(0xffffff, 0.8);
sun.position.set(5, 10, 7);
scene.add(sun);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.position.set(8, 7, 10);

const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0);

const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.addEventListener("dragging-changed", (e) => (orbit.enabled = !e.value));
scene.add(gizmo.getHelper());

const blocks: Block[] = [];
let selected: Block | null = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let nextId = 0;

const COLORS = [0xc08457, 0x5a8fd0, 0x6fae6f, 0xd07a7a, 0xc9b24a];

function addPrimitive(shape: Block["shape"]): void {
  let geom: THREE.BufferGeometry;
  if (shape === "box") geom = new THREE.BoxGeometry(1, 1, 1);
  else if (shape === "plane") geom = new THREE.PlaneGeometry(1, 1);
  else geom = new THREE.CylinderGeometry(0.5, 0.5, 1, 24);
  const mat = new THREE.MeshStandardMaterial({
    color: COLORS[blocks.length % COLORS.length],
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(0, shape === "plane" ? 0.01 : 0.5, 0);
  if (shape === "box") mesh.scale.set(2, 1, 2);
  if (shape === "cylinder") mesh.scale.set(1, 2, 1);
  scene.add(mesh);
  const block: Block = { mesh, shape, label: labelInput.value || shape };
  blocks.push(block);
  select(block);
  updateStatus();
}

function select(block: Block | null): void {
  selected = block;
  if (block) {
    gizmo.attach(block.mesh);
    labelInput.value = block.label;
  } else {
    gizmo.detach();
  }
}

function deleteSelected(): void {
  if (!selected) return;
  gizmo.detach();
  scene.remove(selected.mesh);
  blocks.splice(blocks.indexOf(selected), 1);
  selected = null;
  updateStatus();
}

function updateStatus(msg?: string): void {
  statusEl.textContent = msg ?? `${blocks.length} block${blocks.length === 1 ? "" : "s"}`;
}

function exportLayout(): unknown {
  return {
    style_prompt: styleInput.value,
    blocks: blocks.map((b) => {
      b.mesh.updateMatrix();
      return {
        id: `b${nextId++}`,
        shape: b.shape,
        label: b.label,
        transform: Array.from(b.mesh.matrix.elements),
      };
    }),
  };
}

async function generate(): Promise<void> {
  if (!blocks.length) return updateStatus("add a block first");
  updateStatus("generating…");
  const res = await fetch("/worlds", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layout: exportLayout(), backend: "mock", resolution: 384 }),
  });
  const { job_id, world_id } = (await res.json()) as { job_id: string; world_id: string };
  for (;;) {
    await new Promise((r) => setTimeout(r, 400));
    const job = (await (await fetch(`/jobs/${job_id}`)).json()) as { status: string };
    if (job.status === "done") break;
    if (job.status === "error") return updateStatus("generation failed");
  }
  window.location.href = `/index.html?world=${world_id}`;
}

// --- UI wiring ---
document.querySelectorAll<HTMLButtonElement>("[data-add]").forEach((b) =>
  b.addEventListener("click", () => addPrimitive(b.dataset.add as Block["shape"])),
);
document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
  b.addEventListener("click", () =>
    gizmo.setMode(b.dataset.mode as "translate" | "rotate" | "scale"),
  ),
);
(document.getElementById("del") as HTMLButtonElement).addEventListener("click", deleteSelected);
(document.getElementById("gen") as HTMLButtonElement).addEventListener("click", () => void generate());
labelInput.addEventListener("input", () => {
  if (selected) selected.label = labelInput.value;
});

renderer.domElement.addEventListener("pointerdown", (ev) => {
  if (gizmo.dragging) return;
  pointer.set((ev.clientX / window.innerWidth) * 2 - 1, -(ev.clientY / window.innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(blocks.map((b) => b.mesh), false)[0];
  if (hit) select(blocks.find((b) => b.mesh === hit.object) ?? null);
});

function resize(): void {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

renderer.setAnimationLoop(() => {
  orbit.update();
  renderer.render(scene, camera);
});

// Exposed for automated verification.
(window as unknown as { __blocks: () => number; __exportLayout: () => unknown }).__blocks = () =>
  blocks.length;
(window as unknown as { __exportLayout: () => unknown }).__exportLayout = exportLayout;
