import * as THREE from "three";

/**
 * Lightweight free-look navigation: click/drag or pointer-lock to look (mouse),
 * one-finger drag to look (touch), WASD/arrows to move, Shift to sprint,
 * Space/C for up/down. Disabled automatically while an immersive XR session is
 * presenting (the headset drives the camera then).
 */
export class FreeLookControls {
  private yaw = 0;
  private pitch = 0;
  private readonly keys = new Set<string>();
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private readonly speed = 2.5;
  private readonly lookSpeed = 0.0025;

  constructor(
    private readonly camera: THREE.PerspectiveCamera,
    dom: HTMLElement,
  ) {
    const e = camera.rotation;
    this.yaw = e.y;
    this.pitch = e.x;

    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    dom.style.touchAction = "none";
  }

  private onPointerDown = (ev: PointerEvent): void => {
    this.dragging = true;
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
  };

  private onPointerUp = (): void => {
    this.dragging = false;
  };

  private onPointerMove = (ev: PointerEvent): void => {
    if (!this.dragging) return;
    this.yaw -= (ev.clientX - this.lastX) * this.lookSpeed;
    this.pitch -= (ev.clientY - this.lastY) * this.lookSpeed;
    this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
    this.lastX = ev.clientX;
    this.lastY = ev.clientY;
  };

  private onKeyDown = (ev: KeyboardEvent): void => {
    this.keys.add(ev.key.toLowerCase());
  };

  private onKeyUp = (ev: KeyboardEvent): void => {
    this.keys.delete(ev.key.toLowerCase());
  };

  update(dt: number, presenting: boolean): void {
    if (presenting) return;
    this.camera.rotation.set(this.pitch, this.yaw, 0, "YXZ");

    const move = new THREE.Vector3();
    if (this.keys.has("w") || this.keys.has("arrowup")) move.z -= 1;
    if (this.keys.has("s") || this.keys.has("arrowdown")) move.z += 1;
    if (this.keys.has("a") || this.keys.has("arrowleft")) move.x -= 1;
    if (this.keys.has("d") || this.keys.has("arrowright")) move.x += 1;
    if (this.keys.has(" ")) move.y += 1;
    if (this.keys.has("c")) move.y -= 1;

    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(this.speed * dt * (this.keys.has("shift") ? 3 : 1));
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
      this.camera.position
        .addScaledVector(right, move.x)
        .addScaledVector(forward, -move.z)
        .add(new THREE.Vector3(0, move.y, 0));
    }
  }
}
