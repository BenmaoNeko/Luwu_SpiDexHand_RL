import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { FACES, SIZE_MM, faceletSlots } from "./cube-model.js";
import { quaternion, reorientationPose, supportHeight } from "./planner.js";

const CELL = SIZE_MM / 3;
const Z = new THREE.Vector3(0, 0, 1);

function labelMaterial(text, color = "#334155") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.font = "bold 60px system-ui";
  ctx.fillStyle = color;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.SpriteMaterial({ map: texture, depthWrite: false });
}

export class CubeScene {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#edf2f6");
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute(
      "aria-label",
      "魔方三维视图，可拖动观察",
    );
    container.append(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(38, 1, 1, 1500);
    this.camera.up.copy(Z);
    this.camera.position.set(133, 176, 142);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0, 26);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 125;
    this.controls.maxDistance = 420;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.update();

    const ambient = new THREE.HemisphereLight(0xffffff, 0x758496, 2.7);
    ambient.position.set(0, 0, 1);
    this.scene.add(ambient);
    const light = new THREE.DirectionalLight(0xffffff, 3.2);
    light.position.set(70, -50, 170);
    light.castShadow = true;
    light.shadow.mapSize.set(2048, 2048);
    Object.assign(light.shadow.camera, {
      left: -100,
      right: 100,
      top: 100,
      bottom: -100,
      near: 1,
      far: 350,
    });
    light.shadow.normalBias = 0.3;
    this.scene.add(light);
    const fill = new THREE.DirectionalLight(0xdcefff, 1.1);
    fill.position.set(-90, 80, 80);
    this.scene.add(fill);
    const table = new THREE.Mesh(
      new THREE.PlaneGeometry(900, 900),
      new THREE.MeshStandardMaterial({ color: "#e7edf3", roughness: 1 }),
    );
    table.receiveShadow = true;
    table.position.z = -0.15;
    this.scene.add(table);
    const grid = new THREE.GridHelper(440, 44, "#c5d2df", "#d3dde6");
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.1;
    this.scene.add(grid);
    const station = new THREE.Mesh(
      new THREE.RingGeometry(48, 48.35, 96),
      new THREE.MeshBasicMaterial({ color: "#adc6da", side: THREE.DoubleSide }),
    );
    station.position.z = -0.05;
    this.scene.add(station);
    for (const [dir, color, text] of [
      [[1, 0, 0], 0xe35a60, "X"],
      [[0, 1, 0], 0x3c9a80, "Y"],
      [[0, 0, 1], 0x357ab8, "Z"],
    ]) {
      const origin = new THREE.Vector3(-62, -50, 0.3);
      this.scene.add(
        new THREE.ArrowHelper(
          new THREE.Vector3(...dir),
          origin,
          19,
          color,
          3,
          1.8,
        ),
      );
      const sprite = new THREE.Sprite(
        labelMaterial(text, `#${color.toString(16)}`),
      );
      sprite.position
        .copy(origin)
        .addScaledVector(new THREE.Vector3(...dir), 24);
      sprite.scale.set(11, 5.5, 1);
      this.scene.add(sprite);
    }
    const up = new THREE.ArrowHelper(
      Z,
      new THREE.Vector3(0, 0, 83),
      16,
      0x347eaf,
      4,
      2,
    );
    this.scene.add(up);
    const upText = new THREE.Sprite(labelMaterial("+Z", "#357ab8"));
    upText.position.set(0, 0, 106);
    upText.scale.set(20, 10, 1);
    this.scene.add(upText);

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.cubieGeometry = new RoundedBoxGeometry(
      CELL - 0.35,
      CELL - 0.35,
      CELL - 0.35,
      3,
      1,
    );
    this.stickerGeometry = new RoundedBoxGeometry(
      CELL - 1.25,
      CELL - 1.25,
      0.25,
      3,
      0.11,
    );
    this.plastic = new THREE.MeshStandardMaterial({
      color: "#141e2b",
      roughness: 0.36,
      metalness: 0.04,
    });
    this.stickerMaterials = Object.fromEntries(
      Object.entries(FACES).map(([face, spec]) => [
        face,
        new THREE.MeshPhysicalMaterial({
          color: spec.color,
          roughness: 0.38,
          metalness: 0,
          clearcoat: 0.12,
        }),
      ]),
    );
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.animation = null;
    this.tick = this.tick.bind(this);
    requestAnimationFrame(this.tick);
    this.resize();
  }

  resize() {
    const { width, height } = this.container.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  setState(facelets, orientation) {
    this.root.clear();
    this.root.quaternion.copy(quaternion(orientation));
    this.root.position.set(0, 0, supportHeight(orientation));
    const cubies = new Map();
    for (let x = -1; x <= 1; x++)
      for (let y = -1; y <= 1; y++)
        for (let z = -1; z <= 1; z++) {
          if (x === 0 && y === 0 && z === 0) continue;
          const group = new THREE.Group();
          group.position.set(x * CELL, y * CELL, z * CELL);
          group.userData.position = [x, y, z];
          const body = new THREE.Mesh(this.cubieGeometry, this.plastic);
          body.castShadow = true;
          body.receiveShadow = true;
          group.add(body);
          cubies.set([x, y, z].join(","), group);
          this.root.add(group);
        }
    for (const slot of faceletSlots()) {
      const sticker = new THREE.Mesh(
        this.stickerGeometry,
        this.stickerMaterials[facelets[slot.index]],
      );
      const normal = new THREE.Vector3(...slot.normal);
      sticker.quaternion.setFromUnitVectors(Z, normal);
      sticker.position.copy(normal).multiplyScalar(CELL / 2 - 0.06);
      sticker.castShadow = true;
      cubies.get(slot.position.join(",")).add(sticker);
    }
  }

  animateAction(action, durationMs, onProgress = () => {}) {
    if (this.animation) return Promise.reject(new Error("上一动作尚未完成"));
    let pivot = null;
    if (action.type === "turn_top") {
      pivot = new THREE.Group();
      const n = new THREE.Vector3(...FACES[action.targetFace].normal);
      const selected = this.root.children.filter(
        (child) =>
          child.userData.position &&
          new THREE.Vector3(...child.userData.position).dot(n) > 0.5,
      );
      if (selected.length !== 9)
        return Promise.reject(new Error("顶层选择必须恰好包含 9 个小块"));
      this.root.add(pivot);
      selected.forEach((child) => pivot.add(child));
    }
    return new Promise((resolve) => {
      this.animation = {
        action,
        pivot,
        durationMs: Math.max(1, durationMs),
        start: performance.now(),
        resolve,
        onProgress,
      };
    });
  }

  tick(now) {
    if (this.animation) {
      const a = this.animation;
      const t = Math.min(1, Math.max(0, (now - a.start) / a.durationMs));
      if (a.action.type === "reorient") {
        const pose = reorientationPose(a.action, t);
        this.root.quaternion.copy(quaternion(pose.quaternion));
        this.root.position.fromArray(pose.centerMm);
      } else {
        const ease = t * t * (3 - 2 * t);
        a.pivot.quaternion.setFromAxisAngle(
          new THREE.Vector3(...FACES[a.action.targetFace].normal),
          ((a.action.angleDeg * Math.PI) / 180) * ease,
        );
      }
      a.onProgress(t);
      if (t === 1) {
        this.animation = null;
        a.resolve();
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.tick);
  }
}
