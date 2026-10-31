import { Matrix4, Quaternion, Vector3 } from "three";
import {
  FACES,
  FACE_ORDER,
  SIZE_MM,
  parseMoves,
  cubeFromFacelets,
} from "./cube-model.js";

const Z = new Vector3(0, 0, 1);
const radToDeg = 180 / Math.PI;

export function quaternion(value = [0, 0, 0, 1]) {
  if (
    !Array.isArray(value) ||
    value.length !== 4 ||
    !value.every(Number.isFinite)
  ) {
    throw new Error("姿态必须为有限数值四元数 [x,y,z,w]");
  }
  const q = new Quaternion(...value);
  if (q.lengthSq() < 1e-12) throw new Error("姿态四元数不能为零");
  return q.normalize();
}

export function yawOrientation(degrees) {
  if (!Number.isFinite(degrees)) throw new Error("偏航角必须为有限数值");
  return new Quaternion().setFromAxisAngle(Z, degrees / radToDeg).toArray();
}

export function rotationMatrix(qArray) {
  const e = new Matrix4().makeRotationFromQuaternion(
    quaternion(qArray),
  ).elements;
  return [
    [e[0], e[4], e[8]],
    [e[1], e[5], e[9]],
    [e[2], e[6], e[10]],
  ];
}

export function topFace(qArray) {
  const q = quaternion(qArray);
  return (
    FACE_ORDER.find(
      (face) =>
        new Vector3(...FACES[face].normal).applyQuaternion(q).dot(Z) > 1 - 1e-7,
    ) ?? null
  );
}

export function minimalAlignment(qArray, face) {
  if (!FACES[face]) throw new Error("未知目标面");
  const from = quaternion(qArray);
  const normal = new Vector3(...FACES[face].normal)
    .applyQuaternion(from)
    .normalize();
  const dot = Math.max(-1, Math.min(1, normal.dot(Z)));
  const angleRad = Math.acos(dot);
  // Three.js supplies the shortest vector-to-vector rotation, including the
  // opposite-vector case where any horizontal half-turn axis is optimal.
  const delta = new Quaternion().setFromUnitVectors(normal, Z);
  const to = delta.clone().multiply(from).normalize();
  const axis = new Vector3(delta.x, delta.y, delta.z);
  if (axis.lengthSq() > 1e-16) axis.normalize();
  else axis.set(1, 0, 0);
  return {
    fromQuaternion: from.toArray(),
    toQuaternion: to.toArray(),
    deltaQuaternion: delta.toArray(),
    axisWorld: axis.toArray(),
    sourceNormalWorld: normal.toArray(),
    angleDeg: angleRad * radToDeg,
  };
}

/** Minimum height of cube center over z=0; exact support function of a cube. */
export function supportHeight(qArray, sizeMm = SIZE_MM) {
  return (
    (sizeMm / 2) *
    rotationMatrix(qArray)[2].reduce((sum, value) => sum + Math.abs(value), 0)
  );
}

export function reorientationPose(action, progress, sizeMm = SIZE_MM) {
  const t = Math.max(0, Math.min(1, progress));
  const ease = t * t * (3 - 2 * t);
  const q = quaternion(action.fromQuaternion).slerp(
    quaternion(action.toQuaternion),
    ease,
  );
  const clearance = action.requiresMotion ? 8 * Math.sin(Math.PI * ease) : 0;
  return {
    quaternion: q.toArray(),
    centerMm: [0, 0, supportHeight(q.toArray(), sizeMm) + clearance],
  };
}

export function compileRobotPlan(
  facelets,
  algorithm,
  initialQuaternion = [0, 0, 0, 1],
) {
  const cube = cubeFromFacelets(facelets);
  const moves = parseMoves(algorithm);
  const proof = cube.clone().move(moves.map((move) => move.token).join(" "));
  if (!proof.isSolved())
    throw new Error("输入步骤不能还原当前魔方，拒绝生成机器人计划");
  let orientation = quaternion(initialQuaternion).toArray();
  const actions = [];
  const centerMm = [0, 0, SIZE_MM / 2];
  for (const [moveIndex, move] of moves.entries()) {
    const alignment = minimalAlignment(orientation, move.face);
    const common = { moveIndex, notation: move.token, targetFace: move.face };
    actions.push({
      index: actions.length,
      type: "reorient",
      ...common,
      ...alignment,
      requiresMotion: alignment.angleDeg > 1e-6,
      targetRotationMatrix: rotationMatrix(alignment.toQuaternion),
      targetCenterMm: centerMm.slice(),
      targetNormalWorld: [0, 0, 1],
    });
    orientation = alignment.toQuaternion;
    actions.push({
      index: actions.length,
      type: "turn_top",
      ...common,
      command: move.command,
      angleDeg: move.angleDeg,
      axisWorld: [0, 0, 1],
      quaternion: orientation.slice(),
      centerMm: centerMm.slice(),
    });
  }
  return {
    schemaVersion: 1,
    cubeSizeMm: SIZE_MM,
    stateFormat: "54 facelets, URFDLB, row-major, centers fixed",
    frameConvention:
      "p_world = R_world_cube * p_cube + t_mm; world Z up; table z=0",
    faceDefinitions: Object.fromEntries(
      FACE_ORDER.map((face) => [
        face,
        {
          axisCube: FACES[face].normal.slice(),
          colorName: FACES[face].colorName,
          color: FACES[face].color,
        },
      ]),
    ),
    costModel:
      "Sum of minimum whole-cube SO(3) rotation angles; free yaw; fixed solution sequence",
    initialFacelets: cube.asString(),
    initialQuaternion: quaternion(initialQuaternion).toArray(),
    solution: moves.map((move) => move.token),
    actions,
    statistics: {
      faceTurns: moves.length,
      robotSteps: actions.length,
      reorientations: actions.filter(
        (a) => a.type === "reorient" && a.requiresMotion,
      ).length,
      reorientationCostDeg: actions
        .filter((a) => a.type === "reorient")
        .reduce((sum, a) => sum + a.angleDeg, 0),
    },
    finalQuaternion: orientation,
  };
}

/** Run one abstract robot action, checking the current frame instead of trusting face labels. */
export function executeRobotAction(cube, qArray, action) {
  const q = quaternion(qArray);
  if (action.type === "reorient") {
    if (q.angleTo(quaternion(action.fromQuaternion)) > 1e-6)
      throw new Error("重定向起始位姿与当前位姿不一致");
    const target = quaternion(action.toQuaternion);
    if (topFace(target.toArray()) !== action.targetFace)
      throw new Error("重定向后目标面未对齐世界 +Z");
    return target.toArray();
  }
  if (action.type !== "turn_top")
    throw new Error("机器人只允许 reorient / turn_top");
  if (
    q.angleTo(quaternion(action.quaternion)) > 1e-6 ||
    topFace(q.toArray()) !== action.targetFace
  ) {
    throw new Error("当前最上层不是计划中的目标面");
  }
  const move = parseMoves(action.notation)[0];
  if (
    !move ||
    move.face !== action.targetFace ||
    move.command !== action.command ||
    move.angleDeg !== action.angleDeg
  ) {
    throw new Error("顶层动作指令不一致");
  }
  cube.move(move.token);
  return q.toArray();
}
