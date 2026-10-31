import Cube from "cubejs";

export { Cube };
export const SIZE_MM = 56;
export const FACE_ORDER = ["U", "R", "F", "D", "L", "B"];
// Right-handed cube frame, shared with ../config/default.json.
// F = -Y makes standard Singmaster R/U/F handedness agree with cube.js.
export const FACES = Object.freeze({
  U: {
    normal: [0, 0, 1],
    right: [1, 0, 0],
    up: [0, 1, 0],
    color: "#f5be00",
    colorName: "黄",
    axis: "+Z",
  },
  R: {
    normal: [1, 0, 0],
    right: [0, 1, 0],
    up: [0, 0, 1],
    color: "#008cd7",
    colorName: "蓝",
    axis: "+X",
  },
  F: {
    normal: [0, -1, 0],
    right: [1, 0, 0],
    up: [0, 0, 1],
    color: "#7d1487",
    colorName: "紫",
    axis: "−Y",
  },
  D: {
    normal: [0, 0, -1],
    right: [1, 0, 0],
    up: [0, -1, 0],
    color: "#ebf0e1",
    colorName: "白",
    axis: "−Z",
  },
  L: {
    normal: [-1, 0, 0],
    right: [0, -1, 0],
    up: [0, 0, 1],
    color: "#9be600",
    colorName: "绿",
    axis: "−X",
  },
  B: {
    normal: [0, 1, 0],
    right: [-1, 0, 0],
    up: [0, 0, 1],
    color: "#f00a23",
    colorName: "红",
    axis: "+Y",
  },
});
export const SOLVED = FACE_ORDER.map((face) => face.repeat(9)).join("");

export function parseMoves(algorithm) {
  const tokens =
    typeof algorithm === "string"
      ? algorithm.trim().split(/\s+/).filter(Boolean)
      : algorithm;
  if (!Array.isArray(tokens)) throw new Error("步骤必须为字符串或字符串数组");
  return tokens.map((token) => {
    if (typeof token !== "string" || !/^[URFDLB](2|')?$/.test(token)) {
      throw new Error(
        `不支持动作 ${token}；只允许 U/R/F/D/L/B 的 90°、逆 90°、180° 外层转动`,
      );
    }
    const suffix = token.slice(1);
    return {
      token,
      face: token[0],
      command:
        suffix === "2" ? "HALF_180" : suffix === "'" ? "CCW_90" : "CW_90",
      // Standard clockwise is viewed from outside the selected face.
      angleDeg: suffix === "2" ? 180 : suffix === "'" ? 90 : -90,
    };
  });
}

function parity(permutation) {
  let result = 0;
  for (let i = 0; i < permutation.length; i++) {
    for (let j = i + 1; j < permutation.length; j++)
      result ^= Number(permutation[i] > permutation[j]);
  }
  return result;
}

function isPermutation(values, length) {
  return (
    Array.isArray(values) &&
    values.length === length &&
    values.every((v) => Number.isInteger(v) && v >= 0 && v < length) &&
    new Set(values).size === length
  );
}

export function cubeFromFacelets(raw) {
  if (typeof raw !== "string") throw new Error("状态须为 54 字符面片字符串");
  const facelets = raw.replace(/\s/g, "").toUpperCase();
  if (facelets.length !== 54 || /[^URFDLB]/.test(facelets)) {
    throw new Error("请输入 54 个 U/R/F/D/L/B 字符，按 URFDLB 六面顺序排列");
  }
  for (const [i, face] of FACE_ORDER.entries()) {
    if ([...facelets].filter((c) => c === face).length !== 9)
      throw new Error(`面色 ${face} 必须恰好出现 9 次`);
    if (facelets[i * 9 + 4] !== face)
      throw new Error(
        `第 ${face} 面的中心必须为 ${face}，请先按中心颜色命名各面`,
      );
  }
  const cube = Cube.fromString(facelets);
  const { cp, co, ep, eo } = cube.toJSON();
  if (
    !isPermutation(cp, 8) ||
    !isPermutation(ep, 12) ||
    !co.every((v) => Number.isInteger(v) && v >= 0 && v < 3) ||
    !eo.every((v) => v === 0 || v === 1) ||
    cube.asString() !== facelets
  ) {
    throw new Error("角块或棱块颜色组合不合法，或存在重复/缺失块");
  }
  if (co.reduce((a, b) => a + b, 0) % 3)
    throw new Error("角块扭转总和不合法（mod 3 必须为 0）");
  if (eo.reduce((a, b) => a + b, 0) % 2)
    throw new Error("棱块翻转总和不合法（mod 2 必须为 0）");
  if (parity(cp) !== parity(ep))
    throw new Error("角块与棱块置换奇偶性不一致，此状态不能靠外层转动还原");
  return cube;
}

export function faceletSlots() {
  return FACE_ORDER.flatMap((face, faceIndex) => {
    const { normal, right, up } = FACES[face];
    return Array.from({ length: 9 }, (_, index) => {
      const row = Math.floor(index / 3),
        col = index % 3;
      return {
        index: faceIndex * 9 + index,
        face,
        row,
        col,
        normal,
        position: normal.map(
          (v, k) => v + right[k] * (col - 1) + up[k] * (1 - row),
        ),
      };
    });
  });
}

export const COMMAND_LABELS = {
  CW_90: "顶层顺时针 90°",
  CCW_90: "顶层逆时针 90°",
  HALF_180: "顶层转动 180°",
};
