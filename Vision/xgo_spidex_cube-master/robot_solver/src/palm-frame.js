import { FACES } from './cube-model.js';
// Fixed right-handed palm frame. Matrices are flat row-major 3x3 arrays.
export const A_MATRIX = Object.freeze([1, 0, 0, 0, 0, -1, 0, 1, 0]);
// Physical direction correction: reverse the previous +Y quarter turn.
export const B_MATRIX = Object.freeze([0, 0, -1, 0, 1, 0, 1, 0, 0]);
export const R0_MATRIX = Object.freeze([1, 0, 0, 0, -1, 0, 0, 0, -1]);

const EPS = 1e-10;
const key = (m) => m.map((x) => Math.round(x)).join(',');
const rad = (d) => d * Math.PI / 180;

export function multiply(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 9 || b.length !== 9) throw new TypeError('matrix must have 9 entries');
  const out = Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
  return out.map((x) => Math.abs(x) < EPS ? 0 : x);
}

export function apply(m, v) {
  if (!Array.isArray(m) || m.length !== 9 || !Array.isArray(v) || v.length !== 3) throw new TypeError('matrix/vector shape');
  return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]].map((x) => Math.abs(x) < EPS ? 0 : x);
}

export function quaternionFromMatrix(m) {
  if (!Array.isArray(m) || m.length !== 9) throw new TypeError('matrix must have 9 entries');
  const tr = m[0] + m[4] + m[8]; let x; let y; let z; let w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = .25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s; }
  else if (m[0] > m[4] && m[0] > m[8]) { const s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; w = (m[7] - m[5]) / s; x = .25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
  else if (m[4] > m[8]) { const s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = .25 * s; z = (m[5] + m[7]) / s; }
  else { const s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = .25 * s; }
  const n = Math.hypot(x, y, z, w); return [x / n, y / n, z / n, w / n];
}
export const matrixToQuaternion = quaternionFromMatrix;
export function matrixFromQuaternion(q) {
  if (!Array.isArray(q) || q.length !== 4 || q.some((x) => !Number.isFinite(x)) || Math.hypot(...q) < EPS) throw new RangeError('quaternion must be finite and non-zero');
  const [x0, y0, z0, w0] = q; const n = Math.hypot(x0, y0, z0, w0); const x = x0 / n; const y = y0 / n; const z = z0 / n; const w = w0 / n;
  return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
}
export const quaternionToMatrix = matrixFromQuaternion;

export function quaternionDistanceDeg(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== 4 || b.length !== 4) throw new TypeError('quaternion must have 4 entries');
  const na = Math.hypot(...a); const nb = Math.hypot(...b); if (!Number.isFinite(na) || !Number.isFinite(nb) || na < EPS || nb < EPS) throw new RangeError('quaternion must be finite and non-zero');
  const dot = Math.min(1, Math.abs(a.reduce((s, x, i) => s + x * b[i], 0) / (na * nb)));
  return 2 * Math.acos(dot) * 180 / Math.PI;
}

const byMatrix = new Map();
const nodes = [{ id: 'o0', matrix: [...R0_MATRIX] }]; byMatrix.set(key(R0_MATRIX), nodes[0]);
for (let i = 0; i < nodes.length; i++) for (const [name, m] of [['A', A_MATRIX], ['B', B_MATRIX]]) {
  const next = multiply(m, nodes[i].matrix); const k = key(next);
  if (!byMatrix.has(k)) { const n = { id: `o${nodes.length}`, matrix: next }; byMatrix.set(k, n); nodes.push(n); }
}
export const orientations = Object.freeze(nodes.map((o) => Object.freeze({ id: o.id, matrix: Object.freeze([...o.matrix]), quaternion: Object.freeze(quaternionFromMatrix(o.matrix)) })));
const orientationById = new Map(orientations.map((o) => [o.id, o]));
export const referenceOrientation = orientations[0];
export const ORIENTATIONS = orientations;
export const REFERENCE_ORIENTATION_ID = referenceOrientation.id;
export const DEMO_CENTER_IDENTITY_MAP = Object.freeze({ white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' });

export function flipOrientation(id, primitive) {
  const node = orientationById.get(id); if (!node) throw new RangeError(`unknown orientation: ${id}`);
  const p = primitive === 'A' || primitive === 'flip_A' ? A_MATRIX : primitive === 'B' || primitive === 'flip_B' ? B_MATRIX : null;
  if (!p) throw new RangeError(`unknown flip primitive: ${primitive}`);
  const found = byMatrix.get(key(multiply(p, node.matrix))); if (!found) throw new Error('orientation graph is incomplete'); return found.id;
}

const faceVectors = Object.freeze(Object.fromEntries(Object.entries(FACES).map(([face, data]) => [face, [...data.normal]])));
export function normalInPalm(id, face) { const o = orientationById.get(id); if (!o || !faceVectors[face]) throw new RangeError('unknown orientation or face'); return apply(o.matrix, faceVectors[face]); }

export function classifyOrientation(quaternion, options = {}) {
  const maxResidualDeg = typeof options === 'number' ? options : (options.maxResidualDeg ?? options.maxErrorDeg ?? 15);
  if (!Array.isArray(quaternion) || quaternion.length !== 4 || quaternion.some((x) => !Number.isFinite(x)) || Math.hypot(...quaternion) < EPS) throw new RangeError('quaternion must be finite and non-zero');
  let best = null; for (const o of orientations) { const d = quaternionDistanceDeg(quaternion, o.quaternion); if (!best || d < best.residualDeg) best = { orientationId: o.id, residualDeg: d, errorDeg: d, quaternion: [...quaternion], ok: d <= maxResidualDeg }; }
  if (!best.ok) { const error = new Error('POSE_OFF_LATTICE'); error.code = 'POSE_OFF_LATTICE'; throw error; }
  return best;
}

const colors = ['white', 'yellow', 'blue', 'green', 'purple', 'red'];
const opposites = { white: 'yellow', yellow: 'white', blue: 'green', green: 'blue', purple: 'red', red: 'purple' };
const solverVec = faceVectors;
export function validateCenterIdentityMap(map) {
  if (!map || typeof map !== 'object') return { ok: false, reason: 'center map must be a six-face bijection' };
  const normalized = { ...map, purple: map.purple ?? map.orange };
  if (colors.some((c) => !Object.hasOwn(normalized, c)) || new Set(colors.map((c) => normalized[c])).size !== 6 || colors.some((c) => !faceVectors[normalized[c]])) return { ok: false, reason: 'center map must be a six-face bijection' };
  const oppositeFace = { U: 'D', D: 'U', R: 'L', L: 'R', F: 'B', B: 'F' };
  if (colors.some((c) => normalized[opposites[c]] !== oppositeFace[normalized[c]])) return { ok: false, reason: 'opposite centers mismatch' };
  const cols = ['blue', 'red', 'yellow'].map((c) => solverVec[normalized[c]]); const det = cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1]) - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1]) + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1]);
  return det === 1 ? { ok: true } : { ok: false, reason: 'center map is mirrored' };
}

export { rad };
