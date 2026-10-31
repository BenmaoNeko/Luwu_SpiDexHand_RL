import { Cube, cubeFromFacelets, parseMoves } from "./cube-model.js";

let initialized = false;
export function initializeSolver() {
  if (!initialized) {
    Cube.initSolver();
    initialized = true;
  }
}

export function solveFacelets(facelets) {
  const cube = cubeFromFacelets(facelets);
  if (cube.isSolved()) return "";
  // cubejs' two-phase solver returns a valid solution, but its first valid
  // result is not guaranteed to be the shortest result for trivial states.
  // Check every legal one-face move first so a cube one turn from solved is
  // represented as exactly that one turn.
  for (const token of [
    "U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2",
    "D", "D'", "D2", "L", "L'", "L2", "B", "B'", "B2",
  ]) {
    if (cube.clone().move(token).isSolved()) return token;
  }
  initializeSolver();
  const algorithm = cube.solve();
  const moves = parseMoves(algorithm);
  if (
    !cube
      .clone()
      .move(moves.map((m) => m.token).join(" "))
      .isSolved()
  )
    throw new Error("求解器返回序列校验失败");
  return moves.map((m) => m.token).join(" ");
}
