import { initializeSolver, solveFacelets } from "./solver-core.js";

self.onmessage = ({ data }) => {
  try {
    const algorithm = solveFacelets(data.facelets);
    self.postMessage({ type: "solution", id: data.id, algorithm });
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, message: error.message });
  }
};
try {
  initializeSolver();
  self.postMessage({ type: "ready" });
} catch (error) {
  self.postMessage({
    type: "error",
    message: `求解器初始化失败：${error.message}`,
  });
}
