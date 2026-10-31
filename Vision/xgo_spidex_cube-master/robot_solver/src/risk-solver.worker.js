import { solveRisk } from './risk-solver.js';

export const run = (payload, send) => {
  try {
    const emit = (plan, type = 'plan') => {
      // Browser bundles are traced at the production artifact level; never
      // pretend that a runtime source digest was computed here.
      plan.sourceFingerprint = null;
      send({ type, id: payload.id, plan });
    };
    const plan = solveRisk(payload.facelets, payload.options || {}, emit);
    emit(plan, 'done');
  } catch (error) {
    send({ type: 'error', id: payload.id, message: error?.message || String(error), code: error?.code });
  }
};

if (typeof self !== 'undefined' && typeof self.postMessage === 'function') {
  self.onmessage = event => run(event.data, message => self.postMessage(message));
}
