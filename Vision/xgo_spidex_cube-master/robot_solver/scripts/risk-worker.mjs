import { parentPort } from 'node:worker_threads';
import { solveRisk } from '../src/risk-solver.js';
import { computeV2Fingerprint } from './v2-fingerprint.mjs';

if (!parentPort) throw new Error('risk worker requires parentPort');
const sourceFingerprint = await computeV2Fingerprint();

const emit = (id, type, plan) => {
  if (plan) plan.sourceFingerprint = sourceFingerprint;
  parentPort.postMessage({ type, id, ...(plan ? { plan } : {}) });
};

parentPort.on('message', payload => {
  try {
    const plan = solveRisk(payload.facelets, payload.options || {}, next => emit(payload.id, 'plan', next));
    emit(payload.id, 'done', plan);
  } catch (error) {
    parentPort.postMessage({ type: 'error', id: payload.id, message: error?.message || String(error), code: error?.code });
  }
});
