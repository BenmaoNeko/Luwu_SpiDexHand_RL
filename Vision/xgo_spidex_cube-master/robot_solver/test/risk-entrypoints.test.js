import test from 'node:test';
import assert from 'node:assert/strict';
import { Cube, SOLVED } from '../src/cube-model.js';

test('risk worker entrypoints expose a v2 plan and callback stream', async () => {
  const { solveRisk } = await import('../src/risk-worker-client.js');
  const seen = [];
  const result = await solveRisk(new Cube().move('R U').asString(), {
    initialQuaternion: [0, 0, 0, 1],
    calibrationId: 'cal-test',
    modelProfileId: 'preview',
    centerIdentityMap: { white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' },
    mode: 'ideal_preview',
    costMode: 'proxy',
    maxSolverCalls: 4,
    maxCandidates: 4,
    maxPrefixDepth: 2,
    beamWidth: 2,
  }, plan => seen.push(plan));
  assert.equal(result.schemaVersion, 2);
  assert.ok(seen.length >= 1);
  assert.ok(seen.every(plan => plan.schemaVersion === 2));
});

test('risk worker client errors when timeout occurs before any plan', async () => {
  const { solveRisk } = await import('../src/risk-worker-client.js');
  await assert.rejects(() => solveRisk(SOLVED, {
    initialQuaternion: [0, 0, 0, 1], calibrationId: 'cal-test', modelProfileId: 'preview',
    centerIdentityMap: { white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' },
    maxTimeMs: 1,
  }), /timed out/);
});

test('risk worker client cancellation settles before a delayed worker is created', async () => {
  const { createRiskWorkerRequest } = await import('../src/risk-worker-client.js');
  let posted = 0;
  const worker = {
    on() {},
    postMessage() { posted += 1; },
    terminate() { this.terminated = true; },
  };
  const request = createRiskWorkerRequest({
    facelets: SOLVED,
    workerFactory: () => new Promise(resolve => setTimeout(() => resolve(worker), 20)),
    timeBudgetMs: 1000,
  });
  request.cancel();
  await assert.rejects(request.promise, error => error.code === 'CANCELLED');
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(posted, 0);
  assert.equal(worker.terminated, true);
});

test('risk worker client ignores late done after cancellation', async () => {
  const { createRiskWorkerRequest } = await import('../src/risk-worker-client.js');
  const listeners = new Map();
  const worker = {
    on(type, callback) { listeners.set(type, callback); },
    postMessage() { setTimeout(() => listeners.get('message')?.({ type: 'done', id: 'late', plan: {} }), 10); },
    terminate() { this.terminated = true; },
  };
  const request = createRiskWorkerRequest({ facelets: SOLVED, id: 'late', workerFactory: () => worker, timeBudgetMs: 1000 });
  await new Promise(resolve => setTimeout(resolve, 0));
  request.cancel();
  await assert.rejects(request.promise, error => error.code === 'CANCELLED');
  await new Promise(resolve => setTimeout(resolve, 25));
  assert.equal(worker.terminated, true);
});

test('risk worker rejects illegal numeric search budgets', async () => {
  const { solveRisk } = await import('../src/risk-worker-client.js');
  await assert.rejects(() => solveRisk(SOLVED, {
    initialQuaternion: [0, 0, 0, 1], calibrationId: 'cal-test', modelProfileId: 'preview',
    centerIdentityMap: { white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' },
    maxSolverCalls: 0,
  }), /Invalid search budget maxSolverCalls/);
});

test('browser worker protocol emits terminal done after plan callbacks', async () => {
  const { run } = await import('../src/risk-solver.worker.js');
  const messages = [];
  run({ id: 'browser', facelets: SOLVED, options: {
    initialQuaternion: [0, 0, 0, 1], calibrationId: 'cal-test', modelProfileId: 'preview',
    centerIdentityMap: { white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' },
    maxSolverCalls: 1, maxCandidates: 1, maxPrefixDepth: 0, beamWidth: 1,
  } }, message => messages.push(message));
  assert.equal(messages.at(-1).type, 'done');
  assert.equal(messages.at(-1).plan.sourceFingerprint, null);
});
