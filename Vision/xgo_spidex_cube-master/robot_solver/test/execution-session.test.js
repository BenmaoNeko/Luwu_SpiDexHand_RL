import test from 'node:test';
import assert from 'node:assert/strict';
import { Cube, SOLVED } from '../src/cube-model.js';
import { referenceOrientation, DEMO_CENTER_IDENTITY_MAP } from '../src/palm-frame.js';
import { compilePlanV2, validatePlanV2 } from '../src/robot-plan-v2.js';
import { ExecutionSession, SESSION_STATES } from '../src/execution-session.js';
import { createFakeAdapter } from '../src/execution-session.js';

function fixture() {
  const plan = compilePlanV2(new Cube().move('U').asString(), "U'", { planId: 'plan-1', initialQuaternion: referenceOrientation.quaternion,
    calibrationId: 'cal-1', centerIdentityMap: DEMO_CENTER_IDENTITY_MAP, modelProfileId: 'profile-1', sourceRevision: 3 });
  const first = plan.actions[0];
  const state = { planId: 'plan-1', calibrationId: 'cal-1', modelProfileId: 'profile-1', sourceRevision: 3, revision: 3,
    timestamp: 100, poseFresh: true, internalStateFresh: true, motionState: 'stable',
    facelets: first.expectedBefore.facelets, orientationId: first.expectedBefore.orientationId, snapshotKnown: true, moveWatermarkKnown: true, moveWatermark: 0 };
  return { plan, state };
}

test('dispatches exactly once and face target feedback does not complete an A/B flip', async () => {
  const { plan, state } = fixture();
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  assert.equal(session.observe(), SESSION_STATES.READY);
  assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
  assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
  assert.equal(adapter.dispatches.length, 1);
  await session.feedback({ type: 'face_target_completed', actionId: plan.actions[0].actionId, planId: 'plan-1', calibrationId: 'cal-1', modelProfileId: 'profile-1', revision: 3 });
  assert.equal(session.state, SESSION_STATES.WAIT_STABLE);
  assert.equal(session.completedActions.length, 0);
});

test('requires a fresh verified A/B observation before advancing', async () => {
  const { plan, state } = fixture();
  const adapter = createFakeAdapter({ ...state }); adapter.simulation = true;
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  session.observe(); await session.dispatch();
  await session.feedback({ type: 'ab_orientation_verified', actionId: plan.actions[0].actionId, planId: 'plan-1', calibrationId: 'cal-1', modelProfileId: 'profile-1', revision: 3 });
  assert.equal(session.completedActions.length, 0);
  adapter.state = { ...state, timestamp: 101, ...plan.actions[0].expectedAfter };
  session.now = () => 101;
  session.observe();
  await session.waitStable();
  await session.verify();
  assert.equal(session.completedActions.length, 1);
  assert.equal(session.state, SESSION_STATES.READY);
});

test('stale state, unknown snapshot, timeout, yaw deviation, and unexpected turn pause without progress', async () => {
  for (const mutation of [
    s => ({ ...s, poseFresh: false }),
    s => ({ ...s, snapshotKnown: false }),
    s => ({ ...s, moveWatermarkKnown: false }),
  ]) {
    const { plan, state } = fixture(); const adapter = createFakeAdapter(mutation(state));
    const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
    assert.equal(session.observe(), SESSION_STATES.PAUSED_FAULT);
    assert.equal(adapter.dispatches.length, 0);
  }
  const { plan, state } = fixture(); const adapter = createFakeAdapter(state); adapter.simulation = true;
  const session = new ExecutionSession({ plan, adapter, now: () => 100, actionTimeoutMs: 5 });
  session.observe(); await session.dispatch();
  adapter.state = { ...state, timestamp: 101, orientationId: 'o0' };
  session.now = () => 106; await session.waitStable();
  assert.equal(session.state, SESSION_STATES.PAUSED_FAULT);
  assert.equal(session.completedActions.length, 0);
});

test('pause requests safe pause but does not imply motor power off and restart never replays', async () => {
  const { plan, state } = fixture(); const adapter = createFakeAdapter(state); adapter.simulation = true;
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  session.observe(); await session.dispatch(); await session.pause('manual');
  assert.equal(adapter.safePauses, 1); assert.equal(adapter.powerOffs, 0);
  const restored = ExecutionSession.restore({ plan, adapter, persisted: session.snapshot(), now: () => 100, stableWindowMs: 0 });
  assert.equal(restored.state, SESSION_STATES.PAUSED_FAULT);
  await restored.dispatch();
  assert.equal(adapter.dispatches.length, 1);
});

test('locks dispatch before an async adapter resolves and pauses rejected or thrown dispatches', async () => {
  const { plan, state } = fixture();
  let resolveDispatch;
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  adapter.dispatchAction = () => new Promise(resolve => { resolveDispatch = resolve; });
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  session.observe();
  const first = session.dispatch();
  assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
  assert.equal(adapter.dispatches.length, 0);
  resolveDispatch({ accepted: false });
  assert.equal(await first, SESSION_STATES.PAUSED_FAULT);

  const secondAdapter = createFakeAdapter(state);
  secondAdapter.dispatchAction = async () => { throw new Error('wire'); };
  const second = new ExecutionSession({ plan, adapter: secondAdapter, now: () => 100, stableWindowMs: 0 });
  second.observe();
  assert.equal(await second.dispatch(), SESSION_STATES.PAUSED_FAULT);
  assert.equal(await second.feedback({ actionId: plan.actions[0].actionId }), SESSION_STATES.PAUSED_FAULT);
});

test('checks expectedBefore and pauses on changed inflight calibration or stale/future observations', async () => {
  const { plan, state } = fixture();
  const bad = createFakeAdapter({ ...state, facelets: 'other' });
  const pre = new ExecutionSession({ plan, adapter: bad, now: () => 100, stableWindowMs: 0 });
  pre.observe();
  assert.equal(await pre.dispatch(), SESSION_STATES.PAUSED_FAULT);

  const adapter = createFakeAdapter(state); adapter.simulation = true;
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  session.observe(); await session.dispatch();
  adapter.state = { ...state, calibrationId: 'cal-2', timestamp: 101 };
  session.now = () => 101;
  assert.equal(await session.waitStable(), SESSION_STATES.PAUSED_FAULT);
  const future = createFakeAdapter({ ...state, timestamp: 101 });
  const futureSession = new ExecutionSession({ plan, adapter: future, now: () => 100, stableWindowMs: 0 });
  assert.equal(futureSession.observe(), SESSION_STATES.PAUSED_FAULT);
});

test('turns require turn_verified and a stable observation window', async () => {
  const { plan, state } = fixture();
  // The first action is a real compiled flip; a turn-specific completion must not pass.
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  let clock = 100;
  const session = new ExecutionSession({ plan, adapter, now: () => clock, stableWindowMs: 10 });
  session.observe(); await session.dispatch();
  await session.feedback({ type: 'turn_verified', actionId: plan.actions[0].actionId, planId: 'plan-1', calibrationId: 'cal-1', modelProfileId: 'profile-1', revision: 3 });
  adapter.state = { ...state, timestamp: 101, orientationId: 'o1' }; clock = 101;
  await session.waitStable();
  assert.equal(session.state, SESSION_STATES.WAIT_STABLE);
  clock = 111; await session.waitStable();
  assert.equal(session.state, SESSION_STATES.VERIFY);
  assert.equal(await session.verify(), SESSION_STATES.PAUSED_FAULT);
});

test('zero-action plans only solve when the observed facelets match initial state', () => {
  const { state } = fixture();
  const plan = compilePlanV2(SOLVED, '', { planId: 'empty', initialQuaternion: referenceOrientation.quaternion,
    calibrationId: 'cal-1', centerIdentityMap: DEMO_CENTER_IDENTITY_MAP, modelProfileId: 'profile-1', sourceRevision: 3 });
  const adapter = createFakeAdapter({ ...state, facelets: 'not-solved' });
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  assert.equal(session.observe(), SESSION_STATES.PAUSED_FAULT);
});

test('shadow never dispatches; preview and validated execution require explicit simulation/authorization', async () => {
  const { plan, state } = fixture();
  for (const mode of ['shadow', 'ideal_preview']) {
    const candidate = { ...plan, mode };
    const adapter = createFakeAdapter(state); adapter.simulation = true;
    if (mode === 'ideal_preview') adapter.simulation = true;
    const session = new ExecutionSession({ plan: candidate, adapter, now: () => 100, stableWindowMs: 0 });
    session.observe();
    if (mode === 'shadow') assert.equal(await session.dispatch(), SESSION_STATES.PAUSED_FAULT);
    else assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
    assert.equal(adapter.dispatches.length, mode === 'shadow' ? 0 : 1);
  }
  const validated = { ...plan, mode: 'validated_execution' };
  assert.throws(() => validatePlanV2(validated), /costModel/);
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  assert.throws(() => new ExecutionSession({ plan: validated, adapter, now: () => 100, stableWindowMs: 0 }), /costModel/);
});

test('a compiled multi-action plan advances only through verified action boundaries', async () => {
  const { plan, state } = fixture();
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  let clock = 100;
  const session = new ExecutionSession({ plan, adapter, now: () => clock, stableWindowMs: 0 });
  for (const [index, action] of plan.actions.entries()) {
    assert.equal(session.observe(), SESSION_STATES.READY);
    assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
    const type = action.kind === 'turn' ? 'turn_verified' : 'ab_orientation_verified';
    await session.feedback({ type, actionId: action.actionId, planId: plan.planId, calibrationId: plan.calibrationId,
      modelProfileId: plan.modelProfileId, sourceRevision: plan.sourceRevision, revision: 3 });
    clock += 1;
    adapter.state = { ...adapter.state, ...action.expectedAfter, timestamp: clock,
      moveWatermark: adapter.state.moveWatermark + (action.kind === 'turn' ? 1 : 0) };
    await session.waitStable(); await session.verify();
    if (index < plan.actions.length - 1) assert.equal(session.state, SESSION_STATES.READY);
  }
  assert.equal(session.state, SESSION_STATES.SOLVED);
});

test('requires explicit accepted true, contiguous watermark, and tolerates safe-pause failures', async () => {
  const { plan, state } = fixture();
  const adapter = createFakeAdapter(state); adapter.simulation = true;
  adapter.dispatchAction = async () => ({});
  adapter.safePause = () => Promise.reject(new Error('pause transport')); // must not become unhandled
  const rejected = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  rejected.observe();
  assert.equal(await rejected.dispatch(), SESSION_STATES.PAUSED_FAULT);

  const gapAdapter = createFakeAdapter(state); gapAdapter.simulation = true;
  const gap = new ExecutionSession({ plan, adapter: gapAdapter, now: () => 100, stableWindowMs: 0 });
  gap.observe(); await gap.dispatch();
  gapAdapter.state = { ...state, timestamp: 101, moveWatermark: 2 };
  gap.now = () => 101;
  assert.equal(await gap.waitStable(), SESSION_STATES.PAUSED_FAULT);
});

test('clones the validated plan and dispatch payload, and rejects missing pre-dispatch watermark', async () => {
  const { plan, state } = fixture();
  const originalPrimitive = plan.actions[0].primitive;
  const adapter = createFakeAdapter({ ...state }); adapter.simulation = true;
  adapter.dispatchAction = async action => { action.primitive = 'turn_CCW90'; action.expectedBefore.facelets = 'tampered'; return { accepted: true }; };
  const session = new ExecutionSession({ plan, adapter, now: () => 100, stableWindowMs: 0 });
  plan.actions[0].primitive = 'free_yaw';
  session.observe();
  assert.equal(await session.dispatch(), SESSION_STATES.WAIT_STABLE);
  assert.equal(session.plan.actions[0].primitive, originalPrimitive);

  const missing = createFakeAdapter({ ...state, moveWatermark: undefined }); missing.simulation = true;
  const blocked = new ExecutionSession({ plan: session.plan, adapter: missing, now: () => 100, stableWindowMs: 0 });
  blocked.observe();
  assert.equal(await blocked.dispatch(), SESSION_STATES.PAUSED_FAULT);
});
