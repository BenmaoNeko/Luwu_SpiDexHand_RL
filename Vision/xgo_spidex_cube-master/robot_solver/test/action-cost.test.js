import test from 'node:test';
import assert from 'node:assert/strict';
import { costAction, compareCosts, scorePlan, validateProfile, edgeAllowed, createCostModel } from '../src/action-cost.js';

test('proxy cost is deterministic and counts quarter turns', () => {
  const a = scorePlan([{ kind: 'flip', primitive: 'flip_A', durationMs: 20 }, { kind: 'turn', primitive: 'turn_CW90', durationMs: 5 }], { mode: 'proxy' });
  const b = scorePlan([{ kind: 'flip', primitive: 'flip_B', durationMs: 20 }, { kind: 'turn', primitive: 'turn_CW90', durationMs: 5 }], { mode: 'proxy' });
  assert.deepEqual(a.tuple.slice(0, 3), [1, 1, 25]);
  assert.equal(compareCosts(a, b), -1);
  assert.equal(costAction({ primitive: 'turn_CW180' }, { mode: 'proxy' }).quarterTurns, 2);
});

test('risk cost rejects unknown/zero probabilities and checks profile', () => {
  const profile = { profileId: 'p1', calibrationId: 'c1', edges: { 'flip_A@o0': { pSafe: 0.8, durationMs: 100 } } };
  assert.equal(validateProfile(profile, { profileId: 'p1', calibrationId: 'c1' }).ok, true);
  assert.equal(validateProfile(profile, { profileId: 'other', calibrationId: 'c1' }).ok, false);
  assert.equal(edgeAllowed('flip_A@o0', profile, { mode: 'risk' }), true);
  assert.equal(edgeAllowed('flip_B@o0', profile, { mode: 'risk' }), false);
  assert.throws(() => costAction({ primitive: 'flip_B', startOrientationId: 'o0' }, { mode: 'risk', profile }), /unknown|profile/i);
  assert.equal(scorePlan([{ primitive: 'flip_A', startOrientationId: 'o0' }], { mode: 'risk', profile }).executable, true);
});

test('risk score uses negative log probability plus time', () => {
  const profile = { profileId: 'p', calibrationId: 'c', edges: { 'flip_A@o0': { pSafe: 0.5, durationMs: 200 } } };
  const c = scorePlan([{ primitive: 'flip_A', startOrientationId: 'o0' }], { mode: 'risk', profile, lambdaTime: 0.01 });
  assert.ok(Math.abs(c.value - (Math.log(2) + 2)) < 1e-9);
});

test('validated execution requires bound, enabled, validated profile edges', () => {
  const preview = createCostModel({ mode: 'proxy', executionMode: 'ideal_preview' });
  assert.deepEqual(preview.actionCost({ primitive: 'flip_A', orientationId: 'o0' }), [1, 0, 0]);
  const live = createCostModel({ mode: 'proxy', executionMode: 'validated_execution', profile: { profileId: 'p', calibrationId: 'c', edges: { 'flip_A@o0': { validated: true, durationMs: 10 } } }, modelProfileId: 'p', calibrationId: 'c' });
  assert.deepEqual(live.actionCost({ primitive: 'flip_A', orientationId: 'o0' }), [1, 0, 10]);
  assert.equal(live.actionCost({ primitive: 'flip_B', orientationId: 'o0' }), null);
  const disabled = createCostModel({ mode: 'proxy', executionMode: 'validated_execution', profile: { profileId: 'p', calibrationId: 'c', edges: { 'flip_A@o0': { validated: false, durationMs: 10 } } }, modelProfileId: 'p', calibrationId: 'c' });
  assert.equal(disabled.actionCost({ primitive: 'flip_A', orientationId: 'o0' }), null);
});

test('risk requires finite nonnegative duration and lambda', () => {
  const profile = { profileId: 'p', calibrationId: 'c', edges: { 'flip_A@o0': { validated: true, pSafe: 0.5, durationMs: 20 } } };
  assert.equal(createCostModel({ mode: 'risk', profile, modelProfileId: 'p', calibrationId: 'c' }).actionCost({ primitive: 'flip_A', orientationId: 'o0' })[0] > 0, true);
  assert.equal(createCostModel({ mode: 'risk', profile: { ...profile, edges: { 'flip_A@o0': { pSafe: 0.5, durationMs: -1 } } }, modelProfileId: 'p', calibrationId: 'c' }).actionCost({ primitive: 'flip_A', orientationId: 'o0' }), null);
});

test('cost models reject negative proxy duration and snapshot their profile', () => {
  assert.throws(() => costAction({primitive:'flip_A',durationMs:-1}), /duration/);
  const profile={profileId:'p',calibrationId:'c',edges:{'flip_A@o0':{pSafe:0.8,durationMs:20}}};
  const model=createCostModel({mode:'risk',profile});
  profile.edges['flip_A@o0'].pSafe=0;
  assert.ok(model.actionCost({primitive:'flip_A',orientationId:'o0'}));
  assert.equal(model.contextIndependent,true);
  assert.throws(()=>createCostModel({executionMode:'misspelled'}),/execution mode/i);
});
