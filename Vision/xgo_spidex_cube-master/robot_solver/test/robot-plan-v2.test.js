import test from 'node:test';
import assert from 'node:assert/strict';
import { Cube, SOLVED } from '../src/cube-model.js';
import { ORIENTATIONS, referenceOrientation, DEMO_CENTER_IDENTITY_MAP } from '../src/palm-frame.js';
import { compilePlanV2, validatePlanV2, replayPlanV2, expandTurns } from '../src/robot-plan-v2.js';

export const options = { initialQuaternion: referenceOrientation.quaternion, calibrationId: 'demo-cal',
  centerIdentityMap: DEMO_CENTER_IDENTITY_MAP, modelProfileId: 'preview', sourceRevision: 0 };

test('all six faces expand clockwise into three counterclockwise quarters', () => {
  for (const face of 'URFDLB') {
    const turns = expandTurns(`${face} ${face}2 ${face}'`);
    assert.deepEqual(turns.map(t => t.notation), Array(6).fill(`${face}'`));
    assert.deepEqual(turns.map(t => t.symbolicMoveIndex), [0,0,0,1,1,2]);
    for (const [move, count] of [[face,3],[`${face}2`,2],[`${face}'`,1]]) {
      const initial = new Cube().move(move).asString();
      const inverse = move.endsWith('2') ? move : move.endsWith("'") ? face : `${face}'`;
      const p = compilePlanV2(initial, inverse, options);
      assert.ok(p.actions.filter(a=>a.kind==='turn').every(a=>a.primitive==='turn_CCW90' && a.notation===`${face}'`));
      assert.equal(p.statistics.quarterTurns, count === 3 ? 1 : count === 1 ? 3 : 2);
      assert.equal(replayPlanV2(p).facelets,SOLVED);
    }
  }
});

test('solved cube has no actions and preserves the reference frame', () => {
  const p = compilePlanV2(SOLVED, '', options);
  assert.equal(p.schemaVersion, 2); assert.deepEqual(p.actions, []);
  assert.equal(p.initialState.orientationId, referenceOrientation.id);
  assert.equal(validatePlanV2(p), true);
});

test('white top needs no flip and half-turn is two verified quarters', () => {
  const p = compilePlanV2(new Cube().move('D2').asString(), 'D2', options);
  assert.equal(p.statistics.reorientations, 0);
  assert.equal(p.statistics.quarterTurns, 2);
  assert.deepEqual(p.expandedTurns.map(t => t.notation), ["D'", "D'"]);
  assert.equal(replayPlanV2(p).facelets, SOLVED);
});

test('all initial orientations compile valid A/B-only plans', () => {
  const scramble = "R U F2 L' D";
  const cube = new Cube().move(scramble);
  const inverse = "D' L F2 U' R'";
  for (const orientation of ORIENTATIONS) {
    const p = compilePlanV2(cube.asString(), inverse, { ...options, initialQuaternion: orientation.quaternion });
    assert.equal(validatePlanV2(p), true);
    assert.equal(replayPlanV2(p).facelets, SOLVED);
    for (const a of p.actions.filter(a => a.kind === 'flip')) {
      assert.ok(['flip_A', 'flip_B'].includes(a.primitive));
      assert.equal(a.expectedBefore.facelets, a.expectedAfter.facelets);
    }
  }
});

test('rejects invalid solutions, stale maps and off-lattice poses', () => {
  assert.throws(() => compilePlanV2(new Cube().move('R').asString(), 'U', options), /solve|还原/i);
  assert.throws(() => compilePlanV2(SOLVED, '', { ...options, centerIdentityMap: {} }), /center/i);
  const a = Math.PI / 8;
  assert.throws(() => compilePlanV2(SOLVED, '', { ...options, initialQuaternion: [0, 0, Math.sin(a), Math.cos(a)] }), /POSE_OFF_LATTICE/);
});

test('v2 validator rejects tampered intermediate states and primitive directions', () => {
  const p = compilePlanV2(new Cube().move('R').asString(), "R'", options);
  for (const change of [
    x => { x.actions[0].primitive = 'free_yaw'; },
    x => { x.actions[0].expectedAfter.orientationId = x.actions[0].expectedBefore.orientationId; },
    x => { x.actions.at(-1).expectedAfter.facelets = p.initialState.facelets; },
    x => { x.actions.at(-1).symbolicMoveIndex = 90; },
  ]) { const q = structuredClone(p); change(q); assert.throws(() => validatePlanV2(q)); }
});

test('v2 validator rejects tampered metadata, statistics and proxy cost', () => {
  const p = compilePlanV2(new Cube().move('R').asString(), "R'", options);
  for (const change of [
    x => { x.sourceRevision = -1; },
    x => { x.status = 'executing'; },
    x => { x.calibrationId = ''; },
    x => { x.statistics.robotSteps += 1; },
    x => { x.cost.selected[0] += 1; },
    x => { x.cost.predictedProbability = 2; },
  ]) { const q = structuredClone(p); change(q); assert.throws(() => validatePlanV2(q)); }
});

test('validated execution requires an explicit cost model and matching bindings', () => {
  const profile = { profileId: 'live', calibrationId: 'cal', edges: {} };
  const costModel = { mode: 'proxy', executionMode: 'validated_execution', zero: [0, 0, 0], compare: (a,b) => a[0]-b[0] || a[1]-b[1] || a[2]-b[2], add: (a,b) => a.map((x,i)=>x+b[i]), actionCost: () => [0,0,0], probability: () => null };
  assert.throws(() => compilePlanV2(SOLVED, '', { ...options, mode: 'validated_execution', modelProfileId: 'live', calibrationId: 'cal' }), /costModel/i);
  const p = compilePlanV2(SOLVED, '', { ...options, mode: 'validated_execution', modelProfileId: 'live', calibrationId: 'cal', costModel });
  assert.throws(() => validatePlanV2(p), /costModel/i);
  assert.throws(() => validatePlanV2(p, { costModel: { ...costModel, executionMode: 'ideal_preview' } }), /mode/i);
  assert.throws(() => validatePlanV2(p, { costModel, calibrationId: 'other', modelProfileId: 'live' }), /binding/i);
  assert.equal(validatePlanV2(p, { costModel, calibrationId: 'cal', modelProfileId: 'live', sourceRevision: 0 }), true);
});

test('DP agrees with an independent breadth-first action oracle for short solutions', () => {
  // Search state is (turn index, orientation); every flip is cost 1, turns cost 0.
  const faces = { U:[0,0,1], R:[1,0,0], F:[0,-1,0], D:[0,0,-1], L:[-1,0,0], B:[0,1,0] };
  const A = m => [m[0],m[1],m[2],-m[6],-m[7],-m[8],m[3],m[4],m[5]];
  const B = m => [-m[6],-m[7],-m[8],m[3],m[4],m[5],m[0],m[1],m[2]];
  const algorithm = "R' U' F'"; const targets = ['R','U','F'];
  const cube = new Cube().move('F U R');
  const queue = [{ i:0, m:[...referenceOrientation.matrix], cost:0 }]; const seen = new Set(); let answer;
  while (queue.length) {
    queue.sort((a,b) => a.cost-b.cost); const s=queue.shift(); const key=`${s.i}:${s.m}`;
    if (seen.has(key)) continue; seen.add(key);
    if (s.i === targets.length) { answer=s.cost; break; }
    const n=faces[targets[s.i]];
    if (s.m[6]*n[0]+s.m[7]*n[1]+s.m[8]*n[2]===1) queue.push({ ...s, i:s.i+1 });
    queue.push({ i:s.i,m:A(s.m),cost:s.cost+1 },{ i:s.i,m:B(s.m),cost:s.cost+1 });
  }
  assert.equal(compilePlanV2(cube.asString(),algorithm,options).statistics.reorientations,answer);
});
