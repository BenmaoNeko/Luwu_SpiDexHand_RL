import test from 'node:test';
import assert from 'node:assert/strict';
import {
  A_MATRIX, B_MATRIX, R0_MATRIX, orientations, referenceOrientation,
  apply, multiply, flipOrientation, normalInPalm, classifyOrientation,
  quaternionFromMatrix, quaternionDistanceDeg, validateCenterIdentityMap,
} from '../src/palm-frame.js';
import { FACES } from '../src/cube-model.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const vecClose = (a, b) => a.forEach((x, i) => close(x, b[i]));

test('reference frame and A/B are left multiplied', () => {
  vecClose(apply(multiply(A_MATRIX, R0_MATRIX), [0, 0, -1]), [0, -1, 0]);
  vecClose(apply(multiply(B_MATRIX, R0_MATRIX), [-1, 0, 0]), [0, 0, -1]);
  assert.equal(flipOrientation(referenceOrientation.id, 'flip_A'), flipOrientation(referenceOrientation.id, 'A'));
  assert.notDeepEqual(multiply(A_MATRIX, multiply(B_MATRIX, R0_MATRIX)), multiply(B_MATRIX, multiply(A_MATRIX, R0_MATRIX)));
});

test('B reverses the previous command at every orientation while A stays unchanged', () => {
  const previousB = [0, 0, 1, 0, 1, 0, -1, 0, 0];
  assert.deepEqual(A_MATRIX, [1, 0, 0, 0, 0, -1, 0, 1, 0]);
  for (const o of orientations) {
    const reversed = orientations.find(x => x.id === flipOrientation(o.id, 'flip_B'));
    vecClose(multiply(previousB, reversed.matrix), o.matrix);
  }
});

test('enumerates the 24 orientation nodes and directed graph', () => {
  assert.equal(orientations.length, 24);
  assert.equal(new Set(orientations.map((o) => o.id)).size, 24);
  for (const o of orientations) {
    assert.equal(o.matrix.length, 9);
    assert.equal(o.quaternion.length, 4);
    assert.equal(flipOrientation(flipOrientation(flipOrientation(flipOrientation(o.id, 'A'), 'A'), 'A'), 'A'), o.id);
    assert.equal(quaternionDistanceDeg(o.quaternion, o.quaternion.map((x) => -x)), 0);
  }
});

test('normalInPalm and classification retain measured quaternion', () => {
  assert.deepEqual(normalInPalm(referenceOrientation.id, 'D'), [0, 0, 1]);
  const q = quaternionFromMatrix(multiply(A_MATRIX, R0_MATRIX));
  const result = classifyOrientation(q, { maxResidualDeg: 0.01 });
  assert.equal(result.orientationId, flipOrientation(referenceOrientation.id, 'A'));
  assert.ok(result.residualDeg < 1e-7);
  assert.deepEqual(result.quaternion, q);
  assert.throws(() => classifyOrientation([0, 0, 0, 0]), /finite|zero/i);
  assert.throws(() => classifyOrientation([0, 0, 0.01, 1], { maxResidualDeg: 0 }), /POSE_OFF_LATTICE/);
});

test('center map validates fixed identities without renaming spatial U', () => {
  assert.deepEqual(FACES.F.normal, [0, -1, 0]);
  assert.deepEqual(FACES.B.normal, [0, 1, 0]);
  assert.equal(validateCenterIdentityMap({ white: 'D', yellow: 'U', blue: 'R', green: 'L', purple: 'F', red: 'B' }).ok, true);
  assert.equal(validateCenterIdentityMap({ white: 'D', yellow: 'U', blue: 'R', green: 'L', red: 'B', orange: 'F' }).ok, true);
  assert.equal(validateCenterIdentityMap({ white: 'D', yellow: 'U', blue: 'R', green: 'L', red: 'F', orange: 'B' }).ok, false);
  assert.equal(validateCenterIdentityMap({ white: 'D', yellow: 'D', blue: 'R', green: 'L', red: 'B', orange: 'F' }).ok, false);
});
