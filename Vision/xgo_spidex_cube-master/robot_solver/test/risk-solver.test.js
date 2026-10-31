import test from 'node:test';
import assert from 'node:assert/strict';
import { Cube } from '../src/cube-model.js';
import { referenceOrientation, DEMO_CENTER_IDENTITY_MAP } from '../src/palm-frame.js';
import { solveRisk, normalizeAlgorithm } from '../src/risk-solver.js';

const options={ initialQuaternion:referenceOrientation.quaternion, centerIdentityMap:DEMO_CENTER_IDENTITY_MAP,
  calibrationId:'demo',modelProfileId:'preview',maxSolverCalls:2 };
test('normalization combines adjacent faces but never reorders noncommuting moves', () => {
  assert.equal(normalizeAlgorithm("R R U U' F F F"), "R2 F'");
  assert.equal(normalizeAlgorithm("R U R'"), "R U R'");
});
test('candidate search selects a strictly cheaper valid plan and streams the baseline', () => {
  const cube=new Cube().move('R U'); const stream=[]; let calls=0;
  const identity=Array(6).fill("R U R' U'").join(' ');
  const p=solveRisk(cube.asString(), { ...options, solver:()=> ++calls===1 ? "U' R' "+identity : "D' U' R'" }, x=>stream.push(x));
  assert.equal(calls,2); assert.equal(p.search.calls,2);
  assert.ok(p.statistics.reorientations < stream[0].statistics.reorientations);
  assert.ok(new Cube().move('R U').move(p.solution.join(' ')).isSolved());
  assert.equal(p.search.optimality,'candidate-limited');
});
test('a one-call budget returns a verified baseline and invalid candidates never escape', () => {
  const s=new Cube().move('R').asString();
  const p=solveRisk(s,{...options,maxSolverCalls:1,solver:()=>"R'"});
  assert.equal(p.search.calls,1); assert.equal(p.search.candidateCount,1);
  assert.throws(()=>solveRisk(s,{...options,solver:()=> 'U'}),/solve|还原/i);
  const q=solveRisk(s,{...options,solver:state=>state===s?"R'":'U'});
  assert.equal(q.search.invalidCandidates,1);
  assert.deepEqual(q.solution,["R'"]);
});
test('budgets and cancellation are validated and deterministic', () => {
  const s=new Cube().move('R').asString();
  assert.throws(()=>solveRisk(s,{...options,maxSolverCalls:0}),/budget/i);
  const p=solveRisk(s,{...options,solver:()=>"R'",isCancelled:()=>true});
  assert.equal(p.search.calls,1); assert.equal(p.search.cancelled,true);
});
