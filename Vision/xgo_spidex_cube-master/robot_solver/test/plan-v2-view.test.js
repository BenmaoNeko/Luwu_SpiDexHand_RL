import test from 'node:test';
import assert from 'node:assert/strict';
import { Cube, SOLVED } from '../src/cube-model.js';
import { referenceOrientation, DEMO_CENTER_IDENTITY_MAP } from '../src/palm-frame.js';
import { compilePlanV2 } from '../src/robot-plan-v2.js';
import { toPreviewAction, executePreviewAction, moveProgress } from '../src/plan-v2-view.js';

const options={initialQuaternion:referenceOrientation.quaternion,calibrationId:'demo',modelProfileId:'preview',centerIdentityMap:DEMO_CENTER_IDENTITY_MAP};
test('preview replays multiple flips per move using explicit mapping',()=>{
  const cube=new Cube().move('R U F2');
  const p=compilePlanV2(cube.asString(),"F2 U' R'",options);
  let q=[...referenceOrientation.quaternion];
  for(const [i,a] of p.actions.entries()) {
    const view=toPreviewAction(a);
    if(a.kind==='flip') {
      assert.equal(view.angleDeg,90);
      assert.deepEqual(view.axisWorld,a.primitive==='flip_A'?[1,0,0]:[0,-1,0]);
    }
    q=executePreviewAction(cube,q,a);
    const progress=moveProgress(p,i+1);
    assert.ok(progress.completed>=0 && progress.completed<=p.solution.length);
  }
  assert.equal(cube.asString(),SOLVED);
  assert.deepEqual(moveProgress(p,p.actions.length),{completed:p.solution.length,current:null});
});
test('preview rejects state drift and invalid actions',()=>{
  const p=compilePlanV2(new Cube().move('R').asString(),"R'",options);
  assert.throws(()=>executePreviewAction(new Cube(),referenceOrientation.quaternion,p.actions[0]),/state/i);
  assert.throws(()=>toPreviewAction({...p.actions[0],primitive:'free_yaw'}),/primitive/i);
});
