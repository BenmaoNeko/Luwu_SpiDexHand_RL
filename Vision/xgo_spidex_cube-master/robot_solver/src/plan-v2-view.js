import { parseMoves, SIZE_MM } from './cube-model.js';
import { ORIENTATIONS, quaternionDistanceDeg, flipOrientation } from './palm-frame.js';
import { executeRobotAction } from './planner.js';

const orientation = id => {
  const o=ORIENTATIONS.find(x=>x.id===id);
  if(!o) throw new Error('Unknown preview orientation');
  return o.quaternion;
};

/** Geometry adapter only. Animation completion is never hardware feedback. */
export function toPreviewAction(action) {
  const from=[...orientation(action.expectedBefore.orientationId)];
  const to=[...orientation(action.expectedAfter.orientationId)];
  if(action.kind==='flip' && ['flip_A','flip_B'].includes(action.primitive)) {
    if(flipOrientation(action.expectedBefore.orientationId,action.primitive)!==action.expectedAfter.orientationId) throw new Error('Invalid flip primitive transition');
    return {type:'reorient',targetFace:action.targetFace,fromQuaternion:from,toQuaternion:to,
      requiresMotion:true,angleDeg:90,axisWorld:action.primitive==='flip_A'?[1,0,0]:[0,-1,0],
      targetCenterMm:[0,0,SIZE_MM/2]};
  }
  if(action.kind==='turn' && action.primitive==='turn_CCW90') {
    const move=parseMoves(action.notation)[0];
    if(!move || move.face!==action.targetFace || !move.token.endsWith("'")) throw new Error('Invalid turn primitive');
    return {type:'turn_top',targetFace:action.targetFace,notation:action.notation,
      command:move.command,angleDeg:move.angleDeg,quaternion:from,axisWorld:[0,0,1]};
  }
  throw new Error('Unknown preview primitive');
}

export function executePreviewAction(cube, quaternion, action) {
  if(cube.asString()!==action.expectedBefore.facelets || quaternionDistanceDeg(quaternion,orientation(action.expectedBefore.orientationId))>0.001) throw new Error('Preview state mismatch');
  const next=executeRobotAction(cube,quaternion,toPreviewAction(action));
  if(cube.asString()!==action.expectedAfter.facelets) throw new Error('Preview resulting state mismatch');
  return next;
}

export function moveProgress(plan,cursor) {
  const next=plan.actions[cursor];
  return {completed:next?next.symbolicMoveIndex:plan.solution.length,current:next?.symbolicMoveIndex??null};
}
