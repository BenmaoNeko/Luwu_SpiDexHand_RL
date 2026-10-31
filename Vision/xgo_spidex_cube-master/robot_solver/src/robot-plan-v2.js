import { cubeFromFacelets, parseMoves } from './cube-model.js';
import { ORIENTATIONS, classifyOrientation, flipOrientation, normalInPalm,
  validateCenterIdentityMap } from './palm-frame.js';
import { createCostModel } from './action-cost.js';

const FLIPS = ['flip_A', 'flip_B'];
const stationaryRoutes = new WeakMap();
const node = id => {
  const value = ORIENTATIONS.find(o => o.id === id);
  if (!value) throw new Error(`Unknown orientation ${id}`);
  return value;
};
const top = id => ['U','R','F','D','L','B'].find(f => normalInPalm(id, f)[2] === 1);

export function expandTurns(algorithm) {
  return parseMoves(algorithm).flatMap((move, symbolicMoveIndex) =>
    Array(move.token.endsWith('2') ? 2 : move.token.endsWith("'") ? 1 : 3).fill(move.face + "'")
      .map(notation => ({ notation, symbolicMoveIndex })));
}

function shortestPaths(start, facelets, costModel) {
  // The built-in cost model snapshots a pose/primitive-only profile. Custom
  // models may depend on internal state and intentionally bypass this cache.
  if (costModel.contextIndependent && stationaryRoutes.get(costModel)?.has(start)) return stationaryRoutes.get(costModel).get(start);
  const best = new Map([[start, { cost: [...costModel.zero], path: [] }]]);
  const visited = new Set();
  while (visited.size < best.size) {
    let current;
    for (const [id, record] of best) {
      if (!visited.has(id) && (!current || costModel.compare(record.cost, best.get(current).cost) < 0)) current = id;
    }
    visited.add(current);
    const previous = best.get(current);
    for (const primitive of FLIPS) {
      const next = flipOrientation(current, primitive);
      const edge = costModel.actionCost({ kind:'flip', primitive, orientationId:current, facelets, targetFace:top(next) });
      if (!edge) continue;
      const cost = costModel.add(previous.cost, edge);
      if (!best.has(next) || costModel.compare(cost, best.get(next).cost) < 0) {
        best.set(next, { cost, path:[...previous.path, primitive] });
      }
    }
  }
  if (costModel.contextIndependent) {
    if (!stationaryRoutes.has(costModel)) stationaryRoutes.set(costModel,new Map());
    stationaryRoutes.get(costModel).set(start,best);
  }
  return best;
}

export function compilePlanV2(facelets, algorithm, options = {}) {
  const cube = cubeFromFacelets(facelets);
  const solution = parseMoves(algorithm).map(m => m.token);
  if (!cube.clone().move(solution.join(' ')).isSolved()) throw new Error('Solution does not solve the input cube');
  const mapping = validateCenterIdentityMap(options.centerIdentityMap);
  if (!mapping.ok) throw new Error(`Invalid center identity map: ${mapping.reason}`);
  if (!options.calibrationId || !options.modelProfileId) throw new Error('calibrationId and modelProfileId are required');
  if (options.mode === 'validated_execution' && !options.costModel) throw new Error('validated execution requires costModel');
  const initial = classifyOrientation(options.initialQuaternion, options.maxPoseErrorDeg ?? 15);
  const costModel = options.costModel ?? createCostModel({ ...options, mode:options.costMode ?? 'proxy', executionMode:options.mode ?? 'ideal_preview' });
  const turns = expandTurns(solution);
  let states = new Map([[initial.orientationId, { cost:[...costModel.zero], stages:[] }]]);
  let internal = cube.clone();
  for (const turn of turns) {
    const nextStates = new Map();
    const face = turn.notation[0];
    const primitive = 'turn_CCW90';
    for (const [orientationId, state] of states) {
      for (const [destination, route] of shortestPaths(orientationId, internal.asString(), costModel)) {
        if (top(destination) !== face) continue;
        const turnCost = costModel.actionCost({ kind:'turn', primitive, orientationId:destination, facelets:internal.asString(), targetFace:face });
        if (!turnCost) continue;
        const cost = costModel.add(costModel.add(state.cost, route.cost), turnCost);
        if (!nextStates.has(destination) || costModel.compare(cost, nextStates.get(destination).cost) < 0) {
          nextStates.set(destination, { cost, stages:[...state.stages, { ...turn, destination, path:route.path, primitive }] });
        }
      }
    }
    if (!nextStates.size) { const error=new Error('NO_VALIDATED_ROUTE'); error.code='NO_VALIDATED_ROUTE'; throw error; }
    states=nextStates; internal.move(turn.notation);
  }
  const selected = [...states.values()].reduce((a,b) => costModel.compare(a.cost,b.cost) <= 0 ? a : b);
  const planId = options.planId ?? globalThis.crypto.randomUUID();
  const actions=[];
  let orientationId=initial.orientationId;
  internal=cube.clone();
  const append=(kind, primitive, stage, expandedTurnIndex, afterId, afterFacelets) => {
    const measuredCost = costModel.actionCost({ kind, primitive, orientationId, facelets: internal.asString(), targetFace: kind === 'flip' ? top(afterId) : stage.notation[0] });
    if (!measuredCost) throw new Error('NO_VALIDATED_ROUTE');
    const durationMs = Array.isArray(measuredCost) ? (costModel.mode === 'risk' ? measuredCost[1] : measuredCost[2]) : measuredCost.durationMs;
    actions.push({ actionId:`${planId}:${actions.length}`, actionIndex:actions.length,
      symbolicMoveIndex:stage.symbolicMoveIndex, expandedTurnIndex, kind, primitive,
      notation:kind==='turn' ? stage.notation : null,
      targetFace:kind==='flip' ? top(afterId) : stage.notation[0],
      expectedBefore:{ facelets:internal.asString(), orientationId },
      expectedAfter:{ facelets:afterFacelets, orientationId:afterId },
      predictedQuaternionPC_xyzw:[...node(afterId).quaternion],
      completionPolicyId:kind==='flip' ? 'ab_orientation_verified' : 'turn_verified',
      timeoutPolicyId:'pause_on_timeout', durationMs });
    orientationId=afterId;
  };
  for (const [index, stage] of selected.stages.entries()) {
    for (const primitive of stage.path) append('flip',primitive,stage,index,flipOrientation(orientationId,primitive),internal.asString());
    const after=internal.clone().move(stage.notation).asString();
    append('turn',stage.primitive,stage,index,orientationId,after);
    internal.move(stage.notation);
  }
  const probabilities=actions.map(a => costModel.probability({ ...a, orientationId:a.expectedBefore.orientationId }));
  const probability=costModel.mode==='risk' && probabilities.every(p => p > 0 && p <= 1)
    ? probabilities.reduce((a,b) => a*b,1) : null;
  const plan={ schemaVersion:2, planId, mode:options.mode ?? 'ideal_preview', status:'planned',
    calibrationId:options.calibrationId, modelProfileId:options.modelProfileId,
    centerIdentityMap:{ ...options.centerIdentityMap }, sourceRevision:options.sourceRevision ?? 0,
    initialState:{ facelets:cube.asString(), quaternionPC_xyzw:[...options.initialQuaternion], orientationId:initial.orientationId },
    solution, expandedTurns:turns, actions,
    cost:{ mode:costModel.mode, selected:selected.cost, baseline:null, predictedProbability:probability },
    search:{ optimality:'fixed-candidate' },
    statistics:{ faceTurns:solution.length, quarterTurns:turns.length, robotSteps:actions.length,
      reorientations:actions.filter(a => a.kind==='flip').length },
    finalQuaternion:[...node(orientationId).quaternion] };
  validatePlanV2(plan, { costModel });
  return plan;
}

export function replayPlanV2(plan) {
  if (plan?.schemaVersion !== 2 || !plan.planId || !Array.isArray(plan.actions)) throw new Error('Invalid v2 plan');
  const map=validateCenterIdentityMap(plan.centerIdentityMap);
  if (!map.ok) throw new Error('Invalid center identity map');
  const cube=cubeFromFacelets(plan.initialState.facelets);
  const initial=classifyOrientation(plan.initialState.quaternionPC_xyzw);
  let orientationId=initial.orientationId;
  if (orientationId!==plan.initialState.orientationId) throw new Error('Initial orientation mismatch');
  const turns=expandTurns(plan.solution);
  if (JSON.stringify(turns)!==JSON.stringify(plan.expandedTurns)) throw new Error('Expanded turn mapping mismatch');
  let turnIndex=0;
  const ids=new Set();
  for (const [index,a] of plan.actions.entries()) {
    if (!a.actionId || ids.has(a.actionId) || a.actionIndex!==index) throw new Error('Invalid action index or ID');
    ids.add(a.actionId);
    if (a.expectedBefore.facelets!==cube.asString() || a.expectedBefore.orientationId!==orientationId) throw new Error('Action state chain mismatch');
    const turn=turns[turnIndex];
    if (!turn || a.expandedTurnIndex!==turnIndex || a.symbolicMoveIndex!==turn.symbolicMoveIndex) throw new Error('Action turn mapping mismatch');
    if (a.kind==='flip' && FLIPS.includes(a.primitive)) {
      orientationId=flipOrientation(orientationId,a.primitive);
      if (a.targetFace!==top(orientationId)) throw new Error('Flip target mismatch');
    } else if (a.kind==='turn') {
      const primitive='turn_CCW90';
      if (a.notation!==turn.notation || a.primitive!==primitive || a.targetFace!==turn.notation[0] || top(orientationId)!==a.targetFace) throw new Error('Turn target or direction mismatch');
      cube.move(turn.notation); turnIndex++;
    } else throw new Error('Unknown primitive');
    if (a.expectedAfter.facelets!==cube.asString() || a.expectedAfter.orientationId!==orientationId) throw new Error('Action resulting state mismatch');
    if (classifyOrientation(a.predictedQuaternionPC_xyzw,0.001).orientationId!==orientationId) throw new Error('Predicted quaternion mismatch');
  }
  if (turnIndex!==turns.length || !cube.isSolved()) throw new Error('Plan does not solve cube');
  return { facelets:cube.asString(), orientationId, quaternion:[...node(orientationId).quaternion] };
}

export function validatePlanV2(plan, options = {}) {
  if (!plan || plan.schemaVersion !== 2 || typeof plan.planId !== 'string' || !plan.planId) throw new Error('Invalid v2 metadata');
  if (!['ideal_preview', 'shadow', 'validated_execution'].includes(plan.mode) || plan.status !== 'planned') throw new Error('Invalid v2 mode/status');
  if (typeof plan.calibrationId !== 'string' || !plan.calibrationId || typeof plan.modelProfileId !== 'string' || !plan.modelProfileId) throw new Error('Invalid calibration/profile binding');
  if (!Number.isSafeInteger(plan.sourceRevision) || plan.sourceRevision < 0) throw new Error('Invalid sourceRevision');
  if (!plan.initialState || typeof plan.initialState.facelets !== 'string' || !Array.isArray(plan.initialState.quaternionPC_xyzw) || plan.initialState.quaternionPC_xyzw.length !== 4 || plan.initialState.quaternionPC_xyzw.some(x => !Number.isFinite(x)) || typeof plan.initialState.orientationId !== 'string') throw new Error('Invalid initial state');
  if (!Array.isArray(plan.solution) || !Array.isArray(plan.expandedTurns) || !Array.isArray(plan.actions) || !plan.statistics || !plan.cost) throw new Error('Invalid v2 schema');
  const map = validateCenterIdentityMap(plan.centerIdentityMap); if (!map.ok) throw new Error('Invalid center identity map');
  if (options.calibrationId !== undefined && options.calibrationId !== plan.calibrationId || options.modelProfileId !== undefined && options.modelProfileId !== plan.modelProfileId || options.sourceRevision !== undefined && options.sourceRevision !== plan.sourceRevision) throw new Error('Plan binding mismatch');
  if (plan.mode === 'validated_execution' && !options.costModel) throw new Error('validated plan requires costModel');
  const expectedStats = { faceTurns: plan.solution.length, quarterTurns: plan.expandedTurns.length, robotSteps: plan.actions.length, reorientations: plan.actions.filter(a => a.kind === 'flip').length };
  for (const [key, value] of Object.entries(expectedStats)) if (plan.statistics[key] !== value) throw new Error('Statistics mismatch');
  if (!Array.isArray(plan.cost.selected) || plan.cost.selected.some(x => !Number.isFinite(x) || x < 0)) throw new Error('Invalid selected cost');
  if (plan.cost.predictedProbability !== null && (!Number.isFinite(plan.cost.predictedProbability) || plan.cost.predictedProbability < 0 || plan.cost.predictedProbability > 1)) throw new Error('Invalid predicted probability');
  const costModel = options.costModel ?? createCostModel({ mode: plan.cost.mode, executionMode: plan.mode });
  if (plan.mode === 'validated_execution' && costModel.executionMode !== 'validated_execution') throw new Error('Execution cost model mode mismatch');
  if (costModel.mode !== plan.cost.mode) throw new Error('Cost mode mismatch');
  const recomputed = [...costModel.zero]; let flips = 0; let turns = 0; let duration = 0;
  for (const action of plan.actions) {
    if (!Number.isFinite(action.durationMs) || action.durationMs < 0) throw new Error('Invalid action duration');
    const c = costModel.actionCost({ ...action, orientationId: action.expectedBefore.orientationId, startOrientationId: action.expectedBefore.orientationId, durationMs: action.durationMs });
    if (!c) throw new Error('Cost model rejected action');
    const values = Array.isArray(c) ? c : costModel.mode === 'risk' ? [c.risk, c.durationMs] : [c.flips, c.quarterTurns, c.durationMs];
    for (let i = 0; i < recomputed.length; i++) recomputed[i] += values[i];
    if (action.kind === 'flip') flips++; if (action.kind === 'turn') turns++; duration += action.durationMs;
  }
  if (costModel.mode === 'proxy' && (plan.cost.selected[0] !== flips || plan.cost.selected[1] !== turns || plan.cost.selected[2] !== duration)) throw new Error('Proxy cost mismatch');
  if (plan.cost.selected.length !== recomputed.length || recomputed.some((x, i) => Math.abs(x - plan.cost.selected[i]) > 1e-9)) throw new Error('Selected cost mismatch');
  replayPlanV2(plan); return true;
}
