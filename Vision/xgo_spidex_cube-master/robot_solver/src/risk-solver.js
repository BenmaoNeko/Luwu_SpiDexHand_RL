import { cubeFromFacelets, parseMoves } from './cube-model.js';
import { solveFacelets } from './solver-core.js';
import { compilePlanV2 } from './robot-plan-v2.js';
import { createCostModel } from './action-cost.js';
import { classifyOrientation, normalInPalm } from './palm-frame.js';

export function normalizeAlgorithm(algorithm) {
  const stack=[];
  for (const move of parseMoves(algorithm)) {
    const amount=move.token.endsWith('2') ? 2 : move.token.endsWith("'") ? 3 : 1;
    if (stack.at(-1)?.face===move.face) {
      const previous=stack.pop(); const total=(previous.amount+amount)%4;
      if (total) stack.push({face:move.face,amount:total});
    } else stack.push({face:move.face,amount});
  }
  return stack.map(m=>m.face+(m.amount===2?'2':m.amount===3?"'":'')).join(' ');
}

/** Synchronous core; callers isolate this in a Worker so it can be terminated. */
export function solveRisk(facelets, options = {}, onPlan = () => {}) {
  const started=performance.now();
  const cube=cubeFromFacelets(facelets);
  const budget={ maxSolverCalls:64,maxCandidates:64,maxPrefixDepth:2,beamWidth:18,...options };
  for (const name of ['maxSolverCalls','maxCandidates','maxPrefixDepth','beamWidth']) {
    const min=name==='maxPrefixDepth'?0:1;
    if (!Number.isSafeInteger(budget[name]) || budget[name]<min) throw new Error(`Invalid search budget ${name}`);
  }
  if (budget.maxPrefixDepth>4) throw new Error('Search prefix budget exceeds supported depth 4');
  const costModel=options.costModel ?? createCostModel({...options,mode:options.costMode??'proxy',executionMode:options.mode??'ideal_preview'});
  const solver=options.solver ?? solveFacelets;
  const orientationId=classifyOrientation(options.initialQuaternion,options.maxPoseErrorDeg??15).orientationId;
  const top=['U','R','F','D','L','B'].find(f=>normalInPalm(orientationId,f)[2]===1);
  const faces=[top,...['U','R','F','D','L','B'].filter(f=>f!==top)];
  const tokens=faces.flatMap(f=>[f,f+"'",f+'2']);
  const seen=new Set(); const candidates=[];
  let calls=0,invalidCandidates=0,unreachableCandidates=0,cancelled=false,best=null,baseline=null;
  const compile=algorithm=>compilePlanV2(cube.asString(),algorithm,{...options,costModel});
  const publish=plan=>onPlan(structuredClone(plan));
  const baselineAlgorithm=normalizeAlgorithm(solver(cube.asString())); calls++;
  // Invalid baseline is a solver contract error, not a reason to emit a partial plan.
  if (!cube.clone().move(baselineAlgorithm).isSolved()) throw new Error('Baseline does not solve input cube');
  seen.add(baselineAlgorithm);
  try { baseline=compile(baselineAlgorithm); best=baseline; candidates.push(baseline); }
  catch(error) { if(error.code!=='NO_VALIDATED_ROUTE') throw error; unreachableCandidates++; }
  const metadata=()=>({ calls,candidateCount:seen.size,invalidCandidates,unreachableCandidates,cancelled,
    timedOut:false,optimality:'candidate-limited',elapsedMs:performance.now()-started,
    budget:Object.fromEntries(['maxSolverCalls','maxCandidates','maxPrefixDepth','beamWidth'].map(k=>[k,budget[k]])) });
  const decorate=plan=>{
    plan.search=metadata(); plan.cost.baseline=baseline?.cost.selected??null;
    return plan;
  };
  if(best) publish(decorate(best));
  let frontier=[[]];
  outer: for(let depth=1;depth<=budget.maxPrefixDepth;depth++) {
    const next=[];
    for(const prefix of frontier) for(const token of tokens) {
      if(prefix.at(-1)?.[0]===token[0]) continue;
      if(options.isCancelled?.()) { cancelled=true; break outer; }
      if(calls>=budget.maxSolverCalls || seen.size>=budget.maxCandidates) break outer;
      const extended=[...prefix,token];
      if(next.length<budget.beamWidth) next.push(extended);
      const after=cube.clone().move(extended.join(' '));
      const tail=solver(after.asString()); calls++;
      let algorithm;
      try {
        algorithm=normalizeAlgorithm([...extended,...parseMoves(tail).map(m=>m.token)]);
        if(!cube.clone().move(algorithm).isSolved()) { invalidCandidates++; continue; }
      } catch { invalidCandidates++; continue; }
      if(seen.has(algorithm)) continue;
      seen.add(algorithm);
      let plan;
      try { plan=compile(algorithm); }
      catch(error) { if(error.code==='NO_VALIDATED_ROUTE') { unreachableCandidates++; continue; } throw error; }
      candidates.push(plan);
      if(!best || costModel.compare(plan.cost.selected,best.cost.selected)<0) {
        best=plan; publish(decorate(best));
      }
    }
    frontier=next;
  }
  if(!best) { const error=new Error('NO_VALIDATED_ROUTE'); error.code='NO_VALIDATED_ROUTE'; throw error; }
  const result=decorate(best);
  if(options.includeCandidates) result.candidateSolutions=candidates.map(p=>p.solution);
  return result;
}
