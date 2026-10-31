const FLIPS = new Set(['flip_A', 'flip_B']);
const TURNS = new Set(['turn_CW90', 'turn_CCW90', 'turn_CW180', 'turn_CCW180']);

export function validateProfile(profile, expected = {}) {
  if (!profile || typeof profile !== 'object') return { ok: false, reason: 'missing profile' };
  if (expected.profileId != null && profile.profileId !== expected.profileId) return { ok: false, reason: 'profile mismatch' };
  if (expected.calibrationId != null && profile.calibrationId !== expected.calibrationId) return { ok: false, reason: 'calibration mismatch' };
  if (!profile.edges || typeof profile.edges !== 'object') return { ok: false, reason: 'missing edge table' };
  return { ok: true };
}

export function costAction(action = {}, options = {}) {
  const primitive = action.primitive;
  if (!FLIPS.has(primitive) && !TURNS.has(primitive)) throw new RangeError(`unknown primitive: ${primitive}`);
  const quarterTurns = TURNS.has(primitive) ? (primitive.endsWith('180') ? 2 : 1) : 0;
  const mode = options.mode ?? 'proxy';
  if (mode === 'proxy') {
    const durationMs = action.durationMs ?? 0;
    if (!Number.isFinite(durationMs) || durationMs < 0) throw new Error('duration must be finite and non-negative');
    return { flips: FLIPS.has(primitive) ? 1 : 0, quarterTurns, durationMs, tieBreak: String(action.actionId ?? primitive), executable: true, primitive };
  }
  if (mode !== 'risk') throw new RangeError(`unknown cost mode: ${mode}`);
  const profile = options.profile; const check = validateProfile(profile, options);
  if (!check.ok) throw new Error(check.reason);
  const edgeKey = `${primitive}@${action.startOrientationId ?? ''}`; const edge = profile.edges[edgeKey];
  if (!edge || edge.disabled === true || !Number.isFinite(edge.pSafe) || edge.pSafe <= 0 || edge.pSafe > 1) throw new Error(`unknown or disabled risk edge: ${edgeKey}`);
  const durationMs = Number(edge.durationMs ?? action.durationMs ?? 0); if (!Number.isFinite(durationMs) || durationMs < 0) throw new Error('duration must be finite and non-negative');
  return { flips: FLIPS.has(primitive) ? 1 : 0, quarterTurns, durationMs, risk: -Math.log(edge.pSafe), pSafe: edge.pSafe, tieBreak: String(action.actionId ?? edgeKey), executable: true, primitive };
}

export function edgeAllowed(edgeKey, profile, options = {}) {
  if (options.mode !== 'risk') return true;
  const edge = profile?.edges?.[edgeKey]; return !!edge && Number.isFinite(edge.pSafe) && edge.pSafe > 0 && edge.pSafe <= 1;
}

export function scorePlan(actions = [], options = {}) {
  const mode = options.mode ?? 'proxy'; const costs = actions.map((a) => costAction(a, options));
  const flips = costs.reduce((s, c) => s + c.flips, 0); const quarterTurns = costs.reduce((s, c) => s + c.quarterTurns, 0); const durationMs = costs.reduce((s, c) => s + c.durationMs, 0);
  if (mode === 'risk') { const value = costs.reduce((s, c) => s + c.risk, 0) + (options.lambdaTime ?? 0) * durationMs; return { mode, value, costs, executable: costs.every((c) => c.executable), tuple: [value, flips, quarterTurns, durationMs, costs.map((c) => c.tieBreak).join('|')] }; }
  const tuple = [flips, quarterTurns, durationMs, costs.map((c) => c.tieBreak).join('|')]; return { mode, value: tuple, costs, executable: true, tuple };
}

export function compareCosts(a, b) {
  const aa = a.tuple; const bb = b.tuple; for (let i = 0; i < Math.min(aa.length, bb.length); i++) { if (aa[i] < bb[i]) return -1; if (aa[i] > bb[i]) return 1; } return 0;
}

// Small protocol-facing facade used by the planner. Null means the edge is
// not admissible; unknown risk data is never fabricated.
export function createCostModel({ mode = 'proxy', executionMode = 'ideal_preview', profile = null, modelProfileId, calibrationId, lambdaTime = 0 } = {}) {
  if (!['proxy', 'risk'].includes(mode)) throw new RangeError(`unknown cost mode: ${mode}`);
  if (!['ideal_preview','shadow','validated_execution'].includes(executionMode)) throw new RangeError('unknown execution mode');
  profile = profile ? structuredClone(profile) : null;
  if (!Number.isFinite(lambdaTime) || lambdaTime < 0) throw new RangeError('lambdaTime must be finite and non-negative');
  if (mode === 'risk' && executionMode !== 'ideal_preview' && !validateProfile(profile, { profileId: modelProfileId, calibrationId }).ok) throw new Error('invalid risk profile');
  const zero = mode === 'proxy' ? [0, 0, 0] : [0, 0];
  const add = (a, b) => a.map((x, i) => x + b[i]);
  const compare = (a, b) => { for (let i = 0; i < a.length; i++) { if (a[i] < b[i]) return -1; if (a[i] > b[i]) return 1; } return 0; };
  const probability = (action) => { const e = profile?.edges?.[`${action.primitive}@${action.orientationId ?? action.startOrientationId ?? ''}`]; return Number.isFinite(e?.pSafe) ? e.pSafe : null; };
  const actionCost = (action = {}) => {
    try {
      if (executionMode === 'validated_execution') {
        if (!validateProfile(profile, { profileId: modelProfileId, calibrationId }).ok) return null;
        const edge = profile.edges[`${action.primitive}@${action.orientationId ?? action.startOrientationId ?? ''}`];
        if (!edge || edge.disabled === true || edge.validated !== true) return null;
        action = { ...action, durationMs: edge.durationMs };
      }
      const c = costAction({ ...action, startOrientationId: action.orientationId ?? action.startOrientationId }, { mode, profile, profileId: modelProfileId, calibrationId, lambdaTime });
      return mode === 'proxy' ? [c.flips, c.quarterTurns, c.durationMs] : [c.risk + lambdaTime * c.durationMs, c.durationMs];
    } catch { return null; }
  };
  return { mode, executionMode, zero, add, compare, actionCost, probability, contextIndependent: true };
}
