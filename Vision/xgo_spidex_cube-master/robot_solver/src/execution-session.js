import { validatePlanV2 as validateRobotPlanV2 } from './robot-plan-v2.js';

/**
 * Action-level execution owner for v2 plans.
 * The adapter is deliberately tiny so this module can be tested without hardware.
 */
export const SESSION_STATES = Object.freeze({
  IDLE: 'idle', READY: 'ready', WAIT_STABLE: 'wait_stable', VERIFY: 'verify',
  SOLVED: 'solved', PAUSED_FAULT: 'paused_fault',
});

function bindingMatches(value, plan, state) {
  return value && value.planId === plan.planId && value.calibrationId === plan.calibrationId &&
    value.modelProfileId === plan.modelProfileId && value.revision === state.revision &&
    (value.sourceRevision === undefined || value.sourceRevision === plan.sourceRevision);
}

function actionBinding(plan, state) {
  return { planId: plan.planId, calibrationId: plan.calibrationId,
    modelProfileId: plan.modelProfileId, sourceRevision: plan.sourceRevision, revision: state.revision };
}

export class ExecutionSession {
  constructor({ plan, adapter, costModel, now = () => Date.now(), actionTimeoutMs = 30000, maxObservationAgeMs = 1000, stableWindowMs = 400 } = {}) {
    validateRobotPlanV2(plan, { costModel });
    if (!Number.isFinite(actionTimeoutMs) || actionTimeoutMs <= 0 || !Number.isFinite(maxObservationAgeMs) || maxObservationAgeMs <= 0 || !Number.isFinite(stableWindowMs) || stableWindowMs < 0) throw new RangeError('invalid execution timing');
    if (!adapter || typeof adapter.readState !== 'function') throw new TypeError('adapter.readState is required');
    this.plan = structuredClone(plan); this.adapter = adapter; this.costModel = costModel; this.now = now;
    this.actionTimeoutMs = actionTimeoutMs; this.maxObservationAgeMs = maxObservationAgeMs; this.stableWindowMs = stableWindowMs;
    this.state = SESSION_STATES.IDLE; this.current = null; this.observation = null;
    this.completedActions = []; this.dispatchedActionIds = new Set(); this.events = [];
    this.lastVerifiedRevision = plan.sourceRevision;
  }

  _pause(reason, detail) {
    this.state = SESSION_STATES.PAUSED_FAULT;
    this.fault = { reason, detail, at: this.now() };
    this.events.push({ type: 'paused', reason, detail });
    if (typeof this.adapter.safePause === 'function') {
      try {
        const result = this.adapter.safePause({ reason, detail });
        if (result && typeof result.catch === 'function') result.catch(() => {});
      } catch { /* pausing is best-effort; fault state remains authoritative */ }
    }
    return this.state;
  }

  _fresh(s) {
    return s && s.poseFresh === true && s.internalStateFresh === true && Number.isFinite(s.timestamp) &&
      s.timestamp <= this.now() && this.now() - s.timestamp <= this.maxObservationAgeMs;
  }

  _stateBindingValid(s) {
    return s && s.planId === this.plan.planId && s.calibrationId === this.plan.calibrationId &&
      s.modelProfileId === this.plan.modelProfileId && Number.isSafeInteger(s.revision) && s.revision >= 0 &&
      s.sourceRevision === this.plan.sourceRevision;
  }

  _executionAllowed(action) {
    if (this.plan.mode === 'shadow') return false;
    if (this.plan.mode === 'ideal_preview') return this.adapter.simulation === true;
    if (this.plan.mode === 'validated_execution') {
      return this.costModel?.executionMode === 'validated_execution' &&
        typeof this.adapter.canExecuteAction === 'function' && this.adapter.canExecuteAction(action, this.plan) === true;
    }
    return false;
  }

  _readState() {
    const state = this.adapter.readState();
    if (state && typeof state.then === 'function') throw new TypeError('readState must be synchronous');
    return state;
  }

  observe() {
    if (this.state === SESSION_STATES.PAUSED_FAULT || this.state === SESSION_STATES.SOLVED) return this.state;
    let s;
    try { s = this._readState(); } catch (error) { return this._pause('STATE_READ_FAILED', error); }
    this.observation = s;
    if (!s || !this._fresh(s) || !this._stateBindingValid(s) || s.snapshotKnown !== true || s.moveWatermarkKnown !== true || s.motionState === 'unknown') return this._pause('OBSERVATION_UNSAFE');
    if (this.inflight && s.motionState === 'moving') return this.state;
    if (!this.current) {
      if (this.completedActions.length >= this.plan.actions.length) {
        if (this.plan.initialState?.facelets !== undefined && s.facelets !== this.plan.initialState.facelets) return this._pause('SOLVED_STATE_MISMATCH');
        return (this.state = SESSION_STATES.SOLVED);
      }
      this.current = this.plan.actions[this.completedActions.length];
    }
    if (!this.current) return (this.state = SESSION_STATES.SOLVED);
    if (s.motionState !== 'stable') return this._pause('NOT_STABLE');
    // An observation during an in-flight action is a barrier refresh, never a
    // new READY transition (which could permit a second dispatch).
    if (this.inflight) return this.state;
    this.state = SESSION_STATES.READY;
    return this.state;
  }

  async dispatch() {
    if (this.state !== SESSION_STATES.READY || !this.current) return this.state;
    const action = this.current;
    if (this.dispatchedActionIds.has(action.actionId)) return this._pause('ACTION_ALREADY_DISPATCHED');
    let s;
    try { s = this._readState(); } catch (error) { return this._pause('STATE_READ_FAILED', error); }
    if (!this._stateBindingValid(s) || s.sourceRevision !== this.plan.sourceRevision || s.revision !== this.lastVerifiedRevision || !this._fresh(s) || s.snapshotKnown !== true || s.moveWatermarkKnown !== true || s.motionState !== 'stable' || s.unexpectedTurn) return this._pause('BINDING_CHANGED');
    if (!this._executionAllowed(action)) return this._pause('EXECUTION_NOT_AUTHORIZED');
    if (this.plan.mode === 'validated_execution' && typeof this.adapter.safePause !== 'function') return this._pause('SAFE_PAUSE_UNAVAILABLE');
    if (action.expectedBefore?.facelets !== undefined && s.facelets !== action.expectedBefore.facelets) return this._pause('PRECONDITION_MISMATCH');
    if (action.expectedBefore?.orientationId !== undefined && s.orientationId !== action.expectedBefore.orientationId) return this._pause('PRECONDITION_MISMATCH');
    const watermark = s.moveWatermark ?? s.moveSeq;
    if (!Number.isSafeInteger(watermark) || watermark < 0) return this._pause('MOVE_WATERMARK_UNKNOWN');
    const binding = actionBinding(this.plan, s);
    this.dispatchedActionIds.add(action.actionId);
    // Lock before awaiting the adapter: a second caller must never dispatch the
    // same action while the first request is still in flight.
    this.inflight = { actionId: action.actionId, dispatchedAt: this.now(), feedback: new Set(), binding,
      beforeRevision: s.revision, beforeTimestamp: s.timestamp, beforeSnapshotId: s.snapshotId,
      beforeMoveWatermark: watermark, beforeSequence: s.sequence ?? s.seq };
    this.state = SESSION_STATES.WAIT_STABLE;
    let result;
    try {
      result = typeof this.adapter.dispatchAction === 'function'
        ? await this.adapter.dispatchAction(structuredClone(action), structuredClone(binding)) : await this.adapter.dispatch(structuredClone(action), structuredClone(binding));
    } catch (error) { return this._pause('DISPATCH_FAILED', error); }
    if (this.state === SESSION_STATES.PAUSED_FAULT || !result || result.accepted !== true) return this._pause('DISPATCH_REJECTED', result);
    this.inflight.result = result;
    this.events.push({ type: 'dispatch', actionId: action.actionId });
    return this.state;
  }

  async feedback(feedback) {
    if (this.state === SESSION_STATES.PAUSED_FAULT || !this.inflight) return this.state;
    if (feedback?.actionId !== this.inflight.actionId) return this._pause('FEEDBACK_MISMATCH');
    if (!bindingMatches(feedback, this.plan, this.inflight.binding)) return this._pause('FEEDBACK_BINDING_MISMATCH');
    if (feedback.type === 'face_target_completed') this.inflight.feedback.add(feedback.type);
    else if (feedback.type === 'ab_orientation_verified' || feedback.type === 'turn_verified') this.inflight.feedback.add(feedback.type);
    else if (feedback.type === 'unexpected_turn') return this._pause('UNEXPECTED_TURN');
    else if (feedback.type === 'yaw_deviation') return this._pause('POSE_DEVIATION');
    return this.state;
  }

  async waitStable() {
    if (this.state !== SESSION_STATES.WAIT_STABLE || !this.inflight) return this.state;
    if (this.now() - this.inflight.dispatchedAt >= this.actionTimeoutMs) return this._pause('ACTION_TIMEOUT');
    let s;
    try { s = this._readState(); } catch (error) { return this._pause('STATE_READ_FAILED', error); }
    this.observation = s;
    if (!this._fresh(s) || !this._stateBindingValid(s) || s.snapshotKnown !== true || s.moveWatermarkKnown !== true || s.motionState === 'unknown') return this._pause('OBSERVATION_UNSAFE');
    const newerObservation = (s.timestamp !== undefined && s.timestamp > this.inflight.beforeTimestamp) ||
      (s.sequence !== undefined && s.sequence > (this.inflight.beforeSequence ?? -Infinity)) ||
      (s.seq !== undefined && s.seq > (this.inflight.beforeSequence ?? -Infinity));
    if (!newerObservation) return this._pause('STALE_SNAPSHOT');
    const watermark = s.moveWatermark ?? s.moveSeq;
    if (!Number.isSafeInteger(this.inflight.beforeMoveWatermark) || !Number.isSafeInteger(watermark)) return this._pause('MOVE_WATERMARK_UNKNOWN');
    const expectedWatermark = this.current.kind === 'turn' ? this.inflight.beforeMoveWatermark + 1 : this.inflight.beforeMoveWatermark;
    if (watermark !== expectedWatermark) return this._pause(watermark < expectedWatermark ? 'MOVE_WATERMARK_REGRESSION' : 'MOVE_WATERMARK_GAP');
    if (s.unexpectedTurn || s.internalTurnDetected) return this._pause('UNEXPECTED_TURN');
    if (s.motionState !== 'stable') { this.inflight.stableSince = undefined; return this.state; }
    if (this.stableWindowMs > 0) {
      if (this.inflight.stableSince === undefined) this.inflight.stableSince = this.now();
      if (this.now() - this.inflight.stableSince < this.stableWindowMs) return this.state;
    }
    this.state = SESSION_STATES.VERIFY;
    return this.state;
  }

  async verify() {
    if (this.state !== SESSION_STATES.VERIFY || !this.inflight || !this.observation) return this.state;
    const s = this.observation; const action = this.current;
    if (!this._fresh(s) || !this._stateBindingValid(s) || s.revision < this.inflight.binding.revision || s.snapshotKnown !== true || s.moveWatermarkKnown !== true) return this._pause('OBSERVATION_UNSAFE');
    if (s.yawDeviation || (action.expectedAfter.orientationId && s.orientationId !== action.expectedAfter.orientationId)) return this._pause('POSE_DEVIATION');
    const expectedFeedback = action.kind === 'turn' ? 'turn_verified' : 'ab_orientation_verified';
    if (!this.inflight.feedback.has(expectedFeedback)) return this._pause('ACTION_NOT_VERIFIED');
    if (action.expectedAfter.facelets !== undefined && s.facelets !== action.expectedAfter.facelets) return this._pause('INTERNAL_STATE_MISMATCH');
    this.completedActions.push(action.actionId); this.events.push({ type: 'verified', actionId: action.actionId });
    this.lastVerifiedRevision = Math.max(this.lastVerifiedRevision, s.revision);
    this.inflight = null; this.current = null;
    this.state = this.completedActions.length === this.plan.actions.length ? SESSION_STATES.SOLVED : SESSION_STATES.READY;
    return this.state;
  }

  async pause(reason = 'MANUAL_PAUSE') { return this._pause(reason); }

  snapshot() { return { state: SESSION_STATES.PAUSED_FAULT, completedActions: [...this.completedActions], dispatchedActionIds: [...this.dispatchedActionIds], lastVerifiedRevision: this.lastVerifiedRevision, fault: this.fault }; }

  static restore({ plan, adapter, persisted, ...options }) {
    const session = new ExecutionSession({ plan, adapter, ...options });
    session.completedActions = [...(persisted?.completedActions || [])];
    session.dispatchedActionIds = new Set(persisted?.dispatchedActionIds || []);
    session.lastVerifiedRevision = persisted?.lastVerifiedRevision ?? plan.sourceRevision;
    session.fault = persisted?.fault || { reason: 'PROCESS_RESTART' };
    session.state = SESSION_STATES.PAUSED_FAULT;
    if (typeof adapter.safePause === 'function') {
      try { const result = adapter.safePause({ reason: 'PROCESS_RESTART' }); if (result?.catch) result.catch(() => {}); } catch { /* remain paused */ }
    }
    return session;
  }
}

export function createFakeAdapter(initialState = {}) {
  const adapter = {
    state: { ...initialState }, dispatches: [], safePauses: 0, powerOffs: 0,
    readState() { return this.state; },
    async dispatchAction(action, binding) { this.dispatches.push({ action, binding }); return { accepted: true }; },
    safePause() { this.safePauses += 1; },
    powerOff() { this.powerOffs += 1; },
  };
  return adapter;
}

export default ExecutionSession;
