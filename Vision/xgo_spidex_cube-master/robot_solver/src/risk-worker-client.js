function makeWorker(workerFactory) {
  if (workerFactory) return Promise.resolve(workerFactory());
  if (typeof process !== 'undefined' && process.versions?.node) {
    // Keep the Node-only module out of the browser/Vite dependency graph.
    const dynamicImport = Function('specifier', 'return import(specifier)');
    return dynamicImport('node:worker_threads').then(({ Worker }) => new Worker(new URL('../scripts/risk-worker.mjs', import.meta.url), { execArgv: [] }));
  }
  return Promise.resolve(new Worker(new URL('./risk-solver.worker.js', import.meta.url), { type: 'module' }));
}

export function createRiskWorkerRequest({ facelets, options = {}, id = globalThis.crypto?.randomUUID?.() ?? `risk-${Date.now()}`, timeBudgetMs = options.maxTimeMs ?? 30000, onPlan = () => {}, onDone = () => {}, onError = () => {}, workerFactory } = {}) {
  let worker;
  let cancelled = false;
  let settled = false;
  let timer;
  let latestPlan;
  let rejectPromise;
  const promise = makeWorker(workerFactory).then(instance => new Promise((resolve, reject) => {
    worker = instance;
    rejectPromise = reject;
    if (cancelled) { settled = true; void worker.terminate(); const error = new Error('risk solver cancelled'); error.code = 'CANCELLED'; reject(error); return; }
    const close = () => { if (typeof worker.terminate === 'function') void worker.terminate(); };
    const fail = (error) => { if (settled) return; settled = true; clearTimeout(timer); close(); onError(error); reject(error); };
    timer = setTimeout(() => {
      if (cancelled || settled) return;
      settled = true; cancelled = true; close();
      if (latestPlan) { const timedOut = structuredClone(latestPlan); timedOut.search = { ...(timedOut.search || {}), timedOut: true, error: 'TIMEOUT' }; onDone(timedOut); resolve(timedOut); }
      else { const error = new Error('risk solver timed out'); error.code = 'TIMEOUT'; onError(error); reject(error); }
    }, timeBudgetMs);
    const receive = message => {
      const data = message?.data ?? message;
      if (data.id !== id || cancelled || settled) return;
      if (data.type === 'plan') { latestPlan = data.plan; onPlan(data.plan); }
      else if (data.type === 'done') { settled = true; clearTimeout(timer); close(); onDone(data.plan); resolve(data.plan); }
      else if (data.type === 'error') { const error = new Error(data.message); error.code = data.code; fail(error); }
    };
    if (typeof worker.on === 'function') worker.on('message', receive); else worker.onmessage = receive;
    if (typeof worker.on === 'function') { worker.on('error', fail); worker.on('exit', code => { if (code !== 0 && !settled) fail(new Error(`risk worker exited with code ${code}`)); }); }
    else worker.onerror = event => fail(event.error || new Error(event.message || 'risk worker error'));
    worker.postMessage({ id, facelets, options });
  })).catch(error => {
    if (settled) throw error;
    settled = true;
    clearTimeout(timer);
    onError(error);
    throw error;
  });
  const cancel = () => {
    if (cancelled || settled) return;
    cancelled = true;
    const error = new Error('risk solver cancelled'); error.code = 'CANCELLED';
    if (worker) void worker.terminate();
    if (rejectPromise) { settled = true; clearTimeout(timer); onError(error); rejectPromise(error); }
  };
  return { id, promise, cancel };
}

export function solveRisk(facelets, options = {}, onPlan = () => {}) {
  return createRiskWorkerRequest({ facelets, options, timeBudgetMs: options.maxTimeMs ?? 30000, onPlan }).promise;
}
