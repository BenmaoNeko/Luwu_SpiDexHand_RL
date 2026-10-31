import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { Cube, cubeFromFacelets, parseMoves } from '../src/cube-model.js';
import { DEMO_CENTER_IDENTITY_MAP } from '../src/palm-frame.js';
import { solveRisk } from '../src/risk-worker-client.js';

const usage = 'Usage: node scripts/plan-risk.mjs (--facelets <54> | --scramble <moves>) --initial-quat-xyzw x,y,z,w --calibration-id ID (--demo-center-map | --center-map FILE) [--mode MODE] [--cost-mode MODE] [--profile FILE] [--output FILE]';
const args = process.argv.slice(2); const opts = {};
for (let i = 0; i < args.length; i += 1) {
  const key = args[i];
  if (key === '--demo-center-map') opts.demo = true;
  else if (['--facelets','--scramble','--initial-quat-xyzw','--calibration-id','--model-profile-id','--output','--max-time-ms','--time-budget-ms','--mode','--cost-mode','--profile','--center-map','--max-solver-calls','--max-candidates','--max-prefix-depth','--beam-width'].includes(key)) opts[key.slice(2)] = args[++i];
  else if (key === '--help') { console.log(usage); process.exit(0); }
  else throw new Error(`Unknown argument: ${key}`);
}
if ((opts.facelets === undefined) === (opts.scramble === undefined)) throw new Error('Choose exactly one of --facelets or --scramble');
if (!opts['initial-quat-xyzw'] || !opts['calibration-id']) throw new Error('--initial-quat-xyzw and --calibration-id are required');
if (!opts.demo && !opts['center-map']) throw new Error('one of --demo-center-map or --center-map is required');
const initialQuaternion = opts['initial-quat-xyzw'].split(',').map(Number);
if (initialQuaternion.length !== 4 || initialQuaternion.some(v => !Number.isFinite(v))) throw new Error('invalid --initial-quat-xyzw');
const cube = opts.scramble !== undefined ? new Cube().move(parseMoves(opts.scramble).map(m => m.token).join(' ')) : cubeFromFacelets(opts.facelets);
const centerIdentityMap = opts.demo ? DEMO_CENTER_IDENTITY_MAP : JSON.parse(await readFile(resolve(opts['center-map']), 'utf8'));
const numberOption = name => opts[name] === undefined ? undefined : Number(opts[name]);
const timeBudgetMs = numberOption('time-budget-ms') ?? numberOption('max-time-ms') ?? 30000;
if (!Number.isSafeInteger(timeBudgetMs) || timeBudgetMs < 1) throw new Error('--time-budget-ms must be a positive safe integer');
const searchOptions = Object.fromEntries(Object.entries({ maxSolverCalls: numberOption('max-solver-calls'), maxCandidates: numberOption('max-candidates'), maxPrefixDepth: numberOption('max-prefix-depth'), beamWidth: numberOption('beam-width') }).filter(([, value]) => value !== undefined));
for (const [name, value] of Object.entries(searchOptions)) {
  const min = name === 'maxPrefixDepth' ? 0 : 1;
  if (!Number.isSafeInteger(value) || value < min) throw new Error(`--${name.replace(/[A-Z]/g, letter => `-${letter.toLowerCase()}`)} must be a valid integer`);
}
const profile = opts.profile ? JSON.parse(await readFile(resolve(opts.profile), 'utf8')) : undefined;
const plan = await solveRisk(cube.asString(), {
  initialQuaternion, calibrationId: opts['calibration-id'], modelProfileId: opts['model-profile-id'] ?? 'preview',
  centerIdentityMap, mode: opts.mode ?? 'ideal_preview', costMode: opts['cost-mode'] ?? 'proxy',
  ...(profile ? { profile } : {}),
  ...searchOptions,
  maxTimeMs: timeBudgetMs,
});
const json = `${JSON.stringify(plan, null, 2)}\n`;
if (opts.output) await writeFile(resolve(opts.output), json); else process.stdout.write(json);
