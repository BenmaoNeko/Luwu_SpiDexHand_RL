import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const V2_FINGERPRINT_FILES = [
  'package.json',
  'package-lock.json',
  'src/action-cost.js',
  'src/cube-model.js',
  'src/palm-frame.js',
  'src/robot-plan-v2.js',
  'src/risk-solver.js',
  'src/solver-core.js',
  'src/risk-solver.worker.js',
  'scripts/risk-worker.mjs',
];

export async function computeV2Fingerprint() {
  const hash = createHash('sha256');
  for (const relative of V2_FINGERPRINT_FILES) {
    const bytes = await readFile(resolve(ROOT, relative));
    hash.update(relative);
    hash.update('\0');
    hash.update(bytes);
    hash.update('\0');
  }
  return `sha256:${hash.digest('hex')}`;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(await computeV2Fingerprint());
}
