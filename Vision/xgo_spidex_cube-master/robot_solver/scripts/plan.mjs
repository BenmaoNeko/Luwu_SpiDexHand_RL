import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, resolve } from "node:path";
import { Cube, cubeFromFacelets, parseMoves } from "../src/cube-model.js";
import { compileRobotPlan, yawOrientation } from "../src/planner.js";
import { solveFacelets } from "../src/solver-core.js";

const usage = `Usage:
  node scripts/plan.mjs --scramble "R U F2 L' D" --yaw 25 --output outputs/plan.json
  node scripts/plan.mjs --facelets <54 URFDLB characters> --output outputs/plan.json
  node scripts/plan.mjs --state-file state.txt --output outputs/plan.json
  node scripts/plan.mjs --random --output outputs/plan.json
  node scripts/plan.mjs --pool-count 256 --seed 20261008 --output outputs/pool.json
`;

const SOLVER_SOURCES = [
  "scripts/plan.mjs",
  "src/cube-model.js",
  "src/solver-core.js",
  "src/planner.js",
];

async function solverFingerprint() {
  const hash = createHash("sha256");
  for (const name of SOLVER_SOURCES) {
    hash.update(name);
    hash.update("\0");
    hash.update(await readFile(new URL(`../${name}`, import.meta.url)));
    hash.update("\0");
  }
  return hash.digest("hex");
}

function makeRandom(seed) {
  let state = seed >>> 0;
  if (state === 0) state = 0x6d2b79f5;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function randomScramble(random, minLength, maxLength) {
  const faces = "URFDLB";
  const suffixes = ["", "'", "2"];
  const length = minLength + Math.floor(random() * (maxLength - minLength + 1));
  const moves = [];
  for (let i = 0; i < length; i++) {
    let face;
    do face = faces[Math.floor(random() * faces.length)];
    while (moves.length > 0 && moves.at(-1)[0] === face);
    moves.push(face + suffixes[Math.floor(random() * suffixes.length)]);
  }
  return moves.join(" ");
}

async function makePlanPool({ count, seed, minLength, maxLength }) {
  if (!Number.isInteger(count) || count < 1)
    throw new Error("--pool-count must be a positive integer");
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff)
    throw new Error("--seed must be an unsigned 32-bit integer");
  if (
    !Number.isInteger(minLength) ||
    !Number.isInteger(maxLength) ||
    minLength < 1 ||
    maxLength < minLength
  )
    throw new Error("Scramble length bounds must be positive and ordered");

  const { initializeSolver } = await import("../src/solver-core.js");
  initializeSolver();
  const plans = [];
  const seen = new Set();
  let attempts = 0;
  while (plans.length < count) {
    if (++attempts > count * 100)
      throw new Error("Could not generate enough distinct scramble states");
    const planSeed = (seed + attempts * 0x9e3779b9) >>> 0;
    const random = makeRandom(planSeed);
    const scramble = randomScramble(random, minLength, maxLength);
    const cube = new Cube().move(scramble);
    const initialFacelets = cube.asString();
    if (seen.has(initialFacelets)) continue;
    seen.add(initialFacelets);
    const solution = solveFacelets(initialFacelets);
    const plan = compileRobotPlan(initialFacelets, solution);
    plans.push({
      seed: planSeed,
      scramble,
      initialFacelets,
      solution: plan.solution,
      stateFormat: plan.stateFormat,
      actions: plan.actions.map(({ type, notation, targetFace, moveIndex }) => ({
        type,
        notation,
        targetFace,
        moveIndex,
      })),
    });
  }
  return {
    schema_version: 1,
    solver_fingerprint: await solverFingerprint(),
    seed,
    scramble_length: [minLength, maxLength],
    plans,
  };
}
try {
  const options = {};
  const args = process.argv.slice(2);
  if (args.includes("--help")) {
    console.log(usage);
    process.exit(0);
  }
  for (let i = 0; i < args.length; i++) {
    const key = args[i];
    if (key === "--random") options.random = true;
    else if (
      [
        "--scramble",
        "--facelets",
        "--state-file",
        "--yaw",
        "--output",
        "--pool-count",
        "--seed",
        "--scramble-min",
        "--scramble-max",
      ].includes(key) &&
      args[i + 1] !== undefined
    )
      options[key.slice(2)] = args[++i];
    else throw new Error(`Unknown/missing argument: ${key}`);
  }
  const inputs = ["scramble", "facelets", "state-file", "random"].filter(
    (key) => options[key] !== undefined,
  );
  if (options["pool-count"] !== undefined) {
    if (inputs.length !== 0)
      throw new Error("--pool-count cannot be combined with a single-plan input");
    const pool = await makePlanPool({
      count: Number(options["pool-count"]),
      seed: Number(options.seed ?? 20261008),
      minLength: Number(options["scramble-min"] ?? 2),
      maxLength: Number(options["scramble-max"] ?? 6),
    });
    const json = JSON.stringify(pool, null, 2) + "\n";
    if (options.output) {
      const path = resolve(options.output);
      await mkdir(dirname(path), { recursive: true });
      await writeFile(path, json);
      console.log(`Saved plan pool: ${path} (${pool.plans.length} unique states)`);
    } else console.log(json);
    process.exit(0);
  }
  if (inputs.length !== 1)
    throw new Error(
      "Choose exactly one input: --scramble / --facelets / --state-file / --random",
    );
  let cube;
  if (options.scramble !== undefined)
    cube = new Cube().move(
      parseMoves(options.scramble)
        .map((m) => m.token)
        .join(" "),
    );
  else if (options.random) cube = Cube.random();
  else
    cube = cubeFromFacelets(
      options.facelets ?? (await readFile(options["state-file"], "utf8")),
    );
  const solution = solveFacelets(cube.asString());
  const plan = compileRobotPlan(
    cube.asString(),
    solution,
    yawOrientation(Number(options.yaw ?? 0)),
  );
  const json = JSON.stringify(plan, null, 2) + "\n";
  if (options.output) {
    const path = resolve(options.output);
    await mkdir(dirname(path), { recursive: true });
    await writeFile(path, json);
    console.log(`Solution: ${plan.solution.join(" ") || "(already solved)"}`);
    console.log(
      `Moves: ${plan.statistics.faceTurns}; robot steps: ${plan.statistics.robotSteps}; reorientation: ${plan.statistics.reorientationCostDeg.toFixed(1)} deg`,
    );
    console.log(`Saved: ${path}`);
  } else console.log(json);
} catch (error) {
  console.error(error.message + "\n" + usage);
  process.exitCode = 1;
}
