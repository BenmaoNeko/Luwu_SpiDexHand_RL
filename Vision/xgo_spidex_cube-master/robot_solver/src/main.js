import "./style.css";
import { expandTurns } from './robot-plan-v2.js';
import {
  Cube,
  FACES,
  FACE_ORDER,
  SOLVED,
  COMMAND_LABELS,
  cubeFromFacelets,
} from "./cube-model.js";
import {
  compileRobotPlan,
  executeRobotAction,
  topFace,
  yawOrientation,
} from "./planner.js";
import { CubeScene } from "./scene.js";
import { referenceOrientation, DEMO_CENTER_IDENTITY_MAP } from "./palm-frame.js";
import { toPreviewAction, executePreviewAction, moveProgress } from "./plan-v2-view.js";

document.querySelector("#app").innerHTML = `
  <header class="topbar">
    <div class="brand"><span class="brand-mark" aria-hidden="true">◇</span><div><h1>魔方 · 机器人还原</h1><p>整体重定向 / 仅转顶层</p></div></div>
    <nav class="toolbar" aria-label="主要操作">
      <button id="scramble"><span aria-hidden="true">⤨</span>随机打乱</button>
      <button id="plan" class="primary" disabled><span aria-hidden="true">⌘</span>规划还原步骤</button>
      <button id="step" disabled><span aria-hidden="true">▷|</span>逐步还原</button>
      <button id="play" class="play" disabled><span aria-hidden="true">▶</span>连续还原</button>
    </nav>
  </header>
  <main class="workspace">
    <section class="workbench" aria-label="三维工作台">
      <div class="bench-heading"><div><span class="eyebrow">三维工作台</span><strong id="top-face">黄 U 面朝上</strong></div><span class="dimension">56 × 56 × 56 mm</span></div>
      <div class="stage-wrap">
        <div id="scene"></div>
        <div class="stage-status"><span class="status-tag" id="phase-tag">已还原</span><span id="action-caption">随机打乱，开始一次规划</span></div>
        <div class="view-hint">拖动观察 · 滚轮缩放</div>
        <div class="table-label">桌面 Z = 0</div>
      </div>
      <div class="playback">
        <div class="playback-line"><span id="progress-label">尚未规划</span><label class="speed-label">速度 <select id="speed" aria-label="播放速度"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label></div>
        <div id="progress" class="progress" role="progressbar" aria-label="还原进度" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"><div id="progress-fill"></div></div>
        <div class="metrics"><div><span>外层转动</span><strong id="turn-count">—</strong></div><div><span>整体重定向</span><strong id="reorient-count">—</strong></div><div><span>累计重定向</span><strong id="rotation-cost">—</strong></div></div>
      </div>
      <div class="state-controls">
        <label>规划方式 <select id="planner-mode" aria-label="规划方式"><option value="risk">掌心 A/B · 优先少翻面</option><option value="legacy">原版最小角度</option></select></label>
        <button id="align-palm">对齐当前朝向到掌心</button>
        <label>初始偏航 <input id="yaw" type="number" min="-180" max="180" step="5" value="0" aria-label="初始偏航角" disabled> °</label>
        <span id="cost-label">参考：白上、蓝右、红前 · 离线预览</span>
        <span id="solver-status" role="status">求解器准备中…</span>
      </div>
      <details class="input-panel"><summary>导入任意魔方状态 <span>54 面片 · URFDLB</span></summary>
        <p>每面从正面看按行排列，每面 9 个字符。中心：U 黄、R 蓝、F 紫、D 白、L 绿、B 红。</p>
        <label for="facelet-input">面片状态</label><textarea id="facelet-input" rows="3" spellcheck="false" aria-label="面片状态">${SOLVED}</textarea>
        <div class="input-actions"><button id="import-state">载入状态</button><button id="copy-current">填入当前状态</button></div>
        <label for="current-state">当前状态</label><textarea id="current-state" rows="2" readonly aria-label="当前魔方面片状态">${SOLVED}</textarea>
      </details>
      <p id="message" class="message" role="status" aria-live="polite"></p>
    </section>
    <aside class="sequence-panel" aria-label="还原步骤序列">
      <div class="sequence-heading"><div><span class="eyebrow">执行计划</span><h2>还原步骤</h2></div><button id="export" class="quiet" disabled>导出 JSON</button></div>
      <div class="notation-box"><span class="small-label">标准面转序列</span><div id="notation" class="notation empty-notation">等待规划</div></div>
      <div class="action-list-heading"><span>机器人动作</span><span id="action-count">0 步</span></div>
      <ol id="actions" class="action-list"><li class="empty"><div class="empty-icon" aria-hidden="true">↻</div><strong>从一个打乱状态开始</strong><p>先随机打乱，再规划还原步骤。每次面转会分成重定向与顶层转动。</p></li></ol>
      <div class="legend">${FACE_ORDER.map((face) => `<span><i style="background:${FACES[face].color}"></i>${face} ${FACES[face].colorName} <small>${FACES[face].axis}</small></span>`).join("")}</div>
      <p class="plan-note">A：上→前→下→后；B：上→左→下→右（已反向）。顺 / 逆时针从操作面外侧看。本页为离线预览，动画不代表实机反馈。</p>
    </aside>
  </main>
  <footer><span>cube.js · 候选解筛选 + 掌心姿态规划</span><span id="search-summary">代理代价：基本翻面优先 · 未标定成功率</span></footer>
  <dialog id="export-dialog" aria-labelledby="export-title">
    <div class="export-heading"><h2 id="export-title">机器人动作计划</h2><button id="close-export" aria-label="关闭导出">关闭</button></div>
    <p>包含起始魔方状态、目标姿态，以及重定向和顶层转动指令。</p>
    <textarea id="export-json" readonly rows="18" spellcheck="false" aria-label="机器人计划 JSON"></textarea>
    <div class="export-actions"><span id="export-message" role="status">可以复制到控制程序，或保存为文件。</span><button id="copy-plan" class="primary">复制 JSON</button><button id="save-plan">保存文件</button></div>
  </dialog>
`;

const $ = (id) => document.getElementById(id);
let cube = new Cube();
let orientation = [...referenceOrientation.quaternion];
let plan = null,
  cursor = 0,
  busy = false,
  playing = false,
  solving = false,
  ready = false;
let requestId = 0,
  requestSnapshot = null;
let riskWorker = null, riskTimer = null, latestRiskPlan = null;
const isRiskMode = () => $("planner-mode").value === "risk";
let scene;
try {
  scene = new CubeScene($("scene"));
  scene.setState(cube.asString(), orientation);
} catch (error) {
  $("message").textContent =
    `三维视图无法初始化：${error.message}。请使用支持 WebGL 的浏览器。`;
  $("message").classList.add("error");
}

function message(text = "", error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function setProgress(fraction = 0) {
  const total = plan?.actions.length ?? 0;
  const percent = total
    ? Math.min(100, ((cursor + fraction) / total) * 100)
    : plan
      ? 100
      : 0;
  $("progress-fill").style.width = `${percent}%`;
  $("progress").setAttribute("aria-valuenow", String(Math.round(percent)));
}

function refresh() {
  const locked = busy || playing || solving;
  const hasNext = !!plan && cursor < plan.actions.length;
  $("scramble").disabled = locked;
  $("plan").disabled = !ready || locked;
  $("step").disabled = !scene || !hasNext || locked;
  $("play").disabled = !scene || !hasNext || solving || (busy && !playing);
  $("play").innerHTML = playing
    ? '<span aria-hidden="true">Ⅱ</span>暂停播放'
    : '<span aria-hidden="true">▶</span>连续还原';
  $("import-state").disabled = locked;
  $("yaw").disabled = locked || !!plan;
  $("yaw").disabled ||= isRiskMode();
  $("planner-mode").disabled = locked;
  $("align-palm").disabled = locked || !isRiskMode();
  $("export").disabled = !plan || solving;
  $("current-state").value = cube.asString();
  const face = topFace(orientation);
  $("top-face").textContent = face
    ? `${FACES[face].colorName} ${face} 面朝上`
    : "等待对齐桌面";
  $("progress-label").textContent = plan
    ? `机器人动作 ${cursor} / ${plan.actions.length}`
    : "尚未规划";
  $("turn-count").textContent = plan ? String(plan.statistics.quarterTurns ?? plan.statistics.faceTurns) : "—";
  $("reorient-count").textContent = plan
    ? String(plan.statistics.reorientations)
    : "—";
  $("rotation-cost").textContent = plan
    ? `${Math.round(plan.statistics.reorientationCostDeg ?? plan.statistics.reorientations * 90)}°`
    : "—";
  document.querySelectorAll(".action-row").forEach((row, i) => {
    row.classList.toggle("done", i < cursor);
    row.classList.toggle("current", i === cursor);
    if (i === cursor) row.setAttribute("aria-current", "step");
    else row.removeAttribute("aria-current");
  });
  document.querySelectorAll(".move-chip").forEach((chip, i) => {
    const progress = plan?.schemaVersion === 2 ? moveProgress(plan,cursor) : null;
    chip.classList.toggle("done", progress ? i < progress.completed : i * 2 + 1 < cursor);
    chip.classList.toggle("current", progress ? i === progress.current : i === Math.floor(cursor / 2) && hasNext);
  });
  setProgress();
}

function actionText(action) {
  if (action.kind === "flip") return action.primitive === "flip_A" ? "翻面 A · 上→前→下→后" : "翻面 B · 上→左→下→右（已反向）";
  if (action.kind === "turn") return `${action.notation} · ${FACES[action.targetFace].colorName}面转动`;
  const f = FACES[action.targetFace];
  if (action.type === "reorient")
    return action.requiresMotion
      ? `重定向 · ${f.colorName}面朝上`
      : `保持朝向 · ${f.colorName}面已朝上`;
  return COMMAND_LABELS[action.command];
}

function renderPlan() {
  $("action-count").textContent = `${plan.actions.length} 步`;
  $("notation").classList.remove("empty-notation");
  $("notation").innerHTML = plan.solution.length
    ? plan.solution
        .map(
          (move) => `<span class="move-chip">${move}</span>`,
        )
        .join("")
    : '<span class="solved-text">魔方已还原，无需转动</span>';
  $("actions").innerHTML = plan.actions.length
    ? plan.actions
        .map(
          (action) => `
    <li class="action-row ${action.type ?? (action.kind === "flip" ? "reorient" : "turn_top")}">
      <span class="step-number">${String((action.actionIndex ?? action.index) + 1).padStart(2, "0")}</span>
      <span class="action-symbol" aria-hidden="true">${action.type === "reorient" || action.kind === "flip" ? "⤴" : "↺"}</span>
      <div><strong>${actionText(action)}</strong><small>${action.kind ? `符号步骤 ${action.symbolicMoveIndex + 1} · ${action.kind === "flip" ? "固定掌心轴 90°" : "顶层 90°"}` : action.type === "reorient" ? `${Math.round(action.angleDeg * 10) / 10}° · 最短整体旋转` : `${action.notation} · 仅世界 +Z 顶层`}</small></div>
      <span class="face-dot" style="background:${FACES[action.targetFace].color}"></span>
    </li>`,
        )
        .join("")
    : '<li class="empty"><strong>已还原</strong><p>当前状态不需要机器人动作。</p></li>';
  $("actions").scrollTop = 0;
}

function clearPlan() {
  plan = null;
  cursor = 0;
  playing = false;
  $("notation").textContent = "等待规划";
  $("notation").classList.add("empty-notation");
  $("actions").innerHTML =
    '<li class="empty"><div class="empty-icon" aria-hidden="true">↻</div><strong>魔方已打乱</strong><p>点击「规划还原步骤」，生成面转序列与机器人动作。</p></li>';
  $("action-count").textContent = "0 步";
}

function loadCube(next) {
  cube = next;
  const yaw = Number($("yaw").value);
  orientation = isRiskMode() ? [...referenceOrientation.quaternion] : yawOrientation(Number.isFinite(yaw) ? yaw : 25);
  clearPlan();
  scene?.setState(cube.asString(), orientation);
  $("phase-tag").textContent = cube.isSolved() ? "已还原" : "已打乱";
  $("action-caption").textContent = cube.isSolved()
    ? "当前各面颜色一致"
    : "等待规划还原步骤";
  message();
  refresh();
}

const worker = new Worker(new URL("./solver.worker.js", import.meta.url), {
  type: "module",
});
worker.onmessage = ({ data }) => {
  if (data.type === "ready") {
    ready = true;
    $("solver-status").textContent = "求解器就绪";
    refresh();
    return;
  }
  if (data.id !== requestId && data.id !== undefined) return;
  solving = false;
  if (data.type === "error") {
    message(data.message, true);
    $("solver-status").textContent = ready ? "求解器就绪" : "求解器不可用";
    refresh();
    return;
  }
  try {
    plan = compileRobotPlan(
      requestSnapshot.facelets,
      expandTurns(data.algorithm).map(turn => turn.notation).join(' '),
      requestSnapshot.orientation,
    );
    cursor = 0;
    renderPlan();
    $("phase-tag").textContent = plan.actions.length ? "规划完成" : "已还原";
    $("action-caption").textContent = plan.actions.length
      ? "可逐步检查，或连续播放"
      : "当前各面颜色一致";
    $("solver-status").textContent = "求解器就绪";
    message(
      plan.actions.length
        ? "规划已校验：仅用六个外层面转，完整执行可还原。"
        : "当前状态已还原，无需动作。",
    );
  } catch (error) {
    message(error.message, true);
  }
  refresh();
};
worker.onerror = (event) => {
  solving = false;
  ready = false;
  message(`求解器异常：${event.message}`, true);
  $("solver-status").textContent = "求解器不可用";
  refresh();
};

$("scramble").onclick = () => loadCube(Cube.random());
$("plan").onclick = () => {
  solving = true;
  message();
  $("solver-status").textContent = "正在规划…";
  $("phase-tag").textContent = "求解中";
  requestSnapshot = {
    facelets: cube.asString(),
    orientation: orientation.slice(),
  };
  const id = ++requestId;
  if (isRiskMode()) startRiskPlanning(id);
  else worker.postMessage({ id, facelets: requestSnapshot.facelets });
  refresh();
};
$("import-state").onclick = () => {
  try {
    loadCube(cubeFromFacelets($("facelet-input").value));
  } catch (error) {
    message(error.message, true);
  }
};
$("copy-current").onclick = () => {
  $("facelet-input").value = cube.asString();
};
$("yaw").onchange = () => {
  if (plan || busy || solving) return;
  try {
    orientation = yawOrientation(Number($("yaw").value));
    scene?.setState(cube.asString(), orientation);
    refresh();
  } catch (error) {
    message(error.message, true);
  }
};

function finishRiskPlanning(timedOut = false) {
  clearTimeout(riskTimer);
  riskWorker?.terminate(); riskWorker = null;
  solving = false;
  if (latestRiskPlan) {
    plan = latestRiskPlan;
    plan.search.timedOut ||= timedOut;
    cursor = 0; renderPlan();
    $("phase-tag").textContent = plan.actions.length ? "规划完成" : "已还原";
    $("action-caption").textContent = "掌心 A/B · 离线预览";
    const base = plan.cost.baseline;
    $("search-summary").textContent = `${plan.search.calls} 次求解 · ${plan.search.candidateCount} 个候选 · A/B 翻面 ${base?.[0] ?? "—"} → ${plan.statistics.reorientations} 次`;
    message(timedOut ? "搜索预算已到，保留离线重放校验通过的最佳计划。" : "离线重放通过：转面只用逆时针 90°；概率尚未标定，优先少翻面。");
  } else message("搜索结束，尚无经过校验的计划。", true);
  $("solver-status").textContent = "求解器就绪";
  refresh();
}

function startRiskPlanning(id) {
  riskWorker?.terminate(); clearTimeout(riskTimer); latestRiskPlan = null;
  clearPlan();
  riskWorker = new Worker(new URL("./risk-solver.worker.js", import.meta.url), {type:"module"});
  riskWorker.onmessage = ({data}) => {
    if (data.id !== requestId) return;
    if (data.type === "plan" || data.type === "done") {
      latestRiskPlan = data.plan;
      $("solver-status").textContent = `已找到 ${data.plan.statistics.reorientations} 次翻面的解，继续搜索…`;
      if (data.type === "done") finishRiskPlanning();
    } else if (data.type === "error") {
      if (latestRiskPlan) finishRiskPlanning(true);
      else { clearTimeout(riskTimer); riskWorker?.terminate(); riskWorker=null; solving=false; message(data.message,true); refresh(); }
    }
  };
  riskWorker.onerror = event => {
    clearTimeout(riskTimer); riskWorker?.terminate(); riskWorker=null; solving=false;
    message(`A/B 求解器异常：${event.message}`,true); refresh();
  };
  riskWorker.postMessage({id,facelets:requestSnapshot.facelets,options:{
    initialQuaternion:requestSnapshot.orientation,calibrationId:"demo-palm-v1",modelProfileId:"preview",
    centerIdentityMap:DEMO_CENTER_IDENTITY_MAP,mode:"ideal_preview",costMode:"proxy",sourceRevision:id,
    maxSolverCalls:64,maxCandidates:64,maxPrefixDepth:2,beamWidth:18 }});
  riskTimer = setTimeout(() => { if(id===requestId) finishRiskPlanning(true); },30000);
}

$("planner-mode").onchange = () => {
  clearPlan(); orientation = isRiskMode() ? [...referenceOrientation.quaternion] : yawOrientation(Number($("yaw").value));
  $("cost-label").textContent = isRiskMode() ? "参考：白上、蓝右、红前 · 离线预览" : "原版：最小整体旋转角";
  scene?.setState(cube.asString(),orientation); refresh();
};
$("align-palm").onclick = () => {
  clearPlan(); orientation = [...referenceOrientation.quaternion];
  scene?.setState(cube.asString(),orientation);
  message("预览参考已设为白上、蓝右、红前。魔方内部状态保持当前输入。"); refresh();
};

async function advance() {
  if (busy || solving || !scene || !plan || cursor >= plan.actions.length)
    return;
  busy = true;
  const action = plan.actions[cursor];
  const previewAction = plan.schemaVersion === 2 ? toPreviewAction(action) : action;
  $("phase-tag").textContent =
    previewAction.type === "reorient" ? "重定向" : "顶层转动";
  $("action-caption").textContent = actionText(action);
  refresh();
  const row = $("actions").children[cursor];
  if (row)
    $("actions").scrollTo({
      top: Math.max(0, row.offsetTop - $("actions").offsetTop - 85),
      behavior: "smooth",
    });
  try {
    const duration =
      previewAction.type === "reorient"
        ? previewAction.requiresMotion
          ? 1050
          : 280
        : Math.abs(previewAction.angleDeg) === 180
          ? 1050
          : 760;
    await scene.animateAction(
      previewAction,
      duration / Number($("speed").value),
      setProgress,
    );
    orientation = plan.schemaVersion === 2 ? executePreviewAction(cube, orientation, action) : executeRobotAction(cube, orientation, action);
    cursor++;
    scene.setState(cube.asString(), orientation);
    if (cursor === plan.actions.length) {
      if (!cube.isSolved()) throw new Error("播放结束后的魔方状态校验失败");
      playing = false;
      $("phase-tag").textContent = "已还原";
      $("action-caption").textContent = "全部动作完成 · 六面颜色已还原";
      message("还原完成。魔方平放在桌面上，末次操作面保持朝上。");
    }
  } catch (error) {
    playing = false;
    message(error.message, true);
  }
  busy = false;
  refresh();
}

$("step").onclick = () => {
  message();
  void advance();
};
$("play").onclick = async () => {
  if (playing) {
    playing = false;
    message("将在当前动作完成、平放后暂停。");
    refresh();
    return;
  }
  playing = true;
  message();
  refresh();
  while (playing && plan && cursor < plan.actions.length) await advance();
  playing = false;
  refresh();
};
$("export").onclick = () => {
  $("export-json").value = JSON.stringify(plan, null, 2);
  $("export-message").textContent = plan.schemaVersion === 2 ? "v2 离线计划：需要动作级执行适配器，不能直接交给旧版控制程序。" : "原版计划，可保存为文件。";
  $("export-dialog").showModal();
};
$("close-export").onclick = () => $("export-dialog").close();
$("copy-plan").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("export-json").value);
    $("export-message").textContent = "计划 JSON 已复制。";
  } catch {
    $("export-json").focus();
    $("export-json").select();
    $("export-message").textContent = "已选中计划，请按 Ctrl+C 复制。";
  }
};
$("save-plan").onclick = () => {
  const url = URL.createObjectURL(
    new Blob([$("export-json").value], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "rubiks-robot-plan.json";
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  $("export-message").textContent =
    "已请求保存；若浏览器未下载，可直接复制 JSON。";
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
refresh();
