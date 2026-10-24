'use strict';

/**
 * EAI Studio · 运行环境检测 / 安装（2026-10-24，CUDA 版）
 *
 * 设计口径（用户要求）：
 *   · 一个 bat 同时负责「开 EAI」和「装环境」：真正判断由这里做，界面只是呈现；
 *   · 需要装的东西只有三件：Python 运行时（.venv）→ PyTorch（CUDA 构建）→ 基础依赖；
 *   · 一律装进「应用目录/.venv」——环境跟着应用走，整个文件夹复制到别的机器也能用；
 *   · 只用官方/镜像源，不静默换版本；装完用新进程复验（当前进程里 torch 已加载看不出来）。
 *
 * CUDA 轮子怎么挑（本版关键，用户 2026-10-25）：
 *   · 默认 cu130；算力 < 8.0（GTX 10/16 系、RTX 20 系 = 30 系以下）改走 cu126
 *     —— CUDA 13 的轮子里已经没有 sm_6x/7x 内核，装上去 cuda.is_available() 会是 False；
 *   · 先问 nvidia-smi 的 compute_cap，读不到再按显卡名判断，两者都没有就按默认 cu130；
 *   · 版本与 XPU 版对齐 2.14.1，两个安装包只差后端，曲线和权重可互看。
 *
 * Electron 本身不在安装清单里：没有它连这个界面都起不来，由 Start.bat 在控制台处理。
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PY_VERSION = '3.13';                                  // 与开发机 .venv 对齐（cp313 的 CUDA 轮子齐全）
const TORCH_VERSION = '2.14.1';
const TORCH_INDEX_BASE = 'https://download.pytorch.org/whl/';
const CUDA_INDEX_CURRENT = 'cu130';                         // sm_80+：RTX 30/40/50、A/H 系列
const CUDA_INDEX_LEGACY = 'cu126';                          // sm_<80：GTX 10/16 系、RTX 20 系
const PYPI_MIRROR = 'https://pypi.tuna.tsinghua.edu.cn/simple';
const PYPI_FALLBACK = 'https://pypi.org/simple';            // 镜像抽风时的兜底源
const BASE_DEPS = ['numpy', 'psutil'];

// pip / uv 报错行：命中的话直接推进安装日志，别让用户只看到「退出码 1」
const ERR_LINE = /^\s*(ERROR|error)[:\s]|No matching distribution|Could not find a version|SSLError|ProxyError|Read timed out|WinError|PermissionError|Could not install|No space left/i;

const IS_WIN = process.platform === 'win32';

function venvPython(venvDir) {
  return IS_WIN ? path.join(venvDir, 'Scripts', 'python.exe') : path.join(venvDir, 'bin', 'python');
}

/** 运行环境位置：应用目录内优先，其次上级目录（兼容老的开发布局），最后 PATH 里的 python。 */
function resolvePython(appDir) {
  const local = venvPython(path.join(appDir, '.venv'));
  if (fs.existsSync(local)) return { py: local, venv: path.join(appDir, '.venv'), kind: 'local' };
  const parentVenv = path.join(path.dirname(appDir), '.venv');
  const parent = venvPython(parentVenv);
  if (fs.existsSync(parent)) return { py: parent, venv: parentVenv, kind: 'parent' };
  return { py: IS_WIN ? 'python' : 'python3', venv: null, kind: 'path' };
}

/** 跑一条命令，收齐输出（供探测用，不打印）。 */
function run(cmd, args, opts) {
  return new Promise((resolve) => {
    let out = '';
    let err = '';
    let child;
    try {
      child = spawn(cmd, args, {
        cwd: (opts && opts.cwd) || undefined,
        windowsHide: true,
        env: Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }, (opts && opts.env) || {}),
      });
    } catch (e) {
      return resolve({ code: -1, out: '', err: String((e && e.message) || e) });
    }
    const timer = setTimeout(() => { try { child.kill(); } catch (e) { /* ignore */ } }, (opts && opts.timeout) || 60000);
    child.stdout.on('data', (b) => { out += b.toString('utf8'); });
    child.stderr.on('data', (b) => { err += b.toString('utf8'); });
    child.on('error', (e) => { clearTimeout(timer); resolve({ code: -1, out: out, err: err + String((e && e.message) || e) }); });
    child.on('close', (code) => { clearTimeout(timer); resolve({ code: code == null ? -1 : code, out: out, err: err }); });
  });
}

/** 流式跑一条命令：每一行输出交给 onLine，最后给 onDone。 */
function runStream(cmd, args, opts, onLine, onDone) {
  let child;
  try {
    child = spawn(cmd, args, {
      cwd: (opts && opts.cwd) || undefined,
      windowsHide: true,
      env: Object.assign({}, process.env, {
        PYTHONIOENCODING: 'utf-8',
        PYTHONUNBUFFERED: '1',
        // uv：给足超时/并发，国内直连 pytorch 源也能跑满带宽
        UV_HTTP_TIMEOUT: '300',
        UV_CONCURRENT_DOWNLOADS: '8',
        PIP_DISABLE_PIP_VERSION_CHECK: '1',
        PIP_NO_INPUT: '1',
      }, (opts && opts.env) || {}),
    });
  } catch (e) {
    onDone && onDone(-1, String((e && e.message) || e));
    return null;
  }
  let tail = '';
  const feed = (buf) => {
    const text = buf.toString('utf8');
    tail = (tail + text).slice(-4000);
    text.split(/\r?\n|\r/).forEach((line) => {
      const s = line.trim();
      if (s) onLine && onLine(s);
    });
  };
  child.stdout.on('data', feed);
  child.stderr.on('data', feed);
  child.on('error', (e) => onDone && onDone(-1, String((e && e.message) || e) + '\n' + tail));
  child.on('close', (code) => onDone && onDone(code == null ? -1 : code, tail));
  return child;
}

let uvCache = null;
async function hasUv() {
  if (uvCache !== null) return uvCache;
  const r = await run('uv', ['--version'], { timeout: 15000 });
  uvCache = r.code === 0 && /\buv\s+\d/.test(r.out);
  return uvCache;
}

/** 探测：{ ok, version } */
async function probePy(py, code) {
  const r = await run(py, ['-c', code], { timeout: 30000 });
  if (r.code !== 0) return { ok: false, out: '', err: (r.err || r.out || '').trim() };
  return { ok: true, out: r.out.trim(), err: '' };
}

/** 显卡清单（Windows WMI；非 Windows 返回空）。 */
async function gpuNames() {
  if (!IS_WIN) return [];
  const r = await run('powershell.exe', ['-NoProfile', '-Command',
    '(Get-CimInstance Win32_VideoController).Name'], { timeout: 20000 });
  return String(r.out || '').split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

/** 算力兜底：显卡名里带这些的都是 sm_6x / sm_7x（30 系以下）。 */
const LEGACY_GPU_KEYS = ['gtx 10', 'gtx 16', 'mx1', 'mx2', 'mx3', 'mx4', 'mx5',
  'quadro p', 'quadro t', 'quadro m', 'quadro k', 'titan x', 'titan v',
  'p40', 'p100', 'v100', 'rtx 20'];

let gpuCache = null;
/** { names, joined, cap, hasNvidia } —— 只探测一次，安装/扫描共用。 */
async function detectGpu() {
  if (gpuCache) return gpuCache;
  const names = await gpuNames();
  const joined = names.join(' ').toLowerCase();
  let cap = 0;
  const smi = await run('nvidia-smi', ['--query-gpu=compute_cap', '--format=csv,noheader'], { timeout: 15000 });
  String(smi.out || '').split(/\r?\n/).forEach((line) => {
    const v = parseFloat(line.trim());
    if (!cap && !isNaN(v)) cap = v;
  });
  if (!cap) cap = LEGACY_GPU_KEYS.some((k) => joined.includes(k)) ? 6.1 : 0;
  gpuCache = {
    names: names,
    joined: joined,
    cap: cap,
    hasNvidia: /nvidia|geforce|quadro|tesla/.test(joined),
  };
  return gpuCache;
}

/** 该装哪个 CUDA 轮子：30 系以下（算力 < 8.0）→ cu126，其余（含读不到算力）→ cu130。 */
function cudaIndexFor(gpu) {
  if (gpu && gpu.cap && gpu.cap < 8.0) return CUDA_INDEX_LEGACY;
  return CUDA_INDEX_CURRENT;
}

function torchSpecFor(index) {
  return TORCH_VERSION + '+' + index;
}

/** 已装的 torch 是不是本版本的 CUDA 构建（+cu130 / +cu126 …，不是 +xpu / +cpu）。 */
function isCudaBuild(version) {
  return /\+cu\d+/.test(String(version || ''));
}

/** 从命令输出尾巴里挑出「为什么失败」那一两句（2026-10-25：原来失败只报退出码，等于没报）。 */
function pickErrorLines(tail, max) {
  const lines = String(tail || '').split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const hits = lines.filter((s) => ERR_LINE.test(s));
  const pick = (hits.length ? hits : lines.slice(-2)).slice(-(max || 2));
  return pick.join(' | ').slice(0, 240);
}

/**
 * 环境扫描：返回给安装界面直接渲染的结构。
 * 顺序 = 安装顺序，界面上也按这个顺序排。
 */
async function scanEnv(appDir) {
  const resolved = resolvePython(appDir);
  const items = [];

  // ① Electron（只报告，不在这里装：没 Electron 就没有这个界面）
  const electronExe = path.join(appDir, 'node_modules', 'electron', 'dist',
    IS_WIN ? 'electron.exe' : 'electron');
  items.push({
    id: 'electron',
    installable: false,
    ok: fs.existsSync(electronExe),
    detail: fs.existsSync(electronExe) ? 'node_modules/electron' : '',
  });

  // ②③④ 一次进程问完：Python 版本 / torch / 基础依赖（2026-10-24：本来三次 spawn，
  //        fresh 环境里 import torch 要好几秒，串起来界面会「正在检测」停太久）
  const probeCode = [
    'import sys',
    'print("py=" + "%d.%d.%d" % sys.version_info[:3])',
    'try:',
    '    import torch; print("torch=" + torch.__version__)',
    'except Exception:',
    '    print("torch=")',
    'try:',
    '    import torch; print("dev=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""))',
    'except Exception:',
    '    print("dev=")',
    'try:',
    '    import numpy, psutil; print("deps=" + numpy.__version__ + " / psutil " + psutil.__version__)',
    'except Exception:',
    '    print("deps=")',
  ].join('\n');

  const pyOk = resolved.kind !== 'path' && fs.existsSync(resolved.py);
  let pyVersion = '';
  let torchVersion = '';
  let torchDevice = '';
  let depsDetail = '';
  if (pyOk) {
    const r = await run(resolved.py, ['-c', probeCode], { timeout: 90000 });
    (r.out || '').split(/\r?\n/).forEach((line) => {
      const t = line.trim();
      if (t.startsWith('py=')) pyVersion = t.slice(3).trim();
      else if (t.startsWith('torch=')) torchVersion = t.slice(6).trim();
      else if (t.startsWith('dev=')) torchDevice = t.slice(4).trim();
      else if (t.startsWith('deps=')) depsDetail = t.slice(5).trim();
    });
  }

  const gpu = await detectGpu();
  const torchIndex = cudaIndexFor(gpu);
  const torchSpec = torchSpecFor(torchIndex);
  // warn 只传「代码」，中英文由安装页决定：not-cuda-build = 这个 torch 不是 CUDA 构建；
  // no-cuda-device = 装了 CUDA 构建但探测不到设备（多为驱动 / 轮子与算力不匹配）
  let torchWarn = '';
  if (torchVersion && !isCudaBuild(torchVersion)) torchWarn = 'not-cuda-build';
  else if (torchVersion && !torchDevice) torchWarn = 'no-cuda-device';

  items.push({
    id: 'python',
    installable: true,
    ok: pyOk,
    detail: pyOk ? (pyVersion || resolved.py) : '',
  });
  items.push({
    id: 'torch',
    installable: true,
    ok: Boolean(torchVersion),
    detail: torchDevice ? (torchVersion + ' · ' + torchDevice) : torchVersion,
    warn: torchWarn,
  });
  items.push({ id: 'deps', installable: true, ok: Boolean(depsDetail), detail: depsDetail });

  return {
    ok: items.every((it) => it.ok),
    python: resolved,
    venvDir: resolved.venv || path.join(appDir, '.venv'),
    torchSpec: torchSpec,                        // 安装页显示用：2.14.1+cu130 / 2.14.1+cu126
    torchIndex: torchIndex,
    gpu: { names: gpu.names, cap: gpu.cap, hasNvidia: gpu.hasNvidia },
    items: items,
  };
}

/**
 * 依次安装，onProgress({ id, pct, text, state })：
 *   state: 'run' | 'ok' | 'error'
 * 返回 { ok, error }
 */
async function runInstall(appDir, onProgress) {
  const venvDir = path.join(appDir, '.venv');
  const py = venvPython(venvDir);
  const uv = await hasUv();

  // 进度事件只带「文案代码 + 参数」，具体语言由渲染层决定（中英切换时能重刷），
  // 不要把中文写进事件里 —— 否则切到英文全是中文提示（2026-10-24 实测）。
  const emit = (id, pct, code, arg, state) => {
    try {
      onProgress({
        id: id,
        pct: Math.max(0, Math.min(100, Math.round(pct))),
        code: code || '',
        arg: arg == null ? '' : String(arg).slice(0, 120),
        state: state || 'run',
      });
    } catch (e) { /* 渲染层挂了不该拖垮安装 */ }
  };

  /** 长下载阶段：里程碑之间匀速爬，避免看起来卡死。 */
  function creep(id, from, to, ms, code, arg) {
    let pct = from;
    let last = Date.now();
    const rate = (to - from) / ms;
    emit(id, pct, code, arg);
    const timer = setInterval(() => {
      const now = Date.now();
      pct = Math.min(to, pct + rate * (now - last));
      last = now;
      emit(id, pct, code, arg);
    }, 700);
    return {
      set(next, code2, arg2) {
        if (next > pct) {
          pct = Math.min(to, next);
          emit(id, pct, code2 || code, arg2 == null ? arg : arg2);
          last = Date.now();
        }
      },
      stop() { clearInterval(timer); },
    };
  }

  // ---------- ① Python 运行时 ----------
  if (fs.existsSync(py)) emit('python', 45, 'python.exists');
  else {
    const bar = creep('python', 5, 80, 90000, uv ? 'python.prepare' : 'python.prepareVenv', PY_VERSION);
    let code;
    if (uv) {
      code = await new Promise((resolve) => {
        runStream('uv', ['venv', '--python', PY_VERSION, venvDir], { cwd: appDir }, (line) => {
          bar.set(60, 'uv.line', line);
        }, (c) => resolve(c));
      });
      if (code !== 0) { bar.stop(); emit('python', 0, 'python.uvFail', '', 'error'); return { ok: false, error: 'uv venv failed' }; }
    } else {
      const sysPy = IS_WIN ? 'python' : 'python3';
      const r = await run(sysPy, ['-m', 'venv', venvDir], { cwd: appDir, timeout: 180000 });
      if (r.code !== 0) {
        bar.stop();
        emit('python', 0, 'python.noPython', '', 'error');
        return { ok: false, error: 'no python host' };
      }
      bar.set(70, 'python.created');
    }
    bar.stop();
    if (!fs.existsSync(py)) { emit('python', 0, 'python.noVenvPy', '', 'error'); return { ok: false, error: 'venv python missing' }; }
  }
  const v = await probePy(py, 'import sys;print("%d.%d.%d" % sys.version_info[:3])');
  emit('python', 100, v.ok ? 'python.version' : 'python.ready', v.ok ? v.out : '', 'ok');

  // ---------- ② PyTorch（CUDA：30 系以上 cu130 / 30 系以下 cu126） ----------
  const gpu = await detectGpu();
  const torchIndex = cudaIndexFor(gpu);
  const torchSpec = torchSpecFor(torchIndex);
  const torchUrl = TORCH_INDEX_BASE + torchIndex;
  const have = await probePy(py, 'import torch;print(torch.__version__)');
  if (have.ok) {
    emit('torch', 100, 'torch.version', have.out, 'ok');
  } else {
    const bar = creep('torch', 3, 88, 8 * 60 * 1000, 'torch.start', torchSpec);
    const argsUv = ['pip', 'install', '--python', py,
      '--index-url', torchUrl, '--extra-index-url', PYPI_MIRROR,
      '--extra-index-url', PYPI_FALLBACK,
      '--index-strategy', 'unsafe-best-match',
      'torch==' + torchSpec];
    const argsPip = ['-m', 'pip', 'install', '--no-input',
      '--index-url', torchUrl, '--extra-index-url', PYPI_MIRROR,
      '--extra-index-url', PYPI_FALLBACK,
      'torch==' + torchSpec];
    let torchTail = '';
    const code = await new Promise((resolve) => {
      runStream(uv ? 'uv' : py, uv ? argsUv : argsPip, { cwd: appDir }, (line) => {
        if (/Resolved \d+ package/i.test(line)) bar.set(12, 'torch.resolved');
        else if (/Prepared \d+ package/i.test(line)) bar.set(45, 'torch.prepared');
        else if (/Downloading/i.test(line)) bar.set(20, 'torch.download', line.replace(/^.*Downloading\s*/i, ''));
        else if (/Installing|Installed \d+ package/i.test(line)) bar.set(80, 'torch.install');
        // 只认真正的错误行；uv 的 hardlink 回退是警告，不算失败
        else if (ERR_LINE.test(line)) bar.set(88, 'uv.line', line);
      }, (c, tail) => { torchTail = tail || ''; resolve(c); });
    });
    bar.stop();
    const after = await probePy(py, 'import torch;print(torch.__version__)');
    if (!after.ok) {
      const why = pickErrorLines(torchTail) || (code === 0 ? 'import failed' : 'exit ' + code);
      emit('torch', 0, 'torch.fail', why, 'error');
      return { ok: false, id: 'torch', code: 'torch.fail', error: why };
    }
    emit('torch', 100, 'torch.version', after.out, 'ok');
  }
  // 装了 CUDA 构建却拿不到设备：不拦着进主界面（CPU 还能训），但必须让人看见原因
  const dev = await probePy(py, 'import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")');
  if (dev.ok && !dev.out) {
    emit('torch', 100, 'torch.noDevice', torchIndex, 'ok');
  }

  // ---------- ③ 基础依赖（先镜像，镜像挂了换官方源再试一次） ----------
  const depsHave = await probePy(py, 'import numpy,psutil');
  if (depsHave.ok) {
    emit('deps', 100, 'deps.ready', '', 'ok');
  } else {
    let why = '';
    let installed = false;
    for (const index of [PYPI_MIRROR, PYPI_FALLBACK]) {
      emit('deps', 10, 'deps.start');
      const args = uv
        ? ['pip', 'install', '--python', py, '--index-url', index].concat(BASE_DEPS)
        : ['-m', 'pip', 'install', '--no-input', '--index-url', index].concat(BASE_DEPS);
      let tail = '';
      const code = await new Promise((resolve) => {
        runStream(uv ? 'uv' : py, args, { cwd: appDir }, (line) => {
          if (/Installed|Installing|Prepared/i.test(line)) emit('deps', 80, 'deps.install');
          else if (ERR_LINE.test(line)) emit('deps', 80, 'uv.line', line);
        }, (c, t) => { tail = t || ''; resolve(c); });
      });
      const after = code === 0 ? await probePy(py, 'import numpy,psutil;print("ok")') : { ok: false };
      if (after.ok) { installed = true; break; }
      why = pickErrorLines(tail) || ('exit ' + code);
      emit('deps', 30, 'uv.line', '源失败，换下一个：' + index);
    }
    if (!installed) {
      emit('deps', 0, 'deps.fail', why, 'error');
      return { ok: false, id: 'deps', code: 'deps.fail', error: why };
    }
    emit('deps', 100, 'deps.ready', '', 'ok');
  }

  return { ok: true };
}

module.exports = {
  scanEnv, runInstall, resolvePython, venvPython, hasUv,
  detectGpu, cudaIndexFor, isCudaBuild,
  TORCH_VERSION, CUDA_INDEX_CURRENT, CUDA_INDEX_LEGACY, PY_VERSION,
};
