"""
server/app.py — OpenEnv HTTP API for Enterprise Employee Onboarding Agent
Exposes reset() / step() / state() on port 7860 (HF Space compatible).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from onboarding_env import (
    OnboardingAction, OnboardingEnv,
    TASK_IDS, _TASK_CONFIG,
)

app = FastAPI(
    title="OpenEnv: Enterprise Employee Onboarding Agent",
    version="1.0.0",
    description=(
        "An OpenEnv environment simulating enterprise employee onboarding "
        "across 3–12 IT/HR systems with realistic policy drift. "
        "Three tasks: easy → medium → hard."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_env: Optional[OnboardingEnv] = None
_current_task: str = "basic_onboarding"


class ResetRequest(BaseModel):
    task: Optional[str] = None
    seed: Optional[int] = None


@app.post("/reset", summary="Start a new onboarding episode")
async def reset(body: Optional[ResetRequest] = None):
    global _env, _current_task
    task = _current_task
    seed = None
    if body:
        if body.task:
            if body.task not in TASK_IDS:
                raise HTTPException(400, f"Unknown task. Valid: {TASK_IDS}")
            task = body.task
            _current_task = task
        seed = body.seed
    _env = OnboardingEnv(task=task, seed=seed)
    obs  = _env.reset()
    return obs.model_dump()


@app.post("/step", summary="Execute one agent action")
async def step(action: OnboardingAction):
    if _env is None:
        raise HTTPException(400, "No active episode. POST /reset first.")
    if _env._ep.get("done", False):
        raise HTTPException(400, "Episode finished. POST /reset to start a new one.")
    result = _env.step(action)
    return result.model_dump()


@app.get("/state", summary="Inspect current episode state")
async def state():
    if _env is None:
        raise HTTPException(400, "No active episode. POST /reset first.")
    return _env.state()


@app.get("/tasks", summary="List available tasks")
async def tasks():
    diffs = ["easy", "medium", "hard"]
    return {
        "tasks": [
            {
                "id":          tid,
                "difficulty":  diffs[i],
                "description": cfg["description"],
                "max_steps":   cfg["max_steps"],
                "systems":     len([s for s in __import__("onboarding_env")._systems_for_task(tid)]),
                "drift_steps": cfg["drift_steps"],
            }
            for i, (tid, cfg) in enumerate(_TASK_CONFIG.items())
        ]
    }


@app.get("/tasks/{task_id}", summary="Get single task metadata")
async def task_detail(task_id: str):
    if task_id not in TASK_IDS:
        raise HTTPException(404, f"Task '{task_id}' not found. Valid: {TASK_IDS}")
    diffs = ["easy", "medium", "hard"]
    idx   = TASK_IDS.index(task_id)
    cfg   = _TASK_CONFIG[task_id]
    import onboarding_env as oe
    return {
        "id":          task_id,
        "difficulty":  diffs[idx],
        "description": cfg["description"],
        "max_steps":   cfg["max_steps"],
        "systems":     len(oe._systems_for_task(task_id)),
        "drift_steps": cfg["drift_steps"],
        "start_policy": cfg["start_policy"],
    }


@app.get("/render", summary="ASCII snapshot of current onboarding state")
async def render():
    if _env is None:
        raise HTTPException(400, "No active episode. POST /reset first.")
    obs = _env._make_obs()
    lines = [
        f"╔══ ONBOARDING  step={obs.step}/{obs.max_steps}  "
        f"emp={obs.employee_id}  role={obs.employee_role} ══╗",
        f"  Policy: {obs.current_policy.version.value}  "
        f"security={obs.current_policy.it_security_level}  "
        f"device={obs.current_policy.device_type}",
        f"  Progress: {obs.completed_count}/{obs.total_required} required  "
        f"({obs.completion_pct:.1f}%)  violations={obs.episode_violations}",
        "",
    ]
    status_icon = {
        "pending": "○", "in_progress": "◎", "complete": "✓",
        "blocked": "⊘", "failed": "✗",
    }
    for s in obs.systems:
        req  = "REQ" if s.required else "opt"
        icon = status_icon.get(s.status.value, "?")
        err  = f"  ← {s.error_msg[:50]}" if s.error_msg else ""
        dep  = f"  [needs: {s.dependencies}]" if s.dependencies and s.status.value == "blocked" else ""
        lines.append(
            f"  {icon} [{req}] {s.id:<24} {s.status.value:<12}{err}{dep}"
        )
    if obs.policy_drift_event:
        lines += ["", f"  ⚡ DRIFT: {obs.policy_drift_event[:80]}"]
    lines += [
        "",
        f"  Last: {obs.last_action_result[:70]}",
        f"╚{'═' * 62}╝",
    ]
    return {"render": "\n".join(lines)}


@app.get("/health", summary="Liveness probe")
async def health():
    return {"status": "ok", "service": "openenv-onboarding"}


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/web")


@app.get("/web", include_in_schema=False)
async def web():
    PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enterprise Onboarding Agent — OpenEnv</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#050f1c;--s1:#0a1828;--s2:#0d1f30;--bd:#1a3045;--txt:#e2edf8;--mut:#4a6a85;--pur:#a78bfa;--pur2:#7c3aed;--tel:#22d3a0;--yel:#fbbf24;--red:#f43f5e;--blu:#3b82f6}
html,body{min-height:100vh}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--txt);overflow-x:hidden}
canvas{position:fixed;inset:0;z-index:0;opacity:.25;pointer-events:none}
main{position:relative;z-index:1;max-width:900px;margin:0 auto;padding:52px 24px 72px}
.hero{text-align:center;margin-bottom:40px}
.live-badge{display:inline-flex;align-items:center;gap:7px;background:rgba(34,211,160,.08);border:1px solid rgba(34,211,160,.2);color:var(--tel);border-radius:999px;padding:5px 16px;font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:20px}
.pulse{width:7px;height:7px;border-radius:50%;background:var(--tel);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.7)}}
h1{font-size:clamp(2rem,5vw,3.2rem);font-weight:800;letter-spacing:-.04em;background:linear-gradient(135deg,#e2edf8 0%,#a78bfa 50%,#22d3a0 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.15;margin-bottom:14px}
.tagline{color:var(--mut);font-size:.95rem;max-width:520px;margin:0 auto 24px;line-height:1.75}
.status-bar{display:flex;align-items:center;justify-content:center;gap:8px;background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:11px 22px;font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--mut);margin-bottom:36px}
.sdot{width:9px;height:9px;border-radius:50%;background:var(--tel);flex-shrink:0}
.sdot.err{background:var(--red)}

.section-lbl{font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.1em;color:var(--mut);margin-bottom:14px}
.flow-wrap{background:var(--s1);border:1px solid var(--bd);border-radius:16px;padding:20px;margin-bottom:20px}
.systems-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
.sc{background:var(--bg);border:1px solid var(--bd);border-radius:10px;padding:12px 8px;text-align:center;transition:all .3s;position:relative}
.sc.complete{border-color:#0f6e56;background:rgba(34,211,160,.04)}
.sc.active{border-color:var(--pur);background:rgba(167,139,250,.07);animation:ag 1.8s ease-in-out infinite}
.sc.failed{border-color:#a32d2d;background:rgba(244,63,94,.05)}
.sc.blocked{opacity:.4}
@keyframes ag{0%,100%{border-color:var(--pur)}50%{border-color:#c4b5fd}}
.si{width:30px;height:30px;border-radius:8px;margin:0 auto 7px;display:flex;align-items:center;justify-content:center}
.sn{font-size:.65rem;font-family:'JetBrains Mono',monospace;color:var(--mut);line-height:1.4}
.ss{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;margin-top:5px}
.ss-complete{color:var(--tel)}.ss-active{color:var(--pur)}.ss-failed{color:var(--red)}.ss-blocked,.ss-pending{color:var(--mut)}
.prog-row{display:flex;align-items:center;gap:12px}
.prog-bg{flex:1;height:5px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden}
.prog-fg{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--pur),var(--tel));transition:width .6s ease}
.prog-lbl{font-size:.72rem;font-family:'JetBrains Mono',monospace;color:var(--tel);white-space:nowrap}

.drift{background:rgba(251,191,36,.05);border:1px solid rgba(251,191,36,.2);border-radius:12px;padding:12px 16px;margin-bottom:20px;display:flex;gap:12px;align-items:flex-start;transition:opacity .5s}
.drift-ic{width:20px;height:20px;border-radius:50%;background:rgba(251,191,36,.12);display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}
.drift-title{font-size:.76rem;font-weight:600;color:var(--yel);margin-bottom:4px}
.drift-body{font-size:.68rem;font-family:'JetBrains Mono',monospace;color:rgba(251,191,36,.55);line-height:1.6}

.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.stat{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:18px 12px;text-align:center}
.sv{font-size:2rem;font-weight:800;letter-spacing:-.04em}
.sl{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mut);margin-top:5px}
.vp{color:var(--pur)}.vt{color:var(--tel)}.vy{color:var(--yel)}.vb{color:#60a5fa}

.tasks{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:20px}
.tc{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:18px;transition:transform .15s,border-color .15s}
.tc:hover{transform:translateY(-3px)}
.tc.e:hover{border-color:var(--tel)}.tc.m:hover{border-color:var(--yel)}.tc.h:hover{border-color:var(--red)}
.tch{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.tcn{font-family:'JetBrains Mono',monospace;font-size:.72rem;font-weight:600;color:var(--txt)}
.tcd{font-size:.63rem;font-weight:700;padding:2px 9px;border-radius:999px}
.de{background:rgba(34,211,160,.1);color:var(--tel)}.dm{background:rgba(251,191,36,.1);color:var(--yel)}.dh{background:rgba(244,63,94,.1);color:var(--red)}
.tcb{font-size:.78rem;color:var(--mut);line-height:1.65;margin-bottom:10px}
.tcm{font-size:.65rem;font-family:'JetBrains Mono',monospace;color:#1a3045}

.term{background:#020c18;border:1px solid var(--bd);border-radius:14px;overflow:hidden;margin-bottom:20px}
.term-top{background:var(--s1);padding:10px 16px;display:flex;align-items:center;gap:7px;border-bottom:1px solid var(--bd)}
.tdot{width:10px;height:10px;border-radius:50%}
.tl{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--mut);margin-left:8px}
.term-body{padding:16px 18px;font-family:'JetBrains Mono',monospace;font-size:.75rem;line-height:2;min-height:120px}
.tc-cmd{color:#60a5fa}.tc-out{color:var(--tel)}.tc-warn{color:var(--yel)}.tc-mut{color:var(--mut)}
.blink{animation:blink 1s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}

.ep-wrap{background:var(--s1);border:1px solid var(--bd);border-radius:14px;overflow:hidden;margin-bottom:24px}
.ep{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid var(--bd);transition:background .15s}
.ep:last-child{border-bottom:none}
.ep:hover{background:rgba(167,139,250,.04)}
.mth{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:700;padding:3px 8px;border-radius:5px;min-width:40px;text-align:center}
.mp{background:rgba(34,211,160,.1);color:var(--tel)}.mg{background:rgba(59,130,246,.1);color:#60a5fa}
.ep-p{font-family:'JetBrains Mono',monospace;font-size:.82rem}
.ep-d{font-size:.74rem;color:var(--mut);margin-left:auto}

.ctas{display:flex;gap:12px;justify-content:center}
.bpri{background:var(--pur2);color:#fff;border:none;padding:13px 28px;border-radius:10px;font-size:.87rem;font-weight:600;cursor:pointer;font-family:'Inter',sans-serif;transition:all .15s;text-decoration:none;display:inline-flex;align-items:center}
.bpri:hover{background:#6d28d9;transform:translateY(-1px)}
.bsec{background:transparent;color:var(--mut);border:1px solid var(--bd);padding:13px 28px;border-radius:10px;font-size:.87rem;font-weight:500;cursor:pointer;font-family:'Inter',sans-serif;transition:all .15s;text-decoration:none;display:inline-flex;align-items:center}
.bsec:hover{border-color:var(--mut);color:var(--txt)}
</style>
</head>
<body>
<canvas id="cv"></canvas>
<main>
  <div class="hero">
    <div class="live-badge"><div class="pulse"></div>OpenEnv · Grand Finale 2026</div>
    <h1>Enterprise Employee<br>Onboarding Agent</h1>
    <p class="tagline">An AI agent navigates 12 enterprise IT &amp; HR systems to onboard a new hire — while company policies drift mid-episode without warning.</p>
    <div class="status-bar">
      <div class="sdot" id="sdot"></div>
      <span id="stxt">Checking server&hellip;</span>
    </div>
  </div>

  <div class="section-lbl">Live simulation — dept_onboarding · policy drift active</div>
  <div class="flow-wrap">
    <div class="systems-grid" id="sg">
      <div class="sc complete"><div class="si" style="background:rgba(34,211,160,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="2" y="2" width="12" height="12" rx="2" stroke="#22d3a0" stroke-width="1.5"/><path d="M5 8l2 2 4-4" stroke="#22d3a0" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">ad_account</div><div class="ss ss-complete">complete</div></div>
      <div class="sc complete"><div class="si" style="background:rgba(34,211,160,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M2 4h12v8a1 1 0 01-1 1H3a1 1 0 01-1-1V4z" stroke="#22d3a0" stroke-width="1.5"/><path d="M2 4l6 5 6-5" stroke="#22d3a0" stroke-width="1.5"/></svg></div><div class="sn">email_setup</div><div class="ss ss-complete">complete</div></div>
      <div class="sc complete"><div class="si" style="background:rgba(34,211,160,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="6" r="3" stroke="#22d3a0" stroke-width="1.5"/><path d="M2 14c0-3.314 2.686-5 6-5s6 1.686 6 5" stroke="#22d3a0" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">hrms_reg</div><div class="ss ss-complete">complete</div></div>
      <div class="sc failed"><div class="si" style="background:rgba(244,63,94,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="5.5" stroke="#f43f5e" stroke-width="1.5"/><path d="M6 6l4 4M10 6l-4 4" stroke="#f43f5e" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">badge_access</div><div class="ss ss-failed">failed</div></div>
      <div class="sc active"><div class="si" style="background:rgba(167,139,250,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="2" y="5" width="12" height="8" rx="1.5" stroke="#a78bfa" stroke-width="1.5"/><path d="M5 5V4a3 3 0 016 0v1" stroke="#a78bfa" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">payroll</div><div class="ss ss-active">active</div></div>
      <div class="sc active"><div class="si" style="background:rgba(167,139,250,.1)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><rect x="3" y="2" width="10" height="12" rx="1.5" stroke="#a78bfa" stroke-width="1.5"/><path d="M5 6h6M5 9h4" stroke="#a78bfa" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">device_alloc</div><div class="ss ss-active">active</div></div>
      <div class="sc blocked"><div class="si" style="background:rgba(255,255,255,.04)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="5.5" stroke="#4a6a85" stroke-width="1.5"/><path d="M5 8h6" stroke="#4a6a85" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">vpn_setup</div><div class="ss ss-blocked">blocked</div></div>
      <div class="sc blocked"><div class="si" style="background:rgba(255,255,255,.04)"><svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M8 2v4M8 10v4M2 8h4M10 8h4" stroke="#4a6a85" stroke-width="1.5" stroke-linecap="round"/></svg></div><div class="sn">compliance</div><div class="ss ss-blocked">blocked</div></div>
    </div>
    <div class="prog-row">
      <div class="prog-bg"><div class="prog-fg" id="pf" style="width:37%"></div></div>
      <div class="prog-lbl" id="pl">3 / 6 required</div>
    </div>
  </div>

  <div class="drift" id="driftBox">
    <div class="drift-ic"><svg width="10" height="10" viewBox="0 0 10 10" fill="none"><path d="M5 1L9 9H1L5 1z" stroke="#fbbf24" stroke-width="1.2"/><path d="M5 4v2M5 7.2v.4" stroke="#fbbf24" stroke-width="1.2" stroke-linecap="round"/></svg></div>
    <div><div class="drift-title">Policy drift detected — v1 to v2</div><div class="drift-body">security=enhanced &middot; badge_zones+=server_room &middot; leave_policy=enhanced<br>badge_access and device_allocation invalidated. Call check_policy then resubmit.</div></div>
  </div>

  <div class="stats">
    <div class="stat"><div class="sv vp">12</div><div class="sl">Systems</div></div>
    <div class="stat"><div class="sv vt">3</div><div class="sl">Task levels</div></div>
    <div class="stat"><div class="sv vy">2</div><div class="sl">Policy drifts</div></div>
    <div class="stat"><div class="sv vb">55</div><div class="sl">Max steps</div></div>
  </div>

  <div class="tasks">
    <div class="tc e"><div class="tch"><span class="tcn">basic_onboarding</span><span class="tcd de">Easy</span></div><p class="tcb">3 systems — AD, email, HRMS. No drift. Agent learns the action loop.</p><div class="tcm">3 systems &middot; 20 steps &middot; score 0.807</div></div>
    <div class="tc m"><div class="tch"><span class="tcn">dept_onboarding</span><span class="tcd dm">Medium</span></div><p class="tcb">6 systems. Policy drifts once at step 15, invalidating completed work.</p><div class="tcm">6 systems &middot; 35 steps &middot; score 0.866</div></div>
    <div class="tc h"><div class="tch"><span class="tcn">enterprise_onboarding</span><span class="tcd dh">Hard</span></div><p class="tcb">12 systems. Two drifts. Systems get invalidated twice.</p><div class="tcm">12 systems &middot; 55 steps &middot; score 0.751</div></div>
  </div>

  <div class="section-lbl" style="margin-bottom:14px">Live agent terminal</div>
  <div class="term">
    <div class="term-top">
      <div class="tdot" style="background:#f43f5e"></div>
      <div class="tdot" style="background:#fbbf24"></div>
      <div class="tdot" style="background:#22d3a0"></div>
      <span class="tl">agent · dept_onboarding · recovering from drift</span>
    </div>
    <div class="term-body" id="tb">
      <div><span class="tc-mut">$</span> <span class="tc-cmd">POST /step</span> <span class="tc-mut">{"action":"check_policy"}</span></div>
      <div id="tlines"></div>
      <span class="blink">█</span>
    </div>
  </div>

  <div class="section-lbl" style="margin-bottom:14px">API endpoints</div>
  <div class="ep-wrap">
    <div class="ep"><span class="mth mp">POST</span><span class="ep-p">/reset</span><span class="ep-d">Start a new onboarding episode</span></div>
    <div class="ep"><span class="mth mp">POST</span><span class="ep-p">/step</span><span class="ep-d">Execute one agent action</span></div>
    <div class="ep"><span class="mth mg">GET</span><span class="ep-p">/state</span><span class="ep-d">Inspect current episode state</span></div>
    <div class="ep"><span class="mth mg">GET</span><span class="ep-p">/render</span><span class="ep-d">ASCII onboarding checklist</span></div>
    <div class="ep"><span class="mth mg">GET</span><span class="ep-p">/tasks</span><span class="ep-d">List all tasks with metadata</span></div>
    <div class="ep"><span class="mth mg">GET</span><span class="ep-p">/health</span><span class="ep-d">Liveness probe</span></div>
  </div>

  <div class="ctas">
    <a href="/docs" class="bpri">Interactive API docs</a>
    <a href="https://github.com/TheKshitij/openenv-onboarding" target="_blank" class="bsec">GitHub</a>
  </div>
</main>

<script>
async function ping(){
  const d=document.getElementById('sdot'),t=document.getElementById('stxt');
  try{
    const r=await fetch('/health'),j=await r.json();
    d.className='sdot';d.style.background='#22d3a0';
    t.innerHTML='<span style="color:#22d3a0">LIVE</span>&nbsp;&nbsp;status: "'+j.status+'" &middot; service: "'+j.service+'"';
  }catch(e){
    d.style.background='#f43f5e';
    t.textContent='Server unreachable';
  }
}
ping();setInterval(ping,10000);

const termSeq=[
  {c:'tc-out',t:'200 OK — Policy v2 active'},
  {c:'tc-out',t:'security=enhanced  device=laptop  leave=enhanced'},
  {c:'tc-out',t:'badge_zones=[main_lobby, floor_3, server_room]'},
  {c:'tc-warn',t:'badge_access FAILED — old zones do not match v2'},
  {c:'tc-cmd',t:'> escalate badge_access zone_mismatch_with_policy_v2'},
  {c:'tc-out',t:'Escalated. badge_access reset to IN_PROGRESS.'},
  {c:'tc-cmd',t:'> submit badge_access zones=main_lobby,floor_3,server_room,access_level=standard'},
  {c:'tc-out',t:'badge_access COMPLETE. reward=+0.20'},
];
let ti=0;
function typeNext(){
  if(ti>=termSeq.length){ti=0;document.getElementById('tlines').innerHTML='';}
  const d=document.createElement('div');
  d.className=termSeq[ti].c;d.textContent=termSeq[ti].t;
  document.getElementById('tlines').appendChild(d);
  ti++;
  setTimeout(typeNext,700+Math.random()*500);
}
setTimeout(typeNext,1000);

const states=[
  {cards:['complete','complete','complete','failed','active','active','blocked','blocked'],p:'37%',l:'3 / 6 required'},
  {cards:['complete','complete','complete','active','complete','active','blocked','blocked'],p:'50%',l:'3 / 6 required'},
  {cards:['complete','complete','complete','active','complete','complete','pending','blocked'],p:'50%',l:'3 / 6 required'},
  {cards:['complete','complete','complete','complete','complete','complete','active','blocked'],p:'67%',l:'4 / 6 required'},
  {cards:['complete','complete','complete','complete','complete','complete','complete','pending'],p:'83%',l:'5 / 6 required'},
];
let si=0;
function cycleCards(){
  si=(si+1)%states.length;
  const s=states[si];
  document.querySelectorAll('#sg .sc').forEach((c,i)=>{
    c.className='sc '+s.cards[i];
    const ss=c.querySelector('.ss');
    ss.className='ss ss-'+s.cards[i];
    ss.textContent=s.cards[i];
  });
  document.getElementById('pf').style.width=s.p;
  document.getElementById('pl').textContent=s.l;
}
setInterval(cycleCards,2800);

let dv=true;
setInterval(()=>{
  dv=!dv;
  document.getElementById('driftBox').style.opacity=dv?'1':'0.2';
  document.getElementById('driftBox').style.transition='opacity .5s';
},3500);

const cv=document.getElementById('cv'),cx=cv.getContext('2d');
let W,H,nodes=[];
function initCanvas(){
  W=cv.width=innerWidth;H=cv.height=innerHeight;nodes=[];
  for(let i=0;i<20;i++)nodes.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.4,vy:(Math.random()-.5)*.4});
}
function drawCanvas(){
  cx.clearRect(0,0,W,H);
  nodes.forEach(n=>{n.x+=n.vx;n.y+=n.vy;if(n.x<0||n.x>W)n.vx*=-1;if(n.y<0||n.y>H)n.vy*=-1});
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
    const d=Math.hypot(nodes[i].x-nodes[j].x,nodes[i].y-nodes[j].y);
    if(d<200){cx.strokeStyle='rgba(26,48,69,'+(1-d/200)*.6+')';cx.lineWidth=.8;cx.beginPath();cx.moveTo(nodes[i].x,nodes[i].y);cx.lineTo(nodes[j].x,nodes[j].y);cx.stroke()}
  }
  nodes.forEach(n=>{cx.fillStyle='rgba(167,139,250,.4)';cx.beginPath();cx.arc(n.x,n.y,2,0,Math.PI*2);cx.fill()});
  requestAnimationFrame(drawCanvas);
}
window.addEventListener('resize',initCanvas);
initCanvas();drawCanvas();
</script>
</body>
</html>"""
    return HTMLResponse(content=PAGE)


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
