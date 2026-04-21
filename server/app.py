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
:root{--bg:#06101e;--surf:#0d1b2e;--bdr:#1a2d45;--txt:#e2edf8;--mut:#5a7a9a;--blue:#3b82f6;--green:#22d3a0;--yel:#fbbf24;--red:#f43f5e;--purple:#a78bfa}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh}
main{max-width:860px;margin:0 auto;padding:48px 20px 72px}
.hd{text-align:center;margin-bottom:40px}
.pill{display:inline-flex;align-items:center;gap:6px;background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.3);color:var(--purple);border-radius:999px;padding:4px 14px;font-size:.72rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:16px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pd 2s ease-in-out infinite}
@keyframes pd{0%,100%{opacity:1}50%{opacity:.3}}
h1{font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;letter-spacing:-.03em;background:linear-gradient(135deg,#e2edf8,#a78bfa 50%,var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px}
.sub{color:var(--mut);font-size:.9rem;max-width:520px;margin:0 auto;line-height:1.75}
.sbar{display:flex;align-items:center;justify-content:center;gap:8px;background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:10px 20px;margin-bottom:28px}
.sdot{width:8px;height:8px;border-radius:50%;background:var(--mut);flex-shrink:0;transition:background .3s}
.sdot.ok{background:var(--green)}.sdot.err{background:var(--red)}
#stxt{font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--mut)}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:24px}
.scard{background:var(--surf);border:1px solid var(--bdr);border-radius:10px;padding:16px;text-align:center}
.sv{font-size:1.7rem;font-weight:800;color:var(--purple)}
.sl{font-size:.68rem;color:var(--mut);text-transform:uppercase;letter-spacing:.07em;margin-top:3px}
.sec{font-size:.7rem;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.tcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-bottom:24px}
.tc{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:20px;transition:border-color .2s,transform .15s}
.tc:hover{transform:translateY(-2px)}
.tc.e:hover{border-color:var(--green)}.tc.m:hover{border-color:var(--yel)}.tc.h:hover{border-color:var(--red)}
.th{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.tn{font-family:'JetBrains Mono',monospace;font-size:.76rem;font-weight:600}
.dif{font-size:.66rem;font-weight:700;padding:2px 8px;border-radius:999px}
.dif.e{background:rgba(34,211,160,.1);color:var(--green)}.dif.m{background:rgba(251,191,36,.1);color:var(--yel)}.dif.h{background:rgba(244,63,94,.1);color:var(--red)}
.td{font-size:.8rem;color:var(--mut);line-height:1.6}
.tm{margin-top:8px;font-size:.68rem;color:#2a4a6a;font-family:'JetBrains Mono',monospace}
.ep-box{background:var(--surf);border:1px solid var(--bdr);border-radius:12px;padding:16px 20px;margin-bottom:24px}
.ep{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid var(--bdr)}
.ep:last-child{border-bottom:none}
.mth{font-family:'JetBrains Mono',monospace;font-size:.66rem;font-weight:700;padding:2px 7px;border-radius:4px;min-width:38px;text-align:center}
.mth.p{background:rgba(34,211,160,.1);color:var(--green)}.mth.g{background:rgba(59,130,246,.1);color:var(--blue)}
.epath{font-family:'JetBrains Mono',monospace;font-size:.8rem}
.edsc{font-size:.76rem;color:var(--mut);margin-left:auto}
.acts{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:6px;padding:11px 24px;border-radius:8px;font-size:.87rem;font-weight:600;text-decoration:none;border:none;font-family:inherit;cursor:pointer;transition:all .15s}
.bpri{background:var(--purple);color:#fff}.bpri:hover{background:#8b5cf6;transform:translateY(-1px)}
.bgho{background:transparent;color:var(--mut);border:1px solid var(--bdr)}.bgho:hover{border-color:var(--mut);color:var(--txt)}
</style>
</head>
<body>
<main>
  <div class="hd">
    <div class="pill"><div class="dot"></div>OpenEnv Environment</div>
    <h1>Enterprise Employee<br>Onboarding Agent</h1>
    <p class="sub">An OpenEnv environment where an AI agent navigates 12 enterprise IT &amp; HR systems to onboard a new hire — while company policies drift mid-episode without warning.</p>
  </div>
  <div class="sbar"><div class="sdot" id="sdot"></div><span id="stxt">Checking server&hellip;</span></div>
  <div class="stats">
    <div class="scard"><div class="sv">12</div><div class="sl">Systems</div></div>
    <div class="scard"><div class="sv">3</div><div class="sl">Task Levels</div></div>
    <div class="scard"><div class="sv">55</div><div class="sl">Max Steps</div></div>
    <div class="scard"><div class="sv">2x</div><div class="sl">Policy Drifts</div></div>
  </div>
  <div class="sec">Tasks</div>
  <div class="tcards">
    <div class="tc e"><div class="th"><span class="tn">basic_onboarding</span><span class="dif e">Easy</span></div><p class="td">3-system onboarding: AD, email, HRMS. No policy drift. Agent learns the action space.</p><div class="tm">3 systems &middot; 20 steps &middot; no drift</div></div>
    <div class="tc m"><div class="th"><span class="tn">dept_onboarding</span><span class="dif m">Medium</span></div><p class="td">6 systems including payroll, badge access, and device allocation. Policy changes once at step 15.</p><div class="tm">6 systems &middot; 35 steps &middot; 1 drift</div></div>
    <div class="tc h"><div class="th"><span class="tn">enterprise_onboarding</span><span class="dif h">Hard</span></div><p class="td">12 systems with VPN, compliance training, and health insurance. Two policy drifts invalidate completed work.</p><div class="tm">12 systems &middot; 55 steps &middot; 2 drifts</div></div>
  </div>
  <div class="sec">API Endpoints</div>
  <div class="ep-box">
    <div class="ep"><span class="mth p">POST</span><span class="epath">/reset</span><span class="edsc">Start a new onboarding episode</span></div>
    <div class="ep"><span class="mth p">POST</span><span class="epath">/step</span><span class="edsc">Execute one action</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/state</span><span class="edsc">Inspect current episode state</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/tasks</span><span class="edsc">List all tasks</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/render</span><span class="edsc">ASCII onboarding checklist snapshot</span></div>
    <div class="ep"><span class="mth g">GET</span><span class="epath">/health</span><span class="edsc">Liveness probe</span></div>
  </div>
  <div class="acts">
    <a href="/docs" class="btn bpri">&#9889; Interactive API Docs</a>
    <a href="https://github.com/TheKshitij/openenv-onboarding" target="_blank" class="btn bgho">GitHub &rarr;</a>
  </div>
</main>
<script>
async function ping(){
  const d=document.getElementById('sdot'),t=document.getElementById('stxt');
  try{
    const r=await fetch('/health'),j=await r.json();
    d.className='sdot ok';
    t.innerHTML='<span style="color:#22d3a0">&#9679; LIVE</span>&nbsp;&nbsp;'+j.status+' &middot; '+j.service;
  }catch(e){d.className='sdot err';t.textContent='Server unreachable';}
}
ping();setInterval(ping,10000);
</script>
</body>
</html>"""
    return HTMLResponse(content=PAGE)


def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
