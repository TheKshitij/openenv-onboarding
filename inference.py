"""
inference.py — Baseline LLM agent for Enterprise Employee Onboarding
Runs across all 3 tasks and emits [START]/[STEP]/[END] stdout format.

Environment variables:
    API_BASE_URL   (default: https://router.huggingface.co/v1)
    MODEL_NAME     (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key
"""

import os, sys, textwrap, re
from typing import List, Optional

from openai import OpenAI
from onboarding_env import OnboardingAction, OnboardingEnv, OnboardingObservation, TASK_IDS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "importkk/openenv-onboarding-model")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "no-key")
BENCHMARK    = "enterprise-onboarding"
MAX_STEPS    = {"basic_onboarding": 20, "dept_onboarding": 35, "enterprise_onboarding": 55}
TEMPERATURE  = 0.2
MAX_TOKENS   = 80
SEED         = 42


# ─── Loggers ─────────────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    ev = error if error else "null"
    dv = str(done).lower()
    print(f"[STEP] step={step} action={action.replace(chr(10),' ')!r} "
          f"reward={reward:.2f} done={dv} error={ev}", flush=True)

def log_end(success, steps, score, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} "
          f"score={score:.3f} rewards={rs}", flush=True)


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM = textwrap.dedent("""
    You are an enterprise IT onboarding specialist.
    Your job is to complete an employee's onboarding across all required systems.

    AVAILABLE ACTIONS (output exactly one per turn, no explanation):
      provision <system_id>
      submit <system_id> <field>=<value>,<field>=<value>,...
      check_policy
      escalate <system_id> <reason>
      verify <system_id>
      hold

    CRITICAL RULES:
    - Always check_policy after a POLICY DRIFT EVENT before resubmitting any system.
    - provision a system before submit.
    - Dependencies must be COMPLETE before you can provision a dependent system.
    - Submit fields must match CURRENT policy exactly (security_level, device_type,
      leave_policy, badge zones, compliance modules).
    - If a system is FAILED, escalate it first, then reprovision and resubmit.
    - Output ONLY the action string. No explanation, no markdown, no quotes.

    Field reference (use current policy values):
      ad_account:          username=<emp_id>, security_level=<policy.it_security_level>
      email_setup:         alias=<firstname.lastname>, quota=50gb
      hrms_registration:   role=<role>, department=<dept>, grade=L3
      payroll_enrollment:  bank_account=XXXXXXXX, pan_number=ABCDE1234F, leave_policy=<policy.leave_policy>
      badge_access:        zones=<comma-separated policy.badge_zones>, access_level=standard
      device_allocation:   device_type=<policy.device_type>, asset_tag=IT<emp_id>
      vpn_setup:           profile=corporate, mfa_method=totp
      compliance_training: modules=<comma-separated policy.compliance_modules>, deadline_days=30
      health_insurance:    plan=family, dependents=0
      it_security_training:level=<policy.it_security_level>, certification=yes
      project_assignment:  project_code=PRJ001, manager_id=MGR001
      mentor_assignment:   mentor_id=MNT001, meeting_cadence=weekly
""").strip()


# ─── Observation formatter ────────────────────────────────────────────────────

def _get_hint(obs: OnboardingObservation) -> str:
    """Compute the exact correct next command using live policy values."""
    p = obs.current_policy
    emp = obs.employee_id.lower().replace("-", "")
    fields = {
        "ad_account":           f"username={emp},security_level={p.it_security_level}",
        "email_setup":          f"alias={emp},quota=50gb",
        "hrms_registration":    f"role={obs.employee_role},department={obs.department},grade=L3",
        "payroll_enrollment":   f"bank_account=XXXXXXXX,pan_number=ABCDE1234F,leave_policy={p.leave_policy}",
        "badge_access":         f"zones={','.join(p.badge_zones)},access_level=standard",
        "device_allocation":    f"device_type={p.device_type},asset_tag=IT{emp}",
        "vpn_setup":            "profile=corporate,mfa_method=totp",
        "compliance_training":  f"modules={','.join(p.compliance_modules)} deadline_days=30",
        "it_security_training": f"level={p.it_security_level},certification=yes",
        "health_insurance":     "plan=family,dependents=0",
        "project_assignment":   "project_code=PRJ001,manager_id=MGR001",
        "mentor_assignment":    "mentor_id=MNT001,meeting_cadence=weekly",
    }
    if obs.policy_drift_event:
        return "check_policy"
    for s in obs.systems:
        if not s.required:
            continue
        status = s.status.value
        if status == "failed":
            return f"escalate {s.id} retry_after_failure"
        if status == "pending":
            return f"provision {s.id}"
        if status in ("provisioned", "in_progress"):
            f = fields.get(s.id, "")
            return f"submit {s.id} {f}" if f else f"submit {s.id}"
    return "hold"


def _fmt(obs: OnboardingObservation, step: int, hist: List[str]) -> str:
    p = obs.current_policy
    lines = [
        f"=== STEP {step} ===",
        f"Employee: {obs.employee_id}  role={obs.employee_role}  dept={obs.department}",
        f"Progress: {obs.completed_count}/{obs.total_required} required ({obs.completion_pct:.0f}%)",
        f"Violations: {obs.episode_violations}",
        "",
        f"CURRENT POLICY ({p.version.value}):",
        f"  it_security_level = {p.it_security_level}",
        f"  device_type       = {p.device_type}",
        f"  leave_policy      = {p.leave_policy}",
        f"  badge_zones       = {p.badge_zones}",
        f"  compliance_modules= {p.compliance_modules}",
        "",
    ]
    if obs.policy_drift_event:
        lines += [f"*** POLICY DRIFT: {obs.policy_drift_event[:120]} ***", ""]

    lines.append("SYSTEMS:")
    for s in obs.systems:
        req  = "REQ" if s.required else "opt"
        # FORCE UPPERCASE so model recognises pending systems as action items
        status_str = s.status.value.upper()
        err  = f" ERROR: {s.error_msg[:60]}" if s.error_msg else ""
        dep  = f" [blocked by: {s.dependencies}]" if s.status.value == "blocked" else ""
        lines.append(f"  [{req}] {s.id:<24} {status_str:<12} attempts={s.attempts}{err}{dep}")

    lines += ["", "RECENT HISTORY:"]
    for h in hist[-4:]:
        lines.append(f"  {h}")

    hint = _get_hint(obs)
    lines += ["", f"SUGGESTED NEXT ACTION: {hint}", ""]
    lines += [f"Last result: {obs.last_action_result[:80]}", "", "Your action:"]
    return "\n".join(lines)


# ─── LLM call ────────────────────────────────────────────────────────────────

def get_action(client: OpenAI, obs: OnboardingObservation, step: int, hist: List[str]) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _fmt(obs, step, hist)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = resp.choices[0].message.content or "hold"
        
        # Extract action from <action> tags if present (reasoning model format)
        match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if match:
            # Take only the first line to block chatty/long responses
            act = match.group(1).strip().split('\n')[0].strip()
        else:
            # Fallback: find first line starting with a known command
            act = "hold"
            for line in content.strip().splitlines():
                line = line.strip()
                if any(line.startswith(v) for v in ["provision", "submit", "check_policy", "escalate", "verify", "hold"]):
                    act = line
                    break
            
        return act if act else "hold"
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return "hold"


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task: str) -> None:
    env     = OnboardingEnv(task=task, seed=SEED)
    obs     = env.reset()
    rewards: List[float] = []
    hist:   List[str]   = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS[task] + 1):
            if env._ep.get("done", False):
                break
            action_str = get_action(client, obs, step, hist)
            result     = env.step(OnboardingAction(action=action_str))
            reward, done = result.reward, result.done
            err = None
            msg = result.info.get("action_result", "")
            if msg.startswith("Error:") or msg.startswith("Parse error"):
                err = msg[:60]

            rewards.append(reward)
            steps_taken = step
            score       = result.score

            log_step(step=step, action=action_str, reward=reward, done=done, error=err)
            hist.append(f"step={step} {action_str!r} → {msg[:50]}")
            obs = result.observation
            if done:
                break

        success = score >= 0.30
    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASK_IDS:
        run_episode(client, task)
        print("", flush=True)

if __name__ == "__main__":
    main()
