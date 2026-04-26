"""
inference_local.py — Local GPU inference for Enterprise Employee Onboarding
Runs the model directly on the Colab GPU using Unsloth.
"""

import os, sys, re
from unsloth import FastLanguageModel
from onboarding_env import OnboardingAction, OnboardingEnv, OnboardingObservation, TASK_IDS

# --- Configuration ---
MODEL_PATH = "importkk/openenv-onboarding-model"
MAX_STEPS = {"basic_onboarding": 20, "dept_onboarding": 35, "enterprise_onboarding": 55}
SEED = 42

# 1. Load Model
print(f"--- Loading model {MODEL_PATH} into GPU ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_PATH,
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# 2. Reasoning Prompt
SYSTEM_PROMPT = """You are an enterprise IT onboarding specialist. 
Your job is to complete an employee's onboarding across all required systems.

PRIMARY GOAL: Provision and Submit all REQ systems until Progress is 100%.
AVAILABLE ACTIONS:
  provision <id>
  submit <id> <field>=<value>,<field>=<value>...
  check_policy
  escalate <id> <reason>
  verify <id>
  hold

FIELD REFERENCE (Only use these fields for each system):
  ad_account:          username=<emp_id>, security_level=<policy.it_security_level>
  email_setup:         alias=<first.last>, quota=50gb
  hrms_registration:   role=<role>, department=<dept>, grade=L3
  payroll_enrollment:  bank_account=XXXXXXXX, pan_number=ABCDE1234F, leave_policy=<policy.leave_policy>
  badge_access:        zones=<comma-separated policy.badge_zones>, access_level=standard
  device_allocation:   device_type=<policy.device_type>, asset_tag=IT<emp_id>
  vpn_setup:           profile=corporate, mfa_method=totp
  compliance_training: modules=<comma-separated policy.compliance_modules>, deadline_days=30
  it_security_training:level=<policy.it_security_level>, certification=yes

RULES:
1. Use check_policy once when a drift occurs.
2. If a system is COMPLETED, move to the next PENDING/REQ system. DO NOT repeat provision/submit on a COMPLETED system.
3. Use submit with ONLY the fields listed in the FIELD REFERENCE above.
4. Output reasoning in <thought> and the command in <action> tags."""

def get_hint(obs):
    """Compute the exact correct next command using live policy values."""
    p = obs.current_policy
    emp = obs.employee_id.lower().replace("-", "")

    # Exact field mapping — uses ACTUAL policy values, not templates
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

    # Policy drift takes priority
    if obs.policy_drift_event:
        return "check_policy"

    # Find the first actionable required system in order
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


def fmt_obs(obs, step, hist):
    """Build the full observation string matching training data format."""
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
        status_str = str(s.status.value).upper()
        err  = f" ERROR: {s.error_msg[:60]}" if s.error_msg else ""
        dep  = f" [blocked by: {s.dependencies}]" if s.status.value == "blocked" else ""
        lines.append(f"  [{req}] {s.id:<24} {status_str:<12} attempts={s.attempts}{err}{dep}")

    lines += ["", "RECENT HISTORY:"]
    for h in hist[-4:]:
        lines.append(f"  {h}")

    # Inject the concrete suggested next action
    hint = get_hint(obs)
    lines += ["", f"SUGGESTED NEXT ACTION: {hint}", ""]
    lines += [f"Last result: {obs.last_action_result[:80]}", "", "Your action:"]
    return "\n".join(lines)

def get_local_action(obs, step, hist):
    obs_str = fmt_obs(obs, step, hist)

    full_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{obs_str}<|im_end|>\n<|im_start|>assistant\n<thought>\n"
    
    inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.1)
    
    new_tokens = outputs[0][input_len:]
    content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Extract action from <action> tags
    action = "hold"
    match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
    if match:
        action = match.group(1).strip().split('\n')[0].strip()
    else:
        for line in content.splitlines():
            line = line.strip()
            if any(line.startswith(v) for v in ["provision", "submit", "check_policy", "escalate", "verify", "hold"]):
                action = line
                break
    
    # BULLETPROOF CLEANING: Remove spaces after commas in the command
    # e.g. "submit id key=val, key=val" -> "submit id key=val,key=val"
    if "submit" in action:
        action = re.sub(r",\s+", ",", action)
    
    return action

def run_episode(task):
    env = OnboardingEnv(task=task, seed=SEED)
    obs = env.reset()
    print(f"\n[START] task={task} model=LOCAL_GPU")
    
    final_score = 0.0
    hist = []
    for step in range(1, MAX_STEPS[task] + 1):
        action_str = get_local_action(obs, step, hist)
        result = env.step(OnboardingAction(action=action_str))
        final_score = result.score
        
        msg = result.info.get("action_result", "")
        hist.append(f"step={step} {action_str!r} → {msg[:50]}")
        
        print(f"[STEP] {step}: {action_str!r} -> Reward: {result.reward:.2f} Score: {result.score:.3f}")
        obs = result.observation
        if result.done: break
        
    print(f"[END] task={task} Final Score: {final_score:.3f}")

if __name__ == "__main__":
    for task in TASK_IDS:
        run_episode(task)
