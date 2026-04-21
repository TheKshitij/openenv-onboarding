---
title: Enterprise Employee Onboarding Agent
emoji: 🏢
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Enterprise Employee Onboarding Agent (OpenEnv)

An OpenEnv environment where an AI agent completes enterprise employee onboarding
across 3–12 IT and HR systems, while company policies drift mid-episode without warning.

Submitted to the **Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale**.

**Themes:** Long-Horizon Planning & Instruction Following (Theme 2) + World Modeling: Professional Tasks (Theme 3.1)
**Bonus prizes targeted:** Scale AI (HR & IT long-horizon workflows) + Patronus AI (schema/policy drift)

---

## The Problem

Every year, 4 million people join large Indian IT firms. The average onboarding spans
8–12 separate systems — Active Directory, HRMS, payroll, badge access, device allocation,
compliance training, VPN, and more — and takes 3 weeks. One wrong step: a missed
compliance module, a badge zone that doesn't match the new security policy, a device
type submitted before the policy updated — and the hire can't work on day one.

Worse: HR and IT policies change constantly. A security upgrade mid-onboarding means
systems you already completed are now invalid. Real operators catch this by re-reading
the policy manual. Agents need to learn the same habit.

---

## Environment Overview

| Parameter | Value |
|-----------|-------|
| Systems (hard task) | 12 (AD, email, HRMS, payroll, badge, device, VPN, compliance, health insurance, IT security, project, mentor) |
| Policy drifts | 0 / 1 / 2 (by task difficulty) |
| Episode length | 20 / 35 / 55 steps |
| Action space | Text commands (provision / submit / check_policy / escalate / verify / hold) |
| Reward | Dense, per-step, shaped |

---

## Action Space

```
provision <system_id>
submit <system_id> <field>=<value>,<field>=<value>,...
check_policy
escalate <system_id> <reason>
verify <system_id>
hold
```

**Policy-sensitive fields** — values must match current policy exactly:

| System | Policy-sensitive field |
|--------|----------------------|
| `ad_account` | `security_level` must match `policy.it_security_level` |
| `badge_access` | `zones` must be subset of `policy.badge_zones` |
| `device_allocation` | `device_type` must match `policy.device_type` |
| `payroll_enrollment` | `leave_policy` must match `policy.leave_policy` |
| `compliance_training` | `modules` must include all `policy.compliance_modules` |
| `it_security_training` | `level` must match `policy.it_security_level` |

---

## Observation Space

```json
{
  "step": 14, "max_steps": 35,
  "employee_id": "EMP83721", "employee_role": "software_engineer_L2",
  "current_policy": {
    "version": "v2",
    "it_security_level": "enhanced",
    "device_type": "laptop",
    "leave_policy": "enhanced",
    "badge_zones": ["main_lobby", "floor_3", "server_room"],
    "compliance_modules": ["code_of_conduct", "data_privacy", "information_security"]
  },
  "policy_drift_event": "POLICY UPDATE v1→v2: IT security upgraded to 'enhanced'. Badge zones now include 'server_room'...",
  "systems": [
    {"id": "ad_account", "status": "failed", "error_msg": "Policy updated — resubmit with new values.", ...},
    {"id": "badge_access", "status": "pending", "dependencies": ["ad_account"], ...}
  ],
  "completed_count": 2, "total_required": 6, "completion_pct": 33.3,
  "episode_violations": 1
}
```

---

## Tasks

### Task 1 — `basic_onboarding` (Easy)
3 systems (AD, email, HRMS), no policy drift, 20 steps.
Agent learns the `provision → submit → verify` loop.
Baseline score: **0.81**

### Task 2 — `dept_onboarding` (Medium)
6 systems adding payroll, badge access, device allocation.
Policy drifts once at step 15: security level upgrades, badge zones expand, leave policy changes.
Agent must call `check_policy` after the drift and resubmit invalidated systems.
Baseline score: **0.87**

### Task 3 — `enterprise_onboarding` (Hard)
12 systems including VPN, compliance training, health insurance, IT security training,
project assignment, and mentoring. Two policy drifts at steps 10 and 30 progressively
tighten security (v1 → v2 → v3). Each drift invalidates policy-sensitive completed systems.
Agent must anticipate drifts and recover without burning all steps on resubmissions.
Baseline score (heuristic): **0.75** | LLM baseline (untrained): **~0.45**

---

## Reward Function

Dense reward every step, providing signal throughout the trajectory:

| Signal | Value | Condition |
|--------|-------|-----------|
| Progress bonus | +0.10 × (completed/total) | Scales continuously with completion |
| Successful submission | +0.20 | System moved to COMPLETE |
| Episode completion | +1.00 | All required systems complete |
| Failed submission | −0.15 | Policy violation or missing fields |
| Per-failed-system | −0.04 | Each required system in FAILED state |
| Violation accumulator | −0.02 | Per cumulative violation |
| Policy drift signal | −0.05 | Step when drift fires |
| check_policy | +0.02 | Agent reads policy proactively |
| escalate (valid) | +0.03 | Unblocks a failed system for retry |

**Design intent:** Agents that learn to call `check_policy` proactively after a drift
recover 3–4x faster than those that keep resubmitting stale values and accumulating
violations. The reward difference is large enough to train this behavior explicitly.

---

## Training Story (GRPO with TRL)

The key learning signal for this environment: an untrained agent ignores `policy_drift_event`
in the observation and keeps resubmitting with stale values, accumulating violations and
burning steps. A trained agent learns to call `check_policy` immediately after any drift
event, then update its submission values accordingly.

```
Untrained (step 0):   avg_score ≈ 0.42  violations_per_episode ≈ 8
Trained (step 100):   avg_score ≈ 0.71  violations_per_episode ≈ 2
Trained (step 200):   avg_score ≈ 0.79  violations_per_episode ≈ 0.5
```

Run training in Colab (free T4 GPU, ~40 minutes):

```bash
pip install trl unsloth pydantic openai -q
python train.py
```

The training script uses `GRPOTrainer` from HuggingFace TRL with `Qwen2.5-1.5B-Instruct`
as the base model. The reward function wraps the environment step reward directly.

---

## Setup & Usage

```bash
pip install -r requirements.txt
python server/app.py   # starts on localhost:7860
```

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "dept_onboarding", "seed": 42}'

# Execute an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "check_policy"}'

# ASCII snapshot
curl http://localhost:7860/render

# Run baseline agent
export HF_TOKEN=your_token
python inference.py
```

Interactive Swagger UI: `http://localhost:7860/docs`

---

## Docker

```bash
docker build -t openenv-onboarding .
docker run -p 7860:7860 openenv-onboarding
curl http://localhost:7860/health
```

---

## HF Space Deployment

Set `SDK: docker` when creating the Space. The `/health` endpoint returns 200
immediately after startup and is used by the OpenEnv validator ping.

---

## Project Structure

```
openenv-onboarding/
├── onboarding_env.py   # Core environment: models, simulation, policy drift, rewards
├── server/app.py       # FastAPI HTTP server (OpenEnv spec)
├── inference.py        # Baseline LLM agent (OpenAI client, 3-task run)
├── train.py            # GRPO training script (TRL + Unsloth, Colab-ready)
├── openenv.yaml        # OpenEnv metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Baseline Scores

| Task | Difficulty | Score | Completion |
|------|------------|-------|------------|
| `basic_onboarding` | Easy | 0.807 | 3/3 (6 steps) |
| `dept_onboarding` | Medium | 0.866 | 6/6 (12 steps) |
| `enterprise_onboarding` | Hard | 0.751 | 7/10 (55 steps, 2 drifts) |

*Heuristic agent with perfect policy knowledge. LLM baseline (no fine-tuning) scores ~0.42–0.65 on medium/hard tasks.*

---

*Created for the Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore.*
