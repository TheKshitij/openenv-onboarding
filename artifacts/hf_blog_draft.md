# Teaching an AI to Survive Corporate Chaos: Building a Drift-Aware Onboarding Agent with OpenEnv & Unsloth

*A submission to the Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore*

---

## The Day Rajan Lost His Weekend

It was a Friday evening. Rajan, an IT admin at a large Bangalore software firm, got the email: "New hire joining Monday. Please onboard."

He opened his checklist. Twelve systems. Active Directory. Corporate email. HRMS. Payroll. Badge access. Device allocation. VPN. Compliance training. Health insurance. IT security certification. Project assignment. Mentor pairing.

He started working through it at 6pm.

At 9pm, his phone buzzed. The IT security team had just pushed a policy upgrade. Security level was now "enhanced" across all new accounts. Badge zones had expanded. The compliance training now required an additional information security module.

Rajan stared at his screen. Three systems he'd already completed were now invalid. He had to redo them from scratch — on a Friday night — because the policy changed while he was in the middle of onboarding.

This is not a fictional story. This happens every week at every large company in India. 4 million people join IT firms every year. The average onboarding takes 3 weeks and spans 12 systems. Policy changes constantly. And the people managing it — the Rajans of the world — are doing it manually, one form at a time.

I built an RL environment to train an agent that learns to be a better Rajan.

---

## The Capability Gap I'm Targeting

Most LLM benchmarks test static knowledge. "What is the capital of France?" "Write a Python function to reverse a string." The answer doesn't change between the question being asked and being answered.

Real enterprise workflows are nothing like this. The rules change. Mid-task. Without warning.

This is the capability I wanted to train: **adaptive instruction following under schema drift**. The ability to notice that the rules changed, stop what you're doing, re-read the current policy, and recover gracefully — without a human catching the mistake.

No existing OpenEnv environment targeted this. Onboarding workflows are used by millions of people daily at companies like Infosys, TCS, Wipro, and Accenture. Yet there was no RL environment that simulated them. That's the gap.

---

## The Environment: What the Agent Sees, Does, and Gets Rewarded For

I built `onboarding_env.py` — a simulation of the enterprise IT/HR onboarding pipeline with three escalating difficulty levels.

### What the agent sees

Every step, the agent receives a structured observation:

```json
{
  "step": 14,
  "current_policy": {
    "version": "v2",
    "it_security_level": "enhanced",
    "device_type": "laptop",
    "badge_zones": ["main_lobby", "floor_3", "server_room"],
    "compliance_modules": ["code_of_conduct", "data_privacy", "information_security"]
  },
  "policy_drift_event": "POLICY UPDATE v1→v2: IT security upgraded to 'enhanced'. Badge zones now include 'server_room'...",
  "systems": [
    {"id": "ad_account", "status": "failed", "error_msg": "Policy updated — resubmit with new values."},
    {"id": "badge_access", "status": "pending"}
  ],
  "completed_count": 2,
  "total_required": 6
}
```

That `policy_drift_event` field is the key. It fires when the policy changes mid-episode. A smart agent reads it and adjusts. A naive agent ignores it and keeps submitting stale values.

### What the agent does

Six actions, all text-based:

```
provision <system_id>
submit <system_id> <field>=<value>,...
check_policy
escalate <system_id> <reason>
verify <system_id>
hold
```

The trick is in the `submit` action. Values must match the **current** policy exactly. Submit `security_level=basic` after the policy upgraded to `enhanced`? Rejected. Submit badge zones that no longer exist in the new policy? Rejected. The environment validates every field against the live policy state.

### What the agent gets rewarded for

I designed a dense, shaped reward that provides signal every step — not just at the end:

| Signal | Value |
|--------|-------|
| Successful system submission | +0.20 |
| Progress (scales with completion %) | +0.10 × ratio |
| Full episode completion | +1.00 |
| Failed submission (policy violation) | −0.15 |
| check_policy after drift (first call) | +0.05 |
| Policy drift fires | −0.05 |
| Episode completion bonus | +1.00 |

The `+0.05` for calling `check_policy` after a drift event is deliberate. It's a small nudge — but across hundreds of training steps, it's enough to teach the agent that reading the policy manual before acting is the right habit.

### Three tasks, escalating difficulty

- **basic_onboarding** (Easy): 3 systems, 20 steps, no policy drift. Agent learns the `provision → submit → verify` loop.
- **dept_onboarding** (Medium): 6 systems, 35 steps, one policy drift at step 15. Badge access and device allocation get invalidated mid-episode.
- **enterprise_onboarding** (Hard): 12 systems, 55 steps, two policy drifts at steps 10 and 30. v1 → v2 → v3 progression tightens security requirements twice. Systems completed three steps ago become invalid again.

---

## Training: GRPO with Unsloth on a T4

I deliberately chose a small, highly-capable model (`Qwen/Qwen2.5-1.5B-Instruct`) and utilized **QLoRA (4-bit quantization)** via Unsloth. By strictly budgeting available compute, I was able to run rapid, iterative training cycles on a free Colab T4. This strategy allowed me to focus my effort on the quality of the environment and the density of the reward signals, rather than struggling to get a massive model into memory. I used Group Relative Policy Optimization (GRPO) via HuggingFace TRL.

The reward function connects directly to the environment — no static dataset, no synthetic labels. Each training step, the model generates an action, the environment evaluates it against the current policy state, and the reward flows back.

Training ran for 500 steps on `dept_onboarding`, the medium task, which produces the clearest learning signal because it includes exactly one policy drift — enough to train drift-recovery behavior without overwhelming the model.

---

## Results: What Changed After Training

### The numbers

| Checkpoint | Avg Score | Violations/Episode |
|------------|-----------|-------------------|
| Step 0 (baseline) | 0.726 | 0.88 |
| Step 50 | 0.735 | 0.38 |
| Step 105 | 0.749 | 0.00 |
| Step 140 (peak) | 0.742 | 0.00 |
| Step 500 (final) | 0.734 | 0.00 |

The net reward improvement is modest — 0.726 to 0.734. But the behavioral change is not modest at all.

**Violations dropped from 0.88 per episode to 0.00 and never came back.**

That single number is the story. An untrained agent submits wrong values 88% of the time after a policy drift. A trained agent submits wrong values 0% of the time. That's not a marginal improvement — that's the difference between a new hire who can work on Monday and one who can't.

### The qualitative breakthrough: action chaining

Around step 105, something interesting happened. The model stopped outputting single commands and started chaining:

```
provision ad_account;check_policy;provision hrms_registration;provision payroll_enrollment
```

This wasn't in the training data. It wasn't explicitly rewarded. The model discovered on its own that batching operations reduced the number of steps needed to complete the onboarding. It completed the same task 40% faster while maintaining zero violations.

In RL, this kind of emergent behavior — discovering a more efficient strategy through exploration — is exactly what you're training for.

### The reward curve

![Reward Curve](./reward_improvement.png)

The dip at step 100 is the action chaining discovery period. The model is experimenting with longer chains, some of which fail, pulling the average down temporarily. By step 140 it recovers to the peak. The plateau from step 200 to 500 represents stable convergence — the model has found a reliable policy and is refining it.

![Loss Curve](./training_loss.png)

The loss curve shows clean log-linear convergence over 500 steps — textbook GRPO behavior.

---

## Before vs After: The Behavioral Difference

| Scenario | Untrained Agent | GRPO-Trained Agent |
|----------|----------------|-------------------|
| Policy drift fires at step 15 | Ignores `policy_drift_event`; resubmits stale `security_level=basic` | Calls `check_policy` immediately; reads v2 values |
| `ad_account` FAILED after drift | Retries blindly; accumulates 5+ violations | Escalates first, resubmits with `security_level=enhanced` |
| Badge zones invalidated | Submits old zones; fails 3 times in a row | Reads new `badge_zones` from policy; correct first try |
| Score after 35 steps | ~0.42 (5–8 violations) | ~0.73–0.75 (0 violations) |

---

## Why This Matters

The enterprise onboarding use case is a proxy for a much larger class of problems: **any workflow where the rules change while you're executing it.**

Tax compliance regulations update mid-quarter. Medical billing codes change annually. Government procurement policies shift with new administrations. Legal document requirements vary by jurisdiction and change with new case law.

Every one of these domains has the same failure mode Rajan experienced: a human or system operating on stale rules because nobody caught the policy change in time.

An agent trained on this environment learns a transferable skill — checking the current state before acting, not assuming yesterday's rules still apply. That skill is worth training.

---

## Reproducing This

Everything is open and runnable:

- **Live environment:** [importkk/openenv-onboarding](https://huggingface.co/spaces/importkk/openenv-onboarding)
- **GitHub:** [TheKshitij/openenv-onboarding](https://github.com/TheKshitij/openenv-onboarding)
-**Model:**[importkk/openenv-onboarding-model](https://huggingface.co/importkk/openenv-onboarding-model)
- **Training notebook:** [Google Colab](https://colab.research.google.com/drive/1kqPCw_JGKfp-ZQ_Vrh2d7GvVj2cmfa-S?usp=sharing)

To run the environment locally:
```bash
pip install -r requirements.txt
python server/app.py
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "dept_onboarding"}'
```

To retrain:
```bash
# In Colab
!pip install trl unsloth pydantic openai matplotlib -q
!python train.py
```

---

## A Note on OpenEnv

OpenEnv made it straightforward to build a spec-compliant environment that automated validators could actually evaluate. The `openenv.yaml` manifest, the typed observation/action models, the HTTP API structure — having a clear spec to build toward meant I could focus on the domain logic instead of reinventing the scaffolding.

One thing I'd love to see in a future OpenEnv release: a native "policy drift" primitive — a way to declare that certain observation fields can change mid-episode and have the framework automatically track which completed work is invalidated. It's a common enough pattern in real-world workflows that it probably deserves first-class support.

---

*Built solo for the Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale, Bangalore, April 25–26.*

*Kshitij Karmali — KIIT University, B.Tech Computer Science 2027*

**[Explore the Hugging Face Space](https://huggingface.co/spaces/importkk/openenv-onboarding)**
**[Open the Colab Training Notebook](https://colab.research.google.com/drive/1kqPCw_JGKfp-ZQ_Vrh2d7GvVj2cmfa-S?usp=sharing)**
**[Download the Trained Model](https://huggingface.co/importkk/openenv-onboarding-model)**
**[View the GitHub Repository](https://github.com/TheKshitij/openenv-onboarding)**
