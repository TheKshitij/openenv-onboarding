"""
train.py — GRPO fine-tuning for the Enterprise Onboarding Agent
Uses HuggingFace TRL (GRPOTrainer) with a reward function derived from
the OnboardingEnv step rewards.

Run in Colab:
    !pip install trl unsloth pydantic openai -q
    !python train.py

Or with Unsloth (2x faster on T4):
    from unsloth import FastLanguageModel  # uncomment USE_UNSLOTH below
"""

import os, sys, json, textwrap
from typing import List

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen2.5-1.5B-Instruct"   # small enough for free Colab T4
USE_UNSLOTH   = True                             # set True if unsloth is installed
MAX_NEW_TOKENS= 80
BATCH_SIZE    = 4
GRAD_ACCUM    = 2
LR            = 5e-6
NUM_STEPS     = 500                              # ~40 min on T4 with Unsloth; increase for better results
SAVE_STEPS    = 100
TASK          = "dept_onboarding"               # medium task shows clearest improvement
SEED          = 42
OUTPUT_DIR    = "./onboarding-grpo-ckpt"

# ── Imports ───────────────────────────────────────────────────────────────────
import torch
from onboarding_env import OnboardingEnv, OnboardingAction, TASK_IDS

if USE_UNSLOTH:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_ID, max_seq_length=1024, dtype=None, load_in_4bit=True
    )
    # Attach LoRA adapters — required before fine-tuning a quantized model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                          # LoRA rank (16 is a good balance)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    FastLanguageModel.for_training(model)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto",
    )

from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ── System prompt (same as inference.py) ─────────────────────────────────────
SYSTEM = textwrap.dedent("""
    You are an enterprise IT onboarding specialist.
    Your job is to complete an employee's onboarding across all required systems.

    AVAILABLE ACTIONS (output exactly one, no explanation):
      provision <system_id>
      submit <system_id> <field>=<value>,...
      check_policy
      escalate <system_id> <reason>
      verify <system_id>
      hold

    CRITICAL: Always check_policy after a POLICY DRIFT EVENT.
    Submit field values must match CURRENT policy exactly.
    Output ONLY the action string. No markdown. No explanation.
""").strip()


def _make_prompt(obs_dict: dict) -> str:
    """Convert observation dict to a text prompt for the model."""
    p = obs_dict["current_policy"]
    systems = obs_dict["systems"]

    lines = [
        f"Step {obs_dict['step']}/{obs_dict['max_steps']}",
        f"Employee: {obs_dict['employee_id']}  role={obs_dict['employee_role']}",
        f"Progress: {obs_dict['completed_count']}/{obs_dict['total_required']} required",
        "",
        f"POLICY ({p['version']}):",
        f"  security={p['it_security_level']}  device={p['device_type']}  leave={p['leave_policy']}",
        f"  badge_zones={p['badge_zones']}",
        f"  compliance_modules={p['compliance_modules']}",
        "",
    ]
    if obs_dict.get("policy_drift_event"):
        lines += [f"*** POLICY DRIFT: {obs_dict['policy_drift_event'][:100]} ***", ""]

    lines.append("SYSTEMS:")
    for s in systems:
        err = f" ERR:{s['error_msg'][:40]}" if s.get("error_msg") else ""
        lines.append(f"  [{s['id']:<24}] {s['status']:<12} req={s['required']}{err}")

    lines += ["", f"Last result: {obs_dict.get('last_action_result','')[:60]}", "", "Your action:"]
    return "\n".join(lines)


# ── Reward function ───────────────────────────────────────────────────────────

def env_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """
    Rollout reward: run each completion as an action in a fresh env episode
    and return the cumulative normalised score after 10 steps.
    This is what GRPOTrainer uses to compute advantages.
    """
    rewards = []
    completion_pcts = []
    violation_counts = []

    for prompt, completion in zip(prompts, completions):
        action_str = completion.strip().splitlines()[0].strip()

        env = OnboardingEnv(task=TASK, seed=SEED)
        obs = env.reset()

        total = 0.0
        for _ in range(10):
            if env._ep.get("done", False):
                break
            try:
                result = env.step(OnboardingAction(action=action_str))
                total += result.reward
                obs    = result.observation
            except Exception:
                total -= 0.10
                break

        score = env._normalise_score()
        rewards.append(float(score))
        # ── Collect extra scalars for monitoring (Problem 3a) ───────────────────────────
        completion_pcts.append(obs.completion_pct)
        violation_counts.append(obs.episode_violations)

    if completion_pcts:
        avg_cpl = sum(completion_pcts) / len(completion_pcts)
        avg_vio = sum(violation_counts) / len(violation_counts)
        print(f"  [metrics] avg_score={sum(rewards)/len(rewards):.3f}  "
              f"avg_completion_pct={avg_cpl:.1f}%  avg_violations={avg_vio:.2f}")

    return rewards


# ── Dataset: generate rollout prompts from random env states ──────────────────

def build_dataset(n: int = 500) -> Dataset:
    """
    Generate N onboarding observation prompts split across three episode phases:
      - Bucket A (early, steps 0-10):        model learns the basic provision→submit loop
      - Bucket B (post-drift-1, steps 15-22): model sees policy v2 and must recover
      - Bucket C (post-drift-2, steps 30-38): only for enterprise task; model sees v3 drift
    This ensures the model is trained on drift-recovery scenarios, not just easy starts.
    """
    import random
    rng = random.Random(SEED)
    prompts = []

    is_enterprise = TASK == "enterprise_onboarding"
    # Split: 200 early / 200 post-drift-1 / 100 post-drift-2 (enterprise only)
    bucket_a = n // 2
    bucket_b = n // 2 if not is_enterprise else n * 2 // 5
    bucket_c = 0     if not is_enterprise else n - bucket_a - bucket_b

    # ── Shared warm-up actions (covers basic systems before any drift) ─────────
    def _warm_up_actions(obs):
        return [
            "check_policy",
            f"provision ad_account",
            f"submit ad_account username={obs.employee_id},security_level={obs.current_policy.it_security_level}",
            "provision hrms_registration",
            f"submit hrms_registration role={obs.employee_role},department={obs.department},grade=L3",
            "provision email_setup",
            "submit email_setup alias=emp.user,quota=50gb",
        ]

    def _advance(env, obs, actions, n_steps):
        """Step the env through the first n_steps of a given action list."""
        for j in range(min(n_steps, len(actions))):
            if env._ep.get("done", False):
                break
            try:
                result = env.step(OnboardingAction(action=actions[j]))
                obs = result.observation
            except Exception:
                break
        return obs

    def _make_entry(obs):
        return {
            "prompt": (
                f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                f"<|im_start|>user\n{_make_prompt(obs.model_dump())}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "task": TASK,
        }

    # ── Bucket A: early episode (steps 0-10) ──────────────────────────────────
    for _ in range(bucket_a):
        env = OnboardingEnv(task=TASK, seed=rng.randint(0, 99999))
        obs = env.reset()
        steps = rng.randint(0, 10)
        obs = _advance(env, obs, _warm_up_actions(obs), steps)
        prompts.append(_make_entry(obs))

    # ── Bucket B: post-drift-1 (advance past drift step) ─────────────────────
    drift1 = {"basic_onboarding": 0, "dept_onboarding": 15, "enterprise_onboarding": 10}[TASK]
    if drift1 > 0:
        for _ in range(bucket_b):
            env = OnboardingEnv(task=TASK, seed=rng.randint(0, 99999))
            obs = env.reset()
            # Complete warm-up systems, then hold to trigger drift
            obs = _advance(env, obs, _warm_up_actions(obs), len(_warm_up_actions(obs)))
            while env._ep.get("step", 0) < drift1 + rng.randint(1, 7):
                if env._ep.get("done", False):
                    break
                result = env.step(OnboardingAction(action="hold"))
                obs = result.observation
            prompts.append(_make_entry(obs))

    # ── Bucket C: post-drift-2 (enterprise only, advance past step 30) ────────
    drift2 = 30
    for _ in range(bucket_c):
        env = OnboardingEnv(task=TASK, seed=rng.randint(0, 99999))
        obs = env.reset()
        obs = _advance(env, obs, _warm_up_actions(obs), len(_warm_up_actions(obs)))
        while env._ep.get("step", 0) < drift2 + rng.randint(1, 5):
            if env._ep.get("done", False):
                break
            result = env.step(OnboardingAction(action="hold"))
            obs = result.observation
        prompts.append(_make_entry(obs))

    return Dataset.from_list(prompts)



# ── Training ──────────────────────────────────────────────────────────────────

def main():
    print(f"Building dataset ({TASK})...")
    dataset = build_dataset(500)
    print(f"Dataset size: {len(dataset)}")

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_steps=SAVE_STEPS,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        seed=SEED,
        report_to="none",
        max_completion_length=MAX_NEW_TOKENS,  # renamed from max_new_tokens in newer TRL
        num_generations=4,  # GRPO samples 4 completions per prompt
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[env_reward],
        args=training_args,
        train_dataset=dataset,
    )

    print(f"Training {MODEL_ID} on {TASK} for {NUM_STEPS} steps...")
    print("Watch the 'reward' column — it should climb from ~0.5 to ~0.75+ after 100 steps.")
    trainer.train()

    print("\n=== REWARD IMPROVEMENT SUMMARY ===")
    for log in trainer.state.log_history:
        if 'reward' in log:
            print(f"  step={log.get('step',0):3d}  reward={log['reward']:.4f}")

    # ── Auto-generate reward plot (Problem 3b) ───────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless — works in Colab and containers
        import matplotlib.pyplot as plt

        steps, rewards_log = [], []
        for log in trainer.state.log_history:
            if "reward" in log and "step" in log:
                steps.append(log["step"])
                rewards_log.append(log["reward"])

        if steps:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(steps, rewards_log, color="#7c3aed", linewidth=2, label="Avg Reward")
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Reward (normalised score)", color="#7c3aed")
            ax1.tick_params(axis="y", labelcolor="#7c3aed")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            plt.title(f"GRPO Training Progress — {TASK} ({MODEL_ID})")
            fig.tight_layout()
            plt.savefig("reward_improvement.png", dpi=150)
            plt.close(fig)
            print("\n📈 Reward plot saved → reward_improvement.png  (commit this to your repo!)")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print(f"\nSaving to {OUTPUT_DIR}")
    if USE_UNSLOTH:
        model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
    else:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete.")



if __name__ == "__main__":
    main()
