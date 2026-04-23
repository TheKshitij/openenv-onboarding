"""
onboarding_env.py — Enterprise Employee Onboarding Agent
Core environment: typed models, system simulation, policy drift, reward logic.
Inspired by the real onboarding pipeline at large Indian IT firms (Infosys, TCS, Wipro).
"""

import random
import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class SystemStatus(str, Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE   = "complete"
    BLOCKED    = "blocked"   # dependency not met
    FAILED     = "failed"    # policy violation or wrong data


class PolicyVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


# ─── Observation Models ───────────────────────────────────────────────────────

class SystemState(BaseModel):
    id:           str
    name:         str
    status:       SystemStatus
    required:     bool
    dependencies: List[str] = Field(default_factory=list,
        description="System IDs that must be COMPLETE before this can start")
    attempts:     int = 0
    error_msg:    str = ""


class PolicyState(BaseModel):
    version:         PolicyVersion
    it_security_level: str  = Field(..., description="basic | enhanced | strict")
    leave_policy:    str    = Field(..., description="standard | enhanced | flexi")
    device_type:     str    = Field(..., description="laptop | workstation | remote-kit")
    badge_zones:     List[str] = Field(default_factory=list)
    compliance_modules: List[str] = Field(default_factory=list)


class OnboardingObservation(BaseModel):
    step:               int
    max_steps:          int
    employee_id:        str
    employee_role:      str
    department:         str
    systems:            List[SystemState]
    current_policy:     PolicyState
    policy_drift_event: str  = Field("", description="Description of last policy change, empty if none")
    completed_count:    int
    total_required:     int
    completion_pct:     float
    blocked_systems:    List[str] = Field(default_factory=list)
    failed_systems:     List[str] = Field(default_factory=list)
    last_action_result: str = ""
    episode_violations: int = 0


class OnboardingAction(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One of:\n"
            "  provision <system_id>                     — start provisioning a system\n"
            "  submit <system_id> <field>=<value>,...    — submit required data to complete a system\n"
            "  check_policy                              — read the current policy version\n"
            "  escalate <system_id> <reason>             — escalate a blocked/failed system\n"
            "  verify <system_id>                        — verify a completed system is correct\n"
            "  hold                                      — do nothing this step\n"
            "\nExamples:\n"
            "  provision ad_account\n"
            "  submit hrms_registration role=engineer,department=engineering,grade=L3\n"
            "  submit email_setup alias=john.doe,quota=50gb\n"
            "  check_policy\n"
            "  escalate badge_access zone_mismatch_with_current_policy\n"
            "  verify payroll_enrollment"
        )
    )


class StepResult(BaseModel):
    observation: OnboardingObservation
    reward:      float
    done:        bool
    score:       float
    info:        Dict[str, Any] = Field(default_factory=dict)


# ─── Policy Definitions ───────────────────────────────────────────────────────

_POLICIES: Dict[PolicyVersion, PolicyState] = {
    PolicyVersion.V1: PolicyState(
        version=PolicyVersion.V1,
        it_security_level="basic",
        leave_policy="standard",
        device_type="laptop",
        badge_zones=["main_lobby", "floor_3"],
        compliance_modules=["code_of_conduct", "data_privacy"],
    ),
    PolicyVersion.V2: PolicyState(
        version=PolicyVersion.V2,
        it_security_level="enhanced",
        leave_policy="enhanced",
        device_type="laptop",
        badge_zones=["main_lobby", "floor_3", "server_room"],
        compliance_modules=["code_of_conduct", "data_privacy", "information_security"],
    ),
    PolicyVersion.V3: PolicyState(
        version=PolicyVersion.V3,
        it_security_level="strict",
        leave_policy="flexi",
        device_type="remote-kit",
        badge_zones=["main_lobby"],
        compliance_modules=["code_of_conduct", "data_privacy", "information_security", "ai_ethics"],
    ),
}

# ─── System Definitions per task ─────────────────────────────────────────────

def _systems_for_task(task: str) -> List[Dict]:
    """Return system definitions for a task."""

    base = [
        {
            "id": "ad_account",
            "name": "Active Directory Account",
            "required": True,
            "dependencies": [],
            "valid_fields": {"username", "security_level"},
        },
        {
            "id": "email_setup",
            "name": "Corporate Email Setup",
            "required": True,
            "dependencies": ["ad_account"],
            "valid_fields": {"alias", "quota"},
        },
        {
            "id": "hrms_registration",
            "name": "HRMS Registration",
            "required": True,
            "dependencies": [],
            "valid_fields": {"role", "department", "grade"},
        },
    ]

    medium_extra = [
        {
            "id": "payroll_enrollment",
            "name": "Payroll Enrollment",
            "required": True,
            "dependencies": ["hrms_registration"],
            "valid_fields": {"bank_account", "pan_number", "leave_policy"},
        },
        {
            "id": "badge_access",
            "name": "Badge Access Provisioning",
            "required": True,
            "dependencies": ["ad_account"],
            "valid_fields": {"zones", "access_level"},
        },
        {
            "id": "device_allocation",
            "name": "Device Allocation",
            "required": True,
            "dependencies": ["ad_account"],
            "valid_fields": {"device_type", "asset_tag"},
        },
    ]

    hard_extra = [
        {
            "id": "vpn_setup",
            "name": "VPN & Remote Access Setup",
            "required": True,
            "dependencies": ["ad_account", "device_allocation"],
            "valid_fields": {"profile", "mfa_method"},
        },
        {
            "id": "compliance_training",
            "name": "Compliance Training Enrollment",
            "required": True,
            "dependencies": ["email_setup"],
            "valid_fields": {"modules", "deadline_days"},
        },
        {
            "id": "health_insurance",
            "name": "Health Insurance Registration",
            "required": True,
            "dependencies": ["hrms_registration"],
            "valid_fields": {"plan", "dependents"},
        },
        {
            "id": "it_security_training",
            "name": "IT Security Training",
            "required": True,
            "dependencies": ["compliance_training"],
            "valid_fields": {"level", "certification"},
        },
        {
            "id": "project_assignment",
            "name": "Initial Project Assignment",
            "required": False,
            "dependencies": ["hrms_registration", "ad_account"],
            "valid_fields": {"project_code", "manager_id"},
        },
        {
            "id": "mentor_assignment",
            "name": "Mentor Assignment",
            "required": False,
            "dependencies": ["hrms_registration"],
            "valid_fields": {"mentor_id", "meeting_cadence"},
        },
    ]

    if task == "basic_onboarding":
        return base
    elif task == "dept_onboarding":
        return base + medium_extra
    else:
        return base + medium_extra + hard_extra


# ─── Task Topology ────────────────────────────────────────────────────────────

_TASK_CONFIG: Dict[str, Dict] = {
    "basic_onboarding": {
        "description": "3-system onboarding for a junior hire — no policy drift — easy",
        "max_steps":   20,
        "drift_steps": [],          # no drift
        "start_policy": PolicyVersion.V1,
        "employee_roles": ["software_engineer_L1", "analyst_L1"],
        "departments":    ["engineering", "operations"],
    },
    "dept_onboarding": {
        "description": "6-system departmental onboarding with one mid-episode policy drift — medium",
        "max_steps":   35,
        "drift_steps": [15],        # policy changes at step 15
        "start_policy": PolicyVersion.V1,
        "employee_roles": ["software_engineer_L2", "senior_analyst", "team_lead"],
        "departments":    ["engineering", "finance", "hr"],
    },
    "enterprise_onboarding": {
        "description": "12-system full enterprise onboarding with two policy drifts and system failures — hard",
        "max_steps":   55,
        "drift_steps": [10, 30],    # policies change twice
        "start_policy": PolicyVersion.V1,
        "employee_roles": ["senior_engineer_L4", "principal_engineer", "engineering_manager"],
        "departments":    ["engineering", "security", "platform"],
    },
}

TASK_IDS = list(_TASK_CONFIG.keys())

_DRIFT_SEQUENCE = [PolicyVersion.V1, PolicyVersion.V2, PolicyVersion.V3]
_DRIFT_MESSAGES = {
    PolicyVersion.V2: (
        "POLICY UPDATE v1→v2: IT security level upgraded to 'enhanced'. "
        "Badge zones now include 'server_room'. "
        "Payroll must use leave_policy='enhanced'. "
        "Compliance training now requires 'information_security' module."
    ),
    PolicyVersion.V3: (
        "POLICY UPDATE v2→v3: IT security upgraded to 'strict'. "
        "Device type changed to 'remote-kit'. "
        "Badge access restricted to 'main_lobby' only. "
        "New compliance module 'ai_ethics' now mandatory. "
        "Leave policy changed to 'flexi'."
    ),
}


# ─── Validation Logic ─────────────────────────────────────────────────────────

def _validate_submission(
    sys_id: str,
    fields: Dict[str, str],
    policy: PolicyState,
    systems_def: List[Dict],
) -> Tuple[bool, str]:
    """
    Validate submitted fields against current policy.
    Returns (is_valid, error_message).
    """
    if sys_id == "ad_account":
        sec = fields.get("security_level", "")
        if sec != policy.it_security_level:
            return False, (
                f"security_level='{sec}' does not match current policy "
                f"'{policy.it_security_level}'. Update and resubmit."
            )

    elif sys_id == "badge_access":
        zones_str = fields.get("zones", "")
        submitted_zones = set(z.strip() for z in zones_str.split(",") if z.strip())
        allowed = set(policy.badge_zones)
        invalid = submitted_zones - allowed
        if invalid:
            return False, (
                f"Zone(s) {invalid} not permitted under current policy. "
                f"Allowed zones: {policy.badge_zones}"
            )
        if not submitted_zones:
            return False, "zones field is required and cannot be empty."

    elif sys_id == "device_allocation":
        dt = fields.get("device_type", "")
        if dt != policy.device_type:
            return False, (
                f"device_type='{dt}' does not match policy "
                f"'{policy.device_type}'."
            )

    elif sys_id == "payroll_enrollment":
        lp = fields.get("leave_policy", "")
        if lp != policy.leave_policy:
            return False, (
                f"leave_policy='{lp}' does not match current policy "
                f"'{policy.leave_policy}'."
            )

    elif sys_id == "compliance_training":
        mods_str = fields.get("modules", "")
        submitted = set(m.strip() for m in mods_str.split(",") if m.strip())
        required  = set(policy.compliance_modules)
        missing   = required - submitted
        if missing:
            return False, (
                f"Missing required compliance modules: {missing}. "
                f"Current policy requires: {policy.compliance_modules}"
            )

    elif sys_id == "it_security_training":
        level = fields.get("level", "")
        if level != policy.it_security_level:
            return False, (
                f"Training level='{level}' must match policy "
                f"security level '{policy.it_security_level}'."
            )

    # Check all required fields are present
    sys_def = next((s for s in systems_def if s["id"] == sys_id), None)
    if sys_def:
        for f in sys_def["valid_fields"]:
            if f not in fields or not fields[f]:
                return False, f"Required field '{f}' is missing or empty."

    return True, ""


# ─── Environment ─────────────────────────────────────────────────────────────

class OnboardingEnv:
    """
    OpenEnv environment for Enterprise Employee Onboarding.
    Implements reset() / step() / state() with typed Pydantic models.
    """

    def __init__(self, task: str = "basic_onboarding", seed: Optional[int] = None):
        if task not in _TASK_CONFIG:
            raise ValueError(f"Unknown task '{task}'. Choose from {TASK_IDS}")
        self.task   = task
        self._cfg   = _TASK_CONFIG[task]
        self._rng   = random.Random(seed)
        self._ep: Dict[str, Any] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self) -> OnboardingObservation:
        cfg = self._cfg
        sys_defs = _systems_for_task(self.task)
        systems  = []
        for sd in sys_defs:
            systems.append({
                "id":           sd["id"],
                "name":         sd["name"],
                "required":     sd["required"],
                "dependencies": list(sd["dependencies"]),
                "valid_fields": sd["valid_fields"],
                "status":       SystemStatus.PENDING,
                "attempts":     0,
                "error_msg":    "",
            })

        policy_ver = cfg["start_policy"]
        self._ep = {
            "task":              self.task,
            "step":              0,
            "max_steps":         cfg["max_steps"],
            "employee_id":       f"EMP{self._rng.randint(10000, 99999)}",
            "employee_role":     self._rng.choice(cfg["employee_roles"]),
            "department":        self._rng.choice(cfg["departments"]),
            "systems":           systems,
            "policy_version":    policy_ver,
            "policy":            copy.deepcopy(_POLICIES[policy_ver]),
            "drift_steps":       list(cfg["drift_steps"]),
            "drift_index":       0,
            "policy_drift_event":"",
            "last_checked_policy": "",   # prevents check_policy reward farming
            "total_reward":      0.0,
            "violations":        0,
            "done":              False,
            "last_result":       "Onboarding initiated. Review systems and current policy.",
        }
        return self._make_obs()

    def step(self, action: OnboardingAction) -> StepResult:
        if not self._ep:
            raise RuntimeError("Call reset() before step()")
        if self._ep["done"]:
            raise RuntimeError("Episode complete. Call reset()")

        ep = self._ep
        ep["step"] += 1
        ep["policy_drift_event"] = ""

        # 1. Apply action
        action_reward, msg = self._apply_action(action.action)
        ep["last_result"] = msg

        # 2. Policy drift check
        drift_reward = self._check_drift()

        # 3. Decomposed step reward (Problem 2)
        r_progress   = self._reward_progress()
        r_violations = self._reward_violations()
        r_completion = self._reward_completion()
        step_reward  = round(
            r_progress + r_violations + r_completion + action_reward + drift_reward, 4
        )
        ep["total_reward"] += step_reward

        # 4. Termination
        done = (ep["step"] >= ep["max_steps"]) or self._all_required_complete()
        ep["done"] = done

        obs   = self._make_obs()
        score = self._normalise_score()

        return StepResult(
            observation=obs,
            reward=round(step_reward, 4),
            done=done,
            score=round(score, 4),
            info={
                "step_reward":      round(step_reward, 4),
                "total_reward":     round(ep["total_reward"], 4),
                "violations":       ep["violations"],
                "score":            round(score, 4),
                "action_result":    msg,
                "drift_event":      ep["policy_drift_event"],
                # ── Decomposed reward breakdown (Problem 2b) ──
                "reward_progress":  round(r_progress, 4),
                "reward_violation": round(r_violations, 4),
                "reward_completion":round(r_completion, 4),
                "reward_action":    round(action_reward, 4),
                "reward_drift":     round(drift_reward, 4),
            },
        )

    def state(self) -> Dict[str, Any]:
        if not self._ep:
            return {"error": "No active episode. Call reset() first."}
        return {
            "task":          self._ep["task"],
            "step":          self._ep["step"],
            "max_steps":     self._ep["max_steps"],
            "total_reward":  round(self._ep["total_reward"], 4),
            "violations":    self._ep["violations"],
            "score":         round(self._normalise_score(), 4),
            "done":          self._ep["done"],
            "observation":   self._make_obs().model_dump(),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_obs(self) -> OnboardingObservation:
        ep = self._ep
        systems_obs = []
        blocked, failed = [], []
        completed = 0
        total_req = 0

        for st in ep["systems"]:
            if st["required"]:
                total_req += 1
                if st["status"] == SystemStatus.COMPLETE:
                    completed += 1

            # Recompute blocked status — unblock when dependencies complete
            if st["status"] in (SystemStatus.PENDING, SystemStatus.BLOCKED):
                deps_done = all(
                    any(s["id"] == d and s["status"] == SystemStatus.COMPLETE
                        for s in ep["systems"])
                    for d in st["dependencies"]
                )
                if not deps_done and st["dependencies"]:
                    st["status"] = SystemStatus.BLOCKED
                else:
                    if st["status"] == SystemStatus.BLOCKED:
                        st["status"] = SystemStatus.PENDING

            if st["status"] == SystemStatus.BLOCKED:
                blocked.append(st["id"])
            if st["status"] == SystemStatus.FAILED:
                failed.append(st["id"])

            systems_obs.append(SystemState(
                id=st["id"],
                name=st["name"],
                status=st["status"],
                required=st["required"],
                dependencies=st["dependencies"],
                attempts=st["attempts"],
                error_msg=st["error_msg"],
            ))

        pct = round(completed / total_req * 100, 1) if total_req > 0 else 0.0

        return OnboardingObservation(
            step=ep["step"],
            max_steps=ep["max_steps"],
            employee_id=ep["employee_id"],
            employee_role=ep["employee_role"],
            department=ep["department"],
            systems=systems_obs,
            current_policy=ep["policy"],
            policy_drift_event=ep["policy_drift_event"],
            completed_count=completed,
            total_required=total_req,
            completion_pct=pct,
            blocked_systems=blocked,
            failed_systems=failed,
            last_action_result=ep["last_result"],
            episode_violations=ep["violations"],
        )

    def _apply_action(self, action_str: str) -> Tuple[float, str]:
        # ── Format compliance check (Problem 1b) ─────────────────────────────
        # Penalise the model for outputting markdown, backticks, or multi-line
        # explanations instead of a clean command string.
        _raw = action_str.strip()
        _forbidden = ('```', '**', '##', '\n\n', '* ', '> ')
        if any(tok in _raw for tok in _forbidden):
            return -0.10, "Invalid format: output a single plain-text command only."

        parts   = _raw.split(None, 2)
        if not parts or parts[0] == "hold":
            return 0.0, "hold — no action taken."

        cmd = parts[0].lower()
        ep  = self._ep
        sys_by_id = {s["id"]: s for s in ep["systems"]}

        try:
            # ── provision ────────────────────────────────────────────────────
            if cmd == "provision":
                if len(parts) < 2:
                    return -0.05, "Error: provision requires <system_id>"
                sid = parts[1]
                if sid not in sys_by_id:
                    return -0.08, f"Error: unknown system '{sid}'."
                st = sys_by_id[sid]
                if st["status"] == SystemStatus.COMPLETE:
                    return -0.03, f"{sid} is already complete."
                if st["status"] == SystemStatus.BLOCKED:
                    deps = [d for d in st["dependencies"]
                            if sys_by_id.get(d, {}).get("status") != SystemStatus.COMPLETE]
                    return -0.05, f"{sid} is blocked. Complete first: {deps}"
                st["status"]   = SystemStatus.IN_PROGRESS
                st["attempts"] += 1
                return 0.05, f"Provisioning started for {st['name']}. Now submit required fields."

            # ── submit ───────────────────────────────────────────────────────
            elif cmd == "submit":
                if len(parts) < 3:
                    return -0.05, "Error: submit requires <system_id> <field=value,...>"
                sid        = parts[1]
                fields_str = parts[2]
                if sid not in sys_by_id:
                    return -0.08, f"Error: unknown system '{sid}'."
                st = sys_by_id[sid]
                if st["status"] not in (SystemStatus.IN_PROGRESS, SystemStatus.FAILED):
                    return -0.05, f"Must provision {sid} before submitting."

                # Parse fields
                fields: Dict[str, str] = {}
                for pair in fields_str.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        fields[k.strip()] = v.strip()

                valid, err = _validate_submission(
                    sid, fields, ep["policy"],
                    _systems_for_task(self.task)
                )
                if valid:
                    st["status"]    = SystemStatus.COMPLETE
                    st["error_msg"] = ""
                    return 0.20, f"{st['name']} completed successfully."
                else:
                    st["status"]    = SystemStatus.FAILED
                    st["error_msg"] = err
                    ep["violations"] += 1
                    return -0.15, f"Submission failed for {st['name']}: {err}"

            # ── check_policy ─────────────────────────────────────────────────
            elif cmd == "check_policy":
                p = ep["policy"]
                policy_msg = (
                    f"Current policy {p.version.value}: "
                    f"it_security={p.it_security_level}, "
                    f"device={p.device_type}, "
                    f"leave={p.leave_policy}, "
                    f"badge_zones={p.badge_zones}, "
                    f"compliance_modules={p.compliance_modules}"
                )
                # Only reward the FIRST check per policy version.
                # Repeated checks after already knowing the policy get 0.
                if ep["last_checked_policy"] != p.version.value:
                    ep["last_checked_policy"] = p.version.value
                    return 0.05, policy_msg   # raised from 0.02 — first-check is meaningful
                return 0.0, policy_msg        # already checked this version — no reward

            # ── escalate ────────────────────────────────────────────────────
            elif cmd == "escalate":
                if len(parts) < 3:
                    return -0.03, "Error: escalate requires <system_id> <reason>"
                sid    = parts[1]
                reason = parts[2]
                if sid not in sys_by_id:
                    return -0.05, f"Error: unknown system '{sid}'."
                st = sys_by_id[sid]
                if st["status"] == SystemStatus.FAILED:
                    # Escalation resets a failed system to pending for retry
                    st["status"]    = SystemStatus.IN_PROGRESS
                    st["error_msg"] = ""
                    return 0.03, f"Escalated {sid}. Issue logged: '{reason}'. System reset for retry."
                return -0.02, f"{sid} is not in FAILED state. Escalation not needed."

            # ── verify ───────────────────────────────────────────────────────
            elif cmd == "verify":
                if len(parts) < 2:
                    return -0.03, "Error: verify requires <system_id>"
                sid = parts[1]
                if sid not in sys_by_id:
                    return -0.05, f"Error: unknown system '{sid}'."
                st = sys_by_id[sid]
                if st["status"] == SystemStatus.COMPLETE:
                    return 0.04, f"Verified: {st['name']} is complete and consistent with current policy."
                return 0.0, f"{sid} is not yet complete (status: {st['status'].value})."

        except Exception as exc:
            return -0.05, f"Parse error: {exc}"

        return -0.05, f"Unknown command '{cmd}'."

    def _check_drift(self) -> float:
        """Trigger policy drift at configured steps."""
        ep = self._ep
        drift_steps = ep["drift_steps"]
        idx         = ep["drift_index"]
        if idx < len(drift_steps) and ep["step"] == drift_steps[idx]:
            # Advance policy version
            next_ver_idx = _DRIFT_SEQUENCE.index(ep["policy_version"]) + 1
            if next_ver_idx < len(_DRIFT_SEQUENCE):
                new_ver = _DRIFT_SEQUENCE[next_ver_idx]
                ep["policy_version"]    = new_ver
                ep["policy"]            = copy.deepcopy(_POLICIES[new_ver])
                ep["policy_drift_event"] = _DRIFT_MESSAGES[new_ver]
                ep["drift_index"]       += 1

                # Invalidate completed systems that now violate new policy
                invalidated = []
                for st in ep["systems"]:
                    if st["status"] == SystemStatus.COMPLETE:
                        valid, _ = _validate_submission(
                            st["id"], {},   # empty fields — just check policy fit
                            ep["policy"],
                            _systems_for_task(self.task)
                        )
                        # Only invalidate systems that have policy-sensitive fields
                        if st["id"] in ("ad_account", "badge_access", "device_allocation",
                                        "payroll_enrollment", "compliance_training",
                                        "it_security_training"):
                            st["status"]    = SystemStatus.FAILED
                            st["error_msg"] = "Policy updated — resubmit with new values."
                            invalidated.append(st["id"])

                if invalidated:
                    ep["policy_drift_event"] += (
                        f" INVALIDATED systems (must resubmit): {invalidated}"
                    )
                return -0.05  # small drift penalty to signal disruption
        return 0.0

    # ── Decomposed reward components (Problem 2a) ───────────────────────────

    def _reward_progress(self) -> float:
        """Continuous progress bonus: +0.10 × (completed / total_required)."""
        ep = self._ep
        total_req = sum(1 for s in ep["systems"] if s["required"])
        completed = sum(
            1 for s in ep["systems"]
            if s["required"] and s["status"] == SystemStatus.COMPLETE
        )
        if total_req == 0:
            return 0.0
        return round((completed / total_req) * 0.10, 4)

    def _reward_violations(self) -> float:
        """Penalty for failed systems and cumulative violations."""
        ep = self._ep
        failed = sum(
            1 for s in ep["systems"]
            if s["required"] and s["status"] == SystemStatus.FAILED
        )
        return round(-(failed * 0.04) - (ep["violations"] * 0.02), 4)

    def _reward_completion(self) -> float:
        """One-time +1.0 bonus when every required system is complete."""
        return 1.0 if self._all_required_complete() else 0.0

    def _step_reward(self, action_delta: float) -> float:
        """Legacy wrapper — kept for backward compatibility with external callers."""
        return round(
            self._reward_progress()
            + self._reward_violations()
            + self._reward_completion()
            + action_delta,
            4,
        )

    def _all_required_complete(self) -> bool:
        return all(
            st["status"] == SystemStatus.COMPLETE
            for st in self._ep["systems"]
            if st["required"]
        )

    def _normalise_score(self) -> float:
        ep   = self._ep
        n    = sum(1 for s in ep["systems"] if s["required"])
        msteps = ep["max_steps"]
        # Max: full progress bonus + completion bonus every step possibility
        max_r = msteps * (0.10 + 0.02) + 1.0
        min_r = msteps * (-0.04 * n + -0.02 * 3)
        raw   = ep["total_reward"]
        score = (raw - min_r) / (max_r - min_r)
        return round(max(0.0, min(1.0, score)), 4)
