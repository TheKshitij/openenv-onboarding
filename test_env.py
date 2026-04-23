"""
test_env.py - Validation test suite for the Enterprise Onboarding Environment.

Tests 7 critical behaviours:
  1. Basic episode flow (provision -> submit -> complete)
  2. Dependency blocking (cannot provision before dependency is met)
  3. Policy drift fires at the correct step
  4. Drift invalidates policy-sensitive completed systems
  5. Escalation + recovery from FAILED state
  6. Score is always normalised to [0.0, 1.0]
  7. check_policy reward is granted only once per policy version

Run:  python test_env.py
"""

import sys
from onboarding_env import (
    OnboardingEnv,
    OnboardingAction,
    SystemStatus,
)

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


def run_test(name, fn):
    try:
        fn()
        print("  {}  {}".format(PASS, name))
        results.append((name, True, None))
    except AssertionError as e:
        print("  {}  {}".format(FAIL, name))
        print("         -> {}".format(e))
        results.append((name, False, str(e)))
    except Exception as e:
        print("  {}  {}  [EXCEPTION: {}: {}]".format(FAIL, name, type(e).__name__, e))
        results.append((name, False, "{}: {}".format(type(e).__name__, e)))


# -- Helper -------------------------------------------------------------------

def _get_system(obs, sys_id):
    """Return the SystemState with the given id from an observation."""
    for s in obs.systems:
        if s.id == sys_id:
            return s
    raise KeyError("System '{}' not found in observation.".format(sys_id))


# -- Test 1: Basic Episode Flow -----------------------------------------------

def test_basic_flow():
    """
    Verify the core provision -> submit -> complete loop works for all 3
    systems in basic_onboarding. Episode should terminate as done=True
    and completion_pct should be 100%.
    """
    env = OnboardingEnv(task="basic_onboarding", seed=42)
    obs = env.reset()

    assert obs.step == 0, "Expected step=0, got {}".format(obs.step)
    assert obs.total_required == 3, "Expected 3 required systems, got {}".format(obs.total_required)

    # Complete AD Account (no dependencies)
    env.step(OnboardingAction(action="provision ad_account"))
    result = env.step(OnboardingAction(
        action="submit ad_account username={},security_level=basic".format(obs.employee_id)
    ))
    assert _get_system(result.observation, "ad_account").status == SystemStatus.COMPLETE, \
        "ad_account should be COMPLETE after valid submit"

    # Complete HRMS (no dependencies)
    env.step(OnboardingAction(action="provision hrms_registration"))
    result = env.step(OnboardingAction(
        action="submit hrms_registration role={},department={},grade=L3".format(
            obs.employee_role, obs.department)
    ))
    assert _get_system(result.observation, "hrms_registration").status == SystemStatus.COMPLETE, \
        "hrms_registration should be COMPLETE after valid submit"

    # Complete Email (depends on ad_account -- now complete)
    env.step(OnboardingAction(action="provision email_setup"))
    result = env.step(OnboardingAction(
        action="submit email_setup alias=emp.user,quota=50gb"
    ))
    assert _get_system(result.observation, "email_setup").status == SystemStatus.COMPLETE, \
        "email_setup should be COMPLETE after valid submit"

    assert result.observation.completion_pct == 100.0, \
        "Expected 100% completion, got {}".format(result.observation.completion_pct)
    assert result.done is True, "Episode should be done when all required systems complete"


# -- Test 2: Dependency Blocking ----------------------------------------------

def test_dependency_blocking():
    """
    Verify that email_setup cannot be provisioned until ad_account is COMPLETE.
    The environment must return a blocking error and not advance email_setup.
    """
    env = OnboardingEnv(task="basic_onboarding", seed=42)
    env.reset()

    # Attempt to provision email_setup without ad_account being done
    result = env.step(OnboardingAction(action="provision email_setup"))
    obs_after = result.observation

    email_status = _get_system(obs_after, "email_setup").status
    assert email_status == SystemStatus.BLOCKED, \
        "email_setup should stay BLOCKED when ad_account is not complete, got {}".format(email_status)

    assert result.reward < 0, \
        "Expected negative reward for blocked provision, got {}".format(result.reward)


# -- Test 3: Policy Drift Fires at Correct Step -------------------------------

def test_policy_drift_fires():
    """
    Verify that dept_onboarding fires a policy drift at step 15.
    The observation at step 15 must contain a non-empty policy_drift_event.
    """
    env = OnboardingEnv(task="dept_onboarding", seed=42)
    env.reset()

    drift_observed = False
    for i in range(20):
        result = env.step(OnboardingAction(action="hold"))
        if result.observation.policy_drift_event:
            drift_observed = True
            assert result.observation.step == 15, \
                "Drift should fire at step 15, fired at step {}".format(result.observation.step)
            assert "v2" in result.observation.policy_drift_event, \
                "Drift event should mention policy v2 upgrade"
            break

    assert drift_observed, "No policy drift was observed in 20 steps (expected at step 15)"


# -- Test 4: Drift Invalidates Completed Systems ------------------------------

def test_drift_invalidates_systems():
    """
    Verify that ad_account, completed under policy v1 (security_level=basic),
    is set to FAILED after the v1->v2 drift (requires security_level=enhanced).
    """
    env = OnboardingEnv(task="dept_onboarding", seed=42)
    obs = env.reset()

    # Complete ad_account under v1 (steps 1-2)
    env.step(OnboardingAction(action="provision ad_account"))
    env.step(OnboardingAction(
        action="submit ad_account username={},security_level=basic".format(obs.employee_id)
    ))

    # Advance to step 15 (drift fires there); currently at step 2, need 13 more holds
    for _ in range(13):
        env.step(OnboardingAction(action="hold"))

    result = env.step(OnboardingAction(action="hold"))  # step 16 -- drift has now fired

    ad_status = _get_system(result.observation, "ad_account").status
    assert ad_status == SystemStatus.FAILED, \
        "ad_account should be FAILED after v1->v2 drift, got {}".format(ad_status)

    ad_error = _get_system(result.observation, "ad_account").error_msg
    assert "Policy updated" in ad_error or "resubmit" in ad_error.lower(), \
        "ad_account error_msg should indicate policy update, got: '{}'".format(ad_error)


# -- Test 5: Escalation and Recovery ------------------------------------------

def test_escalation_recovery():
    """
    Verify that a FAILED system can be escalated back to IN_PROGRESS
    and then successfully resubmitted with corrected policy values.
    """
    env = OnboardingEnv(task="dept_onboarding", seed=42)
    obs = env.reset()

    # Complete ad_account under v1 (steps 1-2)
    env.step(OnboardingAction(action="provision ad_account"))
    env.step(OnboardingAction(
        action="submit ad_account username={},security_level=basic".format(obs.employee_id)
    ))

    # Advance to drift at step 15 (13 holds from step 2)
    for _ in range(13):
        env.step(OnboardingAction(action="hold"))
    env.step(OnboardingAction(action="hold"))  # step 16 -- drift has fired

    # ad_account is now FAILED -- escalate it
    result = env.step(OnboardingAction(
        action="escalate ad_account policy_drift_security_upgrade"
    ))
    ad_status = _get_system(result.observation, "ad_account").status
    assert ad_status == SystemStatus.IN_PROGRESS, \
        "After escalation ad_account should be IN_PROGRESS, got {}".format(ad_status)

    # Resubmit with corrected v2 value
    result = env.step(OnboardingAction(
        action="submit ad_account username={},security_level=enhanced".format(obs.employee_id)
    ))
    ad_status = _get_system(result.observation, "ad_account").status
    assert ad_status == SystemStatus.COMPLETE, \
        "ad_account should be COMPLETE after correct v2 resubmit, got {}".format(ad_status)


# -- Test 6: Score Normalisation ----------------------------------------------

def test_score_normalisation():
    """
    Verify that the normalised score stays within [0.0, 1.0] under all
    conditions: at reset, mid-episode with violations, and at completion.
    """
    env = OnboardingEnv(task="enterprise_onboarding", seed=7)
    env.reset()

    # Spam bad actions to accumulate violations and drive reward very negative
    for _ in range(15):
        result = env.step(OnboardingAction(
            action="submit ad_account username=emp,security_level=WRONG_VALUE"
        ))
        assert 0.0 <= result.score <= 1.0, \
            "Score out of bounds after violation: {}".format(result.score)

    # Also check at episode start (state() before any steps)
    env2 = OnboardingEnv(task="basic_onboarding", seed=1)
    env2.reset()
    state = env2.state()
    assert 0.0 <= state["score"] <= 1.0, \
        "Score at reset out of bounds: {}".format(state["score"])


# -- Test 7: check_policy Anti-Farming ----------------------------------------

def test_check_policy_anti_farming():
    """
    Verify that check_policy grants +0.05 reward only on the FIRST call
    per policy version. Subsequent calls must return 0.0 for reward_action.
    """
    env = OnboardingEnv(task="basic_onboarding", seed=42)
    env.reset()

    # First call -- should get reward
    result1 = env.step(OnboardingAction(action="check_policy"))
    action_reward_1 = result1.info.get("reward_action", None)
    assert action_reward_1 is not None, "info dict must contain 'reward_action'"
    assert action_reward_1 == 0.05, \
        "First check_policy should grant +0.05, got {}".format(action_reward_1)

    # Second call -- same policy version, should get 0
    result2 = env.step(OnboardingAction(action="check_policy"))
    action_reward_2 = result2.info.get("reward_action")
    assert action_reward_2 == 0.0, \
        "Repeated check_policy should grant 0.0, got {}".format(action_reward_2)

    # Third call -- still no reward
    result3 = env.step(OnboardingAction(action="check_policy"))
    action_reward_3 = result3.info.get("reward_action")
    assert action_reward_3 == 0.0, \
        "Third check_policy call should still grant 0.0, got {}".format(action_reward_3)


# -- Runner -------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  OpenEnv Onboarding - Environment Validation Suite")
    print("="*60 + "\n")

    run_test("Test 1: Basic episode flow (provision->submit->complete)", test_basic_flow)
    run_test("Test 2: Dependency blocking (email before AD)", test_dependency_blocking)
    run_test("Test 3: Policy drift fires at correct step (dept, step 15)", test_policy_drift_fires)
    run_test("Test 4: Drift invalidates completed systems (ad_account->FAILED)", test_drift_invalidates_systems)
    run_test("Test 5: Escalation + recovery from FAILED state", test_escalation_recovery)
    run_test("Test 6: Score always normalised to [0.0, 1.0]", test_score_normalisation)
    run_test("Test 7: check_policy anti-farming (reward only once per version)", test_check_policy_anti_farming)

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print("="*60)
    print("  Result: {}/{} tests passed".format(passed, total))
    if passed == total:
        print("  All tests green -- environment is stable.")
    else:
        print("  Some tests FAILED -- fix before submission.")
    print("="*60 + "\n")

    sys.exit(0 if passed == total else 1)
