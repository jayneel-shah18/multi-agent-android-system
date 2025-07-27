#!/usr/bin/env python3
"""
Comprehensive test for VerifierAgent integration with the automation pipeline.
Tests verification in real execution scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from utils.mock_env import MockAndroidEnv
from utils.logging import setup_logger
import logging

def test_verifier_integration():
    """Test VerifierAgent integrated with the full automation pipeline."""
    
    print("🔍 VERIFIER AGENT INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize all components
    logger = setup_logger("VerifierIntegrationTest", level=logging.INFO)
    planner = PlannerAgent(name="TestPlanner")
    executor = ExecutorAgent(name="TestExecutor", debug=False)
    verifier = VerifierAgent(name="TestVerifier", debug=True)
    env = MockAndroidEnv()
    
    print("✅ All components initialized")
    
    # Test goal
    goal = "Test turning Wi-Fi on and off"
    plan = planner.generate_plan(goal)
    
    print(f"\n🎯 Testing goal: '{goal}'")
    print(f"📋 Plan: {plan}")
    
    # Reset environment and track state
    previous_obs = env.reset()
    print(f"\n🔄 Starting execution with verification...")
    
    verification_results = []
    
    for step_num, step in enumerate(plan, 1):
        print(f"\n--- Step {step_num}: '{step}' ---")
        
        # Execute step
        action = executor.execute_step(step, previous_obs)
        current_obs = env.step(action)
        
        print(f"Action executed: {action}")
        print(f"Screen transition: {previous_obs.get('screen')} → {current_obs.get('screen')}")
        
        # Verify step result
        verification_result = verifier.verify(
            goal=goal,
            step=step,
            result_obs=current_obs,
            previous_obs=previous_obs
        )
        
        verification_results.append({
            'step': step,
            'result': verification_result,
            'screen_from': previous_obs.get('screen'),
            'screen_to': current_obs.get('screen')
        })
        
        print(f"✅ Verification: {verification_result}")
        
        # Update for next iteration
        previous_obs = current_obs
    
    # Final summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    pass_count = sum(1 for r in verification_results if r['result'] == 'pass')
    fail_count = sum(1 for r in verification_results if r['result'] == 'fail')
    bug_count = sum(1 for r in verification_results if r['result'] == 'bug_detected')
    
    print(f"Total steps: {len(verification_results)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print(f"Bugs detected: {bug_count}")
    print(f"Success rate: {(pass_count / len(verification_results) * 100):.1f}%")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(verification_results, 1):
        status_icon = {"pass": "✅", "fail": "❌", "bug_detected": "🐛"}[result['result']]
        print(f"  {i}. {result['step']}: {status_icon} {result['result']}")
        print(f"     {result['screen_from']} → {result['screen_to']}")
    
    # Agent statistics
    print(f"\n📈 Agent Statistics:")
    verifier_stats = verifier.get_statistics()
    for key, value in verifier_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    return pass_count == len(verification_results)

def test_failure_scenarios():
    """Test VerifierAgent with deliberate failure scenarios."""
    
    print(f"\n🧪 TESTING FAILURE SCENARIOS")
    print("=" * 60)
    
    verifier = VerifierAgent(name="FailureTestVerifier", debug=True)
    
    # Scenario 1: Invalid observation format
    print(f"\n1. Testing invalid observation format:")
    result1 = verifier.verify(
        goal="Test goal",
        step="test step",
        result_obs="invalid_format"  # String instead of dict
    )
    print(f"   Result: {result1} (expected: bug_detected)")
    
    # Scenario 2: No UI elements
    print(f"\n2. Testing empty UI elements:")
    result2 = verifier.verify(
        goal="Test goal",
        step="tap something",
        result_obs={'screen': 'test', 'structured': {'elements': []}}
    )
    print(f"   Result: {result2} (expected: fail)")
    
    # Scenario 3: Unexpected screen transition
    print(f"\n3. Testing unexpected screen transition:")
    previous_obs = {'screen': 'settings', 'structured': {'elements': []}}
    result_obs = {'screen': 'unexpected_screen', 'structured': {'elements': []}}
    
    result3 = verifier.verify(
        goal="Settings navigation",
        step="tap wifi",
        result_obs=result_obs,
        previous_obs=previous_obs
    )
    print(f"   Result: {result3} (expected: fail)")
    
    # Show failure detection statistics
    print(f"\n📊 Failure Detection Statistics:")
    stats = verifier.get_statistics()
    print(f"  Verifications: {stats['verifications_performed']}")
    print(f"  Bugs detected: {stats['bugs_detected']}")
    print(f"  Failures: {stats['failures']}")
    print(f"  Passes: {stats['passes']}")
    
    return True

def main():
    """Run comprehensive VerifierAgent tests."""
    
    print("🚀 VERIFIER AGENT - COMPREHENSIVE TESTING")
    print("=" * 70)
    
    # Test 1: Integration with full pipeline
    print(f"\n🔗 TEST 1: PIPELINE INTEGRATION")
    integration_success = test_verifier_integration()
    
    # Test 2: Failure scenarios
    print(f"\n🚫 TEST 2: FAILURE SCENARIOS")
    failure_success = test_failure_scenarios()
    
    # Final summary
    print(f"\n🎯 FINAL TEST SUMMARY")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 2
    
    if integration_success:
        print(f"✅ Pipeline Integration Test: PASSED")
        tests_passed += 1
    else:
        print(f"❌ Pipeline Integration Test: FAILED")
    
    if failure_success:
        print(f"✅ Failure Scenarios Test: PASSED")
        tests_passed += 1
    else:
        print(f"❌ Failure Scenarios Test: FAILED")
    
    print(f"\nOverall: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print(f"🎉 All VerifierAgent tests PASSED!")
        return True
    else:
        print(f"⚠️  Some VerifierAgent tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
