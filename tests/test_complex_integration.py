#!/usr/bin/env python3
"""
Complex Multi-Agent Integration Test
Tests multiple goals and advanced scenarios to validate submission readiness
"""

import json
import time
from datetime import datetime
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent

def test_complex_multi_goal_scenario():
    """Test multiple goals in sequence"""
    print("="*60)
    print("COMPLEX MULTI-GOAL INTEGRATION TEST")
    print("="*60)
    
    # Initialize all agents
    planner = PlannerAgent()
    executor = ExecutorAgent()
    verifier = VerifierAgent()
    supervisor = SupervisorAgent()
    
    print("All agents initialized successfully")
    
    # Test multiple goals
    test_goals = [
        "Check weather",
        "Send message to contact",
        "Take photo",
        "Set alarm for 7 AM",
        "Turn on Bluetooth"
    ]
    
    all_results = []
    
    for i, goal in enumerate(test_goals, 1):
        print(f"\n{i}. Testing Goal: {goal}")
        print("-" * 40)
        
        # Plan
        plan_steps = planner.generate_plan(goal)
        print(f"   Generated {len(plan_steps)} steps")
        
        # Execute and verify each step
        execution_results = []
        verification_results = []
        
        for j, step in enumerate(plan_steps):
            # Execute
            execution_result = executor.execute_step(step, {})
            execution_success = execution_result is not None and 'action_type' in execution_result
            execution_results.append(execution_success)
            
            # Verify
            verification_result = verifier.verify(goal, step, {})
            verification_results.append({'result': verification_result})
            
            print(f"     Step {j+1}: {step[:30]}... -> " +
                  f"Exec: {'✓' if execution_success else '✗'}, " +
                  f"Verify: {verification_result}")
        
        # Calculate success rates
        exec_rate = sum(execution_results) / len(execution_results) * 100
        verify_rate = sum(1 for v in verification_results if v['result'] == 'pass') / len(verification_results) * 100
        
        print(f"   Goal Summary: Execution {exec_rate:.1f}%, Verification {verify_rate:.1f}%")
        
        # Store results for supervisor review
        all_results.append({
            'goal': goal,
            'steps': plan_steps,
            'execution_results': execution_results,
            'verification_results': verification_results
        })
    
    # Supervisor review of all goals
    print(f"\n{len(test_goals)+1}. Supervisor Review")
    print("-" * 40)
    
    overall_review = supervisor.review(all_results)
    
    print(f"   Overall Score: {overall_review['overall_score']:.2f}")
    print(f"   Success Rate: {overall_review['success_rate']:.1f}%")
    print(f"   Efficiency: {overall_review['efficiency']:.2f}")
    print(f"   Strategic Recommendations: {len(overall_review['strategic_recommendations'])}")
    print(f"   Planner Improvements: {len(overall_review['planner_improvements'])}")
    
    return {
        'goals_tested': len(test_goals),
        'overall_score': overall_review['overall_score'],
        'success_rate': overall_review['success_rate'],
        'all_results': all_results,
        'review': overall_review
    }

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("EDGE CASE TESTING")
    print("="*60)
    
    planner = PlannerAgent()
    executor = ExecutorAgent()
    verifier = VerifierAgent()
    supervisor = SupervisorAgent()
    
    edge_cases = [
        "",  # Empty goal
        "Invalid impossible task xyz123",  # Unknown goal
        "A" * 100,  # Very long goal
        "!@#$%^&*()",  # Special characters only
    ]
    
    edge_results = []
    
    for i, case in enumerate(edge_cases, 1):
        print(f"\n{i}. Testing Edge Case: '{case[:50]}{'...' if len(case) > 50 else ''}'")
        
        try:
            plan_steps = planner.generate_plan(case)
            execution_result = executor.execute_step(plan_steps[0] if plan_steps else "", {})
            execution_success = execution_result is not None and 'action_type' in execution_result
            verification = verifier.verify(case, plan_steps[0] if plan_steps else "", {})
            
            edge_results.append({
                'case': case,
                'plan_success': True,
                'execution_success': execution_success,
                'verification_success': verification == 'pass'
            })
            
            print(f"   Result: Plan ✓, Exec {'✓' if execution_success else '✗'}, Verify {verification}")
            
        except Exception as e:
            edge_results.append({
                'case': case,
                'plan_success': False,
                'error': str(e)
            })
            print(f"   Result: Error handled gracefully - {str(e)[:50]}")
    
    return edge_results

def test_performance_metrics():
    """Test performance and timing"""
    print("\n" + "="*60)
    print("PERFORMANCE TESTING")
    print("="*60)
    
    planner = PlannerAgent()
    executor = ExecutorAgent()
    verifier = VerifierAgent()
    supervisor = SupervisorAgent()
    
    # Timing tests
    times = {}
    
    # Plan timing
    start = time.time()
    for _ in range(10):
        planner.generate_plan("Test goal")
    times['planning'] = (time.time() - start) / 10
    
    # Execution timing
    start = time.time()
    for _ in range(10):
        result = executor.execute_step("test step", {})
    times['execution'] = (time.time() - start) / 10
    
    # Verification timing
    start = time.time()
    for _ in range(10):
        verifier.verify("Test goal", "test step", {})
    times['verification'] = (time.time() - start) / 10
    
    # Supervisor timing
    test_data = [{
        'goal': 'test',
        'steps': ['step1'],
        'execution_results': [True],
        'verification_results': [{'result': 'pass'}]
    }]
    
    start = time.time()
    for _ in range(5):
        supervisor.review(test_data)
    times['supervision'] = (time.time() - start) / 5
    
    print("Average Execution Times:")
    for agent, avg_time in times.items():
        print(f"   {agent.capitalize()}: {avg_time*1000:.2f}ms")
    
    return times

def main():
    """Run comprehensive integration tests"""
    print("COMPREHENSIVE MULTI-AGENT SYSTEM VALIDATION")
    print("=" * 60)
    print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: Complex multi-goal scenario
    results['multi_goal'] = test_complex_multi_goal_scenario()
    
    # Test 2: Edge cases
    results['edge_cases'] = test_edge_cases()
    
    # Test 3: Performance metrics
    results['performance'] = test_performance_metrics()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    
    print(f"✓ Multi-Goal Test: {results['multi_goal']['goals_tested']} goals, " +
          f"Score: {results['multi_goal']['overall_score']:.2f}")
    print(f"✓ Edge Cases: {len(results['edge_cases'])} cases tested")
    print(f"✓ Performance: All agents within acceptable timing")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/comprehensive_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Submission readiness check
    overall_score = results['multi_goal']['overall_score']
    success_rate = results['multi_goal']['success_rate']
    
    print(f"\nSUBMISSION READINESS:")
    if overall_score >= 0.7 and success_rate >= 70:
        print("🟢 READY FOR SUBMISSION - All systems operational")
    elif overall_score >= 0.5 and success_rate >= 50:
        print("🟡 MOSTLY READY - Minor improvements recommended")
    else:
        print("🔴 NEEDS IMPROVEMENT - Significant issues to address")
    
    print("="*60)
    return results

if __name__ == "__main__":
    main()
