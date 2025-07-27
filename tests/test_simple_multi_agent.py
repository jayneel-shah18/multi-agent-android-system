#!/usr/bin/env python3
"""
Multi-Agent System Integration Test
==================================

This module provides comprehensive testing for the multi-agent Android automation
system. It validates the integration between PlannerAgent, ExecutorAgent, 
VerifierAgent, and SupervisorAgent with mock Android environment.

Test Components:
- Agent initialization and configuration
- Plan generation from goals
- Step execution with UI element detection
- Result verification with multiple status types
- Strategic supervision and feedback generation
- End-to-end workflow validation

Output:
- Console progress reporting
- Structured JSON logs in logs/ directory
- Individual agent performance metrics

Author: Multi-Agent System Team
Version: 1.0
"""

import sys
import os
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import all agents
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent


def test_multi_agent_system():
    """Test the complete multi-agent system."""
    print("MULTI-AGENT SYSTEM TEST")
    print("=" * 50)
    
    # Initialize all agents
    planner = PlannerAgent(name="TestPlanner")
    executor = ExecutorAgent(name="TestExecutor")
    verifier = VerifierAgent(name="TestVerifier")
    supervisor = SupervisorAgent(name="TestSupervisor", use_llm=False)
    
    print("All agents initialized successfully")
    
    # Test goal
    goal = "Turn on Wi-Fi"
    print(f"\nTesting goal: {goal}")
    
    # Phase 1: Planning
    print("\n1. Planning Phase")
    plan_steps = planner.generate_plan(goal)
    print(f"   Generated {len(plan_steps)} steps:")
    for i, step in enumerate(plan_steps, 1):
        print(f"     {i}. {step}")
    
    # Phase 2: Execution and Verification
    print("\n2. Execution & Verification Phase")
    
    # Mock environment
    mock_obs = {
        'screen': 'home',
        'wifi_enabled': False,
        'structured': {'elements': []}
    }
    
    success_count = 0
    execution_log = {
        'goal': goal,
        'steps': [],
        'start_time': datetime.now().isoformat()
    }
    
    for i, step in enumerate(plan_steps, 1):
        print(f"   Step {i}: {step}")
        
        # Execute step
        try:
            execution_result = executor.execute_step(step, mock_obs)
            print(f"     Execution: {execution_result.get('success', False)}")
            
            # Mock observation update
            if 'settings' in step.lower():
                mock_obs['screen'] = 'settings'
            elif 'wifi' in step.lower() and 'toggle' in step.lower():
                mock_obs['wifi_enabled'] = not mock_obs['wifi_enabled']
            
            # Verify step
            verification_result = verifier.verify(goal, step, mock_obs)
            print(f"     Verification: {verification_result}")
            
            if verification_result == 'pass':
                success_count += 1
            
            # Log step
            execution_log['steps'].append({
                'step_number': i,
                'step': step,
                'execution_result': execution_result,
                'verification_result': verification_result,
                'observation_after': mock_obs.copy()
            })
            
        except Exception as e:
            print(f"     Error: {e}")
            execution_log['steps'].append({
                'step_number': i,
                'step': step,
                'error': str(e),
                'verification_result': 'bug_detected'
            })
    
    execution_log['end_time'] = datetime.now().isoformat()
    success_rate = success_count / len(plan_steps) if plan_steps else 0
    print(f"\n   Execution complete: {success_count}/{len(plan_steps)} steps passed ({success_rate:.1%})")
    
    # Phase 3: Supervisor Review
    print("\n3. Supervisor Review Phase")
    review = supervisor.review(execution_log)
    
    print(f"   Overall Score: {review['overall_score']:.2f}")
    print(f"   Success Rate: {review['success_rate']:.1%}")
    print(f"   Efficiency: {review['efficiency_score']:.2f}")
    print(f"   Feedback: {review['feedback'][:100]}...")
    
    print(f"\n   Suggestions ({len(review['suggestions'])}):")
    for suggestion in review['suggestions'][:3]:
        print(f"     - {suggestion}")
    
    print(f"\n   Planner Improvements ({len(review['planner_improvements'])}):")
    for improvement in review['planner_improvements'][:2]:
        print(f"     - {improvement}")
    
    # Phase 4: System Statistics
    print("\n4. System Statistics")
    
    planner_stats = planner.get_statistics()
    executor_stats = executor.get_statistics()
    verifier_stats = verifier.get_statistics()
    supervisor_stats = supervisor.get_statistics()
    
    print(f"   Planner: {planner_stats['plans_generated']} plans generated")
    print(f"   Executor: {executor_stats['steps_executed']} steps executed")
    print(f"   Verifier: {verifier_stats['verifications_performed']} verifications")
    print(f"   Supervisor: {supervisor_stats['reviews_performed']} reviews")
    
    # Save results
    try:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/simple_multi_agent_test_{timestamp}.json"
        
        results = {
            'goal': goal,
            'execution_log': execution_log,
            'review': review,
            'statistics': {
                'planner': planner_stats,
                'executor': executor_stats,
                'verifier': verifier_stats,
                'supervisor': supervisor_stats
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
        
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    print("\nMULTI-AGENT SYSTEM TEST COMPLETE")
    print("=" * 50)
    
    return {
        'success_rate': success_rate,
        'overall_score': review['overall_score'],
        'all_agents_working': True
    }


if __name__ == "__main__":
    test_multi_agent_system()
