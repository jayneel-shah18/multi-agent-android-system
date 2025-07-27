#!/usr/bin/env python3
"""
Multi-Agent System Integration Test
==================================

This script demonstrates the complete multi-agent system working together:
- PlannerAgent: Creates test plans
- ExecutorAgent: Executes steps
- VerifierAgent: Verifies results
- SupervisorAgent: Reviews and provides feedback

This showcases the full automation pipeline with all agents collaborating.
"""

import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Import all agents
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from agents.supervisor_agent import SupervisorAgent
from utils.logging import StructuredLogger, setup_logger


class MultiAgentSystem:
    """
    Orchestrates the complete multi-agent automation system.
    """
    
    def __init__(self, debug: bool = True):
        """
        Initialize the multi-agent system.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.logger = setup_logger("MultiAgentSystem", level="DEBUG" if debug else "INFO")
        
        # Initialize all agents
        self.planner = PlannerAgent(name="MainPlanner", log_level="DEBUG" if debug else "INFO")
        self.executor = ExecutorAgent(name="MainExecutor", debug=debug)
        self.verifier = VerifierAgent(name="MainVerifier", debug=debug)
        self.supervisor = SupervisorAgent(name="MainSupervisor", use_llm=False, debug=debug)
        
        # Initialize structured logger (will be set up per session)
        self.structured_logger = None
        
        self.logger.info("Multi-agent system initialized successfully")
    
    def execute_complete_workflow(self, goal: str) -> Dict[str, Any]:
        """
        Execute a complete workflow using all agents.
        
        Args:
            goal: The goal to achieve
            
        Returns:
            Complete workflow results including all agent outputs
        """
        self.logger.info(f"🎯 Starting complete workflow for goal: '{goal}'")
        
        # Start structured logging
        self.structured_logger = StructuredLogger(goal)
        
        workflow_results = {
            'goal': goal,
            'start_time': datetime.now().isoformat(),
            'agents_used': ['PlannerAgent', 'ExecutorAgent', 'VerifierAgent', 'SupervisorAgent'],
            'steps': [],
            'final_status': 'unknown',
            'statistics': {},
            'supervisor_review': {}
        }
        
        try:
            # Phase 1: Planning
            self.logger.info("📋 Phase 1: Planning")
            plan = self._execute_planning_phase(goal)
            workflow_results['plan'] = plan
            
            # Phase 2: Execution with verification
            self.logger.info("⚡ Phase 2: Execution with Verification")
            execution_results = self._execute_execution_phase(goal, plan['steps'])
            workflow_results['steps'] = execution_results['steps']
            workflow_results['final_status'] = execution_results['final_status']
            
            # Phase 3: Supervisor review
            self.logger.info("👑 Phase 3: Supervisor Review")
            supervisor_review = self._execute_review_phase(workflow_results)
            workflow_results['supervisor_review'] = supervisor_review
            
            # Phase 4: Generate statistics
            self.logger.info("📊 Phase 4: Statistics Generation")
            statistics = self._generate_system_statistics()
            workflow_results['statistics'] = statistics
            
            workflow_results['end_time'] = datetime.now().isoformat()
            
            # Save complete results
            self._save_workflow_results(workflow_results)
            
            self.logger.info(f"✅ Complete workflow finished with status: {workflow_results['final_status']}")
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"❌ Workflow failed: {e}")
            workflow_results['final_status'] = 'error'
            workflow_results['error'] = str(e)
            workflow_results['end_time'] = datetime.now().isoformat()
            return workflow_results
    
    def _execute_planning_phase(self, goal: str) -> Dict[str, Any]:
        """Execute the planning phase."""
        self.logger.info("  🧠 Planner creating step-by-step plan...")
        
        # Mock environment for planning
        mock_env = self._create_mock_environment()
        
        # Create plan
        plan_steps = self.planner.generate_plan(goal)
        plan = {
            'goal': goal,
            'steps': plan_steps,
            'step_count': len(plan_steps)
        }
        
        self.logger.info(f"  ✅ Plan created with {len(plan['steps'])} steps")
        
        return plan
    
    def _execute_execution_phase(self, goal: str, steps: List[str]) -> Dict[str, Any]:
        """Execute the execution phase with verification."""
        self.logger.info(f"  ⚡ Executing {len(steps)} steps with verification...")
        
        execution_results = {
            'steps': [],
            'final_status': 'unknown'
        }
        
        # Mock environment state
        current_obs = self._create_mock_environment()
        previous_obs = None
        
        success_count = 0
        
        for i, step in enumerate(steps, 1):
            self.logger.info(f"    Step {i}/{len(steps)}: {step}")
            
            step_start_time = datetime.now()
            
            # Log step start
            self.structured_logger.log_step_start(i, step)
            
            # Execute step
            try:
                execution_result = self.executor.execute_step(step, current_obs)
                
                # Log executor action
                self.structured_logger.log_executor_action(
                    step_num=i,
                    action_type=execution_result.get('action_type', 'unknown'),
                    target=execution_result.get('target', ''),
                    success=execution_result.get('success', False),
                    details=execution_result
                )
                
                # Update observation based on execution
                previous_obs = current_obs.copy()
                current_obs = self._update_mock_environment(current_obs, step, execution_result)
                
                # Verify step
                verification_result = self.verifier.verify(goal, step, current_obs, previous_obs)
                
                # Log verification result
                self.structured_logger.log_verifier_result(
                    step_num=i,
                    verification_status=verification_result,
                    confidence=0.8,  # Mock confidence
                    details={'reason': f'Step {i} verification'}
                )
                
                step_end_time = datetime.now()
                
                # Store step result
                step_result = {
                    'step_number': i,
                    'step': step,
                    'execution_result': execution_result,
                    'verification_result': verification_result,
                    'observation_before': previous_obs,
                    'observation_after': current_obs,
                    'start_time': step_start_time.isoformat(),
                    'end_time': step_end_time.isoformat(),
                    'duration': (step_end_time - step_start_time).total_seconds(),
                    'replanning_triggered': False
                }
                
                execution_results['steps'].append(step_result)
                
                if verification_result == 'pass':
                    success_count += 1
                    self.logger.info(f"      ✅ Step {i} passed verification")
                else:
                    self.logger.warning(f"      ❌ Step {i} failed verification: {verification_result}")
                    
                    # Mock replanning for failed steps
                    if verification_result == 'fail':
                        self.logger.info(f"      🔄 Triggering replanning for step {i}")
                        step_result['replanning_triggered'] = True
                        
                        # Mock alternative step
                        alternative_step = f"Alternative approach: {step}"
                        self.logger.info(f"      🆕 Replanned step: {alternative_step}")
                        
                        # Re-execute with alternative
                        alt_result = self.executor.execute_step(alternative_step, current_obs)
                        alt_verification = self.verifier.verify(goal, alternative_step, current_obs, previous_obs)
                        
                        if alt_verification == 'pass':
                            success_count += 1
                            self.logger.info(f"      ✅ Alternative step {i} passed")
                            step_result['alternative_execution'] = alt_result
                            step_result['final_verification'] = alt_verification
                        else:
                            self.logger.warning(f"      ❌ Alternative step {i} also failed")
                
            except Exception as e:
                self.logger.error(f"      💥 Step {i} execution failed: {e}")
                step_result = {
                    'step_number': i,
                    'step': step,
                    'error': str(e),
                    'verification_result': 'bug_detected',
                    'start_time': step_start_time.isoformat(),
                    'end_time': datetime.now().isoformat()
                }
                execution_results['steps'].append(step_result)
        
        # Determine final status
        success_rate = success_count / len(steps) if steps else 0
        if success_rate >= 0.8:
            execution_results['final_status'] = 'success'
        elif success_rate >= 0.5:
            execution_results['final_status'] = 'partial_success'
        else:
            execution_results['final_status'] = 'failure'
        
        self.logger.info(f"  📊 Execution complete: {success_count}/{len(steps)} steps passed ({success_rate:.1%})")
        
        return execution_results
    
    def _execute_review_phase(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the supervisor review phase."""
        self.logger.info("  👑 Supervisor reviewing execution...")
        
        # Prepare log for supervisor
        supervisor_log = {
            'goal': workflow_results['goal'],
            'start_time': workflow_results['start_time'],
            'end_time': workflow_results.get('end_time', datetime.now().isoformat()),
            'steps': workflow_results['steps']
        }
        
        # Get supervisor review
        review = self.supervisor.review(supervisor_log)
        
        self.logger.info(f"  📋 Review complete - Score: {review['overall_score']:.2f}")
        
        return review
    
    def _generate_system_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive system statistics."""
        stats = {
            'planner_stats': self.planner.get_statistics(),
            'executor_stats': self.executor.get_statistics(),
            'verifier_stats': self.verifier.get_statistics(),
            'supervisor_stats': self.supervisor.get_statistics()
        }
        
        if self.structured_logger:
            stats['structured_logger_stats'] = {'session_active': True}
        else:
            stats['structured_logger_stats'] = {'session_active': False}
            
        return stats
    
    def _create_mock_environment(self) -> Dict[str, Any]:
        """Create a mock environment observation."""
        return {
            'screen': 'home',
            'wifi_enabled': False,
            'bluetooth_enabled': False,
            'step_count': 0,
            'structured': {
                'elements': [
                    {
                        'text': 'Settings',
                        'content_desc': 'Settings app',
                        'class_name': 'android.widget.TextView',
                        'resource_id': 'com.android.launcher:id/settings',
                        'is_clickable': True,
                        'is_enabled': True,
                        'bounds': [100, 200, 300, 280]
                    },
                    {
                        'text': 'Calculator',
                        'content_desc': 'Calculator app',
                        'class_name': 'android.widget.TextView',
                        'resource_id': 'com.android.calculator:id/app',
                        'is_clickable': True,
                        'is_enabled': True,
                        'bounds': [350, 200, 550, 280]
                    }
                ]
            }
        }
    
    def _update_mock_environment(self, current_obs: Dict[str, Any], step: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update mock environment based on executed step."""
        new_obs = current_obs.copy()
        new_obs['step_count'] = current_obs.get('step_count', 0) + 1
        
        step_lower = step.lower()
        
        # Simulate screen transitions
        if 'open settings' in step_lower or 'settings' in step_lower:
            new_obs['screen'] = 'settings'
            new_obs['structured']['elements'] = [
                {
                    'text': 'Wi-Fi',
                    'content_desc': 'WiFi settings',
                    'class_name': 'android.widget.TextView',
                    'resource_id': 'com.android.settings:id/wifi',
                    'is_clickable': True,
                    'is_enabled': True,
                    'bounds': [50, 200, 250, 250]
                },
                {
                    'text': 'OFF' if not new_obs['wifi_enabled'] else 'ON',
                    'content_desc': 'WiFi disabled' if not new_obs['wifi_enabled'] else 'WiFi enabled',
                    'class_name': 'android.widget.Switch',
                    'resource_id': 'com.android.settings:id/wifi_switch',
                    'is_clickable': True,
                    'is_enabled': True,
                    'bounds': [320, 200, 380, 250]
                }
            ]
        
        # Simulate WiFi toggle
        if 'wifi' in step_lower and 'toggle' in step_lower:
            new_obs['wifi_enabled'] = not new_obs['wifi_enabled']
            # Update UI element text
            for element in new_obs['structured']['elements']:
                if 'wifi' in element.get('resource_id', '').lower():
                    element['text'] = 'ON' if new_obs['wifi_enabled'] else 'OFF'
                    element['content_desc'] = 'WiFi enabled' if new_obs['wifi_enabled'] else 'WiFi disabled'
        
        # Simulate back navigation
        if 'back' in step_lower:
            if new_obs['screen'] == 'settings':
                new_obs['screen'] = 'home'
                new_obs['structured']['elements'] = [
                    {
                        'text': 'Settings',
                        'content_desc': 'Settings app',
                        'class_name': 'android.widget.TextView',
                        'resource_id': 'com.android.launcher:id/settings',
                        'is_clickable': True,
                        'is_enabled': True,
                        'bounds': [100, 200, 300, 280]
                    }
                ]
        
        return new_obs
    
    def _save_workflow_results(self, results: Dict[str, Any]):
        """Save complete workflow results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/multi_agent_workflow_{timestamp}.json"
            
            # Ensure logs directory exists
            os.makedirs("logs", exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Workflow results saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow results: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'system_name': 'Multi-Agent Automation System',
            'agents': {
                'planner': str(self.planner),
                'executor': str(self.executor),
                'verifier': str(self.verifier),
                'supervisor': str(self.supervisor)
            },
            'statistics': self._generate_system_statistics()
        }


def main():
    """
    Demonstrate the complete multi-agent system.
    """
    print("🤖 MULTI-AGENT SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize system
    system = MultiAgentSystem(debug=True)
    print(f"✅ Multi-agent system initialized")
    
    # Test goals
    test_goals = [
        "Turn on Wi-Fi",
        "Turn off Wi-Fi after turning it on",
        "Navigate to settings and toggle Wi-Fi twice"
    ]
    
    for i, goal in enumerate(test_goals, 1):
        print(f"\n🎯 TEST {i}: {goal}")
        print("-" * 50)
        
        # Execute complete workflow
        results = system.execute_complete_workflow(goal)
        
        # Display summary
        print(f"\n📊 WORKFLOW SUMMARY:")
        print(f"  Goal: {results['goal']}")
        print(f"  Status: {results['final_status']}")
        print(f"  Steps: {len(results['steps'])}")
        print(f"  Agents: {', '.join(results['agents_used'])}")
        
        # Show supervisor feedback
        review = results.get('supervisor_review', {})
        if review:
            print(f"  Overall Score: {review.get('overall_score', 0):.2f}")
            print(f"  Success Rate: {review.get('success_rate', 0):.1%}")
            print(f"  Efficiency: {review.get('efficiency_score', 0):.2f}")
            
            feedback = review.get('feedback', 'No feedback available')
            if len(feedback) > 100:
                feedback = feedback[:100] + "..."
            print(f"  Feedback: {feedback}")
    
    # System statistics
    print(f"\n🏆 FINAL SYSTEM STATISTICS:")
    print("-" * 40)
    status = system.get_system_status()
    stats = status['statistics']
    
    for agent_name, agent_stats in stats.items():
        if isinstance(agent_stats, dict):
            print(f"\n{agent_name.replace('_', ' ').title()}:")
            for key, value in agent_stats.items():
                if isinstance(value, (int, float)) and key != 'agent_name':
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
    
    print(f"\n🎉 Multi-agent system integration test complete!")
    print(f"Check the logs/ directory for detailed execution logs.")


if __name__ == "__main__":
    main()
