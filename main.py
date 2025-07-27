#!/usr/bin/env python3
"""
Android Automation Pipeline - Complete Integration System
========================================================

This is the main entry point for the Android automation pipeline that integrates
all four agents (Planner, Executor, Verifier, Supervisor) with android_env to
create a complete end-to-end Android automation system.

System Components:
- PlannerAgent: Converts goals to step-by-step plans
- ExecutorAgent: Executes steps as Android actions
- VerifierAgent: Validates execution results
- SupervisorAgent: Provides strategic oversight and feedback
- AndroidEnv/MockEnv: Execution environment

Key Features:
- Environment initialization with FakeSimulatorConfig
- Complete multi-agent workflow orchestration
- Comprehensive logging and monitoring
- Structured QA logging with timestamped outputs
- Error handling and graceful degradation
- Performance metrics and analytics
- Replanning capabilities for failed steps

Output Files:
- logs/qa_run_<timestamp>.json - Structured execution logs
- logs/frame_<step>.png - Screenshot captures (when available)
- logs/<agent>_<timestamp>.log - Individual agent logs

Author: Jayneel Shah
Version: 1.0
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'android_env'))

# Import our agents
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.verifier_agent import VerifierAgent
from utils.logging import setup_logger, StructuredLogger, create_qa_log
from utils.mock_env import MockAndroidEnv

# Android environment imports (with fallback to mock)
ANDROID_ENV_AVAILABLE = False
try:
    from android_env import loader
    from android_env.components import config_classes
    import numpy as np
    ANDROID_ENV_AVAILABLE = True
    print("[OK] Android environment available")
except ImportError as e:
    print(f"[WARNING] Android environment not available: {e}")
    print("   Using MockAndroidEnv as fallback")


class AndroidAutomationPipeline:
    """
    Complete Android automation pipeline integrating planning and execution.
    """
    
    def __init__(self, use_real_env: bool = True, debug: bool = True):
        """
        Initialize the automation pipeline.
        
        Args:
            use_real_env: Whether to use real android_env or mock environment
            debug: Enable debug logging and verbose output
        """
        self.use_real_env = use_real_env and ANDROID_ENV_AVAILABLE
        self.debug = debug
        # Set logging level based on debug flag
        log_level = logging.DEBUG if debug else logging.INFO
        self.logger = setup_logger("AutomationPipeline", level=log_level)
        
        # Initialize components
        self.planner = None
        self.executor = None
        self.verifier = None
        self.env = None
        self.structured_logger = None  # Will be created per goal
        
        # Statistics tracking
        self.stats = {
            'goals_processed': 0,
            'plans_generated': 0,
            'replans_generated': 0,
            'steps_executed': 0,
            'actions_performed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'verifications_performed': 0,
            'bugs_detected': 0,
            'start_time': datetime.now()
        }
        
        self.logger.info(f"Pipeline initialized - Real env: {self.use_real_env}, Debug: {debug}")
    
    def initialize_environment(self) -> bool:
        """
        Initialize the Android environment with FakeSimulatorConfig or MockAndroidEnv.
        
        Returns:
            bool: True if initialization successful
        """
        self.logger.info("Initializing environment...")
        
        if self.use_real_env:
            try:
                # Create AndroidEnv configuration
                config = config_classes.AndroidEnvConfig(
                    simulator=config_classes.FakeSimulatorConfig(),
                    task=config_classes.TaskConfig()
                )
                
                self.env = loader.load(config)
                self.logger.info("[OK] AndroidEnv initialized with FakeSimulatorConfig")
                
                if self.debug:
                    # Get environment specs
                    action_spec = self.env.action_spec()
                    observation_spec = self.env.observation_spec()
                    self.logger.info(f"Action spec keys: {list(action_spec.keys())}")
                    self.logger.info(f"Observation spec keys: {list(observation_spec.keys())}")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize AndroidEnv: {e}")
                self.logger.info("Falling back to MockAndroidEnv...")
                self.use_real_env = False
        
        if not self.use_real_env:
            try:
                self.env = MockAndroidEnv()
                self.logger.info("[OK] MockAndroidEnv initialized")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to initialize MockAndroidEnv: {e}")
                return False
    
    def initialize_agents(self) -> bool:
        """
        Initialize PlannerAgent and ExecutorAgent.
        
        Returns:
            bool: True if initialization successful
        """
        self.logger.info("Initializing agents...")
        
        try:
            # Initialize PlannerAgent
            self.planner = PlannerAgent(name="PipelinePlanner")
            self.logger.info("[OK] PlannerAgent initialized")
            
            # Initialize ExecutorAgent
            self.executor = ExecutorAgent(name="PipelineExecutor", debug=self.debug)
            self.logger.info("[OK] ExecutorAgent initialized")
            
            # Initialize VerifierAgent
            self.verifier = VerifierAgent(name="PipelineVerifier", debug=self.debug)
            self.logger.info("[OK] VerifierAgent initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            return False
    
    def run_automation_goal(self, goal: str) -> Dict[str, Any]:
        """
        Execute a complete automation goal through the pipeline with structured logging.
        
        Args:
            goal: High-level goal description
            
        Returns:
            dict: Execution results and statistics
        """
        self.logger.info(f"[TARGET] Starting automation goal: '{goal}'")
        self.stats['goals_processed'] += 1
        
        # Initialize structured logger for this goal
        self.structured_logger = create_qa_log(goal)
        
        execution_log = {
            'goal': goal,
            'start_time': datetime.now(),
            'plan': [],
            'steps': [],
            'success': False,
            'error': None,
            'final_observation': None,
            'structured_log_path': None
        }
        
        try:
            # Phase 1: Generate execution plan
            print("\n" + "="*60)
            print(f"PHASE 1: PLANNING")
            print("="*60)
            print(f"Goal: {goal}")
            
            plan = self.planner.generate_plan(goal)
            execution_log['plan'] = plan
            self.stats['plans_generated'] += 1
            
            # Log plan to structured logger
            self.structured_logger.log_planner_result(
                step_index=1,  # Planning is considered step 0 or 1
                plan=plan,
                replanning=False,
                context="Initial planning"
            )
            
            print(f"[OK] Generated plan with {len(plan)} steps:")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step}")
            
            self.logger.info(f"Plan generated - {len(plan)} steps")
            
            # Phase 2: Reset environment
            print("\n" + "="*60)
            print(f"[PHONE] PHASE 2: ENVIRONMENT SETUP")
            print("="*60)
            
            obs = self.env.reset()
            current_screen = self._get_current_screen(obs)
            print(f"[OK] Environment reset")
            print(f"  Initial screen: {current_screen}")
            print(f"  Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
            
            if self.debug:
                self._log_observation_details(obs, "Initial")
            
            # Phase 3: Execute plan step by step with verification and structured logging
            print("\n" + "="*60)
            print(f"[ROBOT] PHASE 3: STEP EXECUTION WITH VERIFICATION & LOGGING")
            print("="*60)
            
            remaining_plan = plan.copy()
            step_num = 1
            max_replanning_attempts = 3
            
            while remaining_plan and step_num <= len(plan) + max_replanning_attempts:
                current_step = remaining_plan[0]
                
                # Start structured logging for this step
                step_index = self.structured_logger.log_step_start(current_step, "execution")
                
                # Store observation before step execution for verification
                previous_obs = obs
                
                step_result = self._execute_single_step_with_verification_and_logging(
                    step_num, current_step, obs, previous_obs, execution_log, step_index
                )
                
                # Update observation for next step
                obs = step_result['observation']
                self.stats['steps_executed'] += 1
                
                # Handle verification result with enhanced granularity
                verification_status = step_result.get('verification_status', 'unknown')
                
                if verification_status == 'pass':
                    print(f"[OK] Step {step_num} verified successfully")
                    remaining_plan.pop(0)  # Remove completed step
                    self.stats['successful_executions'] += 1
                    
                    # Complete step logging
                    self.structured_logger.log_step_completion(step_index, "completed_success")
                    
                elif verification_status in ['fail', 'bug_detected']:
                    print(f"Step {step_num} {verification_status} - attempting replanning...")
                    self.stats['failed_executions'] += 1
                    
                    # Complete current step as failed
                    self.structured_logger.log_step_completion(step_index, f"completed_{verification_status}")
                    
                    # Attempt replanning
                    replan_result = self._handle_step_failure_with_logging(
                        current_step, remaining_plan, obs, execution_log, step_num
                    )
                    
                    if replan_result['success']:
                        print(f"Replanning successful - new plan generated")
                        remaining_plan = replan_result['new_plan']
                        self.stats['replans_generated'] += 1
                    else:
                        print(f"[WARNING] Replanning failed - skipping step and continuing")
                        remaining_plan.pop(0)  # Skip failed step
                
                elif verification_status == 'soft_fail':
                    print(f"[SOFT_FAIL] Step {step_num} soft failure - continuing with caution")
                    remaining_plan.pop(0)  # Continue but note the issue
                    self.stats['successful_executions'] += 1  # Count as success but flagged
                    
                    # Complete step logging
                    self.structured_logger.log_step_completion(step_index, "completed_soft_fail")
                
                elif verification_status == 'needs_review':
                    print(f"[NEEDS_REVIEW] Step {step_num} flagged for review - continuing")
                    remaining_plan.pop(0)  # Continue but note for review
                    self.stats['successful_executions'] += 1  # Count as success but flagged
                    
                    # Complete step logging
                    self.structured_logger.log_step_completion(step_index, "completed_needs_review")
                
                else:
                    # Unknown verification status - continue with caution
                    print(f"Step {step_num} verification unclear - continuing")
                    remaining_plan.pop(0)
                    self.structured_logger.log_step_completion(step_index, "completed_unclear")
                
                step_num += 1
            
            execution_log['success'] = True
            execution_log['final_observation'] = obs
            
            # Finalize structured logging
            final_status = "success" if execution_log['success'] else "failed"
            pipeline_stats = self.get_statistics()
            self.structured_logger.finalize_log(final_status, pipeline_stats)
            
            # Save structured log
            log_path = self.structured_logger.save_json_log()
            execution_log['structured_log_path'] = log_path
            
            print(f"\n[OK] Goal execution completed!")
            print(f"[DOC] Structured log saved: {log_path}")
            
        except Exception as e:
            execution_log['error'] = str(e)
            
            # Finalize structured logging even on error
            if self.structured_logger:
                final_status = "error"
                pipeline_stats = self.get_statistics()
                pipeline_stats['error'] = str(e)
                self.structured_logger.finalize_log(final_status, pipeline_stats)
                
                try:
                    log_path = self.structured_logger.save_json_log()
                    execution_log['structured_log_path'] = log_path
                    print(f"Error log saved: {log_path}")
                except Exception as log_error:
                    self.logger.error(f"Failed to save error log: {log_error}")
            
            self.logger.error(f"Goal execution failed: {e}")
            print(f"\nGoal execution failed: {e}")
        
        finally:
            execution_log['end_time'] = datetime.now()
            execution_log['duration'] = (execution_log['end_time'] - execution_log['start_time']).total_seconds()
        
        return execution_log
    
    def _execute_single_step_with_verification(self, step_num: int, step: str, obs: Any, previous_obs: Any, execution_log: Dict) -> Dict[str, Any]:
        """
        Execute a single step with verification and potential replanning.
        
        Args:
            step_num: Step number (1-indexed)
            step: Step description
            obs: Current observation
            previous_obs: Previous observation for verification
            execution_log: Execution log to update
            
        Returns:
            dict: Step execution result with verification status
        """
        print(f"\nStep {step_num}: '{step}'")
        print("-" * 50)
        
        step_result = {
            'step_number': step_num,
            'step': step,
            'start_time': datetime.now(),
            'success': False,
            'action': None,
            'observation': obs,
            'ui_elements_found': 0,
            'matched_element': None,
            'error': None,
            'verification_status': 'unknown',
            'verification_details': None
        }
        
        try:
            # Get current screen info
            current_screen = self._get_current_screen(obs)
            print(f"  Current screen: {current_screen}")
            
            # Extract UI elements for analysis
            ui_elements = self._extract_ui_elements(obs)
            step_result['ui_elements_found'] = len(ui_elements)
            print(f"  UI elements available: {len(ui_elements)}")
            
            if self.debug and ui_elements:
                print("  Available elements:")
                for i, element in enumerate(ui_elements[:5], 1):  # Show first 5
                    element_desc = self._format_element_description(element)
                    print(f"    {i}. {element_desc}")
                if len(ui_elements) > 5:
                    print(f"    ... and {len(ui_elements) - 5} more")
            
            # Execute step with ExecutorAgent
            action = self.executor.execute_step(step, obs)
            step_result['action'] = action
            self.stats['actions_performed'] += 1
            
            # Log action details
            action_desc = self._format_action_description(action)
            print(f"  Action: {action_desc}")
            
            # Find matched UI element (if any)
            matched_element = self._find_matched_element(step, ui_elements)
            if matched_element:
                step_result['matched_element'] = matched_element
                element_desc = self._format_element_description(matched_element)
                print(f"  Matched UI element: {element_desc}")
            else:
                print("  No specific UI element matched")
            
            # Apply action to environment
            new_obs = self.env.step(action)
            step_result['observation'] = new_obs
            
            # Log results
            new_screen = self._get_current_screen(new_obs)
            print(f"  Result: {current_screen} → {new_screen}")
            
            if self.debug:
                self._log_observation_details(new_obs, f"Step {step_num}")
            
            step_result['success'] = True
            
            # PHASE 2: VERIFICATION - Verify step execution
            print(f"  Verifying step execution...")
            
            # Get goal from execution log
            goal = execution_log.get('goal', 'Unknown goal')
            
            # Perform verification
            verification_status = self.verifier.verify(
                goal=goal,
                step=step,
                result_obs=self._convert_obs_for_verification(new_obs),
                previous_obs=self._convert_obs_for_verification(obs)
            )
            
            step_result['verification_status'] = verification_status
            self.stats['verifications_performed'] += 1
            
            # Get detailed verification info if available
            if hasattr(self.verifier, 'verify_detailed'):
                detailed_result = self.verifier.verify_detailed(
                    goal=goal,
                    step=step,
                    result_obs=self._convert_obs_for_verification(new_obs),
                    previous_obs=self._convert_obs_for_verification(obs)
                )
                step_result['verification_details'] = {
                    'confidence': detailed_result.confidence,
                    'reason': detailed_result.reason,
                    'details': detailed_result.details
                }
            
            # Update bug detection stats
            if verification_status == 'bug_detected':
                self.stats['bugs_detected'] += 1
            
            # Log verification result
            status_icon = "[PASS]" if verification_status == "pass" else "[FAIL]" if verification_status == "fail" else "[BUG]"
            print(f"  {status_icon} Verification: {verification_status}")
            
            if step_result['verification_details']:
                confidence = step_result['verification_details']['confidence']
                reason = step_result['verification_details']['reason']
                print(f"    Confidence: {confidence:.2f}, Reason: {reason}")
            
        except Exception as e:
            step_result['error'] = str(e)
            step_result['verification_status'] = 'bug_detected'  # Execution errors are bugs
            self.logger.error(f"Step {step_num} execution failed: {e}")
            print(f" Error: {e}")
        
        finally:
            step_result['end_time'] = datetime.now()
            step_result['duration'] = (step_result['end_time'] - step_result['start_time']).total_seconds()
            execution_log['steps'].append(step_result)
        
        return step_result
    
    def _execute_single_step_with_verification_and_logging(self, step_num: int, step: str, obs: Any, previous_obs: Any, 
                                                         execution_log: Dict, step_index: int) -> Dict[str, Any]:
        """
        Execute a single step with verification and comprehensive structured logging.
        
        Args:
            step_num: Step number (1-indexed)
            step: Step description
            obs: Current observation
            previous_obs: Previous observation for verification
            execution_log: Execution log to update
            step_index: Step index for structured logging
            
        Returns:
            dict: Step execution result with verification status
        """
        print(f"\n Step {step_num}: '{step}'")
        print("-" * 50)
        
        step_result = {
            'step_number': step_num,
            'step': step,
            'start_time': datetime.now(),
            'success': False,
            'action': None,
            'observation': obs,
            'ui_elements_found': 0,
            'matched_element': None,
            'error': None,
            'verification_status': 'unknown',
            'verification_details': None
        }
        
        try:
            # Get current screen info
            current_screen = self._get_current_screen(obs)
            print(f"  Current screen: {current_screen}")
            
            # Extract UI elements for analysis
            ui_elements = self._extract_ui_elements(obs)
            step_result['ui_elements_found'] = len(ui_elements)
            print(f"  UI elements available: {len(ui_elements)}")
            
            if self.debug and ui_elements:
                print("  Available elements:")
                for i, element in enumerate(ui_elements[:5], 1):  # Show first 5
                    element_desc = self._format_element_description(element)
                    print(f"    {i}. {element_desc}")
                if len(ui_elements) > 5:
                    print(f"    ... and {len(ui_elements) - 5} more")
            
            # Execute step with ExecutorAgent
            action = self.executor.execute_step(step, obs)
            step_result['action'] = action
            self.stats['actions_performed'] += 1
            
            # Log action details
            action_desc = self._format_action_description(action)
            print(f"  Action: {action_desc}")
            
            # Find matched UI element (if any)
            matched_element = self._find_matched_element(step, ui_elements)
            if matched_element:
                step_result['matched_element'] = matched_element
                element_desc = self._format_element_description(matched_element)
                print(f"  [MATCH] Matched UI element: {element_desc}")
            else:
                print("  [NO_MATCH] No specific UI element matched")
            
            # Log executor action to structured logger
            self.structured_logger.log_executor_action(
                step_index=step_index,
                action=action,
                element_info=matched_element,
                success=True,
                error=None
            )
            
            # Apply action to environment
            new_obs = self.env.step(action)
            step_result['observation'] = new_obs
            
            # Log results
            new_screen = self._get_current_screen(new_obs)
            print(f"  Result: {current_screen} → {new_screen}")
            
            if self.debug:
                self._log_observation_details(new_obs, f"Step {step_num}")
            
            step_result['success'] = True
            
            # VERIFICATION PHASE - Verify step execution
            print(f"   Verifying step execution...")
            
            # Get goal from execution log
            goal = execution_log.get('goal', 'Unknown goal')
            
            # Perform verification
            verification_status = self.verifier.verify(
                goal=goal,
                step=step,
                result_obs=self._convert_obs_for_verification(new_obs),
                previous_obs=self._convert_obs_for_verification(obs)
            )
            
            step_result['verification_status'] = verification_status
            self.stats['verifications_performed'] += 1
            
            # Get detailed verification info if available
            detailed_result = None
            if hasattr(self.verifier, 'verify_detailed'):
                detailed_result = self.verifier.verify_detailed(
                    goal=goal,
                    step=step,
                    result_obs=self._convert_obs_for_verification(new_obs),
                    previous_obs=self._convert_obs_for_verification(obs)
                )
                step_result['verification_details'] = {
                    'confidence': detailed_result.confidence,
                    'reason': detailed_result.reason,
                    'details': detailed_result.details
                }
            
            # Log verification result to structured logger
            self.structured_logger.log_verifier_result(
                step_index=step_index,
                verification_status=verification_status,
                confidence=detailed_result.confidence if detailed_result else None,
                reason=detailed_result.reason if detailed_result else None,
                details=detailed_result.details if detailed_result else None
            )
            
            # Update stats for enhanced verification status
            if verification_status == 'bug_detected':
                self.stats['bugs_detected'] += 1
            elif verification_status in ['soft_fail', 'needs_review']:
                # Track these as special cases for monitoring
                if 'soft_fails' not in self.stats:
                    self.stats['soft_fails'] = 0
                if 'needs_review' not in self.stats:
                    self.stats['needs_review'] = 0
                if verification_status == 'soft_fail':
                    self.stats['soft_fails'] += 1
                else:
                    self.stats['needs_review'] += 1
            
            # Log verification result with enhanced status icons
            status_icon = "[PASS]" if verification_status == "pass" else "[FAIL]" if verification_status == "fail" else "[SOFT_FAIL]" if verification_status == "soft_fail" else "[REVIEW]" if verification_status == "needs_review" else "[BUG]"
            print(f"  {status_icon} Verification: {verification_status}")
            
            if step_result['verification_details']:
                confidence = step_result['verification_details']['confidence']
                reason = step_result['verification_details']['reason']
                print(f"    Confidence: {confidence:.2f}, Reason: {reason}")
                
                # Show suggestions for soft_fail and needs_review
                if verification_status in ['soft_fail', 'needs_review'] and 'suggestion' in step_result['verification_details']['details']:
                    suggestion = step_result['verification_details']['details']['suggestion']
                    print(f"    Suggestion: {suggestion}")
            
        except Exception as e:
            step_result['error'] = str(e)
            step_result['verification_status'] = 'bug_detected'  # Execution errors are bugs
            
            # Log executor failure to structured logger
            self.structured_logger.log_executor_action(
                step_index=step_index,
                action=None,
                element_info=None,
                success=False,
                error=str(e)
            )
            
            # Log verification failure
            self.structured_logger.log_verifier_result(
                step_index=step_index,
                verification_status='bug_detected',
                confidence=0.0,
                reason=f"Execution error: {e}",
                details={'error_type': 'execution_failure'}
            )
            
            self.logger.error(f"Step {step_num} execution failed: {e}")
            print(f"  Error: {e}")
        
        finally:
            step_result['end_time'] = datetime.now()
            step_result['duration'] = (step_result['end_time'] - step_result['start_time']).total_seconds()
            execution_log['steps'].append(step_result)
        
        return step_result
    
    def _handle_step_failure_with_logging(self, failed_step: str, remaining_plan: List[str], 
                                        current_obs: Any, execution_log: Dict, step_num: int) -> Dict[str, Any]:
        """
        Handle step failure with replanning and structured logging.
        
        Args:
            failed_step: The step that failed
            remaining_plan: Current remaining plan
            current_obs: Current observation state
            execution_log: Execution log for context
            step_num: Current step number
            
        Returns:
            dict: Replanning result with success status and new plan
        """
        replan_result = {
            'success': False,
            'new_plan': remaining_plan,
            'error': None,
            'strategy': 'none'
        }
        
        try:
            print(f"  Attempting replanning after failed step: '{failed_step}'")
            
            # Get current context
            goal = execution_log.get('goal', 'Unknown goal')
            current_screen = self._get_current_screen(current_obs)
            ui_elements = self._extract_ui_elements(current_obs)
            
            print(f"    Current state: screen={current_screen}, elements={len(ui_elements)}")
            
            # Strategy 1: Skip failed step and continue with remaining plan
            if len(remaining_plan) > 1:
                print(f"    Strategy 1: Skip failed step and continue")
                new_plan = remaining_plan[1:]  # Skip failed step
                replan_result.update({
                    'success': True,
                    'new_plan': new_plan,
                    'strategy': 'skip_failed_step'
                })
                
                # Log this replanning event
                self.structured_logger.log_planner_result(
                    step_index=step_num,
                    plan=new_plan,
                    replanning=True,
                    context=f"Skip failed step: {failed_step}"
                )
                
                return replan_result
            
            # Strategy 2: Generate new plan from current state
            context = f"Current screen: {current_screen}. Failed step: '{failed_step}'. "
            context += f"Available UI elements: {len(ui_elements)}. "
            context += f"Previous steps completed: {len(execution_log.get('steps', []))}"
            
            enriched_goal = f"{goal}. Context: {context}"
            
            print(f"    Strategy 2: Generate new plan from current state")
            new_plan = self.planner.generate_plan(enriched_goal)
            
            if new_plan and len(new_plan) > 0:
                print(f"    New plan generated with {len(new_plan)} steps")
                replan_result.update({
                    'success': True,
                    'new_plan': new_plan,
                    'strategy': 'regenerate_full_plan'
                })
                
                # Log this replanning event
                self.structured_logger.log_planner_result(
                    step_index=step_num,
                    plan=new_plan,
                    replanning=True,
                    context=f"Full replan after failure: {failed_step}"
                )
                
                return replan_result
            
            # Strategy 3: Try alternative step for same goal
            print(f"    Strategy 3: Generate alternative for failed step")
            alternative_goal = f"Alternative approach to: {failed_step}. Current context: {context}"
            alternative_plan = self.planner.generate_plan(alternative_goal)
            
            if alternative_plan and len(alternative_plan) > 0:
                print(f"    Alternative approach generated with {len(alternative_plan)} steps")
                # Combine alternative with remaining original plan
                combined_plan = alternative_plan + remaining_plan[1:]
                replan_result.update({
                    'success': True,
                    'new_plan': combined_plan,
                    'strategy': 'alternative_approach'
                })
                
                # Log this replanning event
                self.structured_logger.log_planner_result(
                    step_index=step_num,
                    plan=combined_plan,
                    replanning=True,
                    context=f"Alternative approach for: {failed_step}"
                )
                
                return replan_result
            
            # All strategies failed
            print(f"    All replanning strategies failed")
            replan_result['error'] = "No successful replanning strategy found"
            
        except Exception as e:
            replan_result['error'] = str(e)
            self.logger.error(f"Replanning error: {e}")
            print(f"    Replanning error: {e}")
        
        return replan_result

    def _handle_step_failure(self, failed_step: str, remaining_plan: List[str], current_obs: Any, execution_log: Dict) -> Dict[str, Any]:
        """
        Handle step failure by attempting replanning.
        
        Args:
            failed_step: The step that failed
            remaining_plan: Current remaining plan
            current_obs: Current observation state
            execution_log: Execution log for context
            
        Returns:
            dict: Replanning result with success status and new plan
        """
        replan_result = {
            'success': False,
            'new_plan': remaining_plan,
            'error': None,
            'strategy': 'none'
        }
        
        try:
            print(f"  Attempting replanning after failed step: '{failed_step}'")
            
            # Get current context
            goal = execution_log.get('goal', 'Unknown goal')
            current_screen = self._get_current_screen(current_obs)
            ui_elements = self._extract_ui_elements(current_obs)
            
            print(f"    Current state: screen={current_screen}, elements={len(ui_elements)}")
            
            # Strategy 1: Skip failed step and continue with remaining plan
            if len(remaining_plan) > 1:
                print(f"    Strategy 1: Skip failed step and continue")
                new_plan = remaining_plan[1:]  # Skip failed step
                replan_result.update({
                    'success': True,
                    'new_plan': new_plan,
                    'strategy': 'skip_failed_step'
                })
                return replan_result
            
            # Strategy 2: Generate new plan from current state
            context = f"Current screen: {current_screen}. Failed step: '{failed_step}'. "
            context += f"Available UI elements: {len(ui_elements)}. "
            context += f"Previous steps completed: {len(execution_log.get('steps', []))}"
            
            enriched_goal = f"{goal}. Context: {context}"
            
            print(f"    Strategy 2: Generate new plan from current state")
            new_plan = self.planner.generate_plan(enriched_goal)
            
            if new_plan and len(new_plan) > 0:
                print(f"    New plan generated with {len(new_plan)} steps")
                replan_result.update({
                    'success': True,
                    'new_plan': new_plan,
                    'strategy': 'regenerate_full_plan'
                })
                return replan_result
            
            # Strategy 3: Try alternative step for same goal
            print(f"    Strategy 3: Generate alternative for failed step")
            alternative_goal = f"Alternative approach to: {failed_step}. Current context: {context}"
            alternative_plan = self.planner.generate_plan(alternative_goal)
            
            if alternative_plan and len(alternative_plan) > 0:
                print(f"    Alternative approach generated with {len(alternative_plan)} steps")
                # Combine alternative with remaining original plan
                combined_plan = alternative_plan + remaining_plan[1:]
                replan_result.update({
                    'success': True,
                    'new_plan': combined_plan,
                    'strategy': 'alternative_approach'
                })
                return replan_result
            
            # All strategies failed
            print(f"    All replanning strategies failed")
            replan_result['error'] = "No successful replanning strategy found"
            
        except Exception as e:
            replan_result['error'] = str(e)
            self.logger.error(f"Replanning error: {e}")
            print(f"    [ERROR] Replanning error: {e}")
        
        return replan_result
    
    def _convert_obs_for_verification(self, obs: Any) -> Dict[str, Any]:
        """
        Convert observation to format expected by VerifierAgent.
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            dict: Formatted observation for verification
        """
        try:
            if isinstance(obs, dict):
                # Already in dict format - check if it needs restructuring
                if 'screen' in obs and 'structured' in obs:
                    # MockAndroidEnv format - already compatible
                    return obs
                
                # AndroidEnv format - extract relevant parts
                converted = {
                    'screen': self._get_current_screen(obs),
                    'structured': {
                        'elements': self._extract_ui_elements(obs)
                    }
                }
                
                # Add any state information that might be useful
                if 'wifi_enabled' in obs:
                    converted['wifi_enabled'] = obs['wifi_enabled']
                
                return converted
            
            # Handle other observation formats
            return {
                'screen': self._get_current_screen(obs),
                'structured': {
                    'elements': self._extract_ui_elements(obs)
                }
            }
            
        except Exception as e:
            self.logger.debug(f"Error converting observation for verification: {e}")
            # Return minimal valid observation
            return {
                'screen': 'unknown',
                'structured': {
                    'elements': []
                }
            }

    def _get_current_screen(self, obs: Any) -> str:
        """Extract current screen name from observation."""
        if isinstance(obs, dict):
            # MockAndroidEnv format
            if 'screen' in obs:
                return obs['screen']
            # AndroidEnv format
            if 'observation' in obs and isinstance(obs['observation'], dict):
                return obs['observation'].get('screen', 'unknown')
            # Check for timestep format
            if hasattr(obs, 'observation'):
                return getattr(obs.observation, 'screen', 'unknown')
        
        return 'unknown'
    
    def _extract_ui_elements(self, obs: Any) -> List[Dict]:
        """Extract UI elements from observation."""
        elements = []
        
        try:
            if isinstance(obs, dict):
                # MockAndroidEnv format
                if 'structured' in obs and 'elements' in obs['structured']:
                    elements = obs['structured']['elements']
                # AndroidEnv format
                elif 'observation' in obs and isinstance(obs['observation'], dict):
                    structured = obs['observation'].get('structured', {})
                    elements = structured.get('elements', [])
        except Exception as e:
            self.logger.debug(f"Error extracting UI elements: {e}")
        
        return elements
    
    def _format_element_description(self, element: Dict) -> str:
        """Format UI element for display."""
        text = element.get('text', '')
        desc = element.get('content_desc', '')
        bounds = element.get('bounds', [])
        
        # Format bounds
        bounds_str = ""
        if bounds and len(bounds) >= 4:
            bounds_str = f"({bounds[0]}, {bounds[1]})"
        
        # Create description
        display_text = text or desc or 'No text'
        return f"'{display_text}' {bounds_str}"
    
    def _format_action_description(self, action: Dict) -> str:
        """Format action for display."""
        if not action:
            return "No action"
        
        action_type = action.get('action_type', 'unknown')
        touch_pos = action.get('touch_position', [0, 0])
        
        if action_type == 1:  # Touch action
            return f"Touch at ({touch_pos[0]:.3f}, {touch_pos[1]:.3f})"
        elif action_type == 0:  # Observation/wait
            return "Observation/Wait"
        else:
            return f"Action type {action_type}"
    
    def _find_matched_element(self, step: str, elements: List[Dict]) -> Optional[Dict]:
        """Find the UI element that matches the step (for logging)."""
        try:
            # Use executor's internal logic to find matched element
            from utils.ui_utils import find_best_element_match, UIElement
            
            # Convert to UIElement objects
            ui_elements = []
            for element in elements:
                ui_elem = UIElement(
                    text=element.get('text', ''),
                    content_desc=element.get('content_desc', ''),
                    class_name=element.get('class_name', ''),
                    resource_id=element.get('resource_id', ''),
                    is_clickable=element.get('is_clickable', False),
                    is_enabled=element.get('is_enabled', True),
                    bounds=element.get('bounds', [0, 0, 0, 0])
                )
                ui_elements.append(ui_elem)
            
            # Find best match
            matched = find_best_element_match(ui_elements, step)
            if matched:
                # Convert back to dict format
                return {
                    'text': matched.text,
                    'content_desc': matched.content_desc,
                    'bounds': matched.bounds
                }
        except Exception as e:
            self.logger.debug(f"Error finding matched element: {e}")
        
        return None
    
    def _log_observation_details(self, obs: Any, prefix: str):
        """Log detailed observation information for debugging."""
        try:
            if isinstance(obs, dict):
                keys = list(obs.keys())
                self.logger.debug(f"{prefix} observation keys: {keys}")
                
                # Log structured data details
                if 'structured' in obs:
                    structured = obs['structured']
                    if isinstance(structured, dict) and 'elements' in structured:
                        elem_count = len(structured['elements'])
                        self.logger.debug(f"{prefix} structured elements: {elem_count}")
        except Exception as e:
            self.logger.debug(f"Error logging observation details: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        stats = self.stats.copy()
        stats['runtime_seconds'] = runtime
        stats['success_rate'] = (
            (self.stats['successful_executions'] / max(self.stats['steps_executed'], 1)) * 100
            if self.stats['steps_executed'] > 0 else 0
        )
        stats['verification_rate'] = (
            (self.stats['verifications_performed'] / max(self.stats['steps_executed'], 1)) * 100
            if self.stats['steps_executed'] > 0 else 0
        )
        stats['replanning_rate'] = (
            (self.stats['replans_generated'] / max(self.stats['plans_generated'], 1)) * 100
            if self.stats['plans_generated'] > 0 else 0
        )
        
        return stats
    
    def shutdown(self):
        """Shutdown the pipeline and cleanup resources."""
        self.logger.info("Shutting down automation pipeline...")
        
        if self.env and hasattr(self.env, 'close'):
            try:
                self.env.close()
                self.logger.info("Environment closed")
            except Exception as e:
                self.logger.warning(f"Error closing environment: {e}")
        
        # Log final statistics
        stats = self.get_statistics()
        self.logger.info("Final statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")


def main():
    """
    Main execution function for Phase 3 integration.
    """
    print("[ROCKET] ANDROID AUTOMATION PIPELINE - PHASE 3")
    print("=" * 60)
    print("Integrating PlannerAgent + ExecutorAgent + AndroidEnv")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = AndroidAutomationPipeline(use_real_env=True, debug=True)
    
    try:
        # Initialize components
        print("\n[CLIPBOARD] INITIALIZATION")
        print("-" * 30)
        
        if not pipeline.initialize_environment():
            print("[ERROR] Failed to initialize environment")
            return
        
        if not pipeline.initialize_agents():
            print("[ERROR] Failed to initialize agents")
            return
        
        print("[OK] All components initialized successfully!")
        
        # Test goals for demonstration
        test_goals = [
            "Test turning Wi-Fi on and off",
            "Check battery status",
            "Open calculator and make a simple calculation"
        ]
        
        print(f"\n[TARGET] AUTOMATION GOALS ({len(test_goals)} scenarios)")
        print("-" * 30)
        
        results = []
        
        for i, goal in enumerate(test_goals, 1):
            print(f"\n[FILM] SCENARIO {i}/{len(test_goals)}")
            print("=" * 60)
            
            result = pipeline.run_automation_goal(goal)
            results.append(result)
            
            # Brief pause between scenarios
            if i < len(test_goals):
                print("\n[PAUSE] Pausing between scenarios...")
                time.sleep(2)
        
        # Final summary
        print("\n" + "=" * 60)
        print("[STATS] FINAL SUMMARY")
        print("=" * 60)
        
        stats = pipeline.get_statistics()
        print(f"Goals processed: {stats['goals_processed']}")
        print(f"Plans generated: {stats['plans_generated']}")
        print(f"Replans generated: {stats['replans_generated']}")
        print(f"Steps executed: {stats['steps_executed']}")
        print(f"Actions performed: {stats['actions_performed']}")
        print(f"Verifications performed: {stats['verifications_performed']}")
        print(f"Bugs detected: {stats['bugs_detected']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Verification rate: {stats['verification_rate']:.1f}%")
        print(f"Replanning rate: {stats['replanning_rate']:.1f}%")
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        
        # Results summary
        print(f"\nScenario Results:")
        for i, result in enumerate(results, 1):
            status = "[SUCCESS]" if result['success'] else "[FAILED]"
            duration = result.get('duration', 0)
            steps_count = len(result.get('steps', []))
            print(f"  {i}. {result['goal']}: {status} ({steps_count} steps, {duration:.1f}s)")
        
        print("\n[COMPLETE] Phase 3 integration demonstration complete!")
        
    except KeyboardInterrupt:
        print("\n[STOPPED] Pipeline interrupted by user")
    
    except Exception as e:
        print(f"\n[ERROR] Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
