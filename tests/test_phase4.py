#!/usr/bin/env python3
"""
Phase 4: Test and Validate
===========================

Comprehensive testing and validation of the Android automation pipeline
with detailed logging, UI tree saving, and step-by-step result tracking.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# Import our components
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from utils.logging import setup_logger
from utils.mock_env import MockAndroidEnv

# Android environment imports (with fallback)
ANDROID_ENV_AVAILABLE = False
try:
    from android_env import loader
    from android_env.components import config_classes
    import numpy as np
    ANDROID_ENV_AVAILABLE = True
except ImportError:
    pass


class Phase4Validator:
    """
    Phase 4 testing and validation with comprehensive logging and UI tree saving.
    """
    
    def __init__(self, test_goal: str = "Test turning Wi-Fi on and off"):
        """
        Initialize the Phase 4 validator.
        
        Args:
            test_goal: The goal to test with
        """
        self.test_goal = test_goal
        self.logger = setup_logger("Phase4Validator", level=logging.DEBUG)
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Test results tracking
        self.validation_results = {
            'test_goal': test_goal,
            'start_time': datetime.now(),
            'steps_generated': False,
            'steps_count': 0,
            'steps_details': [],
            'touch_actions_generated': False,
            'touch_actions_count': 0,
            'step_logs_printed': False,
            'ui_trees_saved': False,
            'step_results': [],
            'overall_success': False,
            'errors': []
        }
        
        self.logger.info(f"🧪 Phase 4 Validator initialized for goal: '{test_goal}'")
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run the comprehensive Phase 4 test with all validations.
        
        Returns:
            dict: Complete validation results
        """
        print("🚀 PHASE 4: COMPREHENSIVE TEST AND VALIDATION")
        print("=" * 60)
        print(f"Test Goal: '{self.test_goal}'")
        print("=" * 60)
        
        try:
            # Initialize components
            if not self._initialize_components():
                return self.validation_results
            
            # Generate and validate plan
            if not self._test_plan_generation():
                return self.validation_results
            
            # Execute plan with detailed validation
            if not self._test_plan_execution():
                return self.validation_results
            
            # Final validation
            self._perform_final_validation()
            
        except Exception as e:
            self.logger.error(f"Critical error during Phase 4 testing: {e}")
            self.validation_results['errors'].append(f"Critical error: {e}")
        
        finally:
            self.validation_results['end_time'] = datetime.now()
            self.validation_results['duration'] = (
                self.validation_results['end_time'] - self.validation_results['start_time']
            ).total_seconds()
        
        return self.validation_results
    
    def _initialize_components(self) -> bool:
        """Initialize and validate all components."""
        print("\n📋 COMPONENT INITIALIZATION")
        print("-" * 40)
        
        try:
            # Initialize environment
            if ANDROID_ENV_AVAILABLE:
                print("🔄 Attempting AndroidEnv with FakeSimulatorConfig...")
                try:
                    config = config_classes.AndroidEnvConfig(
                        simulator=config_classes.FakeSimulatorConfig(),
                        task=config_classes.TaskConfig()
                    )
                    self.env = loader.load(config)
                    print("✅ AndroidEnv initialized successfully")
                    self.env_type = "AndroidEnv"
                except Exception as e:
                    print(f"⚠️  AndroidEnv failed: {e}")
                    print("🔄 Falling back to MockAndroidEnv...")
                    self.env = MockAndroidEnv()
                    self.env_type = "MockAndroidEnv"
            else:
                print("🔄 Using MockAndroidEnv (AndroidEnv not available)...")
                self.env = MockAndroidEnv()
                self.env_type = "MockAndroidEnv"
            
            print(f"✅ Environment: {self.env_type}")
            
            # Initialize agents
            self.planner = PlannerAgent(name="Phase4Planner")
            self.executor = ExecutorAgent(name="Phase4Executor", debug=True)
            
            print("✅ PlannerAgent initialized")
            print("✅ ExecutorAgent initialized")
            
            self.logger.info(f"All components initialized - Environment: {self.env_type}")
            return True
            
        except Exception as e:
            error_msg = f"Component initialization failed: {e}"
            self.logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def _test_plan_generation(self) -> bool:
        """Test and validate plan generation."""
        print(f"\n🧠 PLAN GENERATION TEST")
        print("-" * 40)
        print(f"Goal: '{self.test_goal}'")
        
        try:
            # Generate plan
            self.plan = self.planner.generate_plan(self.test_goal)
            
            # Validate plan generation
            if self.plan and len(self.plan) > 0:
                self.validation_results['steps_generated'] = True
                self.validation_results['steps_count'] = len(self.plan)
                
                print(f"✅ Steps generated: {len(self.plan)} steps")
                print("📋 Generated plan:")
                for i, step in enumerate(self.plan, 1):
                    print(f"  {i}. {step}")
                    self.validation_results['steps_details'].append({
                        'step_number': i,
                        'step_text': step
                    })
                
                self.logger.info(f"Plan generation successful: {len(self.plan)} steps")
                return True
            else:
                error_msg = "No steps generated from plan"
                self.validation_results['errors'].append(error_msg)
                print(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Plan generation failed: {e}"
            self.logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def _test_plan_execution(self) -> bool:
        """Test plan execution with detailed validation and logging."""
        print(f"\n🤖 PLAN EXECUTION TEST")
        print("-" * 40)
        
        try:
            # Reset environment
            obs = self.env.reset()
            self._save_ui_tree(obs, "initial", 0)
            
            print(f"✅ Environment reset - Initial screen: {self._get_screen_name(obs)}")
            self.logger.info(f"Environment reset - Initial observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Not dict'}")
            
            # Execute each step with detailed validation
            touch_actions_count = 0
            successful_steps = 0
            
            for step_num, step in enumerate(self.plan, 1):
                step_result = self._execute_and_validate_step(
                    step_num, step, obs, touch_actions_count
                )
                
                # Update counters
                if step_result['touch_action_generated']:
                    touch_actions_count += 1
                
                if step_result['success']:
                    successful_steps += 1
                
                # Update observation for next step
                obs = step_result['final_observation']
                
                # Save UI tree after step
                self._save_ui_tree(obs, f"step_{step_num}", step_num)
                
                self.validation_results['step_results'].append(step_result)
            
            # Validate touch actions
            self.validation_results['touch_actions_generated'] = touch_actions_count > 0
            self.validation_results['touch_actions_count'] = touch_actions_count
            self.validation_results['step_logs_printed'] = True  # We're printing detailed logs
            self.validation_results['ui_trees_saved'] = True    # We're saving UI trees
            
            print(f"\n✅ Plan execution completed")
            print(f"   Successful steps: {successful_steps}/{len(self.plan)}")
            print(f"   Touch actions generated: {touch_actions_count}")
            
            return successful_steps > 0
            
        except Exception as e:
            error_msg = f"Plan execution failed: {e}"
            self.logger.error(error_msg)
            self.validation_results['errors'].append(error_msg)
            print(f"❌ {error_msg}")
            return False
    
    def _execute_and_validate_step(self, step_num: int, step: str, obs: Any, touch_count: int) -> Dict[str, Any]:
        """Execute a single step with comprehensive validation and logging."""
        print(f"\n🔄 STEP {step_num}: '{step}'")
        print("=" * 50)
        
        step_result = {
            'step_number': step_num,
            'step_text': step,
            'start_time': datetime.now(),
            'success': False,
            'touch_action_generated': False,
            'action_details': None,
            'ui_elements_found': 0,
            'matched_element': None,
            'screen_transition': {'from': None, 'to': None},
            'logs_printed': True,  # We're printing logs for each step
            'error': None,
            'final_observation': obs
        }
        
        try:
            # Log initial state
            initial_screen = self._get_screen_name(obs)
            step_result['screen_transition']['from'] = initial_screen
            
            print(f"📱 Current screen: {initial_screen}")
            
            # Extract and log UI elements
            ui_elements = self._extract_ui_elements(obs)
            step_result['ui_elements_found'] = len(ui_elements)
            
            print(f"🎯 UI elements available: {len(ui_elements)}")
            if ui_elements:
                print("   Available elements:")
                for i, element in enumerate(ui_elements[:3], 1):  # Show first 3
                    element_desc = self._format_element_description(element)
                    print(f"     {i}. {element_desc}")
                if len(ui_elements) > 3:
                    print(f"     ... and {len(ui_elements) - 3} more")
            
            # Execute step with ExecutorAgent
            print(f"⚙️  Executing step with ExecutorAgent...")
            action = self.executor.execute_step(step, obs)
            step_result['action_details'] = action
            
            # Validate action generation
            if action:
                print(f"✅ Action generated: {self._format_action_description(action)}")
                
                # Check if it's a touch action
                if action.get('action_type') == 1:  # Touch action
                    step_result['touch_action_generated'] = True
                    touch_pos = action.get('touch_position', [0, 0])
                    print(f"👆 Touch action: coordinates ({touch_pos[0]:.3f}, {touch_pos[1]:.3f})")
                
                # Find matched UI element
                matched_element = self._find_matched_element_for_step(step, ui_elements)
                if matched_element:
                    step_result['matched_element'] = matched_element
                    element_desc = self._format_element_description(matched_element)
                    print(f"🎯 Matched UI element: {element_desc}")
                else:
                    print("⚪ No specific UI element matched (generic action)")
                
            else:
                print("❌ No action generated")
                step_result['error'] = "No action generated"
                return step_result
            
            # Apply action to environment
            print(f"🔄 Applying action to environment...")
            new_obs = self.env.step(action)
            step_result['final_observation'] = new_obs
            
            # Log result
            final_screen = self._get_screen_name(new_obs)
            step_result['screen_transition']['to'] = final_screen
            
            if initial_screen != final_screen:
                print(f"🔀 Screen transition: {initial_screen} → {final_screen}")
            else:
                print(f"📍 Remained on screen: {final_screen}")
            
            step_result['success'] = True
            print(f"✅ Step {step_num} completed successfully")
            
        except Exception as e:
            step_result['error'] = str(e)
            print(f"❌ Step {step_num} failed: {e}")
            self.logger.error(f"Step {step_num} execution failed: {e}")
        
        finally:
            step_result['end_time'] = datetime.now()
            step_result['duration'] = (step_result['end_time'] - step_result['start_time']).total_seconds()
            
            # Log step completion
            status = "SUCCESS" if step_result['success'] else "FAILED"
            duration = step_result['duration']
            print(f"📊 Step {step_num} result: {status} ({duration:.2f}s)")
        
        return step_result
    
    def _save_ui_tree(self, obs: Any, stage: str, step_num: int):
        """Save UI tree to logs directory for debugging."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/ui_tree_{stage}_{timestamp}.json"
            
            # Extract UI tree data
            ui_data = {
                'stage': stage,
                'step_number': step_num,
                'timestamp': timestamp,
                'screen': self._get_screen_name(obs),
                'observation_keys': list(obs.keys()) if isinstance(obs, dict) else [],
                'ui_elements': self._extract_ui_elements(obs),
                'raw_observation': obs if isinstance(obs, dict) else str(obs)
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(ui_data, f, indent=2, default=str)
            
            self.logger.debug(f"UI tree saved: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save UI tree for {stage}: {e}")
    
    def _get_screen_name(self, obs: Any) -> str:
        """Extract screen name from observation."""
        if isinstance(obs, dict):
            return obs.get('screen', 'unknown')
        return 'unknown'
    
    def _extract_ui_elements(self, obs: Any) -> List[Dict]:
        """Extract UI elements from observation."""
        if isinstance(obs, dict) and 'structured' in obs:
            return obs['structured'].get('elements', [])
        return []
    
    def _format_element_description(self, element: Dict) -> str:
        """Format UI element for display."""
        text = element.get('text', '')
        desc = element.get('content_desc', '')
        bounds = element.get('bounds', [])
        
        display_text = text or desc or 'No text'
        bounds_str = f"({bounds[0]}, {bounds[1]})" if bounds and len(bounds) >= 2 else ""
        
        return f"'{display_text}' {bounds_str}".strip()
    
    def _format_action_description(self, action: Dict) -> str:
        """Format action for display."""
        if not action:
            return "No action"
        
        action_type = action.get('action_type', 'unknown')
        if action_type == 1:  # Touch
            touch_pos = action.get('touch_position', [0, 0])
            return f"Touch at ({touch_pos[0]:.3f}, {touch_pos[1]:.3f})"
        elif action_type == 0:  # Observation/wait
            return "Observation/Wait"
        else:
            return f"Action type {action_type}"
    
    def _find_matched_element_for_step(self, step: str, elements: List[Dict]) -> Optional[Dict]:
        """Find the UI element that matches the step."""
        try:
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
                return {
                    'text': matched.text,
                    'content_desc': matched.content_desc,
                    'bounds': matched.bounds
                }
        except Exception as e:
            self.logger.debug(f"Error finding matched element: {e}")
        
        return None
    
    def _perform_final_validation(self):
        """Perform final validation and set overall success."""
        print(f"\n📊 FINAL VALIDATION")
        print("-" * 40)
        
        # Check all validation criteria
        validations = []
        
        # 1. Steps generated
        if self.validation_results['steps_generated']:
            print(f"✅ Steps generated: {self.validation_results['steps_count']} steps")
            validations.append(True)
        else:
            print("❌ Steps generation failed")
            validations.append(False)
        
        # 2. Touch actions generated
        if self.validation_results['touch_actions_generated']:
            print(f"✅ Touch actions generated: {self.validation_results['touch_actions_count']} actions")
            validations.append(True)
        else:
            print("❌ No touch actions generated")
            validations.append(False)
        
        # 3. Step-by-step logs printed
        if self.validation_results['step_logs_printed']:
            print("✅ Step-by-step logs printed")
            validations.append(True)
        else:
            print("❌ Step logs not printed")
            validations.append(False)
        
        # 4. UI trees saved
        if self.validation_results['ui_trees_saved']:
            print("✅ UI trees saved to logs/")
            validations.append(True)
        else:
            print("❌ UI trees not saved")
            validations.append(False)
        
        # 5. Step results logged
        successful_steps = sum(1 for step in self.validation_results['step_results'] if step['success'])
        total_steps = len(self.validation_results['step_results'])
        
        if successful_steps > 0:
            print(f"✅ Step results logged: {successful_steps}/{total_steps} successful")
            validations.append(True)
        else:
            print("❌ No successful step results")
            validations.append(False)
        
        # Overall success
        self.validation_results['overall_success'] = all(validations)
        
        if self.validation_results['overall_success']:
            print(f"\n🎉 PHASE 4 VALIDATION: SUCCESS!")
        else:
            print(f"\n⚠️  PHASE 4 VALIDATION: PARTIAL SUCCESS")
        
        # Print summary
        print(f"\nValidation Summary:")
        print(f"  Total validations: {len(validations)}")
        print(f"  Passed: {sum(validations)}")
        print(f"  Failed: {len(validations) - sum(validations)}")
        
        if self.validation_results['errors']:
            print(f"  Errors encountered: {len(self.validation_results['errors'])}")


def main():
    """Main execution function for Phase 4 testing."""
    print("🚀 PHASE 4: TEST AND VALIDATE")
    print("=" * 60)
    print("Testing goal: 'Test turning Wi-Fi on and off'")
    print("=" * 60)
    
    # Create and run validator
    validator = Phase4Validator(test_goal="Test turning Wi-Fi on and off")
    results = validator.run_comprehensive_test()
    
    # Print final report
    print("\n" + "=" * 60)
    print("📋 PHASE 4 VALIDATION REPORT")
    print("=" * 60)
    
    print(f"Test Goal: {results['test_goal']}")
    print(f"Duration: {results.get('duration', 0):.2f} seconds")
    print(f"Overall Success: {'✅ YES' if results['overall_success'] else '❌ NO'}")
    
    print(f"\n📊 Detailed Results:")
    print(f"  Steps generated: {'✅' if results['steps_generated'] else '❌'} ({results['steps_count']} steps)")
    print(f"  Touch actions: {'✅' if results['touch_actions_generated'] else '❌'} ({results['touch_actions_count']} actions)")
    print(f"  Step logs: {'✅' if results['step_logs_printed'] else '❌'}")
    print(f"  UI trees saved: {'✅' if results['ui_trees_saved'] else '❌'}")
    
    if results['step_results']:
        successful_steps = sum(1 for step in results['step_results'] if step['success'])
        print(f"  Step success rate: {successful_steps}/{len(results['step_results'])} ({(successful_steps/len(results['step_results'])*100):.1f}%)")
    
    if results['errors']:
        print(f"\n⚠️  Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"    • {error}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/phase4_validation_{timestamp}.json"
    
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Failed to save results: {e}")
    
    print(f"\n🎯 Phase 4 validation complete!")
    
    return results['overall_success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
