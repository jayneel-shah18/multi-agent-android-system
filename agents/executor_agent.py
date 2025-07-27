#!/usr/bin/env python3
"""
ExecutorAgent - Step-to-Action Execution Module
===============================================

This module provides the ExecutorAgent class responsible for converting step
descriptions into concrete Android actions. The agent analyzes UI elements
and generates appropriate touch, swipe, and system actions.

Key Features:
- UI element detection and matching
- Action mapping from natural language steps
- Touch coordinate generation with bounds checking
- System gesture support (back, home, etc.)
- Comprehensive logging and error handling
- Debug mode for detailed execution analysis

Author: Jayneel Shah
Version: 1.0
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from utils.logging import setup_logger, log_agent_action
    from utils.ui_utils import (
        UIElement, parse_ui_elements_from_observation, find_best_element_match,
        create_touch_action, create_no_op_action, debug_print_elements,
        extract_step_action_and_target, normalize_coordinates
    )
except ImportError:
    # Fallback imports when running from different directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logging import setup_logger, log_agent_action
    from utils.ui_utils import (
        UIElement, parse_ui_elements_from_observation, find_best_element_match,
        create_touch_action, create_no_op_action, debug_print_elements,
        extract_step_action_and_target, normalize_coordinates
    )


class ExecutorAgent:
    """
    Agent responsible for executing individual steps from plans on Android devices.
    
    The ExecutorAgent serves as the bridge between high-level step descriptions
    and low-level Android actions. It analyzes UI elements, matches them to
    step intentions, and generates precise touch/gesture actions.
    
    Architecture:
        - UI element parsing and analysis
        - Natural language to action mapping
        - Touch coordinate calculation with bounds validation
        - System gesture support (back, home, menu)
        - Comprehensive execution tracking and logging
        - Debug mode for detailed analysis
    
    Attributes:
        name (str): Agent instance identifier
        debug (bool): Enable detailed logging and output
        logger (Logger): Structured logging interface
        steps_executed (int): Total steps attempted
        successful_executions (int): Successfully completed steps
        execution_history (List[Dict]): Historical execution records
    """
    
    def __init__(self, name: str = "ExecutorAgent", debug: bool = False):
        """
        Initialize the ExecutorAgent with logging and metrics tracking.
        
        Args:
            name: Name of the agent instance for identification
            debug: Whether to enable debug logging and detailed output
        """
        self.name = name
        self.debug = debug
        self.logger = setup_logger(name, level=logging.DEBUG if debug else logging.INFO)
        self.steps_executed = 0
        self.successful_executions = 0
        self.execution_history: List[Dict[str, Any]] = []
        
        # Action type mappings for friendly logging
        self.action_type_labels = {
            0: "observation/wait",
            1: "touch",
            2: "swipe", 
            3: "key_press",
            4: "system_action"
        }
        
        log_agent_action(self.logger, self.name, "Initialized", 
                        f"Debug mode: {debug}")
    
    def execute_step(self, step: str, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single step given the current UI state and observation.
        
        This method analyzes the step description, parses available UI elements,
        matches the best element for the intended action, and generates the
        appropriate Android action dictionary.
        
        Args:
            step: Step description (e.g., "tap wifi", "open settings", "go back")
            observation: Environment observation containing UI tree and state data
            
        Returns:
            Dict[str, Any]: Action dictionary containing action_type and parameters
                          for the Android environment execution
            
        Raises:
            ValueError: If step is None or empty
            
        Example:
            >>> executor = ExecutorAgent()
            >>> obs = {'ui_elements': [...]}
            >>> action = executor.execute_step("tap wifi", obs)
            >>> print(action)
            {'action_type': 1, 'touch_position': [0.5, 0.3]}
        """
        if not step or not isinstance(step, str):
            raise ValueError("Step must be a non-empty string")
            
        log_agent_action(self.logger, self.name, "Executing step", f"Step: '{step}'")
        
        # Parse UI elements from observation with error handling
        try:
            ui_elements = parse_ui_elements_from_observation(observation)
        except Exception as e:
            self.logger.error(f"Error parsing UI elements: {e}")
            ui_elements = []
        
        if self.debug:
            print(f"\n[{self.name}] Executing step: '{step}'")
            debug_print_elements(ui_elements, max_elements=5)
        
        # Record execution attempt for analytics
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "ui_elements_count": len(ui_elements),
            "action_taken": None,
            "success": False,
            "element_found": None
        }
        
        # Extract action type and target from step
        action_type, target = extract_step_action_and_target(step)
        
        # Handle different types of actions
        action = self._handle_action(action_type, target, ui_elements, step)
        
        # Update execution record
        execution_record["action_taken"] = action
        execution_record["success"] = action is not None
        
        if action:
            self.successful_executions += 1
            action_label = self.action_type_labels.get(action.get('action_type', 1), 'unknown')
            log_agent_action(self.logger, self.name, "Step executed successfully", 
                           f"Action: {action_label} at {action.get('touch_position', 'N/A')}")
        else:
            log_agent_action(self.logger, self.name, "Step execution failed", 
                           f"No suitable action found for: '{step}'")
        
        self.steps_executed += 1
        self.execution_history.append(execution_record)
        
        return action or create_no_op_action()
    
    def _handle_action(self, action_type: str, target: str, ui_elements: List[UIElement], 
                      original_step: str) -> Optional[Dict[str, Any]]:
        """
        Handle different types of actions based on action type.
        
        Args:
            action_type: Type of action (tap, toggle, scroll, etc.)
            target: Target element description
            ui_elements: Available UI elements
            original_step: Original step description
            
        Returns:
            Action dictionary or None if no action possible
        """
        if action_type in ['tap', 'click', 'press', 'touch', 'toggle', 'open']:
            return self._handle_tap_action(target, ui_elements, original_step, action_type)
        
        elif action_type == 'scroll':
            return self._handle_scroll_action(target, ui_elements)
        
        elif action_type == 'wait':
            return self._handle_wait_action(target)
        
        elif action_type in ['check', 'verify', 'observe']:
            return self._handle_observation_action(target, ui_elements)
        
        elif action_type in ['go', 'back']:
            return self._handle_back_action(ui_elements)
        
        else:
            # Default to tap action with target matching
            return self._handle_tap_action(target or original_step, ui_elements, original_step, 'tap')
    
    def _handle_tap_action(self, target: str, ui_elements: List[UIElement], 
                          original_step: str, action_type: str = 'tap') -> Optional[Dict[str, Any]]:
        """
        Handle tap/click actions by finding and tapping UI elements.
        
        Args:
            target: Target element description
            ui_elements: Available UI elements
            original_step: Original step description
            action_type: Type of action (tap, toggle, etc.)
            
        Returns:
            Touch action dictionary or None
        """
        # Special handling for toggle actions with disambiguation
        if action_type == 'toggle':
            return self._handle_toggle_action(target, ui_elements, original_step)
        
        # Try to find element using target description
        element = find_best_element_match(ui_elements, target)
        
        # If not found with target, try with original step
        if not element and target != original_step:
            element = find_best_element_match(ui_elements, original_step)
        
        if element:
            if self.debug:
                print(f"  → Found element: '{element.text or element.content_desc}' at ({element.center_x:.0f}, {element.center_y:.0f})")
            
            action = create_touch_action(element)
            action_label = self.action_type_labels.get(action.get('action_type', 1), 'unknown')
            
            log_agent_action(self.logger, self.name, "Element found for tap", 
                           f"Text: '{element.text}', Desc: '{element.content_desc}', Position: ({element.center_x:.0f}, {element.center_y:.0f})")
            return action
        
        # Robustness: Fallback when no matching element found
        return self._handle_no_match_fallback(target, original_step, ui_elements)
    
    def _handle_toggle_action(self, target: str, ui_elements: List[UIElement], 
                             original_step: str) -> Optional[Dict[str, Any]]:
        """
        Handle toggle actions with smart disambiguation between ON/OFF states.
        
        Args:
            target: Target element description
            ui_elements: Available UI elements
            original_step: Original step description
            
        Returns:
            Touch action dictionary or None
        """
        # Extract what we're trying to toggle (wifi, bluetooth, etc.)
        toggle_subject = target.lower()
        for keyword in ['wifi', 'bluetooth', 'data', 'location', 'sound', 'vibration']:
            if keyword in toggle_subject:
                toggle_subject = keyword
                break
        
        # Look for current state indicators
        on_indicators = ['on', 'enabled', 'connected', 'active']
        off_indicators = ['off', 'disabled', 'disconnected', 'inactive']
        
        current_state = None
        toggle_element = None
        
        # Find elements related to the toggle subject
        for element in ui_elements:
            element_text = (element.text or '').lower()
            element_desc = (element.content_desc or '').lower()
            
            # Check if this element relates to our toggle subject
            if (toggle_subject in element_text or toggle_subject in element_desc or
                any(keyword in element_text or keyword in element_desc 
                    for keyword in [toggle_subject, 'toggle', 'switch'])):
                
                # Determine current state from element text/description
                if any(indicator in element_text or indicator in element_desc 
                       for indicator in on_indicators):
                    current_state = 'on'
                    toggle_element = element
                elif any(indicator in element_text or indicator in element_desc 
                         for indicator in off_indicators):
                    current_state = 'off'
                    toggle_element = element
                elif element.is_clickable and ('switch' in element.class_name.lower() or 
                                              'toggle' in element.class_name.lower()):
                    # Found a toggle/switch element, use it even if state is unclear
                    toggle_element = element
        
        # If we found a toggle element, use it
        if toggle_element:
            if self.debug:
                state_info = f" (current state: {current_state})" if current_state else ""
                print(f"  → Found toggle element: '{toggle_element.text or toggle_element.content_desc}'{state_info}")
            
            action = create_touch_action(toggle_element)
            action_label = self.action_type_labels.get(action.get('action_type', 1), 'unknown')
            
            log_msg = f"Toggle element found - Text: '{toggle_element.text}', Desc: '{toggle_element.content_desc}'"
            if current_state:
                log_msg += f", Current state: {current_state}"
            
            log_agent_action(self.logger, self.name, "Toggle action", log_msg)
            return action
        
        # Fallback to regular tap behavior if no specific toggle found
        if self.debug:
            print(f"  → No specific toggle found for '{target}', falling back to regular tap")
        
        return self._handle_tap_action(target, ui_elements, original_step, 'tap')
    
    def _handle_no_match_fallback(self, target: str, original_step: str, 
                                 ui_elements: List[UIElement]) -> Optional[Dict[str, Any]]:
        """
        Handle cases where no matching element is found with robust fallback strategies.
        
        Args:
            target: Target element description
            original_step: Original step description
            ui_elements: Available UI elements
            
        Returns:
            Fallback action or None
        """
        if self.debug:
            print(f"  → No suitable element found for: '{target}'")
        
        log_agent_action(self.logger, self.name, "No match found", 
                        f"Target: '{target}', Step: '{original_step}', Available elements: {len(ui_elements)}")
        
        # Strategy 1: Try fuzzy matching with partial keywords
        keywords = target.split()
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short words
                element = find_best_element_match(ui_elements, keyword)
                if element:
                    log_agent_action(self.logger, self.name, "Fallback: Fuzzy match found", 
                                   f"Keyword: '{keyword}' → '{element.text or element.content_desc}'")
                    return create_touch_action(element)
        
        # Strategy 2: Look for any clickable element containing target keywords
        for element in ui_elements:
            if element.is_clickable and element.is_enabled:
                element_text = (element.text or '').lower()
                element_desc = (element.content_desc or '').lower()
                
                if any(word.lower() in element_text or word.lower() in element_desc 
                       for word in target.split() if len(word) > 2):
                    log_agent_action(self.logger, self.name, "Fallback: Partial match found", 
                                   f"Element: '{element.text or element.content_desc}'")
                    return create_touch_action(element)
        
        # Strategy 3: For navigation actions, try common patterns
        if any(nav_word in original_step.lower() 
               for nav_word in ['open', 'go to', 'navigate', 'launch']):
            # Look for any element that might be an app or navigation target
            for element in ui_elements:
                if (element.is_clickable and 
                    any(app_indicator in (element.class_name or '').lower() 
                        for app_indicator in ['button', 'imageview', 'textview'])):
                    log_agent_action(self.logger, self.name, "Fallback: Navigation attempt", 
                                   f"Trying clickable element: '{element.text or element.content_desc}'")
                    return create_touch_action(element)
        
        # Final fallback: Log the failure and return None for upstream handling
        log_agent_action(self.logger, self.name, "All fallback strategies failed", 
                        f"No suitable action found for step: '{original_step}'")
        
        return None
    
    def _handle_scroll_action(self, target: str, ui_elements: List[UIElement]) -> Dict[str, Any]:
        """
        Handle scroll actions.
        
        Args:
            target: Scroll direction/target
            ui_elements: Available UI elements
            
        Returns:
            Scroll action dictionary
        """
        # For now, implement basic scrolling in the center of the screen
        # This can be enhanced to find scrollable elements
        
        action = None
        if 'down' in target.lower():
            # Scroll down: swipe from center-bottom to center-top
            action = {
                'action_type': 2,  # Swipe
                'touch_position': [0.5, 0.7]  # Start lower
            }
        elif 'up' in target.lower():
            # Scroll up: swipe from center-top to center-bottom
            action = {
                'action_type': 2,  # Swipe
                'touch_position': [0.5, 0.3]  # Start higher
            }
        else:
            # Default scroll down
            action = {
                'action_type': 2,  # Swipe
                'touch_position': [0.5, 0.7]
            }
        
        action_label = self.action_type_labels.get(action['action_type'], 'unknown')
        log_agent_action(self.logger, self.name, f"Scroll action: {action_label}", 
                        f"Direction: {target}, Position: {action['touch_position']}")
        
        return action
    
    def _handle_wait_action(self, target: str) -> Dict[str, Any]:
        """
        Handle wait actions.
        
        Args:
            target: Wait duration description
            
        Returns:
            No-op action
        """
        # Extract wait time if specified
        wait_time = 1.0  # Default 1 second
        
        try:
            # Look for numbers in the target
            import re
            numbers = re.findall(r'\d+(?:\.\d+)?', target)
            if numbers:
                wait_time = float(numbers[0])
        except:
            pass
        
        log_agent_action(self.logger, self.name, "Wait action", f"Duration: {wait_time} seconds")
        
        # In a real environment, we might want to actually wait
        # For now, just return a no-op action
        return create_no_op_action()
    
    def _handle_observation_action(self, target: str, ui_elements: List[UIElement]) -> Dict[str, Any]:
        """
        Handle observation/check actions.
        
        Args:
            target: What to observe/check
            ui_elements: Available UI elements
            
        Returns:
            No-op action (observation doesn't require action)
        """
        # Find elements related to the target for logging
        element = find_best_element_match(ui_elements, target)
        
        action = create_no_op_action()
        action_label = self.action_type_labels.get(action['action_type'], 'unknown')
        
        if element:
            log_agent_action(self.logger, self.name, f"Observation successful: {action_label}", 
                           f"Found '{target}': '{element.text or element.content_desc}'")
        else:
            log_agent_action(self.logger, self.name, f"Observation attempted: {action_label}", 
                           f"Target '{target}' not found in UI")
        
        return action
    
    def _handle_back_action(self, ui_elements: List[UIElement]) -> Optional[Dict[str, Any]]:
        """
        Handle back/return actions.
        
        Args:
            ui_elements: Available UI elements
            
        Returns:
            Back action or tap on back button
        """
        # Try to find a back button
        back_keywords = ["back", "return", "close", "<-", "back"]
        
        for keyword in back_keywords:
            element = find_best_element_match(ui_elements, keyword)
            if element and element.is_clickable:
                action = create_touch_action(element)
                action_label = self.action_type_labels.get(action['action_type'], 'unknown')
                log_agent_action(self.logger, self.name, f"Back button found: {action_label}", 
                               f"Text: '{element.text or element.content_desc}'")
                return action
        
        # If no back button found, use system back gesture (bottom-left swipe up)
        action = {
            'action_type': 1,  # Touch
            'touch_position': [0.1, 0.9]  # Bottom-left corner
        }
        action_label = self.action_type_labels.get(action['action_type'], 'unknown')
        log_agent_action(self.logger, self.name, f"Using system back gesture: {action_label}", 
                        "No back button found")
        return action
    
    def execute_plan(self, plan: List[str], env_step_callback) -> List[Dict[str, Any]]:
        """
        Execute a complete plan step by step.
        
        Args:
            plan: List of step descriptions
            env_step_callback: Function to call environment step and get new observation
                             Should accept action dict and return observation dict
            
        Returns:
            List of execution results for each step
        """
        log_agent_action(self.logger, self.name, "Executing plan", f"Steps: {len(plan)}")
        
        results = []
        
        for i, step in enumerate(plan):
            log_agent_action(self.logger, self.name, f"Plan step {i+1}/{len(plan)}", f"Step: '{step}'")
            
            # Get current observation from environment
            try:
                # Execute a no-op first to get current state
                observation = env_step_callback(create_no_op_action())
                
                # Execute the step
                action = self.execute_step(step, observation)
                
                # Take the action in environment
                next_observation = env_step_callback(action)
                
                result = {
                    "step_index": i,
                    "step": step,
                    "action": action,
                    "success": action != create_no_op_action(),
                    "observation": next_observation
                }
                
                results.append(result)
                
                # Brief pause between steps for realistic execution
                time.sleep(0.5)
                
            except Exception as e:
                log_agent_action(self.logger, self.name, f"Plan step {i+1} failed", f"Error: {str(e)}")
                result = {
                    "step_index": i,
                    "step": step,
                    "action": create_no_op_action(),
                    "success": False,
                    "error": str(e)
                }
                results.append(result)
        
        success_count = sum(1 for r in results if r.get("success", False))
        log_agent_action(self.logger, self.name, "Plan execution completed", 
                        f"Successful steps: {success_count}/{len(plan)}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics for the agent.
        
        Returns:
            Dictionary with execution statistics
        """
        success_rate = (self.successful_executions / self.steps_executed) if self.steps_executed > 0 else 0.0
        
        return {
            "steps_executed": self.steps_executed,
            "successful_executions": self.successful_executions,
            "success_rate": success_rate,
            "execution_history_length": len(self.execution_history),
            "most_recent_step": (
                self.execution_history[-1]["step"] if self.execution_history else None
            )
        }
    
    def reset_statistics(self):
        """Reset execution statistics."""
        self.steps_executed = 0
        self.successful_executions = 0
        self.execution_history.clear()
        log_agent_action(self.logger, self.name, "Statistics reset", "All counters reset to zero")
    
    def __str__(self) -> str:
        """String representation of the ExecutorAgent."""
        return f"ExecutorAgent(name={self.name}, executed={self.steps_executed}, success_rate={self.get_statistics()['success_rate']:.2f})"
    
    def __repr__(self) -> str:
        """Developer representation of the ExecutorAgent."""
        stats = self.get_statistics()
        return (f"ExecutorAgent(name='{self.name}', steps_executed={self.steps_executed}, "
                f"success_rate={stats['success_rate']:.2f}, debug={self.debug})")


# Example usage and testing function
def main():
    """Test the ExecutorAgent with mock observations."""
    
    # Create mock observation that mimics android_world structure
    mock_observation = {
        "structured": {
            "elements": [
                {
                    "text": "Settings",
                    "content_desc": "Settings application",
                    "class_name": "android.widget.TextView",
                    "resource_id": "com.android.settings:id/title",
                    "is_clickable": True,
                    "is_enabled": True,
                    "bounds": [100, 200, 300, 250]
                },
                {
                    "text": "Wi-Fi",
                    "content_desc": "WiFi settings",
                    "class_name": "android.widget.TextView",
                    "resource_id": "com.android.settings:id/wifi",
                    "is_clickable": True,
                    "is_enabled": True,
                    "bounds": [50, 300, 250, 350]
                },
                {
                    "text": "OFF",
                    "content_desc": "WiFi toggle",
                    "class_name": "android.widget.Switch",
                    "resource_id": "com.android.settings:id/wifi_switch",
                    "is_clickable": True,
                    "is_enabled": True,
                    "bounds": [300, 300, 350, 350]
                },
                {
                    "text": "Battery",
                    "content_desc": "Battery settings",
                    "class_name": "android.widget.TextView",
                    "resource_id": "com.android.settings:id/battery",
                    "is_clickable": True,
                    "is_enabled": True,
                    "bounds": [50, 400, 250, 450]
                }
            ]
        }
    }
    
    # Test the agent
    agent = ExecutorAgent(name="TestExecutorAgent", debug=True)
    
    test_steps = [
        "open settings",
        "tap wifi",
        "toggle wifi",
        "check battery status",
        "go back"
    ]
    
    print(f"Testing {agent.name}...")
    print(f"Agent: {agent}")
    print()
    
    for step in test_steps:
        print(f"Testing step: '{step}'")
        action = agent.execute_step(step, mock_observation)
        print(f"Generated action: {action}")
        print("-" * 50)
    
    # Print statistics
    stats = agent.get_statistics()
    print("\nAgent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
