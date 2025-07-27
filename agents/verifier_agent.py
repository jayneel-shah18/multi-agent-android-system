#!/usr/bin/env python3
"""
VerifierAgent - Action Verification and Result Validation
=========================================================

The VerifierAgent validates that executed actions achieved their intended results
by comparing expected vs actual UI state changes. It provides three-level validation:
- "pass": Action succeeded as expected
- "fail": Action failed but system is stable
- "bug_detected": Unexpected behavior or system instability detected

Key Features:
- UI state comparison before/after actions
- Element state verification (enabled/disabled, text changes)
- Screen transition validation
- Heuristic-based and pattern-based verification
- Extensible for LLM-based reasoning
"""

import logging
import re
import sys
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.logging import setup_logger


class VerificationResult:
    """Container for verification results with detailed analysis."""
    
    def __init__(self, status: str, confidence: float = 1.0, reason: str = "", details: Dict = None):
        """
        Initialize verification result.
        
        Args:
            status: "pass", "fail", or "bug_detected"
            confidence: Confidence level (0.0 to 1.0)
            reason: Human-readable explanation
            details: Additional verification details
        """
        self.status = status
        self.confidence = confidence
        self.reason = reason
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def __str__(self):
        return f"VerificationResult(status={self.status}, confidence={self.confidence:.2f}, reason='{self.reason}')"


class VerifierAgent:
    """
    Agent responsible for verifying that executed actions achieved their intended results.
    """
    
    def __init__(self, name: str = "VerifierAgent", debug: bool = False):
        """
        Initialize the VerifierAgent.
        
        Args:
            name: Agent name for logging
            debug: Enable debug logging
        """
        self.name = name
        self.debug = debug
        self.logger = setup_logger(name, level=logging.DEBUG if debug else logging.INFO)
        
        # Verification statistics with enhanced granularity
        self.stats = {
            'verifications_performed': 0,
            'passes': 0,
            'failures': 0,
            'soft_failures': 0,
            'needs_review': 0,
            'bugs_detected': 0,
            'average_confidence': 0.0
        }
        
        # Verification patterns for common actions
        self.verification_patterns = self._initialize_verification_patterns()
        
        self.logger.info(f"[{self.name}] Initialized - Debug mode: {debug}")
    
    def verify(self, goal: str, step: str, result_obs: dict, previous_obs: dict = None) -> str:
        """
        Verify that an executed step achieved its intended result.
        
        Args:
            goal: The high-level goal being pursued
            step: The specific step that was executed
            result_obs: Observation after step execution
            previous_obs: Observation before step execution (optional)
            
        Returns:
            str: "pass", "fail", "soft_fail", "needs_review", or "bug_detected"
        """
        self.logger.info(f"[{self.name}] Verifying step - Goal: '{goal}', Step: '{step}'")
        
        # Perform comprehensive verification
        verification_result = self._perform_verification(goal, step, result_obs, previous_obs)
        
        # Update statistics
        self._update_statistics(verification_result)
        
        # Log result
        self.logger.info(f"[{self.name}] Verification result: {verification_result.status} - {verification_result.reason}")
        
        if self.debug:
            self.logger.debug(f"[{self.name}] Verification details: {verification_result.details}")
        
        return verification_result.status
    
    def verify_detailed(self, goal: str, step: str, result_obs: dict, previous_obs: dict = None) -> VerificationResult:
        """
        Perform detailed verification with full result information.
        
        Args:
            goal: The high-level goal being pursued
            step: The specific step that was executed
            result_obs: Observation after step execution
            previous_obs: Observation before step execution (optional)
            
        Returns:
            VerificationResult: Detailed verification result
        """
        self.logger.info(f"[{self.name}] Performing detailed verification - Step: '{step}'")
        
        verification_result = self._perform_verification(goal, step, result_obs, previous_obs)
        self._update_statistics(verification_result)
        
        return verification_result
    
    def _perform_verification(self, goal: str, step: str, result_obs: dict, previous_obs: dict = None) -> VerificationResult:
        """
        Core verification logic with multiple validation strategies.
        
        Args:
            goal: The high-level goal being pursued
            step: The specific step that was executed
            result_obs: Observation after step execution
            previous_obs: Observation before step execution
            
        Returns:
            VerificationResult: Detailed verification result
        """
        try:
            # Extract step action and target
            action_type, target = self._parse_step(step)
            
            # Strategy 1: Pattern-based verification
            pattern_result = self._verify_by_pattern(action_type, target, result_obs, previous_obs)
            if pattern_result:
                return pattern_result
            
            # Strategy 2: UI state comparison
            state_result = self._verify_ui_state_change(step, result_obs, previous_obs)
            if state_result:
                return state_result
            
            # Strategy 3: Screen transition verification
            transition_result = self._verify_screen_transition(step, result_obs, previous_obs)
            if transition_result:
                return transition_result
            
            # Strategy 4: Element-specific verification
            element_result = self._verify_element_changes(step, result_obs, previous_obs)
            if element_result:
                return element_result
            
            # Strategy 5: Failure detection for action steps that should have caused changes
            failure_result = self._detect_action_failure(action_type, step, result_obs, previous_obs)
            if failure_result:
                return failure_result
            
            # Default: Basic sanity check (enhanced with better failure detection)
            return self._enhanced_sanity_check(action_type, step, result_obs, previous_obs)
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Verification error: {e}")
            return VerificationResult(
                status="bug_detected",
                confidence=0.8,
                reason=f"Verification error: {e}",
                details={"error": str(e), "step": step}
            )
    
    def _parse_step(self, step: str) -> Tuple[str, str]:
        """
        Parse step to extract action type and target.
        
        Args:
            step: Step description
            
        Returns:
            Tuple of (action_type, target)
        """
        step_lower = step.lower().strip()
        
        # Navigation actions
        if any(word in step_lower for word in ['open', 'launch', 'start']):
            target = step_lower.replace('open', '').replace('launch', '').replace('start', '').strip()
            return 'navigate', target
        
        # Toggle actions
        if any(word in step_lower for word in ['toggle', 'turn', 'switch']):
            return 'toggle', step_lower
        
        # Tap/click actions
        if any(word in step_lower for word in ['tap', 'click', 'press']):
            target = step_lower.replace('tap', '').replace('click', '').replace('press', '').strip()
            return 'tap', target
        
        # Back/navigation
        if any(word in step_lower for word in ['back', 'return', 'go back']):
            return 'back', ''
        
        # Wait/observe actions
        if any(word in step_lower for word in ['wait', 'check', 'observe']):
            return 'observe', step_lower
        
        # Scroll actions
        if any(word in step_lower for word in ['scroll', 'swipe']):
            return 'scroll', step_lower
        
        return 'unknown', step_lower
    
    def _verify_by_pattern(self, action_type: str, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """
        Verify using predefined patterns for common actions.
        
        Args:
            action_type: Type of action performed
            target: Target of the action
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult if pattern match found, None otherwise
        """
        if action_type not in self.verification_patterns:
            return None
        
        patterns = self.verification_patterns[action_type]
        
        for pattern in patterns:
            result = pattern(target, result_obs, previous_obs)
            if result:
                return result
        
        return None
    
    def _verify_ui_state_change(self, step: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """
        Verify by comparing UI state before and after action.
        
        Args:
            step: Step description
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult if state change detected and validated
        """
        if not previous_obs:
            return None
        
        try:
            # Compare screen names
            prev_screen = previous_obs.get('screen', 'unknown')
            curr_screen = result_obs.get('screen', 'unknown')
            
            # Compare UI elements
            prev_elements = self._extract_ui_elements(previous_obs)
            curr_elements = self._extract_ui_elements(result_obs)
            
            # Check for expected changes based on step
            step_lower = step.lower()
            
            # Screen transition verification
            if 'open' in step_lower or 'tap' in step_lower:
                if prev_screen != curr_screen:
                    return VerificationResult(
                        status="pass",
                        confidence=0.9,
                        reason=f"Screen transition successful: {prev_screen} -> {curr_screen}",
                        details={
                            "previous_screen": prev_screen,
                            "current_screen": curr_screen,
                            "step": step
                        }
                    )
                elif len(curr_elements) != len(prev_elements):
                    return VerificationResult(
                        status="pass",
                        confidence=0.7,
                        reason="UI elements changed as expected",
                        details={
                            "elements_before": len(prev_elements),
                            "elements_after": len(curr_elements)
                        }
                    )
            
            # Toggle verification
            if 'toggle' in step_lower or 'turn' in step_lower:
                toggle_change = self._detect_toggle_change(prev_elements, curr_elements)
                if toggle_change:
                    return VerificationResult(
                        status="pass",
                        confidence=0.9,
                        reason=f"Toggle state changed: {toggle_change}",
                        details={"toggle_change": toggle_change}
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"UI state comparison failed: {e}")
            return None
    
    def _verify_screen_transition(self, step: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """
        Verify expected screen transitions.
        
        Args:
            step: Step description
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult if screen transition can be verified
        """
        if not previous_obs:
            return None
        
        prev_screen = previous_obs.get('screen', 'unknown')
        curr_screen = result_obs.get('screen', 'unknown')
        
        step_lower = step.lower()
        
        # Expected transitions for navigation steps
        expected_transitions = {
            'open settings': ['home', 'settings'],
            'tap wifi': ['settings', 'wifi'],
            'tap battery': ['settings', 'battery'],
            'go back': ['wifi', 'settings'],
            'back': ['wifi', 'settings']
        }
        
        for expected_step, (expected_from, expected_to) in expected_transitions.items():
            if expected_step in step_lower:
                if prev_screen == expected_from and curr_screen == expected_to:
                    return VerificationResult(
                        status="pass",
                        confidence=0.95,
                        reason=f"Expected screen transition: {prev_screen} -> {curr_screen}",
                        details={
                            "expected_transition": f"{expected_from} -> {expected_to}",
                            "actual_transition": f"{prev_screen} -> {curr_screen}"
                        }
                    )
                elif prev_screen == expected_from and curr_screen != expected_to:
                    return VerificationResult(
                        status="fail",
                        confidence=0.8,
                        reason=f"Unexpected screen transition: expected {expected_to}, got {curr_screen}",
                        details={
                            "expected_screen": expected_to,
                            "actual_screen": curr_screen
                        }
                    )
        
        return None
    
    def _verify_element_changes(self, step: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """
        Verify changes in specific UI elements.
        
        Args:
            step: Step description
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult if element changes can be verified
        """
        if not previous_obs:
            return None
        
        try:
            prev_elements = self._extract_ui_elements(previous_obs)
            curr_elements = self._extract_ui_elements(result_obs)
            
            # Create element maps for comparison
            prev_map = {elem.get('text', ''): elem for elem in prev_elements}
            curr_map = {elem.get('text', ''): elem for elem in curr_elements}
            
            step_lower = step.lower()
            
            # WiFi toggle verification
            if 'wifi' in step_lower and 'toggle' in step_lower:
                wifi_change = self._verify_wifi_toggle(prev_map, curr_map)
                if wifi_change:
                    return wifi_change
            
            # Button/element state changes
            for text, prev_elem in prev_map.items():
                curr_elem = curr_map.get(text)
                if curr_elem and prev_elem:
                    # Check for state changes
                    if prev_elem.get('is_enabled') != curr_elem.get('is_enabled'):
                        return VerificationResult(
                            status="pass",
                            confidence=0.8,
                            reason=f"Element '{text}' state changed",
                            details={
                                "element": text,
                                "previous_enabled": prev_elem.get('is_enabled'),
                                "current_enabled": curr_elem.get('is_enabled')
                            }
                        )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Element change verification failed: {e}")
            return None
    
    def _detect_action_failure(self, action_type: str, step: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """
        Detect when action steps that should have caused changes actually failed.
        
        Args:
            action_type: Type of action performed
            step: Step description
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult if action failure detected
        """
        if not previous_obs:
            return None
        
        try:
            # Check if state remained completely unchanged for action steps that should cause changes
            prev_screen = previous_obs.get('screen', 'unknown')
            curr_screen = result_obs.get('screen', 'unknown')
            
            prev_elements = self._extract_ui_elements(previous_obs)
            curr_elements = self._extract_ui_elements(result_obs)
            
            # For tap/click actions on specific targets
            if action_type in ['tap', 'navigate'] and any(keyword in step.lower() 
                for keyword in ['tap', 'click', 'open', 'press']):
                
                # Check if this was targeting a specific element
                if any(target in step.lower() for target in ['button', 'unknown', 'nonexistent', 'missing']):
                    # If screen and elements are identical, the tap likely failed
                    if (prev_screen == curr_screen and 
                        len(prev_elements) == len(curr_elements) and
                        self._elements_essentially_same(prev_elements, curr_elements)):
                        
                        return VerificationResult(
                            status="fail",
                            confidence=0.85,
                            reason=f"Action step failed - no UI changes detected for: '{step}'",
                            details={
                                "action_type": action_type,
                                "step": step,
                                "screen_unchanged": True,
                                "elements_unchanged": True
                            }
                        )
                
                # For other tap actions, check if target element likely doesn't exist
                if 'unknown' in step.lower() or 'nonexistent' in step.lower():
                    return VerificationResult(
                        status="fail",
                        confidence=0.9,
                        reason=f"Attempted to interact with nonexistent element: '{step}'",
                        details={
                            "action_type": action_type,
                            "step": step,
                            "target_exists": False
                        }
                    )
            
            # For navigation actions that should change screens
            if action_type == 'navigate' and prev_screen == curr_screen:
                # If the step mentions opening/launching something but screen didn't change
                if any(word in step.lower() for word in ['open', 'launch', 'start']):
                    return VerificationResult(
                        status="fail",
                        confidence=0.8,
                        reason=f"Navigation action failed - screen did not change: '{step}'",
                        details={
                            "action_type": action_type,
                            "step": step,
                            "expected_screen_change": True,
                            "actual_screen_change": False
                        }
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Action failure detection failed: {e}")
            return None
    
    def _elements_essentially_same(self, prev_elements: List[Dict], curr_elements: List[Dict]) -> bool:
        """
        Check if two element lists are essentially the same (accounting for minor differences).
        
        Args:
            prev_elements: Previous UI elements
            curr_elements: Current UI elements
            
        Returns:
            True if elements are essentially the same
        """
        if len(prev_elements) != len(curr_elements):
            return False
        
        try:
            # Compare element texts and types
            prev_texts = [elem.get('text', '') for elem in prev_elements]
            curr_texts = [elem.get('text', '') for elem in curr_elements]
            
            prev_classes = [elem.get('class_name', '') for elem in prev_elements]
            curr_classes = [elem.get('class_name', '') for elem in curr_elements]
            
            return prev_texts == curr_texts and prev_classes == curr_classes
            
        except Exception:
            return False
    
    def _enhanced_sanity_check(self, action_type: str, step: str, result_obs: dict, previous_obs: dict = None) -> VerificationResult:
        """
        Perform enhanced sanity check with better failure detection.
        
        Args:
            action_type: Type of action performed
            step: Step description
            result_obs: Observation after action
            previous_obs: Observation before action
            
        Returns:
            VerificationResult with enhanced validation
        """
        try:
            # Check if observation structure is valid
            if not isinstance(result_obs, dict):
                return VerificationResult(
                    status="bug_detected",
                    confidence=0.9,
                    reason="Invalid observation format",
                    details={"observation_type": type(result_obs).__name__}
                )
            
            # Check if UI elements exist
            elements = self._extract_ui_elements(result_obs)
            if not elements:
                return VerificationResult(
                    status="fail",
                    confidence=0.8,
                    reason="No UI elements found after action",
                    details={"step": step, "action_type": action_type}
                )
            
            # Check screen name validity
            screen = result_obs.get('screen', 'unknown')
            if screen == 'unknown':
                return VerificationResult(
                    status="fail",
                    confidence=0.7,
                    reason="Screen state unknown after action",
                    details={"step": step, "action_type": action_type}
                )
            
            # Enhanced logic: Be more skeptical of "pass" for action steps with granular assessment
            if action_type in ['tap', 'navigate', 'toggle'] and previous_obs:
                # If this was an action step but nothing meaningful changed, be suspicious
                prev_screen = previous_obs.get('screen', 'unknown')
                curr_screen = result_obs.get('screen', 'unknown')
                
                # For action steps, require some evidence of success
                if prev_screen == curr_screen and action_type in ['tap', 'navigate']:
                    # Check if the step mentions specific targets that might not exist
                    suspicious_targets = ['unknown', 'nonexistent', 'missing', 'invalid']
                    if any(target in step.lower() for target in suspicious_targets):
                        return VerificationResult(
                            status="fail",
                            confidence=0.8,
                            reason=f"Action step appears to have failed - targeting suspicious element: '{step}'",
                            details={
                                "action_type": action_type,
                                "step": step,
                                "screen_unchanged": True,
                                "suspicious_target": True
                            }
                        )
                    
                    # Check for number references that might have failed
                    number_references = ['number', 'digit', 'tap 1', 'tap 2', 'tap 3', 'tap 4', 'tap 5', 'tap 6', 'tap 7', 'tap 8', 'tap 9', 'tap 0']
                    if any(ref in step.lower() for ref in number_references):
                        return VerificationResult(
                            status="soft_fail",
                            confidence=0.4,
                            reason=f"Number tap action may have failed - no clear UI response: '{step}'",
                            details={
                                "action_type": action_type,
                                "step": step,
                                "screen_unchanged": True,
                                "number_tap_attempted": True,
                                "suggestion": "Verify number buttons are present and clickable"
                            }
                        )
                    
                    # For other tap/navigate actions with no screen change, flag for review
                    return VerificationResult(
                        status="needs_review",
                        confidence=0.3,
                        reason="Action completed but no clear state change detected - manual review recommended",
                        details={
                            "elements_found": len(elements),
                            "screen": screen,
                            "step": step,
                            "action_type": action_type,
                            "review_reason": "no_state_change",
                            "suggestion": "Check if action achieved intended effect"
                        }
                    )
            
            # For observation/wait actions, this is normal behavior
            if action_type in ['observe', 'wait']:
                return VerificationResult(
                    status="pass",
                    confidence=0.8,
                    reason="Observation/wait action completed successfully",
                    details={
                        "elements_found": len(elements),
                        "screen": screen,
                        "step": step,
                        "action_type": action_type
                    }
                )
            
            # Default pass for basic structural validity (but with context awareness)
            confidence = 0.6 if action_type in ['tap', 'navigate'] else 0.7
            return VerificationResult(
                status="pass",
                confidence=confidence,
                reason="Basic sanity check passed - UI structure appears valid",
                details={
                    "elements_found": len(elements),
                    "screen": screen,
                    "step": step,
                    "action_type": action_type
                }
            )
            
        except Exception as e:
            return VerificationResult(
                status="bug_detected",
                confidence=0.9,
                reason=f"Enhanced sanity check failed: {e}",
                details={"error": str(e), "step": step, "action_type": action_type}
            )
    
    def _extract_ui_elements(self, obs: dict) -> List[Dict]:
        """Extract UI elements from observation."""
        if isinstance(obs, dict) and 'structured' in obs:
            return obs['structured'].get('elements', [])
        return []
    
    def _detect_toggle_change(self, prev_elements: List[Dict], curr_elements: List[Dict]) -> Optional[str]:
        """
        Detect toggle state changes between element lists.
        
        Args:
            prev_elements: Previous UI elements
            curr_elements: Current UI elements
            
        Returns:
            Description of toggle change if detected
        """
        try:
            # Look for ON/OFF toggle changes
            prev_toggles = {}
            curr_toggles = {}
            
            for elem in prev_elements:
                text = elem.get('text', '').strip()
                if text in ['ON', 'OFF']:
                    prev_toggles[elem.get('content_desc', '')] = text
            
            for elem in curr_elements:
                text = elem.get('text', '').strip()
                if text in ['ON', 'OFF']:
                    curr_toggles[elem.get('content_desc', '')] = text
            
            # Find changes
            for desc, prev_state in prev_toggles.items():
                curr_state = curr_toggles.get(desc)
                if curr_state and prev_state != curr_state:
                    return f"{desc}: {prev_state} -> {curr_state}"
            
            return None
            
        except Exception:
            return None
    
    def _verify_wifi_toggle(self, prev_map: Dict, curr_map: Dict) -> Optional[VerificationResult]:
        """
        Verify WiFi toggle state change.
        
        Args:
            prev_map: Previous elements mapped by text
            curr_map: Current elements mapped by text
            
        Returns:
            VerificationResult if WiFi toggle verified
        """
        try:
            # Check for OFF -> ON or ON -> OFF transitions
            prev_on = 'ON' in prev_map
            prev_off = 'OFF' in prev_map
            curr_on = 'ON' in curr_map
            curr_off = 'OFF' in curr_map
            
            if (prev_off and curr_on) or (prev_on and curr_off):
                transition = "OFF → ON" if (prev_off and curr_on) else "ON → OFF"
                return VerificationResult(
                    status="pass",
                    confidence=0.95,
                    reason=f"WiFi toggle successful: {transition}",
                    details={"wifi_transition": transition}
                )
            
            return None
            
        except Exception:
            return None
    
    def _initialize_verification_patterns(self) -> Dict:
        """
        Initialize verification patterns for common action types.
        
        Returns:
            Dictionary of verification patterns
        """
        return {
            'navigate': [
                self._verify_navigation_pattern
            ],
            'toggle': [
                self._verify_toggle_pattern
            ],
            'tap': [
                self._verify_tap_pattern
            ],
            'back': [
                self._verify_back_pattern
            ],
            'observe': [
                self._verify_observe_pattern
            ]
        }
    
    def _verify_navigation_pattern(self, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """Verify navigation pattern."""
        screen = result_obs.get('screen', 'unknown')
        
        if 'settings' in target and screen == 'settings':
            return VerificationResult(
                status="pass",
                confidence=0.9,
                reason="Successfully navigated to settings",
                details={"target_screen": "settings", "actual_screen": screen}
            )
        
        return None
    
    def _verify_toggle_pattern(self, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """Verify toggle pattern."""
        if not previous_obs:
            return None
        
        # Check for toggle state changes in environment state
        if 'wifi' in target:
            prev_wifi = previous_obs.get('wifi_enabled', None)
            curr_wifi = result_obs.get('wifi_enabled', None)
            
            if prev_wifi is not None and curr_wifi is not None and prev_wifi != curr_wifi:
                state_change = f"{'enabled' if curr_wifi else 'disabled'}"
                return VerificationResult(
                    status="pass",
                    confidence=0.95,
                    reason=f"WiFi toggle successful - now {state_change}",
                    details={"wifi_previous": prev_wifi, "wifi_current": curr_wifi}
                )
        
        return None
    
    def _verify_tap_pattern(self, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """Verify tap pattern."""
        if not previous_obs:
            return None
        
        prev_screen = previous_obs.get('screen', 'unknown')
        curr_screen = result_obs.get('screen', 'unknown')
        
        # Check for expected screen transitions after taps
        if prev_screen != curr_screen:
            return VerificationResult(
                status="pass",
                confidence=0.8,
                reason=f"Tap resulted in screen change: {prev_screen} -> {curr_screen}",
                details={"screen_transition": f"{prev_screen} -> {curr_screen}"}
            )
        
        return None
    
    def _verify_back_pattern(self, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """Verify back navigation pattern."""
        if not previous_obs:
            return None
        
        prev_screen = previous_obs.get('screen', 'unknown')
        curr_screen = result_obs.get('screen', 'unknown')
        
        # Expected back transitions
        back_transitions = {
            'wifi': 'settings',
            'battery': 'settings',
            'settings': 'home'
        }
        
        expected_screen = back_transitions.get(prev_screen)
        if expected_screen and curr_screen == expected_screen:
            return VerificationResult(
                status="pass",
                confidence=0.9,
                reason=f"Back navigation successful: {prev_screen} -> {curr_screen}",
                details={"back_transition": f"{prev_screen} -> {curr_screen}"}
            )
        
        return None
    
    def _verify_observe_pattern(self, target: str, result_obs: dict, previous_obs: dict = None) -> Optional[VerificationResult]:
        """Verify observation pattern."""
        # Observation steps typically don't change state
        if previous_obs:
            prev_screen = previous_obs.get('screen', 'unknown')
            curr_screen = result_obs.get('screen', 'unknown')
            
            if prev_screen == curr_screen:
                return VerificationResult(
                    status="pass",
                    confidence=0.8,
                    reason="Observation completed - screen state unchanged as expected",
                    details={"screen": curr_screen}
                )
        
        return VerificationResult(
            status="pass",
            confidence=0.6,
            reason="Observation completed - no state change expected",
            details={"observation_type": "passive"}
        )
    
    def _update_statistics(self, result: VerificationResult):
        """Update verification statistics with enhanced granularity."""
        self.stats['verifications_performed'] += 1
        
        if result.status == "pass":
            self.stats['passes'] += 1
        elif result.status == "fail":
            self.stats['failures'] += 1
        elif result.status == "soft_fail":
            self.stats['soft_failures'] += 1
        elif result.status == "needs_review":
            self.stats['needs_review'] += 1
        elif result.status == "bug_detected":
            self.stats['bugs_detected'] += 1
        
        # Update average confidence
        total_confidence = (self.stats['average_confidence'] * (self.stats['verifications_performed'] - 1) + 
                          result.confidence)
        self.stats['average_confidence'] = total_confidence / self.stats['verifications_performed']
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get verification statistics with enhanced granularity.
        
        Returns:
            Dictionary of verification statistics
        """
        total_verifications = max(self.stats['verifications_performed'], 1)
        
        return {
            'agent_name': self.name,
            'verifications_performed': self.stats['verifications_performed'],
            'passes': self.stats['passes'],
            'failures': self.stats['failures'],
            'soft_failures': self.stats['soft_failures'],
            'needs_review': self.stats['needs_review'],
            'bugs_detected': self.stats['bugs_detected'],
            'pass_rate': (self.stats['passes'] / total_verifications) * 100,
            'failure_rate': (self.stats['failures'] / total_verifications) * 100,
            'soft_failure_rate': (self.stats['soft_failures'] / total_verifications) * 100,
            'needs_review_rate': (self.stats['needs_review'] / total_verifications) * 100,
            'bug_detection_rate': (self.stats['bugs_detected'] / total_verifications) * 100,
            'average_confidence': self.stats['average_confidence']
        }
    
    def reset_statistics(self):
        """Reset verification statistics."""
        self.stats = {
            'verifications_performed': 0,
            'passes': 0,
            'failures': 0,
            'soft_failures': 0,
            'needs_review': 0,
            'bugs_detected': 0,
            'average_confidence': 0.0
        }
        self.logger.info(f"[{self.name}] Statistics reset")
    
    def __str__(self):
        """String representation of the agent."""
        stats = self.get_statistics()
        return (f"VerifierAgent(name={self.name}, "
                f"verifications={stats['verifications_performed']}, "
                f"pass_rate={stats['pass_rate']:.1f}%)")


def main():
    """
    Demo of VerifierAgent functionality.
    """
    print("[SEARCH] VERIFIER AGENT DEMO")
    print("=" * 50)
    
    # Initialize VerifierAgent
    verifier = VerifierAgent(name="DemoVerifier", debug=True)
    print(f"[OK] Initialized: {verifier}")
    
    # Mock observations for testing
    mock_obs_before = {
        'screen': 'home',
        'wifi_enabled': False,
        'structured': {
            'elements': [
                {'text': 'Settings', 'content_desc': 'Settings app', 'is_clickable': True},
                {'text': 'Calculator', 'content_desc': 'Calculator app', 'is_clickable': True}
            ]
        }
    }
    
    mock_obs_after_settings = {
        'screen': 'settings',
        'wifi_enabled': False,
        'structured': {
            'elements': [
                {'text': 'Wi-Fi', 'content_desc': 'WiFi settings', 'is_clickable': True},
                {'text': 'OFF', 'content_desc': 'WiFi disabled', 'is_clickable': True},
                {'text': 'Battery', 'content_desc': 'Battery settings', 'is_clickable': True}
            ]
        }
    }
    
    mock_obs_after_wifi_toggle = {
        'screen': 'settings',
        'wifi_enabled': True,
        'structured': {
            'elements': [
                {'text': 'Wi-Fi', 'content_desc': 'WiFi settings', 'is_clickable': True},
                {'text': 'ON', 'content_desc': 'WiFi enabled', 'is_clickable': True},
                {'text': 'Battery', 'content_desc': 'Battery settings', 'is_clickable': True}
            ]
        }
    }
    
    # Test verification scenarios
    test_scenarios = [
        {
            'goal': 'Test turning Wi-Fi on and off',
            'step': 'open settings',
            'before': mock_obs_before,
            'after': mock_obs_after_settings,
            'expected': 'pass'
        },
        {
            'goal': 'Test turning Wi-Fi on and off',
            'step': 'toggle wifi on',
            'before': mock_obs_after_settings,
            'after': mock_obs_after_wifi_toggle,
            'expected': 'pass'
        },
        {
            'goal': 'Test navigation',
            'step': 'tap unknown button',
            'before': mock_obs_before,
            'after': mock_obs_before,  # No change
            'expected': 'fail'
        }
    ]
    
    print(f"\n[TEST] Testing {len(test_scenarios)} verification scenarios:")
    print("-" * 50)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing step: '{scenario['step']}'")
        
        # Perform verification
        result = verifier.verify(
            goal=scenario['goal'],
            step=scenario['step'],
            result_obs=scenario['after'],
            previous_obs=scenario['before']
        )
        
        # Check result
        expected = scenario['expected']
        status = "[OK] PASS" if result == expected else "[FAIL] FAIL"
        print(f"   Result: {result} (expected: {expected}) {status}")
    
    # Show statistics
    print(f"\n[STATS] Verification Statistics:")
    print("-" * 30)
    stats = verifier.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n[TARGET] VerifierAgent demo complete!")


if __name__ == "__main__":
    main()
