#!/usr/bin/env python3
"""
SupervisorAgent - Test Execution Review and Analysis
===================================================

The SupervisorAgent provides high-level oversight and analysis of test execution.
It reviews completed test runs, analyzes outcomes, and provides strategic feedback
to improve future test execution and planning.

Key Features:
- Comprehensive test execution analysis
- LLM-powered review and feedback generation
- Planner prompt optimization suggestions
- Edge case and flow recommendations
- Performance metrics analysis
- Automated report generation

Author: Jayneel Shah
Version: 1.0
"""

import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict

# Optional numpy import for frame processing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.logging import setup_logger


@dataclass
class ReviewResult:
    """Container for supervisor review results."""
    
    # Overall assessment
    overall_score: float  # 0.0 to 1.0
    success_rate: float
    efficiency_score: float
    
    # Detailed feedback
    feedback: str
    suggestions: List[str]
    
    # Recommendations
    planner_improvements: List[str]
    new_test_prompts: List[str]
    edge_cases: List[str]
    
    # Analysis details
    step_analysis: Dict[str, Any]
    failure_patterns: List[str]
    optimization_opportunities: List[str]
    
    # Metadata
    review_timestamp: str
    agent_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SupervisorAgent:
    """
    High-level supervisor agent that reviews test execution and provides strategic feedback.
    """
    
    def __init__(self, name: str = "SupervisorAgent", use_llm: bool = False, debug: bool = False):
        """
        Initialize the SupervisorAgent.
        
        Args:
            name: Agent name for logging
            use_llm: Whether to use LLM for advanced analysis (requires API key)
            debug: Enable debug logging
        """
        self.name = name
        self.use_llm = use_llm
        self.debug = debug
        self.logger = setup_logger(name, level=logging.DEBUG if debug else logging.INFO)
        
        # Review statistics
        self.stats = {
            'reviews_performed': 0,
            'total_tests_analyzed': 0,
            'average_success_rate': 0.0,
            'average_efficiency': 0.0,
            'recommendations_generated': 0
        }
        
        # Analysis patterns for common issues
        self.analysis_patterns = self._initialize_analysis_patterns()
        
        # LLM configuration (if enabled)
        if self.use_llm:
            self._initialize_llm()
        
        self.logger.info(f"[{self.name}] Initialized - LLM mode: {use_llm}, Debug: {debug}")
    
    def review(self, log: dict, frames: Optional[List[Any]] = None) -> dict:
        """
        Perform comprehensive review of test execution.
        
        Args:
            log: Test execution log with steps, results, and metadata
            frames: Optional list of screenshot frames from execution (numpy arrays if available)
            
        Returns:
            dict: Comprehensive review results and recommendations
        """
        self.logger.info(f"[{self.name}] Starting review of test execution")
        
        try:
            # Perform comprehensive analysis
            review_result = self._perform_comprehensive_review(log, frames)
            
            # Update statistics
            self._update_statistics(review_result, log)
            
            # Log summary
            self.logger.info(f"[{self.name}] Review complete - Score: {review_result.overall_score:.2f}")
            
            return review_result.to_dict()
            
        except Exception as e:
            self.logger.error(f"[{self.name}] Review failed: {e}")
            return self._create_error_review(str(e))
    
    def _perform_comprehensive_review(self, log: dict, frames: Optional[List[Any]] = None) -> ReviewResult:
        """
        Core review logic with multiple analysis strategies.
        
        Args:
            log: Test execution log
            frames: Optional screenshot frames (numpy arrays if available)
            
        Returns:
            ReviewResult: Comprehensive review analysis
        """
        # Extract key metrics from log
        metrics = self._extract_execution_metrics(log)
        
        # Analyze step execution patterns
        step_analysis = self._analyze_step_execution(log)
        
        # Identify failure patterns
        failure_patterns = self._identify_failure_patterns(log)
        
        # Generate optimization opportunities
        optimization_opportunities = self._identify_optimizations(log, step_analysis)
        
        # Generate planner improvements
        planner_improvements = self._generate_planner_improvements(log, step_analysis)
        
        # Generate new test prompts
        new_test_prompts = self._generate_test_prompts(log, failure_patterns)
        
        # Identify edge cases
        edge_cases = self._identify_edge_cases(log, step_analysis)
        
        # Generate overall feedback
        feedback = self._generate_comprehensive_feedback(metrics, step_analysis, failure_patterns)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(step_analysis, optimization_opportunities)
        
        # Calculate scores
        overall_score = self._calculate_overall_score(metrics, step_analysis)
        efficiency_score = self._calculate_efficiency_score(metrics, step_analysis)
        
        # Create review result
        return ReviewResult(
            overall_score=overall_score,
            success_rate=metrics['success_rate'],
            efficiency_score=efficiency_score,
            feedback=feedback,
            suggestions=suggestions,
            planner_improvements=planner_improvements,
            new_test_prompts=new_test_prompts,
            edge_cases=edge_cases,
            step_analysis=step_analysis,
            failure_patterns=failure_patterns,
            optimization_opportunities=optimization_opportunities,
            review_timestamp=datetime.now().isoformat(),
            agent_version="1.0.0"
        )
    
    def _extract_execution_metrics(self, log: dict) -> Dict[str, Any]:
        """
        Extract key execution metrics from test log.
        
        Args:
            log: Test execution log
            
        Returns:
            Dictionary of execution metrics
        """
        try:
            steps = log.get('steps', [])
            total_steps = len(steps)
            
            if total_steps == 0:
                return {
                    'total_steps': 0,
                    'successful_steps': 0,
                    'failed_steps': 0,
                    'success_rate': 0.0,
                    'average_step_time': 0.0,
                    'total_execution_time': 0.0,
                    'replanning_events': 0
                }
            
            # Count successful/failed steps
            successful_steps = 0
            failed_steps = 0
            step_times = []
            replanning_events = 0
            
            for step in steps:
                verification_result = step.get('verification_result', 'unknown')
                
                if verification_result == 'pass':
                    successful_steps += 1
                elif verification_result in ['fail', 'bug_detected']:
                    failed_steps += 1
                
                # Extract timing if available
                if 'start_time' in step and 'end_time' in step:
                    try:
                        start = datetime.fromisoformat(step['start_time'])
                        end = datetime.fromisoformat(step['end_time'])
                        step_time = (end - start).total_seconds()
                        step_times.append(step_time)
                    except:
                        pass
                
                # Check for replanning
                if step.get('replanning_triggered', False):
                    replanning_events += 1
            
            success_rate = successful_steps / total_steps if total_steps > 0 else 0.0
            average_step_time = sum(step_times) / len(step_times) if step_times else 0.0
            total_execution_time = sum(step_times) if step_times else 0.0
            
            return {
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'failed_steps': failed_steps,
                'success_rate': success_rate,
                'average_step_time': average_step_time,
                'total_execution_time': total_execution_time,
                'replanning_events': replanning_events
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            return {'total_steps': 0, 'successful_steps': 0, 'failed_steps': 0, 'success_rate': 0.0}
    
    def _analyze_step_execution(self, log: dict) -> Dict[str, Any]:
        """
        Analyze step execution patterns and identify issues.
        
        Args:
            log: Test execution log
            
        Returns:
            Dictionary of step execution analysis
        """
        try:
            steps = log.get('steps', [])
            
            analysis = {
                'step_types': {},
                'common_failures': {},
                'execution_patterns': [],
                'verification_analysis': {},
                'screen_transitions': [],
                'action_effectiveness': {}
            }
            
            # Analyze step types and their success rates
            for step in steps:
                step_description = step.get('step', 'unknown')
                verification_result = step.get('verification_result', 'unknown')
                
                # Categorize step type
                step_type = self._categorize_step_type(step_description)
                
                if step_type not in analysis['step_types']:
                    analysis['step_types'][step_type] = {'total': 0, 'passed': 0, 'failed': 0}
                
                analysis['step_types'][step_type]['total'] += 1
                
                if verification_result == 'pass':
                    analysis['step_types'][step_type]['passed'] += 1
                else:
                    analysis['step_types'][step_type]['failed'] += 1
                
                # Track common failures
                if verification_result in ['fail', 'bug_detected']:
                    failure_reason = step.get('verification_details', {}).get('reason', 'unknown')
                    if failure_reason not in analysis['common_failures']:
                        analysis['common_failures'][failure_reason] = 0
                    analysis['common_failures'][failure_reason] += 1
                
                # Track screen transitions
                before_screen = step.get('observation_before', {}).get('screen', 'unknown')
                after_screen = step.get('observation_after', {}).get('screen', 'unknown')
                
                if before_screen != 'unknown' and after_screen != 'unknown':
                    transition = f"{before_screen} → {after_screen}"
                    analysis['screen_transitions'].append({
                        'transition': transition,
                        'step': step_description,
                        'success': verification_result == 'pass'
                    })
            
            # Calculate success rates for step types
            for step_type, data in analysis['step_types'].items():
                if data['total'] > 0:
                    data['success_rate'] = data['passed'] / data['total']
                else:
                    data['success_rate'] = 0.0
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing step execution: {e}")
            return {}
    
    def _categorize_step_type(self, step_description: str) -> str:
        """
        Categorize step into type based on description.
        
        Args:
            step_description: Step description text
            
        Returns:
            Step category string
        """
        step_lower = step_description.lower()
        
        if any(word in step_lower for word in ['open', 'launch', 'start']):
            return 'navigation'
        elif any(word in step_lower for word in ['tap', 'click', 'press']):
            return 'interaction'
        elif any(word in step_lower for word in ['toggle', 'turn', 'switch']):
            return 'toggle'
        elif any(word in step_lower for word in ['back', 'return']):
            return 'back_navigation'
        elif any(word in step_lower for word in ['wait', 'check', 'observe']):
            return 'observation'
        elif any(word in step_lower for word in ['scroll', 'swipe']):
            return 'scroll'
        else:
            return 'other'
    
    def _identify_failure_patterns(self, log: dict) -> List[str]:
        """
        Identify common failure patterns in the execution.
        
        Args:
            log: Test execution log
            
        Returns:
            List of identified failure patterns
        """
        patterns = []
        steps = log.get('steps', [])
        
        try:
            # Pattern 1: Consecutive failures
            consecutive_failures = 0
            max_consecutive = 0
            
            for step in steps:
                if step.get('verification_result') in ['fail', 'bug_detected']:
                    consecutive_failures += 1
                    max_consecutive = max(max_consecutive, consecutive_failures)
                else:
                    consecutive_failures = 0
            
            if max_consecutive >= 3:
                patterns.append(f"Consecutive failures detected (max: {max_consecutive} steps)")
            
            # Pattern 2: Repeated same step failures
            step_failure_counts = {}
            for step in steps:
                if step.get('verification_result') in ['fail', 'bug_detected']:
                    step_desc = step.get('step', 'unknown')
                    step_failure_counts[step_desc] = step_failure_counts.get(step_desc, 0) + 1
            
            for step_desc, count in step_failure_counts.items():
                if count >= 2:
                    patterns.append(f"Repeated failure on step type: '{step_desc}' ({count} times)")
            
            # Pattern 3: Screen transition failures
            transition_failures = {}
            for step in steps:
                if step.get('verification_result') in ['fail', 'bug_detected']:
                    before_screen = step.get('observation_before', {}).get('screen', 'unknown')
                    after_screen = step.get('observation_after', {}).get('screen', 'unknown')
                    transition = f"{before_screen} → {after_screen}"
                    transition_failures[transition] = transition_failures.get(transition, 0) + 1
            
            for transition, count in transition_failures.items():
                if count >= 2:
                    patterns.append(f"Screen transition issues: {transition} ({count} failures)")
            
            # Pattern 4: High replanning frequency
            replanning_count = sum(1 for step in steps if step.get('replanning_triggered', False))
            if replanning_count > len(steps) * 0.3:  # More than 30% replanning
                patterns.append(f"High replanning frequency: {replanning_count}/{len(steps)} steps")
            
        except Exception as e:
            self.logger.error(f"Error identifying failure patterns: {e}")
            patterns.append("Error analyzing failure patterns")
        
        return patterns
    
    def _identify_optimizations(self, log: dict, step_analysis: Dict[str, Any]) -> List[str]:
        """
        Identify optimization opportunities based on execution analysis.
        
        Args:
            log: Test execution log
            step_analysis: Step execution analysis results
            
        Returns:
            List of optimization opportunities
        """
        optimizations = []
        
        try:
            # Optimization 1: Reduce redundant steps
            steps = log.get('steps', [])
            if len(steps) > 10:
                optimizations.append("Consider breaking down long test sequences into smaller, focused tests")
            
            # Optimization 2: Improve low-success step types
            for step_type, data in step_analysis.get('step_types', {}).items():
                if data.get('success_rate', 1.0) < 0.7 and data.get('total', 0) > 2:
                    optimizations.append(f"Improve {step_type} actions (success rate: {data['success_rate']:.1%})")
            
            # Optimization 3: Reduce execution time
            metrics = self._extract_execution_metrics(log)
            avg_time = metrics.get('average_step_time', 0)
            if avg_time > 5.0:  # More than 5 seconds per step
                optimizations.append(f"Optimize step execution time (average: {avg_time:.1f}s per step)")
            
            # Optimization 4: Reduce verification failures
            total_steps = metrics.get('total_steps', 0)
            failed_steps = metrics.get('failed_steps', 0)
            if total_steps > 0 and failed_steps / total_steps > 0.2:
                optimizations.append("Improve verification accuracy or step reliability")
            
            # Optimization 5: Screen transition efficiency
            transitions = step_analysis.get('screen_transitions', [])
            failed_transitions = [t for t in transitions if not t.get('success', True)]
            if len(failed_transitions) > len(transitions) * 0.2:
                optimizations.append("Optimize screen navigation and transition handling")
            
        except Exception as e:
            self.logger.error(f"Error identifying optimizations: {e}")
            optimizations.append("Error analyzing optimization opportunities")
        
        return optimizations
    
    def _generate_planner_improvements(self, log: dict, step_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate specific improvements for the PlannerAgent.
        
        Args:
            log: Test execution log
            step_analysis: Step execution analysis results
            
        Returns:
            List of planner improvement suggestions
        """
        improvements = []
        
        try:
            # Improvement 1: Better step descriptions
            unclear_steps = []
            for step in log.get('steps', []):
                step_desc = step.get('step', '')
                if len(step_desc.split()) < 3 or 'unknown' in step_desc.lower():
                    unclear_steps.append(step_desc)
            
            if unclear_steps:
                improvements.append("Improve step descriptions for clarity and specificity")
            
            # Improvement 2: Step type optimization
            for step_type, data in step_analysis.get('step_types', {}).items():
                if data.get('success_rate', 1.0) < 0.6:
                    improvements.append(f"Refine {step_type} step planning (low success rate)")
            
            # Improvement 3: Goal decomposition
            total_steps = len(log.get('steps', []))
            if total_steps > 15:
                improvements.append("Break down complex goals into smaller, manageable sub-goals")
            
            # Improvement 4: Context awareness
            replanning_events = sum(1 for step in log.get('steps', []) if step.get('replanning_triggered', False))
            if replanning_events > 3:
                improvements.append("Improve initial planning to reduce need for replanning")
            
            # Improvement 5: Error handling
            common_failures = step_analysis.get('common_failures', {})
            if common_failures:
                most_common = max(common_failures.items(), key=lambda x: x[1])
                improvements.append(f"Add error handling for common issue: {most_common[0]}")
            
        except Exception as e:
            self.logger.error(f"Error generating planner improvements: {e}")
            improvements.append("Error analyzing planner improvements")
        
        return improvements
    
    def _generate_test_prompts(self, log: dict, failure_patterns: List[str]) -> List[str]:
        """
        Generate new test prompts based on execution analysis.
        
        Args:
            log: Test execution log
            failure_patterns: Identified failure patterns
            
        Returns:
            List of new test prompt suggestions
        """
        prompts = []
        
        try:
            # Extract original goal for context
            original_goal = log.get('goal', 'Unknown goal')
            
            # Prompt 1: Edge case variations
            if 'WiFi' in original_goal or 'wifi' in original_goal:
                prompts.extend([
                    "Test WiFi toggle when already connected to a network",
                    "Test WiFi settings access from notification panel",
                    "Test WiFi toggle with airplane mode enabled",
                    "Test WiFi reconnection after toggle off/on cycle"
                ])
            
            # Prompt 2: Error recovery scenarios
            if failure_patterns:
                prompts.extend([
                    "Test recovery from unexpected screen transitions",
                    "Test behavior when target elements are not found",
                    "Test handling of slow UI responses",
                    "Test interaction with disabled interface elements"
                ])
            
            # Prompt 3: Alternative paths
            steps = log.get('steps', [])
            navigation_steps = [s for s in steps if 'open' in s.get('step', '').lower()]
            if navigation_steps:
                prompts.extend([
                    "Test alternative navigation paths to the same goal",
                    "Test goal completion using voice commands",
                    "Test goal completion using gestures instead of taps"
                ])
            
            # Prompt 4: Stress testing
            if len(steps) > 5:
                prompts.extend([
                    "Test rapid sequential execution of the same goal",
                    "Test goal execution with multiple apps running",
                    "Test goal execution with low device memory"
                ])
            
            # Prompt 5: Negative testing
            prompts.extend([
                "Test invalid input handling in the workflow",
                "Test behavior when required permissions are denied",
                "Test workflow interruption and resumption"
            ])
            
        except Exception as e:
            self.logger.error(f"Error generating test prompts: {e}")
            prompts.append("Error generating test prompts")
        
        return prompts
    
    def _identify_edge_cases(self, log: dict, step_analysis: Dict[str, Any]) -> List[str]:
        """
        Identify potential edge cases not covered in current testing.
        
        Args:
            log: Test execution log
            step_analysis: Step execution analysis results
            
        Returns:
            List of edge case scenarios
        """
        edge_cases = []
        
        try:
            # Edge case 1: State dependencies
            goal = log.get('goal', '').lower()
            if 'wifi' in goal:
                edge_cases.extend([
                    "WiFi already enabled before test starts",
                    "Multiple WiFi networks available",
                    "WiFi toggle during active connection",
                    "WiFi settings access without admin permissions"
                ])
            
            # Edge case 2: UI state variations
            screen_transitions = step_analysis.get('screen_transitions', [])
            unique_screens = set()
            for transition in screen_transitions:
                screens = transition.get('transition', '').split(' → ')
                unique_screens.update(screens)
            
            for screen in unique_screens:
                if screen not in ['unknown']:
                    edge_cases.append(f"Test behavior when {screen} screen loads slowly")
            
            # Edge case 3: Concurrent operations
            if len(log.get('steps', [])) > 3:
                edge_cases.extend([
                    "Test during incoming call or notification",
                    "Test with device rotation during execution",
                    "Test with low battery warning active"
                ])
            
            # Edge case 4: Network conditions
            edge_cases.extend([
                "Test with weak network connectivity",
                "Test with airplane mode enabled",
                "Test during network switching (WiFi to cellular)"
            ])
            
            # Edge case 5: Accessibility scenarios
            edge_cases.extend([
                "Test with accessibility services enabled",
                "Test with large text size settings",
                "Test with high contrast mode enabled"
            ])
            
        except Exception as e:
            self.logger.error(f"Error identifying edge cases: {e}")
            edge_cases.append("Error identifying edge cases")
        
        return edge_cases
    
    def _generate_comprehensive_feedback(self, metrics: Dict[str, Any], step_analysis: Dict[str, Any], 
                                       failure_patterns: List[str]) -> str:
        """
        Generate comprehensive feedback text.
        
        Args:
            metrics: Execution metrics
            step_analysis: Step analysis results
            failure_patterns: Identified failure patterns
            
        Returns:
            Comprehensive feedback string
        """
        try:
            feedback_parts = []
            
            # Overall performance summary
            success_rate = metrics.get('success_rate', 0.0)
            total_steps = metrics.get('total_steps', 0)
            
            if success_rate >= 0.9:
                feedback_parts.append(f"Excellent test execution with {success_rate:.1%} success rate ({total_steps} steps).")
            elif success_rate >= 0.7:
                feedback_parts.append(f"Good test execution with {success_rate:.1%} success rate ({total_steps} steps).")
            elif success_rate >= 0.5:
                feedback_parts.append(f"Moderate test execution with {success_rate:.1%} success rate ({total_steps} steps). Improvement needed.")
            else:
                feedback_parts.append(f"Poor test execution with {success_rate:.1%} success rate ({total_steps} steps). Significant improvement required.")
            
            # Step type analysis
            step_types = step_analysis.get('step_types', {})
            if step_types:
                best_type = max(step_types.items(), key=lambda x: x[1].get('success_rate', 0))
                worst_type = min(step_types.items(), key=lambda x: x[1].get('success_rate', 1))
                
                feedback_parts.append(f"Best performing action type: {best_type[0]} ({best_type[1].get('success_rate', 0):.1%} success).")
                if worst_type[1].get('success_rate', 1) < 0.8:
                    feedback_parts.append(f"Needs improvement: {worst_type[0]} actions ({worst_type[1].get('success_rate', 0):.1%} success).")
            
            # Timing analysis
            avg_time = metrics.get('average_step_time', 0)
            if avg_time > 0:
                if avg_time < 2.0:
                    feedback_parts.append("Excellent execution speed.")
                elif avg_time < 5.0:
                    feedback_parts.append(f"Good execution speed ({avg_time:.1f}s per step).")
                else:
                    feedback_parts.append(f"Slow execution detected ({avg_time:.1f}s per step). Consider optimization.")
            
            # Failure pattern analysis
            if failure_patterns:
                feedback_parts.append(f"Identified {len(failure_patterns)} failure patterns requiring attention.")
            else:
                feedback_parts.append("No significant failure patterns detected.")
            
            # Replanning analysis
            replanning_events = metrics.get('replanning_events', 0)
            if replanning_events == 0:
                feedback_parts.append("Excellent planning - no replanning required.")
            elif replanning_events <= 2:
                feedback_parts.append(f"Minimal replanning required ({replanning_events} events).")
            else:
                feedback_parts.append(f"High replanning frequency ({replanning_events} events). Consider improving initial planning.")
            
            return " ".join(feedback_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating feedback: {e}")
            return "Error generating comprehensive feedback."
    
    def _generate_suggestions(self, step_analysis: Dict[str, Any], 
                            optimization_opportunities: List[str]) -> List[str]:
        """
        Generate actionable suggestions for improvement.
        
        Args:
            step_analysis: Step analysis results
            optimization_opportunities: Identified optimizations
            
        Returns:
            List of actionable suggestions
        """
        suggestions = []
        
        try:
            # Suggestion 1: Step-specific improvements
            step_types = step_analysis.get('step_types', {})
            for step_type, data in step_types.items():
                success_rate = data.get('success_rate', 1.0)
                if success_rate < 0.8 and data.get('total', 0) > 1:
                    suggestions.append(f"Focus on improving {step_type} action reliability")
            
            # Suggestion 2: Common failure mitigation
            common_failures = step_analysis.get('common_failures', {})
            if common_failures:
                top_failure = max(common_failures.items(), key=lambda x: x[1])
                suggestions.append(f"Implement specific handling for: {top_failure[0]}")
            
            # Suggestion 3: Optimization-based suggestions
            for optimization in optimization_opportunities:
                if "execution time" in optimization:
                    suggestions.append("Add wait optimization and parallel processing where possible")
                elif "verification" in optimization:
                    suggestions.append("Enhance verification strategies with more robust patterns")
                elif "transition" in optimization:
                    suggestions.append("Implement smart screen transition detection and handling")
            
            # Suggestion 4: Testing methodology
            suggestions.extend([
                "Implement pre-condition validation before test execution",
                "Add post-execution cleanup verification",
                "Create baseline performance benchmarks for comparison"
            ])
            
            # Suggestion 5: Error handling
            suggestions.extend([
                "Implement graceful degradation for failed steps",
                "Add automatic retry logic for transient failures",
                "Create detailed error context capture for debugging"
            ])
            
        except Exception as e:
            self.logger.error(f"Error generating suggestions: {e}")
            suggestions.append("Error generating suggestions")
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _calculate_overall_score(self, metrics: Dict[str, Any], step_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall execution score (0.0 to 1.0).
        
        Args:
            metrics: Execution metrics
            step_analysis: Step analysis results
            
        Returns:
            Overall score between 0.0 and 1.0
        """
        try:
            # Base score from success rate
            success_rate = metrics.get('success_rate', 0.0)
            base_score = success_rate
            
            # Adjust for execution efficiency
            avg_time = metrics.get('average_step_time', 5.0)
            if avg_time <= 2.0:
                time_multiplier = 1.0
            elif avg_time <= 5.0:
                time_multiplier = 0.9
            else:
                time_multiplier = 0.8
            
            # Adjust for replanning frequency
            total_steps = metrics.get('total_steps', 1)
            replanning_events = metrics.get('replanning_events', 0)
            replanning_ratio = replanning_events / total_steps
            
            if replanning_ratio <= 0.1:
                planning_multiplier = 1.0
            elif replanning_ratio <= 0.3:
                planning_multiplier = 0.9
            else:
                planning_multiplier = 0.8
            
            # Calculate final score
            overall_score = base_score * time_multiplier * planning_multiplier
            
            return min(max(overall_score, 0.0), 1.0)  # Clamp to [0.0, 1.0]
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.5  # Default neutral score
    
    def _calculate_efficiency_score(self, metrics: Dict[str, Any], step_analysis: Dict[str, Any]) -> float:
        """
        Calculate execution efficiency score (0.0 to 1.0).
        
        Args:
            metrics: Execution metrics
            step_analysis: Step analysis results
            
        Returns:
            Efficiency score between 0.0 and 1.0
        """
        try:
            # Time efficiency
            avg_time = metrics.get('average_step_time', 5.0)
            time_score = max(0.0, min(1.0, (10.0 - avg_time) / 8.0))  # Normalize to [0,1]
            
            # Step efficiency (fewer steps = better for same goal)
            total_steps = metrics.get('total_steps', 10)
            step_score = max(0.0, min(1.0, (20.0 - total_steps) / 15.0))  # Normalize to [0,1]
            
            # Replanning efficiency
            replanning_events = metrics.get('replanning_events', 0)
            replanning_score = max(0.0, 1.0 - (replanning_events * 0.1))
            
            # Combined efficiency score
            efficiency_score = (time_score * 0.4 + step_score * 0.3 + replanning_score * 0.3)
            
            return min(max(efficiency_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency score: {e}")
            return 0.5
    
    def _initialize_analysis_patterns(self) -> Dict[str, Any]:
        """
        Initialize analysis patterns for common issues.
        
        Returns:
            Dictionary of analysis patterns
        """
        return {
            'failure_indicators': [
                'timeout',
                'element not found',
                'screen transition failed',
                'verification failed',
                'unexpected state'
            ],
            'optimization_keywords': [
                'slow',
                'delay',
                'wait',
                'retry',
                'replanning'
            ],
            'success_indicators': [
                'pass',
                'successful',
                'completed',
                'verified',
                'achieved'
            ]
        }
    
    def _initialize_llm(self):
        """
        Initialize LLM configuration for advanced analysis.
        Note: This is a placeholder for LLM integration.
        """
        try:
            # Placeholder for LLM initialization
            # In a real implementation, you would initialize your LLM client here
            # e.g., OpenAI API, local model, etc.
            self.logger.info(f"[{self.name}] LLM mode enabled - using mock implementation")
            
            # Mock LLM configuration
            self.llm_config = {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 1000
            }
            
        except Exception as e:
            self.logger.warning(f"[{self.name}] LLM initialization failed: {e}")
            self.use_llm = False
    
    def _create_error_review(self, error_message: str) -> dict:
        """
        Create error review result.
        
        Args:
            error_message: Error description
            
        Returns:
            Error review dictionary
        """
        return {
            'overall_score': 0.0,
            'success_rate': 0.0,
            'efficiency_score': 0.0,
            'feedback': f"Review failed: {error_message}",
            'suggestions': ['Fix review system error'],
            'planner_improvements': ['Address review system issues'],
            'new_test_prompts': ['Test review system functionality'],
            'edge_cases': ['Review system error scenarios'],
            'step_analysis': {},
            'failure_patterns': [f"Review error: {error_message}"],
            'optimization_opportunities': ['Fix review system'],
            'review_timestamp': datetime.now().isoformat(),
            'agent_version': "1.0.0"
        }
    
    def _update_statistics(self, review_result: ReviewResult, log: dict):
        """Update supervisor statistics."""
        self.stats['reviews_performed'] += 1
        self.stats['total_tests_analyzed'] += 1
        
        # Update averages
        prev_avg_success = self.stats['average_success_rate']
        prev_avg_efficiency = self.stats['average_efficiency']
        
        self.stats['average_success_rate'] = (
            (prev_avg_success * (self.stats['reviews_performed'] - 1) + review_result.success_rate) /
            self.stats['reviews_performed']
        )
        
        self.stats['average_efficiency'] = (
            (prev_avg_efficiency * (self.stats['reviews_performed'] - 1) + review_result.efficiency_score) /
            self.stats['reviews_performed']
        )
        
        # Count recommendations
        total_recommendations = (
            len(review_result.suggestions) +
            len(review_result.planner_improvements) +
            len(review_result.new_test_prompts) +
            len(review_result.edge_cases)
        )
        self.stats['recommendations_generated'] += total_recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get supervisor statistics.
        
        Returns:
            Dictionary of supervisor statistics
        """
        return {
            'agent_name': self.name,
            'reviews_performed': self.stats['reviews_performed'],
            'total_tests_analyzed': self.stats['total_tests_analyzed'],
            'average_success_rate': self.stats['average_success_rate'],
            'average_efficiency': self.stats['average_efficiency'],
            'recommendations_generated': self.stats['recommendations_generated'],
            'llm_enabled': self.use_llm
        }
    
    def reset_statistics(self):
        """Reset supervisor statistics."""
        self.stats = {
            'reviews_performed': 0,
            'total_tests_analyzed': 0,
            'average_success_rate': 0.0,
            'average_efficiency': 0.0,
            'recommendations_generated': 0
        }
        self.logger.info(f"[{self.name}] Statistics reset")
    
    def __str__(self):
        """String representation of the agent."""
        stats = self.get_statistics()
        return (f"SupervisorAgent(name={self.name}, "
                f"reviews={stats['reviews_performed']}, "
                f"avg_success={stats['average_success_rate']:.1%})")


def main():
    """
    Demo of SupervisorAgent functionality.
    """
    print("[CROWN] SUPERVISOR AGENT DEMO")
    print("=" * 50)
    
    # Initialize SupervisorAgent
    supervisor = SupervisorAgent(name="DemoSupervisor", use_llm=False, debug=True)
    print(f"[OK] Initialized: {supervisor}")
    
    # Create mock test execution log
    mock_log = {
        'goal': 'Test turning Wi-Fi on and off',
        'start_time': '2025-07-26T09:39:05.123456',
        'end_time': '2025-07-26T09:39:25.987654',
        'steps': [
            {
                'step': 'Open settings app',
                'verification_result': 'pass',
                'start_time': '2025-07-26T09:39:05.123456',
                'end_time': '2025-07-26T09:39:07.456789',
                'observation_before': {'screen': 'home'},
                'observation_after': {'screen': 'settings'},
                'replanning_triggered': False,
                'verification_details': {'reason': 'Screen transition successful'}
            },
            {
                'step': 'Tap Wi-Fi toggle',
                'verification_result': 'pass',
                'start_time': '2025-07-26T09:39:07.456789',
                'end_time': '2025-07-26T09:39:09.123456',
                'observation_before': {'screen': 'settings', 'wifi_enabled': False},
                'observation_after': {'screen': 'settings', 'wifi_enabled': True},
                'replanning_triggered': False,
                'verification_details': {'reason': 'WiFi toggle successful'}
            },
            {
                'step': 'Tap Wi-Fi toggle again',
                'verification_result': 'fail',
                'start_time': '2025-07-26T09:39:09.123456',
                'end_time': '2025-07-26T09:39:12.789012',
                'observation_before': {'screen': 'settings', 'wifi_enabled': True},
                'observation_after': {'screen': 'settings', 'wifi_enabled': True},
                'replanning_triggered': True,
                'verification_details': {'reason': 'WiFi state did not change'}
            },
            {
                'step': 'Retry Wi-Fi toggle',
                'verification_result': 'pass',
                'start_time': '2025-07-26T09:39:12.789012',
                'end_time': '2025-07-26T09:39:14.345678',
                'observation_before': {'screen': 'settings', 'wifi_enabled': True},
                'observation_after': {'screen': 'settings', 'wifi_enabled': False},
                'replanning_triggered': False,
                'verification_details': {'reason': 'WiFi toggle successful after retry'}
            }
        ]
    }
    
    # Perform review
    print(f"\n[SEARCH] Reviewing test execution:")
    print(f"  Goal: {mock_log['goal']}")
    print(f"  Steps: {len(mock_log['steps'])}")
    
    review_result = supervisor.review(mock_log)
    
    print(f"\n[RESULTS] Review Results:")
    print("-" * 30)
    print(f"Overall Score: {review_result['overall_score']:.2f}")
    print(f"Success Rate: {review_result['success_rate']:.1%}")
    print(f"Efficiency Score: {review_result['efficiency_score']:.2f}")
    
    print(f"\n[CHAT] Feedback:")
    print(f"  {review_result['feedback']}")
    
    print(f"\n[BULB] Suggestions ({len(review_result['suggestions'])}):")
    for i, suggestion in enumerate(review_result['suggestions'][:5], 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\n[TARGET] Planner Improvements ({len(review_result['planner_improvements'])}):")
    for i, improvement in enumerate(review_result['planner_improvements'][:3], 1):
        print(f"  {i}. {improvement}")
    
    print(f"\n[TEST] New Test Prompts ({len(review_result['new_test_prompts'])}):")
    for i, prompt in enumerate(review_result['new_test_prompts'][:3], 1):
        print(f"  {i}. {prompt}")
    
    print(f"\n[WARNING] Edge Cases ({len(review_result['edge_cases'])}):")
    for i, edge_case in enumerate(review_result['edge_cases'][:3], 1):
        print(f"  {i}. {edge_case}")
    
    # Show supervisor statistics
    print(f"\n[CHART] Supervisor Statistics:")
    print("-" * 30)
    stats = supervisor.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n[CROWN] SupervisorAgent demo complete!")


if __name__ == "__main__":
    main()
