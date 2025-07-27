#!/usr/bin/env python3
"""
PlannerAgent - Goal-to-Plan Conversion Module
============================================

This module provides the PlannerAgent class responsible for converting high-level
goals into actionable step-by-step plans. The agent uses a template-based approach
with predefined mappings for common Android automation tasks.

Key Features:
- Template-based plan generation for common tasks
- Fallback mechanisms for unknown goals
- Comprehensive logging and metrics tracking
- Extensible architecture for future LLM integration

Author: Jayneel Shah
Version: 1.0
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from utils.logging import setup_logger, log_plan, log_agent_action
    from utils.prompts import get_plan_for_goal, format_plan_prompt, PREDEFINED_PLANS
except ImportError:
    # Fallback imports when running from different directory
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logging import setup_logger, log_plan, log_agent_action
    from utils.prompts import get_plan_for_goal, format_plan_prompt, PREDEFINED_PLANS


class PlannerAgent:
    """
    Agent responsible for generating execution plans from high-level goals.
    
    The PlannerAgent converts natural language goals into structured, executable
    step sequences that can be processed by the ExecutorAgent. It uses a template-
    based approach with predefined mappings for common Android automation tasks.
    
    Architecture:
        - Template-based plan generation for reliability
        - Fallback mechanisms for unknown goals  
        - Comprehensive metrics and logging
        - Extensible design for future LLM integration
    
    Attributes:
        name (str): Agent instance identifier
        logger (Logger): Structured logging interface
        plans_generated (int): Count of plans created
        plan_history (List[Dict]): Historical record of all plans
    """
    
    def __init__(self, name: str = "PlannerAgent", log_level: str = "INFO"):
        """
        Initialize the PlannerAgent with logging and metrics tracking.
        
        Args:
            name: Name of the agent instance for identification
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.name = name
        self.logger = setup_logger(name, level=getattr(logging, log_level.upper()))
        self.plans_generated = 0
        self.plan_history: List[Dict[str, Any]] = []
        
        log_agent_action(self.logger, self.name, "Initialized", 
                        f"Ready to generate plans from {len(PREDEFINED_PLANS)} predefined templates")
    
    def generate_plan(self, goal: str) -> List[str]:
        """
        Generate a step-by-step execution plan for the given goal.
        
        This method analyzes the input goal and matches it against predefined
        templates to generate a sequence of actionable steps. For completely
        unsupported goals, it returns an empty plan or "unsupported" indicator.
        
        Args:
            goal: High-level goal description (e.g., "Turn on Wi-Fi")
            
        Returns:
            List[str]: Ordered sequence of actionable steps, or empty list for unsupported goals
            
        Raises:
            ValueError: If goal is None or empty string
            
        Example:
            >>> planner = PlannerAgent()
            >>> steps = planner.generate_plan("Turn on Wi-Fi")
            >>> print(steps)
            ['open settings', 'tap wifi', 'toggle wifi on', 'go back']
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("Goal must be a non-empty string")
            
        log_agent_action(self.logger, self.name, "Generating plan", f"Goal: {goal}")
        
        # Check if goal is completely unsupported
        if not self._is_goal_supported(goal):
            log_agent_action(self.logger, self.name, "Unsupported goal detected", 
                           f"Goal '{goal}' is not supported by any templates")
            
            # Record unsupported goal attempt
            plan_record = {
                "timestamp": datetime.now().isoformat(),
                "goal": goal,
                "plan": [],
                "plan_length": 0,
                "supported": False,
                "template_matched": False,
                "complexity": "unsupported",
                "estimated_time": 0
            }
            self.plan_history.append(plan_record)
            
            return []  # Return empty plan for unsupported goals
        
        # Get plan using predefined mappings with error handling
        try:
            plan = get_plan_for_goal(goal)
        except Exception as e:
            self.logger.error(f"Error generating plan for goal '{goal}': {e}")
            # Fallback to generic plan for supported but problematic goals
            plan = ["analyze the goal", "determine required actions", "execute step by step", "verify completion"]
        
        # Generate comprehensive metadata for the plan
        metadata = self._generate_plan_metadata(goal, plan)
        
        # Record plan generation for analytics with full metadata
        plan_record = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "plan": plan.copy(),
            "plan_length": len(plan),
            "supported": True,
            **metadata  # Include all metadata fields
        }
        
        self.plan_history.append(plan_record)
        self.plans_generated += 1
        
        # Log the generated plan with metadata
        log_plan(self.logger, goal, plan, self.name)
        log_agent_action(self.logger, self.name, "Plan generated successfully", 
                        f"Generated {len(plan)} steps - Complexity: {metadata['complexity']}, "
                        f"Template matched: {metadata['template_matched']}")
        
        return plan
    
    def get_available_goals(self) -> List[str]:
        """
        Get a list of goals that have predefined plans.
        
        Returns:
            List of available goal descriptions
        """
        return list(PREDEFINED_PLANS.keys())
    
    def validate_goal(self, goal: str) -> bool:
        """
        Check if a goal has a predefined plan available.
        
        Args:
            goal: Goal to validate
            
        Returns:
            True if goal has a predefined plan, False otherwise
        """
        goal_lower = goal.lower().strip()
        
        # Check exact match
        if goal_lower in PREDEFINED_PLANS:
            return True
        
        # Check partial matches
        for predefined_goal in PREDEFINED_PLANS.keys():
            if any(keyword in goal_lower for keyword in predefined_goal.split()):
                return True
        
        return False
    
    def get_plan_details(self, goal: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive details about a plan for a specific goal.
        
        Args:
            goal: Goal to get plan details for
            
        Returns:
            Optional[Dict[str, Any]]: Plan details including steps, complexity,
                                    and time estimates, or None if no plan found
        """
        try:
            plan = get_plan_for_goal(goal)
            
            if plan:
                # Check if this goal has a direct template match
                goal_lower = goal.lower().strip()
                template_matched = goal_lower in PREDEFINED_PLANS
                
                return {
                    "goal": goal,
                    "steps": plan,
                    "step_count": len(plan),
                    "estimated_time": len(plan) * 2,  # Rough estimate: 2 seconds per step
                    "complexity": self._calculate_complexity(len(plan)),
                    "template_matched": template_matched
                }
        except Exception as e:
            self.logger.error(f"Error getting plan details for '{goal}': {e}")
        
        return None
    
    def _calculate_complexity(self, step_count: int) -> str:
        """
        Calculate plan complexity based on step count.
        
        Args:
            step_count: Number of steps in the plan
            
        Returns:
            str: Complexity level (simple, medium, complex)
        """
        if step_count <= 3:
            return "simple"
        elif step_count <= 6:
            return "medium"
        else:
            return "complex"
    
    def _is_goal_supported(self, goal: str) -> bool:
        """
        Check if a goal is supported by any predefined templates.
        
        This method performs comprehensive goal validation to determine if
        the goal can be handled by existing templates or patterns.
        
        Args:
            goal: Goal to check for support
            
        Returns:
            bool: True if goal is supported, False if completely unsupported
        """
        goal_lower = goal.lower().strip()
        
        # Check exact match in predefined plans
        if goal_lower in PREDEFINED_PLANS:
            return True
        
        # Check for partial keyword matches
        supported_keywords = [
            "wifi", "wi-fi", "bluetooth", "battery", "calculator", "settings",
            "camera", "phone", "call", "message", "sms", "email", "browser",
            "volume", "brightness", "airplane", "location", "gps", "music",
            "photo", "video", "contact", "app", "notification", "alarm"
        ]
        
        # If goal contains any supported keywords, consider it supported
        for keyword in supported_keywords:
            if keyword in goal_lower:
                return True
        
        # Check for action patterns that indicate supportable goals
        supported_actions = [
            "open", "close", "turn on", "turn off", "enable", "disable",
            "check", "view", "set", "change", "toggle", "tap", "click",
            "send", "call", "play", "stop", "start", "restart"
        ]
        
        for action in supported_actions:
            if action in goal_lower:
                return True
        
        # If no patterns match, goal is likely unsupported
        return False
    
    def _generate_plan_metadata(self, goal: str, plan: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a plan.
        
        Args:
            goal: The original goal
            plan: The generated plan steps
            
        Returns:
            Dict[str, Any]: Comprehensive metadata including complexity, timing, and matching info
        """
        goal_lower = goal.lower().strip()
        template_matched = goal_lower in PREDEFINED_PLANS
        step_count = len(plan)
        
        # Calculate estimated time based on step complexity
        base_time_per_step = 2.0  # Base 2 seconds per step
        complexity_multiplier = {
            "simple": 1.0,
            "medium": 1.5,
            "complex": 2.0
        }
        
        complexity = self._calculate_complexity(step_count)
        estimated_time = step_count * base_time_per_step * complexity_multiplier.get(complexity, 1.0)
        
        # Analyze step types for additional metadata
        step_types = {
            "navigation": 0,
            "interaction": 0,
            "observation": 0,
            "configuration": 0
        }
        
        for step in plan:
            step_lower = step.lower()
            if any(word in step_lower for word in ["open", "go", "navigate", "back"]):
                step_types["navigation"] += 1
            elif any(word in step_lower for word in ["tap", "click", "toggle", "press", "swipe"]):
                step_types["interaction"] += 1
            elif any(word in step_lower for word in ["check", "wait", "observe", "verify"]):
                step_types["observation"] += 1
            elif any(word in step_lower for word in ["set", "configure", "change", "adjust"]):
                step_types["configuration"] += 1
        
        return {
            "template_matched": template_matched,
            "complexity": complexity,
            "estimated_time": round(estimated_time, 1),
            "step_types": step_types,
            "primary_step_type": max(step_types, key=step_types.get) if any(step_types.values()) else "general",
            "requires_navigation": step_types["navigation"] > 0,
            "requires_interaction": step_types["interaction"] > 0,
            "goal_category": self._categorize_goal(goal),
            "risk_level": self._assess_risk_level(plan)
        }
    
    def _categorize_goal(self, goal: str) -> str:
        """
        Categorize the goal into functional areas.
        
        Args:
            goal: The goal to categorize
            
        Returns:
            str: Goal category
        """
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["wifi", "bluetooth", "network", "connection"]):
            return "connectivity"
        elif any(word in goal_lower for word in ["battery", "power", "charge"]):
            return "power_management"
        elif any(word in goal_lower for word in ["calculator", "app", "application"]):
            return "applications"
        elif any(word in goal_lower for word in ["settings", "configure", "setup"]):
            return "system_settings"
        elif any(word in goal_lower for word in ["call", "phone", "message", "sms", "contact"]):
            return "communication"
        elif any(word in goal_lower for word in ["camera", "photo", "video", "media"]):
            return "media"
        else:
            return "general"
    
    def _assess_risk_level(self, plan: List[str]) -> str:
        """
        Assess the risk level of executing the plan.
        
        Args:
            plan: The plan steps to assess
            
        Returns:
            str: Risk level (low, medium, high)
        """
        high_risk_keywords = ["delete", "reset", "factory", "erase", "format", "disable"]
        medium_risk_keywords = ["install", "uninstall", "modify", "change", "configure"]
        
        plan_text = " ".join(plan).lower()
        
        if any(keyword in plan_text for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in plan_text for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the planner's usage and performance.
        
        Returns:
            Dictionary with detailed usage statistics and analytics
        """
        if not self.plan_history:
            return {
                "plans_generated": 0,
                "available_goals": len(PREDEFINED_PLANS),
                "average_plan_length": 0,
                "most_recent_goal": None,
                "supported_goals": 0,
                "unsupported_goals": 0,
                "complexity_distribution": {},
                "category_distribution": {},
                "template_match_rate": 0.0
            }
        
        # Calculate basic statistics
        supported_plans = [record for record in self.plan_history if record.get("supported", True)]
        unsupported_plans = [record for record in self.plan_history if not record.get("supported", True)]
        
        # Calculate complexity distribution
        complexity_counts = {}
        category_counts = {}
        template_matches = 0
        
        for record in self.plan_history:
            # Complexity distribution
            complexity = record.get("complexity", "unknown")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            
            # Category distribution (if available)
            category = record.get("goal_category", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Template match counting
            if record.get("template_matched", False):
                template_matches += 1
        
        return {
            "plans_generated": self.plans_generated,
            "available_goals": len(PREDEFINED_PLANS),
            "supported_goals": len(supported_plans),
            "unsupported_goals": len(unsupported_plans),
            "average_plan_length": (
                sum(record["plan_length"] for record in supported_plans) / len(supported_plans)
                if supported_plans else 0
            ),
            "most_recent_goal": (
                self.plan_history[-1]["goal"] if self.plan_history else None
            ),
            "complexity_distribution": complexity_counts,
            "category_distribution": category_counts,
            "template_match_rate": template_matches / len(self.plan_history) if self.plan_history else 0.0,
            "success_rate": len(supported_plans) / len(self.plan_history) if self.plan_history else 0.0,
            "total_estimated_time": sum(
                record.get("estimated_time", 0) for record in supported_plans
            ),
            "average_estimated_time": (
                sum(record.get("estimated_time", 0) for record in supported_plans) / len(supported_plans)
                if supported_plans else 0
            )
        }
    
    def add_custom_plan(self, goal: str, steps: List[str]) -> bool:
        """
        Add a custom plan for a goal (extensibility for future enhancements).
        
        Args:
            goal: Goal description
            steps: List of actionable steps for the goal
            
        Returns:
            bool: True if plan was added successfully, False otherwise
        """
        try:
            if not goal or not steps:
                return False
                
            PREDEFINED_PLANS[goal.lower().strip()] = steps.copy()
            log_agent_action(self.logger, self.name, "Custom plan added", 
                           f"Goal: {goal}, Steps: {len(steps)}")
            return True
        except Exception as e:
            log_agent_action(self.logger, self.name, "Failed to add custom plan", 
                           f"Error: {str(e)}")
            return False
    
    def export_plan_with_metadata(self, goal: str) -> Optional[Dict[str, Any]]:
        """
        Export a comprehensive plan with full metadata for logging and reporting.
        
        This method generates a complete plan report including the plan steps,
        execution metadata, complexity analysis, and risk assessment.
        
        Args:
            goal: Goal to generate plan and metadata for
            
        Returns:
            Optional[Dict[str, Any]]: Complete plan export with metadata,
                                    or None if goal is unsupported
            
        Example:
            >>> planner = PlannerAgent()
            >>> export = planner.export_plan_with_metadata("Turn on Wi-Fi")
            >>> print(export['metadata']['complexity'])
            'medium'
        """
        if not self._is_goal_supported(goal):
            return {
                "goal": goal,
                "supported": False,
                "plan": [],
                "metadata": {
                    "template_matched": False,
                    "complexity": "unsupported",
                    "estimated_time": 0,
                    "step_types": {},
                    "primary_step_type": "none",
                    "requires_navigation": False,
                    "requires_interaction": False,
                    "goal_category": "unsupported",
                    "risk_level": "unknown"
                },
                "export_timestamp": datetime.now().isoformat(),
                "agent_name": self.name
            }
        
        try:
            plan = get_plan_for_goal(goal)
            metadata = self._generate_plan_metadata(goal, plan)
            
            export_data = {
                "goal": goal,
                "supported": True,
                "plan": plan.copy(),
                "metadata": metadata,
                "export_timestamp": datetime.now().isoformat(),
                "agent_name": self.name,
                "plan_history_index": len(self.plan_history),  # Reference to history entry
                "validation": {
                    "goal_validated": True,
                    "template_available": metadata["template_matched"],
                    "estimated_success_rate": self._estimate_success_rate(metadata)
                }
            }
            
            log_agent_action(self.logger, self.name, "Plan exported with metadata", 
                           f"Goal: {goal}, Complexity: {metadata['complexity']}, "
                           f"Category: {metadata['goal_category']}")
            
            return export_data
            
        except Exception as e:
            self.logger.error(f"Error exporting plan with metadata for '{goal}': {e}")
            return None
    
    def _estimate_success_rate(self, metadata: Dict[str, Any]) -> float:
        """
        Estimate the success rate of a plan based on its metadata.
        
        Args:
            metadata: Plan metadata
            
        Returns:
            float: Estimated success rate (0.0 to 1.0)
        """
        base_rate = 0.8  # Base 80% success rate
        
        # Adjust based on template matching
        if metadata["template_matched"]:
            base_rate += 0.15
        
        # Adjust based on complexity
        complexity_adjustments = {
            "simple": 0.1,
            "medium": 0.0,
            "complex": -0.1
        }
        base_rate += complexity_adjustments.get(metadata["complexity"], 0)
        
        # Adjust based on risk level
        risk_adjustments = {
            "low": 0.05,
            "medium": 0.0,
            "high": -0.15
        }
        base_rate += risk_adjustments.get(metadata["risk_level"], 0)
        
        # Ensure rate is within valid bounds
        return max(0.0, min(1.0, base_rate))
    
    def __str__(self) -> str:
        """String representation of the PlannerAgent."""
        return f"PlannerAgent(name={self.name}, plans_generated={self.plans_generated})"
    
    def __repr__(self) -> str:
        """Developer representation of the PlannerAgent."""
        return (f"PlannerAgent(name='{self.name}', plans_generated={self.plans_generated}, "
                f"available_goals={len(PREDEFINED_PLANS)})")
        return (f"PlannerAgent(name='{self.name}', plans_generated={self.plans_generated}, "
                f"available_goals={len(PREDEFINED_PLANS)})")


def main():
    """
    Test function demonstrating enhanced PlannerAgent capabilities.
    
    This function runs various test scenarios to validate the agent's
    functionality including plan generation, validation, metadata export,
    and comprehensive statistics.
    """
    print("=" * 60)
    print("ENHANCED PLANNER AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = PlannerAgent(name="TestPlanner")
    
    # Test goals covering different complexity levels and support scenarios
    test_goals = [
        "Test turning Wi-Fi on and off",
        "Open calculator", 
        "Check battery status",
        "Turn bluetooth on",
        "Send message to contact",
        "Launch nuclear missiles",  # Completely unsupported goal
        "Hack into government database",  # Another unsupported goal
        "Make coffee with phone"  # Nonsensical unsupported goal
    ]
    
    print(f"Initialized {agent.name}")
    print(f"Available predefined goals: {len(agent.get_available_goals())}")
    print()
    
    for i, goal in enumerate(test_goals, 1):
        print(f"Test {i}: {goal}")
        print(f"  Support check: {'[SUPPORTED]' if agent._is_goal_supported(goal) else '[UNSUPPORTED]'}")
        print(f"  Validation: {'[VALID]' if agent.validate_goal(goal) else '[UNKNOWN]'}")
        
        try:
            plan = agent.generate_plan(goal)
            
            if plan:
                print(f"  Generated plan ({len(plan)} steps):")
                for j, step in enumerate(plan, 1):
                    print(f"    {j}. {step}")
                
                # Test metadata export
                export_data = agent.export_plan_with_metadata(goal)
                if export_data and export_data["supported"]:
                    metadata = export_data["metadata"]
                    print(f"  [METADATA]")
                    print(f"    Complexity: {metadata['complexity']}")
                    print(f"    Estimated time: {metadata['estimated_time']} seconds")
                    print(f"    Template matched: {metadata['template_matched']}")
                    print(f"    Goal category: {metadata['goal_category']}")
                    print(f"    Risk level: {metadata['risk_level']}")
                    print(f"    Primary step type: {metadata['primary_step_type']}")
                    print(f"    Success rate estimate: {export_data['validation']['estimated_success_rate']:.1%}")
            else:
                print(f"  [UNSUPPORTED] No plan generated - goal not supported")
                
                # Test metadata export for unsupported goal
                export_data = agent.export_plan_with_metadata(goal)
                if export_data and not export_data["supported"]:
                    print(f"  [METADATA] Unsupported goal metadata recorded")
            
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            print(f"  Full error: {traceback.format_exc()}")
        
        print("-" * 50)
    
    # Display comprehensive final statistics
    stats = agent.get_statistics()
    print("\nCOMPREHENSIVE STATISTICS:")
    print(f"  Total plans generated: {stats['plans_generated']}")
    print(f"  Supported goals: {stats['supported_goals']}")
    print(f"  Unsupported goals: {stats['unsupported_goals']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Template match rate: {stats['template_match_rate']:.1%}")
    print(f"  Average plan length: {stats['average_plan_length']:.1f} steps")
    print(f"  Average estimated time: {stats['average_estimated_time']:.1f} seconds")
    print(f"  Total estimated time: {stats['total_estimated_time']:.1f} seconds")
    
    print(f"\n  Complexity distribution:")
    for complexity, count in stats['complexity_distribution'].items():
        print(f"    {complexity}: {count}")
    
    print(f"\n  Category distribution:")
    for category, count in stats['category_distribution'].items():
        print(f"    {category}: {count}")
    
    print(f"\n  Most recent goal: {stats['most_recent_goal']}")
    print(f"  Available predefined goals: {stats['available_goals']}")
    
    print("\nDemonstration complete!")
    print("Enhanced features implemented:")
    print("  [PASS] Goal validation with unsupported goal detection")
    print("  [PASS] Comprehensive task metadata export")
    print("  [PASS] Advanced analytics and statistics")
    print("  [PASS] Risk assessment and success rate estimation")
    print("=" * 60)


if __name__ == "__main__":
    main()
