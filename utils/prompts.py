"""
Prompt templates and reusable strings for the agent system.
Contains predefined plans and templates for LLM interactions.
"""

from typing import Dict, List

# Predefined goal-to-steps mappings for MVP implementation
PREDEFINED_PLANS: Dict[str, List[str]] = {
    "test turning wi-fi on and off": [
        "open settings",
        "tap wifi",
        "toggle wifi off",
        "wait 2 seconds",
        "toggle wifi on",
        "go back"
    ],
    
    "check wifi status": [
        "open settings",
        "tap wifi",
        "check wifi status",
        "go back"
    ],
    
    "open calculator": [
        "find calculator app",
        "tap calculator app",
        "wait for app to load"
    ],
    
    "open calculator and make a simple calculation": [
        "find calculator app",
        "tap calculator app",
        "wait for app to load",
        "tap number 5",
        "tap plus",
        "tap number 3",
        "tap equals",
        "check result"
    ],
    
    "make a simple calculation": [
        "open calculator",
        "tap number 5",
        "tap plus",
        "tap number 3",
        "tap equals",
        "check result"
    ],
    
    "check device storage": [
        "open settings",
        "scroll to storage",
        "tap storage",
        "check available space",
        "go back"
    ],
    
    "adjust screen brightness": [
        "pull down notification panel",
        "adjust brightness slider",
        "close notification panel"
    ],
    
    "check battery status": [
        "open settings",
        "tap battery",
        "check battery percentage",
        "check battery health",
        "go back"
    ],
    
    "turn bluetooth on": [
        "open settings",
        "tap bluetooth",
        "toggle bluetooth on",
        "go back"
    ],
    
    "turn bluetooth off": [
        "open settings",
        "tap bluetooth",
        "toggle bluetooth off",
        "go back"
    ],
    
    "check current time": [
        "observe status bar",
        "note current time"
    ]
}

# Prompt templates for future LLM integration
PLANNER_SYSTEM_PROMPT = """
You are an expert Android automation planner. Your job is to break down high-level goals into specific, actionable steps that can be executed on an Android device.

Guidelines:
1. Be specific and atomic - each step should be a single, clear action
2. Use standard Android UI terms (tap, swipe, scroll, toggle, etc.)
3. Consider the typical Android settings and app structure
4. Include verification steps when needed
5. Keep steps simple and executable

Example:
Goal: "Turn on airplane mode"
Plan:
1. open settings
2. scroll to network & internet
3. tap airplane mode
4. toggle airplane mode on
5. go back

Now generate a plan for the given goal.
"""

USER_PROMPT_TEMPLATE = """
Goal: {goal}

Please provide a step-by-step plan to accomplish this goal on an Android device.
Return only the numbered steps, one per line.
"""

FALLBACK_PLAN = [
    "analyze the goal",
    "determine required actions",
    "execute step by step",
    "verify completion"
]

def get_plan_for_goal(goal: str) -> List[str]:
    """
    Get a predefined plan for a goal, or return a fallback plan.
    
    Args:
        goal: The goal to find a plan for
    
    Returns:
        List of steps for the goal
    """
    goal_lower = goal.lower().strip()
    
    # Try exact match first
    if goal_lower in PREDEFINED_PLANS:
        return PREDEFINED_PLANS[goal_lower].copy()
    
    # Try partial matches
    for predefined_goal, plan in PREDEFINED_PLANS.items():
        if any(keyword in goal_lower for keyword in predefined_goal.split()):
            return plan.copy()
    
    # Return fallback plan
    return FALLBACK_PLAN.copy()

def format_plan_prompt(goal: str) -> str:
    """
    Format a goal into a prompt template for LLM interaction.
    
    Args:
        goal: The goal to format
    
    Returns:
        Formatted prompt string
    """
    return USER_PROMPT_TEMPLATE.format(goal=goal)
