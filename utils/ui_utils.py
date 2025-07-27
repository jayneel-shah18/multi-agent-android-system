"""
UI utilities for Android automation.
Provides helpers for UI element searching, matching, and coordinate conversion.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class UIElement:
    """Represents a UI element with its properties and location."""
    text: str = ""
    content_desc: str = ""
    class_name: str = ""
    resource_id: str = ""
    is_clickable: bool = False
    is_enabled: bool = True
    is_focusable: bool = False
    is_scrollable: bool = False
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (left, top, right, bottom)
    center_x: float = 0.0
    center_y: float = 0.0
    
    def __post_init__(self):
        """Calculate center coordinates from bounds."""
        if self.bounds != (0, 0, 0, 0):
            left, top, right, bottom = self.bounds
            self.center_x = (left + right) / 2
            self.center_y = (top + bottom) / 2

def normalize_coordinates(x: float, y: float, screen_width: int = 1080, screen_height: int = 1920) -> Tuple[float, float]:
    """
    Convert absolute coordinates to normalized coordinates [0, 1].
    
    Args:
        x: Absolute x coordinate
        y: Absolute y coordinate
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        
    Returns:
        Tuple of normalized (x, y) coordinates
    """
    norm_x = max(0.0, min(1.0, x / screen_width))
    norm_y = max(0.0, min(1.0, y / screen_height))
    return norm_x, norm_y

def parse_ui_elements_from_observation(observation: Dict[str, Any]) -> List[UIElement]:
    """
    Parse UI elements from an observation dictionary.
    Handles both mock observations and real android_world observations.
    
    Args:
        observation: Observation dictionary from environment
        
    Returns:
        List of UIElement objects
    """
    elements = []
    
    # Handle mock structured observations
    if "structured" in observation:
        structured = observation["structured"]
        if isinstance(structured, dict) and "elements" in structured:
            for elem_data in structured["elements"]:
                element = UIElement(
                    text=elem_data.get("text", ""),
                    content_desc=elem_data.get("content_desc", ""),
                    class_name=elem_data.get("class_name", ""),
                    resource_id=elem_data.get("resource_id", ""),
                    is_clickable=elem_data.get("is_clickable", False),
                    is_enabled=elem_data.get("is_enabled", True),
                    bounds=tuple(elem_data.get("bounds", [0, 0, 0, 0]))
                )
                elements.append(element)
    
    # Handle android_world ui_elements format
    elif "ui_elements" in observation:
        ui_elements = observation["ui_elements"]
        for elem in ui_elements:
            # Handle both dict and object formats
            if hasattr(elem, 'text'):
                # Object format
                element = UIElement(
                    text=getattr(elem, 'text', ''),
                    content_desc=getattr(elem, 'content_description', ''),
                    class_name=getattr(elem, 'class_name', ''),
                    resource_id=getattr(elem, 'resource_id', ''),
                    is_clickable=getattr(elem, 'is_clickable', False),
                    is_enabled=getattr(elem, 'is_enabled', True),
                    bounds=getattr(elem, 'bounds', (0, 0, 0, 0))
                )
            else:
                # Dict format
                element = UIElement(
                    text=elem.get('text', ''),
                    content_desc=elem.get('content_description', ''),
                    class_name=elem.get('class_name', ''),
                    resource_id=elem.get('resource_id', ''),
                    is_clickable=elem.get('is_clickable', False),
                    is_enabled=elem.get('is_enabled', True),
                    bounds=tuple(elem.get('bounds', [0, 0, 0, 0]))
                )
            elements.append(element)
    
    return elements

def find_elements_by_text(elements: List[UIElement], text: str, fuzzy: bool = True) -> List[UIElement]:
    """
    Find UI elements by text content with enhanced matching for numbers and partial text.
    
    Args:
        elements: List of UI elements to search
        text: Text to search for
        fuzzy: If True, use substring matching; if False, exact match
        
    Returns:
        List of matching elements
    """
    matches = []
    search_text = text.lower().strip()
    
    for element in elements:
        # Check text field
        element_text = element.text.lower().strip()
        # Check content description
        element_desc = element.content_desc.lower().strip()
        
        if fuzzy:
            # Enhanced fuzzy matching with multiple strategies
            match_found = False
            
            # Strategy 1: Direct substring match (existing logic)
            if (search_text in element_text or search_text in element_desc or
                element_text in search_text or element_desc in search_text):
                match_found = True
            
            # Strategy 2: Number matching - extract numbers from both text and search
            if not match_found:
                search_numbers = re.findall(r'\d+', search_text)
                element_numbers = re.findall(r'\d+', element_text + ' ' + element_desc)
                
                # If search contains numbers, check if element contains those numbers
                if search_numbers and element_numbers:
                    for search_num in search_numbers:
                        if search_num in element_numbers:
                            match_found = True
                            break
            
            # Strategy 3: Word-level partial matching
            if not match_found:
                search_words = re.findall(r'\w+', search_text)
                element_words = re.findall(r'\w+', element_text + ' ' + element_desc)
                
                # Check if any significant search words appear in element
                for search_word in search_words:
                    if len(search_word) >= 2:  # Skip very short words
                        for element_word in element_words:
                            if search_word in element_word or element_word in search_word:
                                match_found = True
                                break
                        if match_found:
                            break
            
            # Strategy 4: Single character matching for buttons (like number buttons)
            if not match_found and len(search_text) == 1 and search_text.isdigit():
                # For single digit searches, exact match on element text
                if element_text == search_text or element_desc == search_text:
                    match_found = True
            
            if match_found:
                matches.append(element)
                
        else:
            # Exact match
            if element_text == search_text or element_desc == search_text:
                matches.append(element)
    
    return matches

def find_elements_by_keywords(elements: List[UIElement], keywords: List[str]) -> List[UIElement]:
    """
    Find UI elements that match any of the given keywords.
    
    Args:
        elements: List of UI elements to search
        keywords: List of keywords to search for
        
    Returns:
        List of matching elements
    """
    matches = []
    
    for keyword in keywords:
        keyword_matches = find_elements_by_text(elements, keyword, fuzzy=True)
        for match in keyword_matches:
            if match not in matches:
                matches.append(match)
    
    return matches

def find_clickable_elements(elements: List[UIElement]) -> List[UIElement]:
    """
    Filter elements to only clickable ones.
    
    Args:
        elements: List of UI elements to filter
        
    Returns:
        List of clickable elements
    """
    return [elem for elem in elements if elem.is_clickable and elem.is_enabled]

def find_best_element_match(elements: List[UIElement], step: str) -> Optional[UIElement]:
    """
    Find the best UI element match for a given step description.
    Uses enhanced keyword extraction, fuzzy matching, and scoring.
    
    Args:
        elements: List of available UI elements
        step: Step description (e.g., "tap wifi", "tap number 5", "open settings")
        
    Returns:
        Best matching UI element or None
    """
    step_lower = step.lower().strip()
    
    # Extract keywords from step with enhanced number handling
    action_words = ['tap', 'click', 'press', 'touch', 'open', 'close', 'toggle', 'scroll', 'swipe', 'go', 'back']
    
    # Split step into words and clean them
    words = [word.strip() for word in re.split(r'[^\w]', step_lower) if word.strip()]
    keywords = [word for word in words if word not in action_words and len(word) > 0]
    
    # Special handling for number references (e.g., "tap number 5", "press 3")
    numbers_in_step = re.findall(r'\b(?:number\s+)?(\d+)\b', step_lower)
    if numbers_in_step:
        # Add just the numbers as high-priority keywords
        for num in numbers_in_step:
            if num not in keywords:
                keywords.insert(0, num)  # Put numbers at the beginning for priority
    
    if not keywords:
        # If no keywords found, try the whole step as keyword
        keywords = [step_lower]
    
    # Find elements matching keywords with enhanced search
    candidates = []
    
    # Try multiple search strategies
    for keyword in keywords:
        # Strategy 1: Direct text search
        matches = find_elements_by_text(elements, keyword, fuzzy=True)
        candidates.extend(matches)
        
        # Strategy 2: For single digits, also try exact match
        if keyword.isdigit() and len(keyword) == 1:
            exact_matches = find_elements_by_text(elements, keyword, fuzzy=False)
            candidates.extend(exact_matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in candidates:
        candidate_id = (candidate.text, candidate.content_desc, candidate.bounds)
        if candidate_id not in seen:
            seen.add(candidate_id)
            unique_candidates.append(candidate)
    
    candidates = unique_candidates
    
    # Filter to clickable elements for action steps
    action_keywords = ['tap', 'click', 'press', 'touch', 'open', 'toggle']
    if any(action in step_lower for action in action_keywords):
        clickable_candidates = find_clickable_elements(candidates)
        if clickable_candidates:  # Only filter if we have clickable options
            candidates = clickable_candidates
    
    if not candidates:
        return None
    
    # Enhanced scoring system
    scored_candidates = []
    for candidate in candidates:
        score = 0
        candidate_text = (candidate.text + " " + candidate.content_desc).lower()
        
        # Score based on keyword matches
        for keyword in keywords:
            if keyword in candidate_text:
                score += 3
                
                # Exact match bonus
                if keyword == candidate.text.lower().strip() or keyword == candidate.content_desc.lower().strip():
                    score += 5
                
                # Number matching bonus - if keyword is a number and element contains that exact number
                if keyword.isdigit():
                    candidate_numbers = re.findall(r'\d+', candidate_text)
                    if keyword in candidate_numbers:
                        score += 4  # High score for number matches
        
        # Clickable elements get bonus for action steps
        if candidate.is_clickable and any(action in step_lower for action in action_keywords):
            score += 2
        
        # UI element type bonuses
        class_name_lower = candidate.class_name.lower()
        if any(ui_type in class_name_lower for ui_type in ['button', 'textview', 'imageview']):
            score += 1
        
        # Button class gets extra bonus for tap actions
        if 'button' in class_name_lower and any(action in step_lower for action in ['tap', 'click', 'press']):
            score += 2
            
        # Penalty for very long text that might be descriptions rather than buttons
        if len(candidate.text) > 50:
            score -= 1
            
        scored_candidates.append((score, candidate))
    
    # Return highest scoring candidate
    if scored_candidates:
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_candidates[0][1]
        best_score = scored_candidates[0][0]
        
        # Only return if score is above threshold to avoid bad matches
        if best_score > 0:
            return best_candidate
    
    return None

def create_touch_action(element: UIElement, screen_width: int = 1080, screen_height: int = 1920) -> Dict[str, Any]:
    """
    Create a touch action dictionary for the given UI element.
    
    Args:
        element: UI element to touch
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
        
    Returns:
        Action dictionary for the environment
    """
    norm_x, norm_y = normalize_coordinates(element.center_x, element.center_y, screen_width, screen_height)
    
    return {
        'action_type': 1,  # Touch action (from android_env ActionType)
        'touch_position': [norm_x, norm_y]
    }

def create_no_op_action() -> Dict[str, Any]:
    """
    Create a no-operation action (lift finger).
    
    Returns:
        No-op action dictionary
    """
    return {
        'action_type': 0,  # Lift action
        'touch_position': [0.0, 0.0]
    }

def debug_print_elements(elements: List[UIElement], max_elements: int = 10):
    """
    Print debug information about UI elements.
    
    Args:
        elements: List of UI elements to print
        max_elements: Maximum number of elements to print
    """
    print(f"Found {len(elements)} UI elements:")
    for i, elem in enumerate(elements[:max_elements]):
        clickable_str = "CLICKABLE" if elem.is_clickable else ""
        enabled_str = "ENABLED" if elem.is_enabled else "DISABLED"
        
        display_text = elem.text or elem.content_desc or elem.class_name or "No text"
        print(f"  {i+1}. {display_text[:50]} [{clickable_str} {enabled_str}] @ ({elem.center_x:.0f}, {elem.center_y:.0f})")
    
    if len(elements) > max_elements:
        print(f"  ... and {len(elements) - max_elements} more elements")

def extract_step_action_and_target(step: str) -> Tuple[str, str]:
    """
    Extract action type and target from a step description.
    
    Args:
        step: Step description (e.g., "tap wifi button", "scroll down")
        
    Returns:
        Tuple of (action, target)
    """
    step_lower = step.lower().strip()
    
    # Define action patterns
    action_patterns = {
        'tap': ['tap', 'click', 'press', 'touch'],
        'toggle': ['toggle', 'switch', 'turn on', 'turn off'],
        'scroll': ['scroll', 'swipe'],
        'open': ['open', 'launch', 'start'],
        'wait': ['wait', 'pause', 'delay'],
        'check': ['check', 'verify', 'see', 'observe'],
        'go': ['go back', 'back', 'return']
    }
    
    detected_action = 'tap'  # Default action
    target = step
    
    for action_type, patterns in action_patterns.items():
        for pattern in patterns:
            if step_lower.startswith(pattern):
                detected_action = action_type
                target = step_lower.replace(pattern, '').strip()
                break
        if detected_action != 'tap':
            break
    
    return detected_action, target if target else step
