"""
Mock Android Environment for testing ExecutorAgent.
Simulates android_env behavior without requiring actual Android emulator.
"""

import time
import random
from typing import Dict, Any, List, Tuple, Optional
# import numpy as np  # Not needed for mock environment


class MockAndroidEnv:
    """
    Mock Android environment that simulates UI interactions.
    Provides realistic UI trees and responds to actions.
    """
    
    def __init__(self):
        """Initialize the mock environment."""
        self.current_screen = "home"
        self.wifi_enabled = False
        self.bluetooth_enabled = False
        self.step_count = 0
        self.last_action = None
        
        # Define screen layouts
        self.screens = {
            "home": {
                "elements": [
                    {
                        "text": "Settings",
                        "content_desc": "Settings app",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.launcher:id/settings",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [100, 200, 300, 280]
                    },
                    {
                        "text": "Calculator",
                        "content_desc": "Calculator app",
                        "class_name": "android.widget.TextView", 
                        "resource_id": "com.android.calculator:id/app",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [350, 200, 550, 280]
                    },
                    {
                        "text": "Camera",
                        "content_desc": "Camera app",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.camera:id/app", 
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [100, 350, 300, 430]
                    }
                ]
            },
            
            "settings": {
                "elements": [
                    {
                        "text": "Settings",
                        "content_desc": "Settings title",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/title",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 100, 400, 150]
                    },
                    {
                        "text": "Wi-Fi",
                        "content_desc": "WiFi settings",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/wifi",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [50, 200, 250, 250]
                    },
                    {
                        "text": "ON" if self.wifi_enabled else "OFF",
                        "content_desc": f"WiFi {'enabled' if self.wifi_enabled else 'disabled'}",
                        "class_name": "android.widget.Switch",
                        "resource_id": "com.android.settings:id/wifi_switch",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [320, 200, 380, 250]
                    },
                    {
                        "text": "Bluetooth",
                        "content_desc": "Bluetooth settings",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/bluetooth",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [50, 300, 250, 350]
                    },
                    {
                        "text": "ON" if self.bluetooth_enabled else "OFF",
                        "content_desc": f"Bluetooth {'enabled' if self.bluetooth_enabled else 'disabled'}",
                        "class_name": "android.widget.Switch",
                        "resource_id": "com.android.settings:id/bluetooth_switch",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [320, 300, 380, 350]
                    },
                    {
                        "text": "Battery",
                        "content_desc": "Battery settings",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/battery",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [50, 400, 250, 450]
                    },
                    {
                        "text": "Storage",
                        "content_desc": "Storage settings",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/storage",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [50, 500, 250, 550]
                    },
                    {
                        "text": "<-",
                        "content_desc": "Navigate back",
                        "class_name": "android.widget.ImageButton",
                        "resource_id": "com.android.settings:id/back",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [20, 50, 70, 100]
                    }
                ]
            },
            
            "wifi": {
                "elements": [
                    {
                        "text": "Wi-Fi",
                        "content_desc": "WiFi settings title",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/title",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 100, 400, 150]
                    },
                    {
                        "text": "ON" if self.wifi_enabled else "OFF",
                        "content_desc": f"WiFi {'on' if self.wifi_enabled else 'off'}",
                        "class_name": "android.widget.Switch",
                        "resource_id": "com.android.settings:id/wifi_main_switch",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [320, 180, 380, 230]
                    },
                    {
                        "text": "Available networks" if self.wifi_enabled else "Turn on Wi-Fi to see available networks",
                        "content_desc": "WiFi networks list",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/wifi_networks",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 280, 400, 320]
                    },
                    {
                        "text": "<-",
                        "content_desc": "Navigate back",
                        "class_name": "android.widget.ImageButton",
                        "resource_id": "com.android.settings:id/back",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [20, 50, 70, 100]
                    }
                ]
            },
            
            "calculator": {
                "elements": [
                    {
                        "text": "Calculator",
                        "content_desc": "Calculator title",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.calculator:id/title",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 100, 400, 150]
                    },
                    {
                        "text": "0",
                        "content_desc": "Display",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.calculator:id/display",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 200, 400, 280]
                    },
                    {
                        "text": "5",
                        "content_desc": "Five",
                        "class_name": "android.widget.Button",
                        "resource_id": "com.android.calculator:id/digit_5",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [150, 350, 200, 400]
                    },
                    {
                        "text": "+",
                        "content_desc": "Plus",
                        "class_name": "android.widget.Button",
                        "resource_id": "com.android.calculator:id/op_add",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [250, 350, 300, 400]
                    },
                    {
                        "text": "3",
                        "content_desc": "Three",
                        "class_name": "android.widget.Button",
                        "resource_id": "com.android.calculator:id/digit_3",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [200, 450, 250, 500]
                    },
                    {
                        "text": "=",
                        "content_desc": "Equals",
                        "class_name": "android.widget.Button",
                        "resource_id": "com.android.calculator:id/op_eq",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [300, 450, 350, 500]
                    }
                ]
            },
            
            "battery": {
                "elements": [
                    {
                        "text": "Battery",
                        "content_desc": "Battery settings title",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/title",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 100, 400, 150]
                    },
                    {
                        "text": f"{random.randint(65, 95)}%",
                        "content_desc": "Battery percentage",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/battery_percentage",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 200, 200, 250]
                    },
                    {
                        "text": "Good",
                        "content_desc": "Battery health status",
                        "class_name": "android.widget.TextView",
                        "resource_id": "com.android.settings:id/battery_health",
                        "is_clickable": False,
                        "is_enabled": True,
                        "bounds": [50, 300, 200, 350]
                    },
                    {
                        "text": "<-",
                        "content_desc": "Navigate back",
                        "class_name": "android.widget.ImageButton",
                        "resource_id": "com.android.settings:id/back",
                        "is_clickable": True,
                        "is_enabled": True,
                        "bounds": [20, 50, 70, 100]
                    }
                ]
            }
        }
    
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        self.current_screen = "home"
        self.wifi_enabled = False
        self.bluetooth_enabled = False
        self.step_count = 0
        self.last_action = None
        
        return self._get_observation()
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take an action in the environment and return new observation.
        
        Args:
            action: Action dictionary with action_type and touch_position
            
        Returns:
            Observation dictionary
        """
        self.step_count += 1
        self.last_action = action
        
        # Simulate action processing delay
        time.sleep(0.1)
        
        # Process the action
        if action.get('action_type') == 1:  # Touch action
            touch_pos = action.get('touch_position', [0, 0])
            self._process_touch(touch_pos)
        
        return self._get_observation()
    
    def _process_touch(self, touch_position: List[float]):
        """
        Process a touch action and update environment state.
        
        Args:
            touch_position: Normalized [x, y] coordinates
        """
        # Convert normalized coordinates to absolute pixels (assuming 1080x1920 screen)
        abs_x = touch_position[0] * 1080
        abs_y = touch_position[1] * 1920
        
        # Find which element was touched
        current_elements = self.screens.get(self.current_screen, {}).get("elements", [])
        
        for element in current_elements:
            bounds = element["bounds"]
            left, top, right, bottom = bounds
            
            if left <= abs_x <= right and top <= abs_y <= bottom and element["is_clickable"]:
                self._handle_element_click(element)
                break
    
    def _handle_element_click(self, element: Dict[str, Any]):
        """
        Handle clicking on a specific element.
        
        Args:
            element: Element that was clicked
        """
        resource_id = element.get("resource_id", "")
        
        # Navigation actions
        if "settings" in resource_id and self.current_screen == "home":
            self.current_screen = "settings"
        
        elif "calculator" in resource_id and self.current_screen == "home":
            self.current_screen = "calculator"
        
        elif "back" in resource_id:
            if self.current_screen in ["wifi", "battery"]:
                self.current_screen = "settings"
            elif self.current_screen == "settings":
                self.current_screen = "home"
            elif self.current_screen == "calculator":
                self.current_screen = "home"
        
        # WiFi actions
        elif "wifi" in resource_id:
            if self.current_screen == "settings" and "wifi" in resource_id and not "switch" in resource_id:
                self.current_screen = "wifi"
            elif "switch" in resource_id:
                self.wifi_enabled = not self.wifi_enabled
        
        # Battery actions  
        elif "battery" in resource_id and self.current_screen == "settings":
            self.current_screen = "battery"
        
        # Bluetooth actions
        elif "bluetooth" in resource_id and "switch" in resource_id:
            self.bluetooth_enabled = not self.bluetooth_enabled
        
        # Update screen elements to reflect state changes
        self._update_dynamic_elements()
    
    def _update_dynamic_elements(self):
        """Update dynamic elements based on current state."""
        # Update WiFi switches in settings screen
        if "settings" in self.screens:
            for element in self.screens["settings"]["elements"]:
                if "wifi_switch" in element.get("resource_id", ""):
                    element["text"] = "ON" if self.wifi_enabled else "OFF"
                    element["content_desc"] = f"WiFi {'enabled' if self.wifi_enabled else 'disabled'}"
                elif "bluetooth_switch" in element.get("resource_id", ""):
                    element["text"] = "ON" if self.bluetooth_enabled else "OFF"
                    element["content_desc"] = f"Bluetooth {'enabled' if self.bluetooth_enabled else 'disabled'}"
        
        # Update WiFi screen
        if "wifi" in self.screens:
            for element in self.screens["wifi"]["elements"]:
                if "wifi_main_switch" in element.get("resource_id", ""):
                    element["text"] = "ON" if self.wifi_enabled else "OFF"
                    element["content_desc"] = f"WiFi {'on' if self.wifi_enabled else 'off'}"
                elif "wifi_networks" in element.get("resource_id", ""):
                    element["text"] = "Available networks" if self.wifi_enabled else "Turn on Wi-Fi to see available networks"
    
    def _get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from the environment.
        
        Returns:
            Observation dictionary with structured UI data
        """
        # Update dynamic elements before returning observation
        self._update_dynamic_elements()
        
        current_elements = self.screens.get(self.current_screen, {}).get("elements", [])
        
        return {
            "structured": {
                "elements": current_elements.copy()
            },
            "screen": self.current_screen,
            "step_count": self.step_count,
            "wifi_enabled": self.wifi_enabled,
            "bluetooth_enabled": self.bluetooth_enabled
        }
    
    def action_spec(self) -> Dict[str, Any]:
        """Return action specification."""
        return {
            'action_type': {
                'minimum': 0,
                'maximum': 1,
                'shape': (),
                'dtype': 'int32'
            },
            'touch_position': {
                'minimum': [0.0, 0.0],
                'maximum': [1.0, 1.0],
                'shape': (2,),
                'dtype': 'float32'
            }
        }
    
    def observation_spec(self) -> Dict[str, Any]:
        """Return observation specification."""
        return {
            'structured': dict,
            'screen': str,
            'step_count': int
        }
    
    def close(self):
        """Close the environment."""
        pass
    
    def get_current_screen_info(self) -> Dict[str, Any]:
        """Get information about current screen state."""
        return {
            "current_screen": self.current_screen,
            "wifi_enabled": self.wifi_enabled,
            "bluetooth_enabled": self.bluetooth_enabled,
            "step_count": self.step_count,
            "elements_count": len(self.screens.get(self.current_screen, {}).get("elements", []))
        }
