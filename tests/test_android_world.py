import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'android_world'))

def test_android_world_import():
    print("Testing android_world imports...")
    
    try:
        from android_world import env
        print("✓ Successfully imported android_world.env")
    except ImportError as e:
        print(f"✗ Failed to import android_world.env: {e}")
    
    try:
        from android_world.env import actuation
        print("✓ Successfully imported android_world.env.actuation")
    except ImportError as e:
        print(f"✗ Failed to import android_world.env.actuation: {e}")
    
    try:
        from android_world.task_evals import task_eval
        print("✓ Successfully imported android_world.task_evals.task_eval")
    except ImportError as e:
        print(f"✗ Failed to import android_world.task_evals.task_eval: {e}")
    
    try:
        from android_world.agents import base_agent
        print("✓ Successfully imported android_world.agents.base_agent")
    except ImportError as e:
        print(f"✗ Failed to import android_world.agents.base_agent: {e}")
        
    print("\nAndroid World import test completed!")

if __name__ == "__main__":
    test_android_world_import() 