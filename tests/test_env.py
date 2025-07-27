import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'android_env'))

from android_env import loader
from android_env.components import config_classes
import numpy as np

def main():
    print("Initializing AndroidEnv...")
    
    # Create a simple configuration for AndroidEnv
    # Using FakeSimulator for testing without actual Android emulator
    config = config_classes.AndroidEnvConfig(
        simulator=config_classes.FakeSimulatorConfig(),
        task=config_classes.TaskConfig()
    )
    
    try:
        env = loader.load(config)
        print("AndroidEnv loaded successfully!")
    except Exception as e:
        print(f"Failed to load AndroidEnv: {e}")
        return

    print("Resetting environment...")
    try:
        obs = env.reset()
        print("Environment reset successful!")
        print("Observation type:", type(obs))
        print("Observation keys:", list(obs.observation.keys()) if hasattr(obs, 'observation') and obs.observation else "No observation data")
    except Exception as e:
        print(f"Failed to reset environment: {e}")
        return

    # Try a simple step
    print("Trying to step with a simple action...")
    try:
        # Get action spec to understand valid actions
        action_spec = env.action_spec()
        print("Action spec:", action_spec)
        
        # # Create a simple touch action
        # action = {
        #     'action_type': 1,  # Touch action
        #     'touch_position': [0.5, 0.5]  # Simple coordinates
        # }
        action = {}
        for key, spec in action_spec.items():
            if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
                # Use shape-aware uniform sampling
                value = np.random.uniform(low=spec.minimum, high=spec.maximum, size=spec.shape)
                action[key] = np.array(value).astype(spec.dtype)
            else:
                action[key] = np.random.randint(low=spec.minimum, high=spec.maximum + 1)
        
        timestep = env.step(action)
        print("Step successful!")
        print("Reward:", timestep.reward)
        print("Step type:", timestep.step_type)
    except Exception as e:
        print(f"Step failed: {e}")

    print("Test completed!")
    env.close()

if __name__ == "__main__":
    main()
