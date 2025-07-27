#!/usr/bin/env python3
"""
System Validation Script
========================

Quick validation script to verify system integrity and readiness.
"""

import sys
import os

def validate_system():
    """Validate system components and dependencies."""
    print("🔍 Multi-Agent System Validation")
    print("=" * 40)
    
    # Check core modules
    try:
        sys.path.append('.')
        from agents.planner_agent import PlannerAgent
        from agents.executor_agent import ExecutorAgent
        from agents.verifier_agent import VerifierAgent
        from agents.supervisor_agent import SupervisorAgent
        print("✅ All agent modules imported successfully")
    except ImportError as e:
        print(f"❌ Agent import failed: {e}")
        return False
    
    # Check utilities
    try:
        from utils.logging import StructuredLogger
        from utils.prompts import PREDEFINED_PLANS
        from utils.mock_env import MockAndroidEnv
        print("✅ All utility modules imported successfully")
    except ImportError as e:
        print(f"❌ Utility import failed: {e}")
        return False
    
    # Check file structure
    required_files = [
        'main.py', 'README.md', 'SUBMISSION_SUMMARY.md',
        'requirements.txt', 'evaluation.md'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    # Check directories
    required_dirs = ['agents', 'utils', 'tests', 'logs']
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            return False
    
    print("\n🎉 System validation completed successfully!")
    print("📋 Ready for submission")
    return True

if __name__ == "__main__":
    validate_system()
