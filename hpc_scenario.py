# Mock Ray implementation since Ray cannot be installed on Python 3.13
# import ray
import logging
import numpy as np
import random
from typing import List, Dict, Any, Tuple
# Commenting out this import as it might still cause issues
# from system_dynamics import system_dynamics_sim

logger = logging.getLogger("flashdna.hpc")

# Mock Ray functions
def init(*args, **kwargs):
    logger.info("Mock Ray initialized (Ray package not available)")
    return True

def remote(func):
    """Mock ray.remote decorator"""
    # In mock mode, we don't need the remote behavior, just return the function
    # But we need to handle the .remote() call syntax
    class RemoteWrapper:
        def __init__(self, func):
            self._func = func
        def remote(self, *args, **kwargs):
            # Execute the function directly and return the result
            return self._func(*args, **kwargs)
    return RemoteWrapper(func)

def get(futures):
    """Mock ray.get function - futures are already computed results"""
    return futures # Results are computed directly in mock mode

def shutdown():
    logger.info("Mock Ray shutdown")
    return True

# Create a mock ray module CLASS DEFINITION
class MockRay:
    @staticmethod
    def init(*args, **kwargs):
        return init(*args, **kwargs)

    @staticmethod
    def remote(func):
        return remote(func)

    @staticmethod
    def get(futures):
        return get(futures)

    @staticmethod
    def shutdown():
        return shutdown()

# Replace actual ray with mock INSTANCE
ray = MockRay()

# Mock system_dynamics_sim if it's not available or causes issues
def mock_system_dynamics_sim(user_initial, months, marketing_spend, referral_rate, churn_rate, seasonality=False):
    # Simple linear growth for mock purposes
    users = [user_initial * (1 + (referral_rate - churn_rate) * i) for i in range(months)]
    revenue = [u * 10 for u in users] # Assuming $10 ARPU
    return {"users": users, "revenue": revenue}

system_dynamics_sim = mock_system_dynamics_sim

# ... rest of the mock functions ...

# Comment out the original @ray.remote decorator
# @ray.remote
def scenario_worker(churn, referral, user_initial=1000):
    """Mock worker function - runs directly"""
    params = {
        "churn_rate": churn,
        "referral_rate": referral,
        "user_initial": user_initial,
        "marketing_spend": 5000, # Mock value
        "months": 12, # Mock value
        "seasonality": False # Mock value
    }
    # Directly call a simplified simulation logic
    # Use the mock or imported system_dynamics_sim
    sim_result = system_dynamics_sim(
        user_initial=user_initial,
        months=params['months'],
        marketing_spend=params['marketing_spend'],
        referral_rate=referral,
        churn_rate=churn,
        seasonality=params['seasonality']
    )
    final_users = sim_result["users"][-1] if sim_result["users"] else user_initial
    return churn, referral, final_users # Return tuple directly

def run_hpc_simulations():
    """
    parallel scenario scanning using Ray (MOCKED)
    """
    # Mock ray.init() call is handled by the mock object, no need to call explicitly
    # ray.init()
    churn_vals= np.linspace(0.01,0.2,5)
    referral_vals= np.linspace(0.01,0.1,5)
    tasks=[]
    for c in churn_vals:
        for r in referral_vals:
            # Call the worker function directly via the mock .remote() syntax
            result = scenario_worker.remote(c,r)
            tasks.append(result)
    # Mock ray.get() simply returns the list of results
    out = ray.get(tasks)
    # Mock ray.shutdown()
    # ray.shutdown()
    return out

def find_optimal_scenario(target_metric="final_users", init_users=1000,
                          current_churn=0.05, current_referral=0.02):
    """
    run HPC sim => find best scenario by final_users or success prob
    """
    all_scen= run_hpc_simulations()
    best= None
    best_val= -1
    for sc in all_scen:
        val= sc.get("final_users",0)
        if val> best_val:
            best_val= val
            best= sc
    return {
        "all_scenarios": all_scen,
        "optimal": best
    }
