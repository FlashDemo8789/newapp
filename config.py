import os

# optional define how many Monte Carlo runs
MC_SIMULATION_RUNS= 2000

MONGO_URI= os.getenv("MONGO_URI","mongodb://localhost:27017/flash_dna")

# Optimized XGBoost parameters for better performance
XGB_PARAMS= {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "eval_metric": "logloss",
    "random_state": 42,
    "enable_categorical": False,
    "use_label_encoder": False
}
