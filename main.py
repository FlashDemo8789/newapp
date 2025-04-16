import sys
import logging
import argparse
import threading
from train_model import train_model_xgb
from hpc_scenario import run_hpc_simulations
from flask import Flask, request, jsonify
import os
# NOTE: changed from `import streamlit.cli as stcli` to the new:
import streamlit.web.cli as stcli

logging.basicConfig(level=logging.INFO)
logger= logging.getLogger("main")

def run_train():
    logger.info("Starting Flash DNA training => model_xgb.pkl")
    train_model_xgb()
    logger.info("Training done => Flash DNA")

def run_hpc():
    logger.info("Running Flash DNA simulations => churn/referral range")
    try:
        results= run_hpc_simulations()
        logger.info(f"HPC => found {len(results)} combos.")
    except Exception as e:
        logger.error(f"HPC error => {str(e)}")

def run_api_server(host="0.0.0.0", port=5000):
    logger.info(f"Starting Flash DNA API on {host}:{port}")
    from advanced_ml import build_feature_vector
    import pickle
    import numpy as np

    app= Flask("flashdna_api")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status":"ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data= request.json
            if not data:
                return jsonify({"error":"no data"}),400
            model_path= "model_xgb.pkl"
            if not os.path.exists(model_path):
                return jsonify({"error":"model_xgb.pkl not found => run train"}),500
            with open(model_path,"rb") as f:
                model= pickle.load(f)
            fv= build_feature_vector(data).reshape(1,-1)
            if hasattr(model,"predict_proba"):
                prob= model.predict_proba(fv)[0,1]
                pred= "pass" if prob>=0.5 else "fail"
                return jsonify({
                    "prediction": pred,
                    "probability": prob,
                    "success_probability": prob*100
                })
            else:
                pred_val= model.predict(fv)[0]
                return jsonify({
                    "prediction":"pass" if pred_val>=0.5 else "fail",
                    "probability": float(pred_val)
                })
        except Exception as e:
            logger.error(f"Predict error => {str(e)}")
            return jsonify({"error":str(e)}),500

    @app.route("/assess", methods=["POST"])
    def assess():
        try:
            data= request.json
            if not data:
                return jsonify({"error":"no data"}),400
            from ml_assessment import StartupAssessmentModel
            assess_path= "assessment_model.pkl"
            if not os.path.exists(assess_path):
                return jsonify({"error":"assessment_model not found => please train"}),500
            import pickle
            with open(assess_path,"rb") as f:
                assess_model= pickle.load(f)
            res= assess_model.assess_startup(data)
            return jsonify(res)
        except Exception as e:
            logger.error(f"Assess error => {str(e)}")
            return jsonify({"error":str(e)}),500

    app.run(host=host, port=port)

def run_streamlit():
    logger.info("Starting Flash DNA Streamlit => analysis_flow.py")
    sys.argv = [
        "streamlit", 
        "run", 
        "analysis_flow.py",
        "--server.address=0.0.0.0",
        "--server.port=8501"
    ]
    stcli._main_run_clExplicit()

def main():
    parser= argparse.ArgumentParser(description="FlashDNA Infinity Flash DNA CLI")
    parser.add_argument("command", choices=["train","hpc","serve","api","streamlit","all"],help="Command")
    parser.add_argument("--host", default="0.0.0.0",help="API host")
    parser.add_argument("--port", type=int, default=5000, help="API port")
    args= parser.parse_args()

    if args.command=="train":
        run_train()
    elif args.command=="hpc":
        run_hpc()
    elif args.command in ["serve","api"]:
        run_api_server(args.host, args.port)
    elif args.command=="streamlit":
        run_streamlit()
    elif args.command=="all":
        th= threading.Thread(target=run_api_server, args=(args.host, args.port))
        th.daemon= True
        th.start()
        run_streamlit()

if __name__=="__main__":
    main()
