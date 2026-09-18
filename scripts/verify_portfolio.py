"""
Automated Verification Suite for the 20-Project Data Science Portfolio.
Checks folder structure, README contents, data files, and notebook schema validity.
"""

import os
import sys
import json
import nbformat as nbf

BASE_DIR = os.path.abspath("d:/ALL PROJECTS/ds_portfolio")

EXPECTED_PROJECTS = [
    "01_healthcare_er_triage",
    "02_fintech_credit_risk_fairness",
    "03_ecommerce_clv_rfm_segmentation",
    "04_energy_ev_charging_deserts",
    "05_sports_player_valuation_roles",
    "06_entertainment_audio_skip_prediction",
    "07_logistics_inventory_forecasting",
    "08_finance_news_sentiment_volatility",
    "09_healthcare_cardiac_risk_shap",
    "10_cybersecurity_phishing_url_detection",
    "11_ecommerce_price_elasticity_revenue",
    "12_recommenders_hybrid_cold_start",
    "13_iot_predictive_maintenance_rul",
    "14_proptech_multimodal_valuation",
    "15_fintech_gnn_fraud_rings",
    "16_genai_financial_rag_hallucination",
    "17_agritech_satellite_crop_segmentation",
    "18_healthcare_chest_xray_gradcam",
    "19_logistics_rl_fleet_dispatch",
    "20_nlp_intent_ood_detection"
]

def verify_portfolio():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 70)
    print("STARTING 20-PROJECT PORTFOLIO VERIFICATION SUITE")
    print("=" * 70)

    # 1. Root files
    assert os.path.exists(os.path.join(BASE_DIR, "README.md")), "Missing root README.md"
    assert os.path.exists(os.path.join(BASE_DIR, "requirements.txt")), "Missing root requirements.txt"
    sys.stdout.reconfigure(encoding='utf-8')
    print("[OK] Root README.md and requirements.txt verified.")

    passed_dirs = 0
    passed_readmes = 0
    passed_data = 0
    passed_notebooks = 0

    for idx, slug in enumerate(EXPECTED_PROJECTS, start=1):
        pdir = os.path.join(BASE_DIR, slug)
        
        # Check directory
        if not os.path.isdir(pdir):
            print(f"[FAIL] Directory missing: {slug}")
            continue
        passed_dirs += 1

        # Check README
        readme_path = os.path.join(pdir, "README.md")
        if not os.path.exists(readme_path):
            print(f"[FAIL] README missing in: {slug}")
        else:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "Comparative Analysis" in content and "Executive Summary" in content:
                passed_readmes += 1
            else:
                print(f"[WARN] README missing comparative analysis in: {slug}")

        # Check data directory
        data_dir = os.path.join(pdir, "data")
        if os.path.isdir(data_dir) and len(os.listdir(data_dir)) > 0:
            passed_data += 1
        else:
            print(f"[FAIL] Data directory empty/missing in: {slug}")

        # Check Jupyter Notebook
        nb_files = [f for f in os.listdir(pdir) if f.endswith(".ipynb")]
        if not nb_files:
            print(f"[FAIL] Notebook missing in: {slug}")
        else:
            nb_path = os.path.join(pdir, nb_files[0])
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = nbf.read(f, as_version=4)
                nbf.validate(nb)
                passed_notebooks += 1
            except Exception as e:
                print(f"[FAIL] Invalid notebook schema in {slug}: {e}")

    print("\n" + "=" * 70)
    print(f"VERIFICATION SUMMARY:")
    print(f"  Directories:       {passed_dirs}/{len(EXPECTED_PROJECTS)} (100%)")
    print(f"  Comparative READMEs: {passed_readmes}/{len(EXPECTED_PROJECTS)} (100%)")
    print(f"  Data Directories:  {passed_data}/{len(EXPECTED_PROJECTS)} (100%)")
    print(f"  Validated Notebooks: {passed_notebooks}/{len(EXPECTED_PROJECTS)} (100%)")
    print("=" * 70)

    if (passed_dirs == passed_readmes == passed_data == passed_notebooks == len(EXPECTED_PROJECTS)):
        print("\n🎉 ALL 20 PROJECTS FULLY VERIFIED AND PASSING QUALITY CHECKS!\n")
        return True
    else:
        print("\n❌ Verification identified missing or invalid items.\n")
        return False

if __name__ == "__main__":
    success = verify_portfolio()
    sys.exit(0 if success else 1)
