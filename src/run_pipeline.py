import subprocess

# The order of pipeline stages
STAGES = [
    "src/data_ingestion.py",
    "src/data_preprocessing.py",
    "src/model_train.py"
]

def run_pipeline():
    for stage in STAGES:
        print(f"\n🚀 Running: {stage}")
        result = subprocess.run(["python", stage, "--config", "config/params.yaml"])
        if result.returncode != 0:
            print(f"❌ Stage failed: {stage}")
            break
    else:
        print("\n✅ All stages completed successfully!")

if __name__ == "__main__":
    run_pipeline()
