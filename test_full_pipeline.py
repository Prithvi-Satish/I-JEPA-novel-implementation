import os
import torch
import train_and_evaluate_jepa as pipeline
import eval_knn

def run_dry_run():
    print("=" * 60)
    print("      RUNNING INTEGRATION VERIFICATION SUITE       ")
    print("=" * 60)
    
    # Temporarily set PRETRAIN_EPOCHS to 2 and PROBE_EPOCHS to 2 for dry run
    pipeline.PRETRAIN_EPOCHS = 2
    pipeline.PROBE_EPOCHS = 2
    
    print("\n[Check 1/4] Running Full Pipeline (Phase 1, Phase 2, Phase 3)...")
    pipeline.main()
    
    print("\n[Check 2/4] Verifying checkpoint was created...")
    assert os.path.exists("./checkpoints/jepa_plant_75epochs.pt"), "Checkpoint missing!"
    print(" Checkpoint verified.")
    
    print("\n[Check 3/4] Verifying confusion matrix image was created...")
    assert os.path.exists("confusion_matrix_jepa.png"), "Confusion matrix plot missing!"
    print(" Confusion matrix image verified.")
    
    print("\n[Check 4/4] Verifying standalone eval_knn.py...")
    eval_knn.evaluate_knn()
    assert os.path.exists("confusion_matrix_knn.png"), "k-NN Confusion matrix plot missing!"
    print(" k-NN evaluation verified.")
    
    print("\n" + "=" * 60)
    print(" ALL CHECKS PASSED PERFECTLY! 100% READY FOR FULL RUN.")
    print("=" * 60)

if __name__ == "__main__":
    run_dry_run()
