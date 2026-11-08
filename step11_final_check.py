# step11_final_check.py
"""
STEP 11: Final Project Verification (FIXED VERSION)

Checks everything is complete and working.
Run this as the LAST step to verify your project.
"""

import os
import json


def count_files_in_dir(dir_path):
    """Count files in a directory safely"""
    if not os.path.exists(dir_path):
        return 0
    if not os.path.isdir(dir_path):
        return 0
    try:
        return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    except Exception:
        return 0


def get_file_size(filepath):
    """Get file size as readable string"""
    try:
        size = os.path.getsize(filepath)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.0f} KB"
        else:
            return f"{size/1024/1024:.1f} MB"
    except Exception:
        return "? KB"


def final_check():
    """Verify all project components are complete"""

    print("=" * 60)
    print("STEP 11: FINAL PROJECT VERIFICATION")
    print("=" * 60)

    total = 0
    passed = 0
    missing = []

    # ==================== CODE FILES ====================
    print("\n  [CODE FILES]")

    code_files = {
        'verify_setup.py': 'Environment verification',
        'step2_prepare_data.py': 'Dataset preparation',
        'step3_data_pipeline.py': 'Data loading pipeline',
        'step4_model.py': 'Model architecture',
        'step5_train.py': 'Training pipeline',
        'step6_explainability.py': 'XAI module',
        'step7_robustness.py': 'Robustness testing',
        'step8_webapp.py': 'Web application',
        'step9_results.py': 'Results compilation',
        'step10_report_content.py': 'Report content generator',
        'requirements.txt': 'Dependencies list',
    }

    for filepath, description in code_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # ==================== MODEL FILES ====================
    print("\n  [MODEL FILES]")

    model_files = {
        'checkpoints/best_model.pth': 'Best trained model weights',
    }

    for filepath, description in model_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # ==================== OUTPUT FILES ====================
    print("\n  [RESULT FILES]")

    output_files = {
        'outputs/training_history.png': 'Training curves plot',
        'outputs/final_confusion_matrix.png': 'Confusion matrix',
        'outputs/final_roc_curve.png': 'ROC curve',
        'outputs/final_pr_curve.png': 'Precision-Recall curve',
        'outputs/final_per_class_metrics.png': 'Per-class metrics chart',
        'outputs/final_confidence_dist.png': 'Confidence distribution',
        'outputs/final_comparison.png': 'Literature comparison',
        'outputs/final_results.json': 'Metrics JSON file',
        'outputs/final_report.txt': 'Text report',
        'outputs/metrics.json': 'Training metrics',
    }

    for filepath, description in output_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # ==================== EXPLAINABILITY FILES ====================
    print("\n  [EXPLAINABILITY FILES]")

    explain_files = {
        'outputs/explanation_normal.png': 'Normal class explanation',
        'outputs/explanation_benign.png': 'Benign class explanation',
        'outputs/explanation_malignant.png': 'Malignant class explanation',
    }

    for filepath, description in explain_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # ==================== ROBUSTNESS FILES ====================
    print("\n  [ROBUSTNESS FILES]")

    robust_files = {
        'outputs/adversarial_robustness.png': 'Adversarial attack results',
        'outputs/corruption_robustness.png': 'Corruption test results',
        'outputs/uncertainty_analysis.png': 'Uncertainty analysis',
    }

    for filepath, description in robust_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # ==================== WEB APP FILES ====================
    print("\n  [WEB APP FILES]")

    webapp_files = {
        'templates/index.html': 'Web interface HTML',
    }

    for filepath, description in webapp_files.items():
        total += 1
        if os.path.exists(filepath):
            passed += 1
            size = get_file_size(filepath)
            print(f"    [OK]   {description:<35s} ({size})")
        else:
            missing.append(filepath)
            print(f"    [MISS] {description:<35s} -> {filepath}")

    # Check directories
    webapp_dirs = {
        'static/uploads': 'Upload directory',
        'static/results': 'Results directory',
    }

    for dirpath, description in webapp_dirs.items():
        total += 1
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            passed += 1
            count = count_files_in_dir(dirpath)
            print(f"    [OK]   {description:<35s} ({count} files)")
        else:
            missing.append(dirpath)
            print(f"    [MISS] {description:<35s} -> {dirpath}")

    # ==================== DATASET ====================
    print("\n  [DATASET]")

    dataset_dirs = {
        'dataset/train/normal': 'Training - Normal',
        'dataset/train/benign': 'Training - Benign',
        'dataset/train/malignant': 'Training - Malignant',
        'dataset/val/normal': 'Validation - Normal',
        'dataset/val/benign': 'Validation - Benign',
        'dataset/val/malignant': 'Validation - Malignant',
        'dataset/test/normal': 'Test - Normal',
        'dataset/test/benign': 'Test - Benign',
        'dataset/test/malignant': 'Test - Malignant',
    }

    total_images = 0
    for dirpath, description in dataset_dirs.items():
        total += 1
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            count = count_files_in_dir(dirpath)
            total_images += count
            if count > 0:
                passed += 1
                print(f"    [OK]   {description:<35s} ({count} images)")
            else:
                missing.append(dirpath)
                print(f"    [WARN] {description:<35s} (0 images - EMPTY)")
        else:
            missing.append(dirpath)
            print(f"    [MISS] {description:<35s} -> {dirpath}")

    print(f"\n    Total dataset images: {total_images}")

    # ==================== LOAD METRICS ====================
    print("\n  [PERFORMANCE METRICS]")

    metrics_path = 'outputs/final_results.json'
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                results = json.load(f)

            overall = results.get('overall_metrics', {})
            print(f"    Accuracy:     {overall.get('accuracy', 'N/A')}")
            print(f"    F1 Score:     {overall.get('f1_weighted', 'N/A')}")
            print(f"    ROC-AUC:      {overall.get('roc_auc_micro', 'N/A')}")
            print(f"    Precision:    {overall.get('precision_weighted', 'N/A')}")
            print(f"    Recall:       {overall.get('recall_weighted', 'N/A')}")
        except Exception as e:
            print(f"    Could not read metrics: {e}")
    else:
        print(f"    Metrics file not found. Run step9_results.py")

    # ==================== SUMMARY ====================
    print(f"\n{'='*60}")
    print(f"  VERIFICATION RESULT: {passed}/{total} checks passed")
    print(f"{'='*60}")

    if len(missing) == 0:
        print(f"""
  +--------------------------------------------------+
  |                                                    |
  |   ALL {total} CHECKS PASSED!                          |
  |                                                    |
  |   Your project is COMPLETE and ready               |
  |   for submission!                                   |
  |                                                    |
  +--------------------------------------------------+

  WHAT TO DO NEXT:
  ================

  1. DEMO: Run the web app
     Command: python step8_webapp.py
     Then open: http://localhost:5000

  2. REPORT: Use these files for your report
     - outputs/report_content.txt (chapter structure)
     - outputs/final_report.txt (metrics summary)
     - All .png files in outputs/ folder (charts & plots)

  3. PPT: Include these images in your presentation
     - outputs/final_confusion_matrix.png
     - outputs/final_roc_curve.png
     - outputs/explanation_normal.png
     - outputs/explanation_malignant.png
     - outputs/adversarial_robustness.png
     - outputs/final_comparison.png
     - Screenshots of web app (http://localhost:5000)

  4. SUBMISSION: Your project folder structure
     lung_cancer_detection/
     +-- dataset/           (training data)
     +-- checkpoints/       (trained model)
     +-- outputs/           (all results & charts)
     +-- templates/         (web app HTML)
     +-- static/            (web app files)
     +-- step2-11 .py files (all code)
     +-- requirements.txt   (dependencies)
""")

    else:
        print(f"\n  Missing items ({len(missing)}):")
        for item in missing:
            print(f"    - {item}")

        print(f"\n  HOW TO FIX:")
        print(f"  -----------")

        # Give specific fix instructions
        fix_map = {
            'checkpoints/best_model.pth': 'python step5_train.py',
            'outputs/training_history.png': 'python step5_train.py',
            'outputs/metrics.json': 'python step5_train.py',
            'outputs/final_confusion_matrix.png': 'python step9_results.py',
            'outputs/final_roc_curve.png': 'python step9_results.py',
            'outputs/final_pr_curve.png': 'python step9_results.py',
            'outputs/final_per_class_metrics.png': 'python step9_results.py',
            'outputs/final_confidence_dist.png': 'python step9_results.py',
            'outputs/final_comparison.png': 'python step9_results.py',
            'outputs/final_results.json': 'python step9_results.py',
            'outputs/final_report.txt': 'python step9_results.py',
            'outputs/explanation_normal.png': 'python step6_explainability.py',
            'outputs/explanation_benign.png': 'python step6_explainability.py',
            'outputs/explanation_malignant.png': 'python step6_explainability.py',
            'outputs/adversarial_robustness.png': 'python step7_robustness.py',
            'outputs/corruption_robustness.png': 'python step7_robustness.py',
            'outputs/uncertainty_analysis.png': 'python step7_robustness.py',
            'outputs/report_content.txt': 'python step10_report_content.py',
            'templates/index.html': 'python step8_webapp.py',
        }

        commands_needed = set()
        for item in missing:
            cmd = fix_map.get(item, None)
            if cmd:
                commands_needed.add(cmd)

        if commands_needed:
            print(f"\n  Run these commands to fix:")
            for cmd in sorted(commands_needed):
                print(f"    {cmd}")
        else:
            print(f"\n  Check that you have run all steps (step2 through step10)")

    print(f"\n  Quick reference commands:")
    print(f"  -------------------------")
    print(f"    python step8_webapp.py          -> Launch web demo")
    print(f"    python step9_results.py         -> Regenerate all charts")
    print(f"    python step10_report_content.py -> Report content")
    print(f"    python step11_final_check.py    -> Run this check again")


if __name__ == "__main__":
    final_check()