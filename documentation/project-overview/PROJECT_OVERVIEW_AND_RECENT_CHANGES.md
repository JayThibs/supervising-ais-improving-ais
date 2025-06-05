# Project Overview and Recent Changes (Main Branch)

This document provides an overview of the `supervising-ais-improving-ais` project, focusing on the `main` branch, recent changes, and overall structure.

## Recent Activity (Last 30 Commits)

Here's a summary of the latest commits to the `main` branch, indicating recent development focus:

* 9f9c107 - Merge pull request #23 from JayThibs/auto-interventions (QuintinPope, 12 hours ago)
* c3cacd8 - Bugfixed issue with wrong number of validation tests, added diversification methods, updated p-value computations (Quintin Pope, 12 hours ago)
* 276b072 - Added updated experiment configurations (Quintin Pope, 12 hours ago)
* 7f8d129 - fixed quantization bug, added required CLIs for diversification, added weight diff analysis (Quintin Pope, 12 hours ago)
* 6b031f3 - Bugfixes, external data as prompts, diversified labels, faster clustering (Quintin Pope, 12 hours ago)
* 44ec7a3 - Added weight diff analysis and loading of external data (Quintin Pope, 12 hours ago)
* 24a76ad - Updated gemini api classes (Quintin Pope, 12 hours ago)
* 7b23594 - Updated gemini api classes (Quintin Pope, 12 hours ago)
* 37c89db - Added the ability to load external decoding runs with specified clustering and cluster metrics, plus using anthropic repo for prompts (Quintin Pope, 3 months ago)
* 4ceaa77 - Updated experiment configs (Quintin Pope, 5 months ago)
* fc9c81e - Added checks to remove invalid ground truths from GT recovery experiments (Quintin Pope, 5 months ago)
* 651cea8 - Added escape char to pandas csv read and write (Quintin Pope, 5 months ago)
* cb394d4 - Added validated_comparison_tools.py to repository (key file replacing validated_analysis.py, should have been tracked before) (Quintin Pope, 5 months ago)
* f6011b8 - Refactor .gitignore and add new files and directories (Jacques Thibodeau, 5 months ago)
* 6cc7011 - Refactor code for improved performance (Jacques Thibodeau, 5 months ago)
* 663aa66 - Bugfixes around handling multi labels, added better control of discriminative evals (Quintin Pope, 5 months ago)
* ef4be8d - Added basic unit tests (Quintin Pope, 6 months ago)
* 859088e - Fixed logging bug, and other small issues, allowed stronger models for cluster labeling (Quintin Pope, 6 months ago)
* 0243605 - Fixed updated README.md (Quintin Pope, 6 months ago)
* 529e40d - Updated README (Quintin Pope, 6 months ago)
* ec127f0 - Cleaned up redundant code (Quintin Pope, 6 months ago)
* 674296f - Added structlog and cleaned up file output paths (Quintin Pope, 6 months ago)
* c5c7d25 - Merge pull request #17 from JayThibs/auto-interventions (QuintinPope, 6 months ago)
* 3ec3b6f - Implemented statistical tests for cluster size differences, bug fixes for equal LM scores, corrections to experiment control logic (Quintin Pope, 6 months ago)
* 9aae673 - Fixed bugs in finetuning experiments control logic, corrected discriminative experiments, and updated experiment parameters (Quintin Pope, 6 months ago)
* fa0b95d - Added automatic retrying for errors during discrim hypothesis testing (Quintin Pope, 6 months ago)
* 353c578 - Major update, with modified validation metrics, additional test, more logging, bugfixes, etc (Quintin Pope, 6 months ago)
* 8d59500 - Merge pull request #16 from JayThibs/soft-prompting (Jacques Thibodeau, 7 months ago)
* 6bcbfc6 - Refactor DivergenceAnalyzer class for improved analysis and reporting (Jacques Thibodeau, 7 months ago)
* ffd8630 - Refactor ExperimentConfig class to improve serialization and deserialization (Jacques Thibodeau, 7 months ago)

**Key Themes from Recent Commits:**
*   Significant work on `auto-interventions`, including bug fixes, new features (diversification, weight diff analysis), and updated configurations.
*   Updates related to Gemini API classes.
*   Improvements in experiment configurations, ground truth recovery, and handling of external data.
*   Ongoing work on `validated_comparison_tools.py`.
*   Earlier work involved refactoring, unit tests, logging improvements, and statistical tests for clustering.

## Project Structure Overview

The project appears to be organized into several key areas, primarily within the `src/` directory:

*   **`src/behavioural_clustering/`**: Contains modules for behavioral clustering, including configuration, evaluation (analysis, clustering, dimensionality reduction, embeddings), models (API and local), and utility functions.
*   **`src/contrastive-decoding/`**: Focuses on contrastive decoding techniques, including divergence analysis, model comparison, n-gram clouds, and related scripts.
*   **`src/interventions/`**: This seems to be a major area of development, particularly the `auto_finetune_eval/` subdirectory. It includes tools for comparing to truth, data handling, helpers, interpretation, training, and a significant `validated_comparison_tools.py`.
*   **`src/soft_prompting/`**: Deals with soft prompting techniques, including analysis (divergence), configuration, core experiment logic, data loading, metrics, model management, and training.
*   **`src/webapp/`**: Contains code for a Streamlit web application, likely for visualizing results, comparing models, and running analyses. It includes components for authentication, configuration, and various pages.

Other important top-level directories and files include:
*   **`data/`**: Likely stores datasets, saved experiment results, embeddings, etc. (Note: many subdirectories are gitignored).
*   **`notebooks/`**: Jupyter notebooks, possibly for exploratory analysis or demonstrations.
*   **`scripts/`**: Utility scripts for various tasks like evaluation, generation, and training.
*   **`tests/`**: Contains test files (though the primary test suites seem to be within the `src` subdirectories like `src/interventions/auto_finetune_eval/tests/`).
*   `README.md`: Main project documentation.
*   `requirements.txt`: Python package dependencies.
*   `Makefile`: For build/automation tasks.

## Important Notes & Potential Issues

### 1. Merge Conflict with `devin/1743606310-report-cards-integration`
An attempt to merge the branch `devin/1743606310-report-cards-integration` into `main` resulted in conflicts. The conflicting files were:
*   `src/interventions/auto_finetune_eval/auto_finetuning_interp.py`
*   `src/interventions/auto_finetune_eval/auto_finetuning_main.py`

These conflicts will need to be resolved manually if the changes from that branch are to be integrated into `main`.

### 2. Embedded Git Repositories
During recent Git operations, warnings were issued about embedded Git repositories:
*   `.ai/dotai`
*   `SoftPromptsForEvaluation`

This means these directories are themselves Git repositories. Standard Git operations on the parent repository (this project) will not track changes within these embedded repositories correctly. Consider whether these should be:
*   Converted to Git submodules (if they are separate projects that need to be versioned independently but included here).
*   Added to the main project's `.gitignore` file (if their contents are not meant to be tracked by this parent repository).
*   Have their `.git` directories removed and their contents committed directly to this parent repository (if they are integral parts of this project and not separate entities).

This overview should provide a good starting point for understanding the current state and recent developments in the `main` branch.
