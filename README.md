#FusionAttnHLAII

## Overview
    This repository is the source code of our paper "FusionAttnHLAII: A Hierarchical Model for Predicting HLA II–Peptide Interactions"

## Environment Setting
    This code is based on Pytorch. You can find the specific environment and packages in the requirements.txt file.

## Running the code
    python main.py: Used for training a new model.
    python test.py: Used for testing trained models on various datasets.


## Files and Functions:
    main.py: Main function running file.
    main_ablation.py: Ablation function running file (need to change file storage paths, names, and models).
    Loader.py: Dataset splitting and packaging.
    train_test.py: Training, validation, and testing functions.
    feature_extraction.py: Contact graph: Residue physicochemical property extraction.
    performance.py: Performance metric calculation.
    test.py: Testing on various datasets.
    FusionAttnHLAII_GRU.py:abaltion study code
    FusionAttnHLAII_NoEncoder.py:ablation study code



