## Source Code

This repository contains the official source code for **Obliviate: Efficient Unlearning in Recommender Systems**.

### File Overview

- **`add_deletion_set.py`**  
  Implements the deletion-interaction construction method. It adds low-preference interactions to simulate more realistic unlearning requests and provides the corresponding evaluation setup.

- **`train_MF.py`**  
  Trains the Matrix Factorization (MF) model with the BPR objective and saves the required triplets for the unlearning stage.

- **`models.py`**  
  Contains the model implementations for MF-BPR and LightGCN.

- **`train_lightGCN.py`**  
  Trains the LightGCN model with the BPR objective and saves the triplets required for unlearning.

- **`unlearning_main.py`**  
  Main end-to-end implementation of Obliviate, including both stages:  
  - LUA: Low-rank Unlearning Adapter  
  - LAC: Locality-aware Calibration

- utils.py  
  Utility functions used across training, deletion-set creation, and unlearning.
