Package Versions:
- torch 2.2.2
- NumPy 1.26.4
- scikit-learn 1.4.1.post1

This code base relies on [HAMPLE](https://github.com/ZhangLab312/Hample/tree/main).

To increase the number of weights, call model creation like: 
`model = Hample(ham_i_dim=(8,64))` 

### Training ID/OOD
The `train_id.py` produces models with in-distribution (ID) training.

The `train_hist_ood.py` produces models with out-of-distribution (OOD) histone modifications used during training.

### Evaluation
Running `eval_ood4.py` creates the models' predictions for the various (robustness) evaluation scenarios: $D_{in}$, $D_{\mathbb{N}}$, $D_{sub}$.
`scripts.calc_auroc.py` calculates the Area Under the Receiver Operating Characteristic Curve (AuROC) scores.

### Notes
Make sure to check the `readme` files in the subfolders to avoid any complications.
