# Cancer Tumor Classification & Dimensionality Reduction

## Project Overview

This project implements an end-to-end machine learning pipeline to identify **malignant tumor profiles** from continuous cellular morphometric measurements. In computational oncology and high-throughput genomics, analyzing high-dimensional tissue matrices frequently presents the **"Curse of Dimensionality"**, where a large number of correlated features can cause models to overfit on noise.

This pipeline addresses that constraint. It utilizes **Principal Component Analysis (PCA)** to compress a 30-dimensional feature space down to 2 components, then trains a classifier to establish a high-accuracy, clinically sound classification boundary.

The project is built twice and compared:

- **scikit-learn:** PCA + a **Linear Support Vector Machine (SVM)**, which achieves a **95.61% test accuracy** on the original 80:20 split.
- **PyTorch:** PCA + a **Multilayer Perceptron (MLP)** trained with a full training loop, a validation set, and early stopping, evaluated across 5 random seeds.

Both models are evaluated with a focus on medical safety: sensitivity (catching malignant tumors) is prioritized over raw accuracy.

## Repository Contents

- `tumor_expression_classifier_sklearn.ipynb`: EDA, standardization, PCA, and Linear SVM.
- `tumor_expression_classifier_pytorch.ipynb`: same data and preprocessing, with a PyTorch MLP.

## Data

The dataset utilized is the classic **Wisconsin Breast Cancer Dataset**, natively integrated into Scikit-Learn (`load_breast_cancer`). It features 569 patient biopsy samples containing 30 continuous floating-point geometric features (e.g., cell radius, smoothness, and area).

## Pipeline

### scikit-learn (Linear SVM)

1. **Data Ingestion**: Unpacks the native scikit-learn biological data bundle (`Bunch` object) into a structured Pandas DataFrame. The raw target classifications are mapped immediately to capitalized, human-readable labels (`Malignant` / `Benign`) to enforce upfront code readability and prevent silent target-inversion bugs during downstream analysis.
2. **Exploratory Data Analysis (EDA)**: Conducts a global missing-value audit to verify dataset completeness (0 null rows found) and diagnostics on the clinical class balance (62.74% Benign vs. 37.26% Malignant). Scale profiling across all 30 features unmasks a massive global variance gap (e.g., `mean smoothness` at ~0.10 vs. `mean area` scaling up to 2,501.00), mandating a structural standardization phase.
3. **Leakage-Proof Data Standardization**: Performs an 80:20 stratified train-test split _before_ data transformations to prevent data leakage. A `StandardScaler` calculates scaling constants ($\mu=0, \sigma=1$) exclusively on the 455 training rows using `.fit_transform()`. These locked parameters are then applied to the 114 test rows via `.transform()`, mirroring how new patient data would be handled at deployment.
4. **Dimensionality Reduction (PCA)**: Initialized with `n_components=2` to compress the 30 continuous feature columns down into a 2D coordinate space. This compression retains **62.64% of the total dataset variance** (PC1: 44.00%, PC2: 18.65%) while stripping away redundant feature noise and exposing a distinct diagonal spatial boundary between tumor classes.
5. **Model Training & Evaluation (SVM)**: Trains a Linear Support Vector Machine to optimize geometric margins. A linear kernel is deliberately selected as the compressed PCA space exhibits clean visual separation, eliminating the need for hyper-flexible kernels (like RBF) that risk memorizing noisy border points on limited clinical samples.

### PyTorch (MLP)

1. **Data and EDA**: Same ingestion and EDA as above.
2. **Train / Validation / Test Split**: A stratified 64/16/20 split (364 / 91 / 114 samples) using the same random seed. The validation set is used only for early stopping, and the test set is touched once for final evaluation.
3. **Leakage-Proof Preprocessing**: `StandardScaler` and PCA (`n_components=2`) are fit on the 364 training rows only, then applied to the validation and test rows with `.transform()`.
4. **Data Pipeline**: Features and labels are wrapped in a `TensorDataset` and fed through a `DataLoader` in mini-batches of 32, with shuffling on the training set only.
5. **Model**: An MLP (inputs → 32 → 16 → 1) with ReLU activations and a dropout layer (p = 0.2). It outputs a single logit, and `BCEWithLogitsLoss` applies the sigmoid internally for numerical stability.
6. **Training**: Adam optimizer with weight decay. Training stops when validation loss has not improved for 20 consecutive epochs, and the weights from the best validation epoch are restored.
7. **Evaluation**: The model is trained with 5 random seeds, and results are reported as mean ± standard deviation.

## Core Performance Metrics

Malignant is treated as the positive class. **Sensitivity** is malignant recall, and **specificity** is benign recall.

| Model                             | Training Samples | Test Accuracy  | Sensitivity    | Specificity    | AUC           |
| --------------------------------- | ---------------- | -------------- | -------------- | -------------- | ------------- |
| Linear SVM (80:20 split)          | 455              | 95.61%         | 92.86%         | 97.22%         | n/a           |
| Linear SVM (64/16/20 split)       | 364              | 93.86%         | 88.10%         | 97.22%         | 0.982         |
| PyTorch MLP (5 seeds, mean ± std) | 364              | 93.33% ± 0.48% | 90.48% ± 0.00% | 95.00% ± 0.76% | 0.979 ± 0.002 |

All three models use the same PCA setup (2 components). The fitted components differ slightly between the 455- and 364-sample runs. Test sets contain 114 samples (42 malignant, 72 benign), so one sample is worth about 0.9 percentage points of accuracy.

**Sources:** The 80:20 SVM row comes from `[V1 notebook filename]`. The 364-sample SVM row comes from `[notebook/cell that produced it]`, and the MLP row from `tumor_expression_classifierV2_pytorch.ipynb`. [Both 364-sample rows use the same split code (`random_state=24`) and therefore the same 114-sample test set.]

### Comparison

The PyTorch MLP does not outperform the linear SVM on this dataset. On the same 364-sample training split, the two models land within one or two test samples of each other, which is within noise at this test-set size. The SVM catches slightly fewer malignant tumors (37 vs. 38 of 42) but makes fewer false alarms (2 vs. 3 to 4 of 72). The original SVM's higher 95.61% may reflect training on more samples (455 vs. 364) [and a different test set], a gap of about 2 test samples.

This is the expected result for a small tabular dataset where the classes are largely separable in the PCA space: a nonlinear model has little to gain. The MLP's seed-to-seed variation is tiny (accuracy varies by at most one test sample), but it reflects only random initialization, not the choice of train/test split.

### Clinical Valuation Analysis

Through a medical oncology lens, minimizing false negatives (missing an active cancer case) matters more than avoiding false positives. The original (80:20) SVM reaches a **sensitivity of 0.93**, catching 39 of 42 malignant cases, with a **malignant precision of 0.95** and a **specificity of 97.22%**, which limits unnecessary follow-up procedures. The PyTorch MLP detects 38 of 42 malignant tumors (90.48%). Neither model should be treated as screening-ready: with 114 test samples, the sensitivity estimates carry wide uncertainty.

## Limitations

- Each model was evaluated on a single train/test split of 569 samples, and the SVM rows use two different splits. Cross-validation would give a more reliable estimate.
- Only 2 principal components are used, retaining about 62% of the total variance. More components may improve sensitivity.
- Default decision thresholds are used (0.5 probability for the MLP, 0 margin for the SVM). Lowering them would trade specificity for sensitivity, which may be preferable in a screening setting.

## Installation & Reproduction

```bash
# Clone the repository
git clone https://github.com/zoechow24/tumor-expression-classifier.git
cd tumor-expression-classifier

# Initialize and activate the isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install scikit-learn pandas numpy matplotlib seaborn torch jupyter ipykernel

# Run the Jupyter Notebook pipelines
jupyter notebook tumor_expression_classifier_sklearn.ipynb
jupyter notebook tumor_expression_classifier_pytorch.ipynb
```
