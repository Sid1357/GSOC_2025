# GSOC 2025 Evaluation Test - Project Repository

Welcome to my evaluation test repository for the GSOC 2025 DeepLense projects. This repository contains a series of Jupyter Notebooks demonstrating implementations for various evaluation tasks for DeepLense. Each task targets a specific challenge(as a part of application process) and provides both detailed explanations within the notebooks as well as summarized information here.

---

## Tasks Overview

- **Task I: Multi-Class Image Classification**  
  This task focuses on classifying gravitational lensing images into multiple classes. The goal is to build a model that distinguishes between different types of lensing images (e.g., strong lensing with no substructure, subhalo substructure, and vortex substructure).  
  **Key Components:**
  - **Dataset:**  
    - Images are provided as `.npy` files that have been normalized using min-max normalization.
    - The dataset is organized into folders corresponding to each class.
  - **Data Pipeline:**  
    - A custom dataset class loads images from these folders and applies necessary preprocessing.
    - Exploratory steps include verifying the folder contents and data cleanup (e.g., removing system files).
  - **Model Architecture:**  
    - A deep learning model (e.g., based on CNN architectures) is designed to perform multi-class classification.
  - **Training & Evaluation:**  
    - The model is trained using Cross-Entropy Loss function.
    - Evaluation metrics include the ROC curve and the AUC score for each class.
    - The final model is selected based on validation performance.

- **Task II: Lens Finding (Classification)**  
  This task focuses on building a classifier to distinguish between gravitational lensing images ("lens") and non-lensing images ("nonlens"). The implementation uses a custom CNN named `LensNet` and addresses dataset imbalance via class weighting and stratified data splitting.
  
  **Key Components:**
  - **Data Pipeline:**  
    - Loads image data stored as `.npy` files from separate directories for lenses and nonlenses.
    - Explores the dataset by printing folder contents, plotting sample channels, and performing cleanup.
    - Splits the data into training, validation, and test sets using stratified sampling.
  - **Dataset & DataLoader:**  
    - A custom `LensDataset` class loads images and maps class names to numeric labels.
    - DataLoaders are configured for training, validation, and testing.
  - **Model Architecture:**  
    - The custom CNN (`LensNet`) comprises three convolutional layers with ReLU activations, max-pooling, dropout, and fully connected layers.
  - **Training & Evaluation:**  
    - Uses Cross-Entropy Loss with computed class weights to mitigate imbalance.
    - Optimization is performed using Adam with a StepLR scheduler.
    - Performance is monitored via loss, accuracy, ROC-AUC score, ROC curve plots, and confusion matrices.
    - Early stopping is applied, and the best model is saved based on validation loss.

- **Task III: Image Super-Resolution**  
  This task involves training and fine-tuning a super-resolution model for gravitational lensing images using an SRCNN-based architecture, augmented with transfer learning.
  
  **Key Components:**
  - **Data Pipeline:**  
    - A custom `SuperResolutionDataset` handles paired high-resolution (HR) and low-resolution (LR) images.
  - **Model Architecture:**  
    - An SRCNN model with three convolutional layers is implemented in PyTorch.
  - **Training, Transfer Learning & Fine-Tuning:**  
    - The initial training is performed on a synthetic or simulated dataset using a 90:10 train-validation split with Mean Squared Error (MSE) loss.
    - **Transfer Learning:**  
      - The pre-trained SRCNN model weights from the initial training phase are saved.
      - These pre-trained weights are then loaded to fine-tune the model on a dataset of real HR/LR image pairs.
    - Fine-tuning is carried out with data augmentation (e.g., random horizontal flips) and a reduced learning rate.
  - **Evaluation Metrics:**  
    - Performance is measured using MSE, Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM).

- **Task IV: Diffusion Models**  
  This task implements a diffusion model to generate gravitational lensing images with data augmentation.
  
  **Key Components:**
  - **Custom Dataset:**  
    - The `AugmentedLensDataset` loads `.npy` files and applies random horizontal/vertical flips and rotations.
  - **Time Embedding & Cosine Noise Schedule:**  
    - Sinusoidal embeddings for time steps are generated.
    - A cosine-based beta schedule controls the diffusion process.
  - **U-Net with Residual Blocks:**  
    - The `DiffLensUNet` model, with residual blocks conditioned on time embeddings, forms the backbone of the diffusion process.
  - **Diffusion Process & EMA:**  
    - Noise addition and image sampling are implemented as part of the diffusion process.
    - An Exponential Moving Average (EMA) is applied to stabilize model training.
  - **Training & Evaluation:**  
    - The training loop incorporates data augmentation, periodically saves sample images and checkpoints, and calculates the Fréchet Inception Distance (FID) as the evaluation metric.
    - The final FID score is reported using the EMA model.

---

## Repository Structure

```plaintext
GSOC_2025/
├── DeepLense_commontask1/common-task-deep-lense_final1           # Notebook for Multi-Class Image Classification
├── DeepLense_specifictask2/specifictask2deeplense_final          # Notebook for Lens Finding (Classification)
├── DeepLense_specifictask3/specific3a-b_final                    # Notebook for Image Super-Resolution
└── DeepLense_specifictask4/dlspt4                                # Notebook for Diffusion Models 
