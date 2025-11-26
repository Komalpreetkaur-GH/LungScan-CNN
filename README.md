🫁 Lung Cancer Nodule Detection Using Deep Learning

Early detection of lung cancer is critical, yet interpreting CT scans can be challenging and subjective. Radiologists must review hundreds of slices per patient, which is time-consuming and prone to fatigue-related errors.
This project explores how convolutional neural networks (CNNs) can assist in detecting lung nodules automatically.

Traditional machine-learning approaches rely on manually engineered features, which may limit performance. Deep learning, however, learns features directly from raw image data, making it a powerful tool for medical imaging—especially when large, high-dimensional data like CT scans are involved.

This project demonstrates the full pipeline of building, training, and evaluating a CNN for lung-nodule detection using data from the LIDC-IDRI dataset and its curated version, LUNA16.

📦 Dataset

The LUNA16 dataset provides:

888 CT scans

Associated annotations containing voxel coordinates of candidate nodules

Standardized .mhd and .raw image formats

Each scan consists of 512 × 512 × n slices (around 200 per scan).
The dataset contains 551,065 candidate annotations, of which 1,351 are labeled as true nodules — resulting in a significant class imbalance.

🏗️ Data Preprocessing

Key preprocessing steps:

1. Reading CT volumes

CT scans are loaded using SimpleITK from .mhd/.raw files. Intensities (Hounsfield units) are normalized for model input.

2. Patch extraction

Instead of training on full scans, 50×50 pixel patches are cropped around each annotation point.
Coordinates are converted from world space to voxel space.

3. Class balancing

To address the heavy imbalance:

Negative samples are undersampled

Positive samples are augmented using 90° and 180° rotations

This results in a more usable ~80:20 class split.

🔄 Data Augmentation

Each original nodule patch generates two additional rotated images:

Original	90° Rotation	180° Rotation

This increases variability without excessively oversampling the minority class.

🧠 Model Architecture

The model is a 3-layer Convolutional Neural Network built using TFLearn (a high-level wrapper around TensorFlow).
The architecture consists of:

Conv layer (32 filters, 5×5)

Max pooling

Conv layer (64 filters, 5×5)

Conv layer (64 filters, 3×3)

Fully connected layer

Softmax output layer

This network is compact but effective for small medical image patches.

🏋️ Training Pipeline

Due to hardware limitations (MacBook Pro, 2012), the dataset is stored in HDF5 format and streamed in batches using h5py.

Training details:

6,878 training images

Batch training for memory efficiency

Model trained for several epochs until convergence

Visualization of early convolutional feature maps shows progressive extraction of edges, textures, and higher-level features.

📊 Evaluation

The model is evaluated on 1,623 test images.

Results:

Accuracy: 93%

Precision: 89.3%

Recall: 71.2%

Specificity: 98.2%

A confusion matrix highlights typical misclassifications, and sample predictions (TP, FP, FN, TN) help interpret model behavior.