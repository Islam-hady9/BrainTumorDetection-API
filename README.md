# Brain Tumor Detection API

<div align="center">
  <p align="center">
    <img src="https://projects.smartinternz.com/cdn/shop/files/5.Brain-tumour-1440x720.jpg?v=1702460420" alt="brain" />
  </p>
<p align="center">
<strong>Brain Tumor Detection using CNN & FastAPI</strong></p>
</div>

---

This repository contains a FastAPI-based API for detecting brain tumors using a Convolutional Neural Network (CNN) model. The model is trained on a dataset of MRI images labeled as either "tumor" or "no tumor." The API allows users to upload an MRI image and receive a prediction indicating whether the image contains a brain tumor.

## Project Overview

Brain tumor detection is a critical task in medical imaging, as early and accurate detection can significantly impact patient outcomes. This project leverages deep learning techniques to build a CNN model capable of classifying MRI images into two categories: "tumor" or "no tumor." The trained model is then deployed as an API using FastAPI, providing an accessible interface for users to interact with the model and make predictions.

## Features

- **Image Upload:** Upload MRI images directly to the API.
- **Prediction:** The API returns a prediction indicating whether the image contains a brain tumor.
- **Model Summary:** A simple CNN model is used, including layers for convolution, pooling, flattening, and dense connections.
- **Data Augmentation:** The model uses data augmentation techniques to improve generalization and robustness.
- **Visualization:** The repository includes scripts for visualizing training results, including loss curves and confusion matrices.

## Installation

### Prerequisites

- Anaconda or Miniconda
- Python 3.8 (included in the Conda environment)

### Clone the Repository

```bash
git clone https://github.com/Islam-hady9/BrainTumorDetection-API.git
cd BrainTumorDetection-API
```

### Create a Conda Virtual Environment

It's recommended to create a Conda virtual environment to manage dependencies.

```bash
conda create -n brain-tumor-detection python=3.8
conda activate brain-tumor-detection
```

### Install Dependencies

Install the required Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

### 1. **Train the Model**

If you haven't trained the model yet, you can do so using the script provided in the repository. Ensure you have the dataset available in the correct format.

### 2. **Run the FastAPI Server**

Start the FastAPI server to expose the API:

```bash
uvicorn main:app --reload
```

The server will start running at `http://127.0.0.1:8000/`.

### 3. **Make Predictions**

You can make predictions by sending a POST request to the `/predict/` endpoint with an image file.

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_image.jpg"
```

### 4. **Test the API**

You can also test the API using the `test_api.py` script provided in the repository:

```bash
python test_api.py
```

Make sure to replace `path_to_your_image.jpg` in the script with the actual path to your MRI image file.

## Project Structure

- `app.py`: The FastAPI application script.
- `test_api.py`: A script to test the API by sending a sample image and printing the result.
- `requirements.txt`: A file listing all Python dependencies required for the project.
- `brain_tumor_detection_using_cnn.ipynb`: Training model file using CNN (Accuracy: 92.16%).
- `brain_tumor_model.h5`: The trained Keras model file (to be generated after training).
- `Dataset/`: Directory containing MRI images for training and testing.
- `README.md`: This file, providing an overview and instructions for the project.

## Dataset

The dataset used for training the model should be placed in the `Dataset/brain_tumor_dataset` directory, with subdirectories `yes` and `no` for images containing tumors and images without tumors, respectively.

- **Yes Tumor Directory:** `Dataset/brain_tumor_dataset/yes`
- **No Tumor Directory:** `Dataset/brain_tumor_dataset/no`

Ensure that all images are in a compatible format (e.g., `.jpg`, `.png`).

## Model Details

The CNN model used in this project has the following architecture:

- **Conv2D Layers:** For feature extraction from the input images.
- **MaxPooling2D Layers:** For downsampling the feature maps.
- **Flatten Layer:** To convert 2D feature maps into 1D feature vectors.
- **Dense Layers:** Fully connected layers for classification.
- **Output Layer:** A softmax layer with 2 units (tumor, no tumor).

## Visualization

The repository includes code for visualizing the training process, such as loss curves and confusion matrices. These visualizations can help in understanding the model's performance and diagnosing potential issues.

## Contributing

Contributions to the project are welcome. If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or inquiries, please contact [Islam Abd_Elhady] at [eslamabdo71239@gmail.com].
