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

- **Image Upload:** Users can upload MRI images directly through the web interface provided by the `index.html` page. The `static/styles.css` file ensures that the page is styled for a better user experience.
- **Prediction:** The FastAPI API endpoint `/predict/` processes the uploaded image and returns a prediction indicating whether the image contains a brain tumor. Predictions are displayed on the web interface below the uploaded image.
- **Model Summary:** A Convolutional Neural Network (CNN) is used for prediction. The model includes layers for convolution, pooling, flattening, and dense connections, and has been trained to achieve a high accuracy rate.
- **Data Augmentation:** The model utilizes data augmentation techniques to enhance generalization and robustness. This helps in improving the model's performance by exposing it to a wider variety of training examples.
- **Visualization:** Scripts are available for visualizing training results, including loss curves and confusion matrices, which help in evaluating the model's performance and understanding its learning progress.
- **User Interface:** A simple and user-friendly web interface allows users to upload images and view predictions. The interface is styled using a separate `styles.css` file for a clean and professional look.

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

### 1. **Train the Model Or Use the Pre-trained Model directly**

If you haven’t trained the model yet, use the Jupyter Notebook `brain_tumor_detection_using_cnn.ipynb` provided in the repository. Ensure you have the MRI images in the `Dataset/` directory and in the correct format before running the notebook or use the pre-trained model `brain_tumor_cnn_model.h5` directly that I trained.

### 2. **Run the FastAPI Server**

Start the FastAPI server to expose the API:

```bash
uvicorn app:app --reload
```

The server will start running at `http://127.0.0.1:8000/`.

### 3. **Access the Web Interface**

Open your browser and navigate to `http://127.0.0.1:8000` to access the HTML interface. You can upload images through this interface and receive predictions.

### 4. **Make Predictions via API**

You can make predictions by sending a POST request to the `/predict/` endpoint with an image file. 

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict/" -F "file=@path_to_your_image.jpg"
```

Replace `path_to_your_image.jpg` with the actual path to your MRI image file.

### 5. **Test the API**

You can test the API using the `test_api.py` script provided in the repository:

```bash
python test_api.py
```

This script will send a sample image to the API and print the prediction result. Make sure to modify the script if necessary to use the correct path to your test image.

## Project Structure

- **`app.py`:** The FastAPI application script that defines the API endpoints and handles image uploads and predictions.
- **`test_api.py`:** A script for testing the API by sending a sample image and printing the prediction result.
- **`requirements.txt`:** A file listing all Python dependencies required for the project.
- **`brain_tumor_detection_using_cnn.ipynb`:** Jupyter Notebook used for training the CNN model (Accuracy: 92.16%).
- **`brain_tumor_model.h5`:** The trained Keras model file (to be generated after training).
- **`Dataset/`:** Directory containing MRI images used for training and testing the model.
- **`templates/`:** Directory containing HTML templates.
  - **`index.html`:** The main HTML file providing the user interface for image uploads and predictions.
- **`static/`:** Directory containing static files such as CSS.
  - **`styles.css`:** The CSS file used for styling the `index.html` page.
- **`README.md`:** This file, providing an overview and instructions for the project.

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

## Web Interface Screens

### Screen 1: Upload Image
![Screen_1-Upload image](https://github.com/Islam-hady9/BrainTumorDetection-API/blob/main/Web%20Interface%20Screens/Screen_1-Upload%20image.png)

### Screen 2: Show Prediction
![Screen_2-Show prediction](https://github.com/Islam-hady9/BrainTumorDetection-API/blob/main/Web%20Interface%20Screens/Screen_2-Show%20prediction.png)

## Contributing

Contributions to the project are welcome. If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Contact

For questions or inquiries, please contact [Islam Abd_Elhady] at [eslamabdo71239@gmail.com].
