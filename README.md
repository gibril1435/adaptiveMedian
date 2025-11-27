# Image Denoising Tool

## 1\. Project Description

This project is a **Flask web application** designed for **Image Denoising**. It allows users to upload an image, apply simulated **salt-and-pepper noise**, and then compare the performance of several common filters in removing that noise. The primary focus is demonstrating the effectiveness of the **Adaptive Median Filter (AMF)**.

-----

## 2\. Features

  * **Image Upload:** Supports image files in PNG, JPG, JPEG, TIFF, and BMP formats.
  * **Noise Application:** Adds customizable **Salt-and-Pepper Noise** to the uploaded image. Users can select noise densities from 10% up to 90%.
  * **Filter Comparison:** Compares the denoising results of four images:
    1.  The **Original Image**.
    2.  The **Noisy Input** image.
    3.  The image processed by a **Box Filter** (using a weighted average kernel).
    4.  The image processed by the **Standard Median Filter**.
    5.  The image processed by the **Adaptive Median Filter (AMF)**.
  * **Quality Metrics:** Calculates and displays quantitative results for each filter, including **PSNR (Peak Signal-to-Noise Ratio)** and **MSE (Mean Squared Error)**, against the original image.
  * **Download:** Allows downloading individual processed images or a **ZIP archive** containing all results.

-----

## 3\. Installation and Setup

### Prerequisites

You must have **Python** installed. The application uses the following external libraries:

  * Flask
  * OpenCV (`cv2`)
  * NumPy
  * scikit-image (`skimage`)
  * werkzeug

### Setup Steps

1.  **Clone the Repository** (or download the files).
2.  **Install Dependencies:** Install the required libraries using pip:
    ```bash
    pip install Flask opencv-python numpy scikit-image werkzeug
    ```

-----

## 4\. Usage

### Running the Application

1.  Ensure you are in the root directory of the project.
2.  Run the main application file:
    ```bash
    python app.py
    ```
    The application will typically start on `http://127.0.0.1:5000/`.

### Web Interface Steps

1.  Open your web browser and navigate to the application's URL.
2.  **Upload an Image:** Use the upload area to select an image file.
3.  **Select Noise Density:** Choose the desired percentage of salt-and-pepper noise to apply (e.g., 30% is the default).
4.  **Process Image:** Click the "Process Image" button.
5.  **View Results:** The application will display the comparison grid of the Original, Noisy, Box Filter, Standard Median, and Adaptive Median Filter results, along with the **Quality Metrics table** (PSNR and MSE).

-----

## 5\. Implementation Details (from `app.py`)

  * **Noise:** Salt and pepper noise is implemented by randomly setting half the noise density pixels to 0 (pepper) and the other half to 255 (salt).
  * **Filters:**
      * **Box Filter:** Uses a custom 3x3 weighted kernel: `[[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16.0`.
      * **Standard Median Filter:** Uses `cv2.medianBlur` with a kernel size of 3.
      * **Adaptive Median Filter (Simplified):** A simple implementation where noise pixels (0 or 255) are replaced with the result of a 3x3 median filter, while other pixels remain unchanged.
  * **Image Processing:** All uploaded images are resized to **512x512** pixels and converted to **grayscale** before processing.
