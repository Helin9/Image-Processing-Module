# Custom Image Processing Module – Imaging Technologies Course

### Project Context & Credits
This repository contains my final project for the **Imaging Technologies** course at SRH University of Applied Sciences. 

**Note on Attribution:** * **Base Application:** The graphical user interface (GUI), file loading system, and main application skeleton (`main_app.py`) were provided by my professor as the course framework.
* **My Contribution:** I developed the **Custom Image Processing Module** (located in `src/modules/helin_men`). I was responsible for designing and implementing the mathematical logic for all the filters described below from scratch.

### The Challenge: Pure NumPy Implementation
The critical constraint of this project was a strict prohibition on using high-level image processing libraries (such as OpenCV or PIL) for the actual filtering logic. All algorithms had to be implemented using **pure NumPy**. This required a deep dive into the underlying mathematics of digital signal processing, matrix manipulation, and linear algebra.

### My Implementation: The Helin Men Module
I integrated five distinct tunable filters into the application. My implementation focuses on both visual accuracy and algorithmic optimization:

1.  **Optimized Gaussian Blur:** Unlike standard 2D convolution implementations which can be slow ($O(N^2)$), I implemented **Separable Convolution**. By splitting the process into two 1D passes (horizontal and vertical), I reduced the computational complexity to linear time ($O(N)$). This optimization allows the filter to handle large kernel sizes (e.g., 15x15) in real-time without interface lag.
    
2.  **Median Filter (Non-Linear):** Designed for noise reduction, this filter uses a sliding window approach to sort pixel values and select the median. It effectively removes "salt and pepper" noise while preserving sharp edges, demonstrating the difference between linear and non-linear filtering.
    
3.  **Laplacian Edge Detection:** Implements a second-derivative operator to calculate the rate of change in pixel intensity. The result visualizes edges as bright lines against a black background, highlighting the structural boundaries within the image.
    
4.  **Posterization (Quantization):** A pixel-wise operation that reduces the continuous color space (256 levels per channel) into a finite set of "buckets." This creates a stylized, retro, or comic-book aesthetic by reducing the image's bit depth.
    
5.  **Sharpening (Unsharp Masking):** Enhances image texture by amplifying high-frequency components. I implemented a tunable kernel that dynamically adjusts weights based on a user-defined strength parameter, allowing for precise control over the sharpening effect.

### Technologies Used
* **Python 3.x**
* **NumPy** (Broadcasting, Matrix Operations)
* **PySide6 / Qt** (UI Components)
* **Git** (Version Control)
