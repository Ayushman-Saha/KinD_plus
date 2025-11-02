# Use TensorFlow GPU image as base
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p checkpoint/decom_net_retrain \
    checkpoint/new_restoration_retrain \
    checkpoint/illumination_adjust_net_retrain \
    decom_net_train_result \
    new_restoration_train_results \
    illumination_adjust_net_train_result \
    test_results

# Set environment variables for better CPU performance
ENV TF_NUM_INTEROP_THREADS=2
ENV TF_NUM_INTRAOP_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Default command (can be overridden)
CMD ["/bin/bash"]