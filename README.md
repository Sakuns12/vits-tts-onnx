# VITS (ONNX CPU Optimization for VoIP)

This project is based on [JK-VITS](https://github.com/kdrkdrkdr/JK-VITS) and has been optimized for **high-speed real-time inference on CPUs (RTF < 0.03)** through ONNX conversion.

It is designed specifically for **VoIP (Voice over IP) applications**, achieving extreme lightness and speed by training at **16kHz** without compromising voice clarity.

## 📌 Key Features

* **Ultra-Fast Inference:**
    * Optimized for **16kHz** sampling rate, which covers 99% of the human voice frequency range.
    * Achieved a **Real Time Factor (RTF) of ~0.02** on a modern CPU, enabling high-density concurrent processing.
* **Enhanced ONNX Compatibility:**
    * Solved PyTorch 2.x `Complex` tensor errors and Dynamic Shape (`aten::lift_fresh`) issues by implementing a custom `stft.py`.
    * Replaced heavy STFT operations with a `Conv1d`-based implementation for maximum ONNX export stability.
* **High Fidelity:**
    * Maintained the `pinv` (pseudo-inverse) method used in original VITS training to prevent voice jitter and metallic artifacts often seen after ONNX conversion.

---

## 🛠️ Installation

### 1. Prerequisites

* Anaconda (or Miniconda)
* Python 3.10+
* CUDA Toolkit (Required for training; optional for inference)

### 2. Create Virtual Environment

```bash
conda create -n vits python=3.10 -y
conda activate vits
```

### 3. Install PyTorch
Install the appropriate version of PyTorch for your system. (Version 2.0 or higher is recommended for ONNX conversion).

```Bash
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

### 4. Install Dependencies
```Bash
pip install -r requirements.txt
pip install scipy librosa matplotlib ipython unidecode Cython onnxruntime

# CMake is required for some packages (e.g., pyopenjtalk)
pip install "cmake<3.27"
pip install pyopenjtalk

# For Korean text processing
pip install ko_pron mecab-python3
```

🚀 Usage</br>
1. Preprocessing & Training</br>
This guide is based on the **KSS Dataset**. You can use your own dataset, but it must follow the specific format of the `train.txt` file (Audio Path | Text).

**⚠️ Important Audio Requirement:**
Since this project targets a **16kHz** model, **all audio files in your dataset must be resampled to 16,000Hz** before training. Using 44.1kHz or 24kHz audio without resampling will result in incorrect pitch or training failure.

**Dataset Format Example:**
```text
dataset/KO/1_0000.wav|[KO]그는 괜찮은 척하려고 애쓰는 것 같았다.[KO]
dataset/KO/1_0001.wav|[KO]안녕하세요.[KO]
...


```Bash
# 1. Text Preprocessing
# Generates filelists/train.txt.cleaned from raw text
python preprocess.py --filelists filelists/train.txt filelists/val.txt

# 2. Audio Preprocessing
# The KSS dataset (audiobook) contains about 1 second of silence at the beginning.
# This script trims silence and adds slight padding (200ms) for faster VoIP response.
python audio_preprocess.py

# 3. Start Training
# For the KSS dataset, 200~300 epochs are sufficient for decent voice quality (approx. 300,000 steps).
# Excessive training may lead to overfitting. It is recommended to listen to generated samples periodically.
./train_16k.sh
```

2. Export ONNX</br>
Rename the final trained checkpoint (e.g., G_300000.pth) to G_0.pth (or modify the script path) and convert it to an ONNX file.

```Bash
# Run export script
python export_onnx.py
Output: ko_16k.onnx (Filename may vary based on configuration)
```

3. ONNX Inference</br>
Perform high-speed inference on CPU using the converted model.

```Bash
python inference_onnx.py
Result: Generates result_onnx.wav and prints the RTF to the console.
```

⚠️ Troubleshooting & Technical Notes</br>
To prevent common ONNX conversion errors (such as aten::lift_fresh and Unknown type: complex), the stft.py file has been modified.

Issue: PyTorch's default stft/istft functions return complex numbers, which causes compatibility issues during ONNX export.

Solution: The TorchSTFT class in stft.py has been replaced with pure real-number operations using Conv1d and ConvTranspose1d.

Warning: Do not overwrite stft.py with the original VITS code, as this will make ONNX conversion impossible.

📊 Performance Benchmark
Environment: Intel Core i9-14900K (ONNX Runtime, CPU Only), Clock limited to 4.8GHz.

Single Thread Inference
```
==============================
Generated Audio Length : 14.86s
Inference Time         : 0.3284s
RTF (Real Time Factor) : 0.0221
==============================
```
Multi-Thread Load Test

```
===============================
Cores Used             : 8
Successful Requests    : 32 / 32
Total Elapsed Time     : 5.98s
Total Audio Generated  : 475.65s
Throughput (Speed)     : 79.49x (Real-time speedup)
----------------------------------------
Average RTF            : 0.1820 (Lower is better)
----------------------------------------
```

🔗 Reference</br>
Original Repository: https://github.com/kdrkdrkdr/JK-VITS

🙏 Acknowledgements</br>
This project is built upon the excellent work of JK-VITS by kdrkdrkdr.

I would like to express my sincere gratitude to the original author for open-sourcing the VITS implementation. Their contribution laid the solid foundation that made this optimization and ONNX conversion project possible.
