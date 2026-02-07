# Device Health Monitoring Through Sound Analysis

## ASME Hackathon 2025 - Predictive Maintenance Challenge

---

## Executive Summary

This project presents an artificial intelligence-based solution for predictive maintenance of industrial and domestic machinery through acoustic signature analysis. The system employs machine learning techniques to detect early-stage equipment failures by analyzing sound patterns, enabling proactive maintenance interventions before catastrophic failures occur.

The solution addresses the critical industrial challenge of unplanned downtime by providing real-time health monitoring capabilities that can be deployed on low-power edge devices, making advanced predictive maintenance accessible across diverse operational environments.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Technical Approach](#technical-approach)
3. [System Architecture](#system-architecture)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Results and Performance](#results-and-performance)
7. [Deployment Considerations](#deployment-considerations)

---

## Problem Statement

### Context

Industrial and domestic machines frequently experience failures that could be prevented through early detection of maintenance issues. These failures manifest as acoustic or vibrational anomalies before reaching critical stages, yet most facilities lack the infrastructure for continuous monitoring and analysis.

### Challenge Objectives

Develop an AI-powered solution capable of:

- Analyzing sound and vibration signatures in real-time
- Distinguishing between normal operating conditions and early-stage failure indicators
- Providing quantitative health scores for monitored equipment
- Identifying probable failure modes with mechanical terminology
- Operating efficiently on low-power edge computing devices

### Success Criteria

- High signal-to-noise ratio robustness
- Precision in anomaly detection with minimal false positives
- Mechanically interpretable failure mode classification
- Resource-efficient implementation suitable for embedded deployment

---

## Technical Approach

### Core Methodology

The solution leverages supervised machine learning combined with domain-specific feature engineering to create a robust anomaly detection system. The approach integrates signal processing techniques from acoustics and vibration analysis with modern deep learning architectures.

### Key Innovation Points

**Multi-Domain Feature Extraction**: The system extracts features across time, frequency, and cepstral domains to capture comprehensive acoustic signatures that correlate with mechanical health states.

**Mechanically-Grounded Classification**: Rather than treating this as a pure pattern recognition problem, the solution incorporates mechanical engineering principles to ensure failure mode classifications align with physical deterioration mechanisms.

**Edge-Optimized Architecture**: The neural network design prioritizes inference efficiency and model compactness to enable deployment on resource-constrained devices without sacrificing detection accuracy.

---

## System Architecture

### Pipeline Overview

```
Audio Input → Preprocessing → Feature Extraction → Neural Network → Health Assessment → Failure Mode Diagnosis
```

### Component Breakdown

**1. Signal Acquisition Layer**
- Accepts audio input from microphones or vibration sensors
- Sampling rate: 22,050 Hz (configurable)
- Supports real-time streaming and batch processing

**2. Preprocessing Module**
- Noise normalization and amplitude scaling
- Signal conditioning for optimal feature extraction
- Temporal windowing for continuous monitoring

**3. Feature Extraction Engine**
- Mel-Frequency Cepstral Coefficients (MFCC): 13 coefficients
- Spectral features: centroid, bandwidth, rolloff, zero-crossing rate
- Statistical aggregation: mean and standard deviation across temporal windows
- Total feature dimensionality: 34 features per sample

**4. Classification Network**
- Input layer: 34 neurons (feature vector)
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation (anomaly probability)
- Regularization: Dropout layers at 30% to prevent overfitting

**5. Decision and Interpretation System**
- Threshold-based anomaly classification
- Health score computation (0-100 scale)
- Rule-based failure mode identification
- Confidence-weighted diagnostic reporting

---

## Methodology

### Feature Engineering Rationale

**Mel-Frequency Cepstral Coefficients (MFCC)**

MFCCs represent the short-term power spectrum of sound signals and are particularly effective for capturing timbral characteristics that change with mechanical degradation. The mel-scale transformation aligns with human auditory perception while maintaining sensitivity to machinery-relevant frequency bands.

**Spectral Centroid**

Indicates the center of mass of the spectrum and shifts toward higher frequencies as components wear, particularly in bearing and gear failures.

**Spectral Bandwidth**

Measures frequency distribution spread, increasing with irregular vibrations from misalignment or imbalance conditions.

**Zero-Crossing Rate**

Correlates with signal noisiness and high-frequency content, useful for detecting bearing surface defects and cavitation.

### Failure Mode Classification Framework

The system identifies three primary failure modes based on spectral characteristics:

**Bearing Wear**: Characterized by elevated high-frequency content (spectral centroid > 1500 Hz) resulting from surface roughness and rolling element defects.

**Mechanical Imbalance**: Identified through dominant low-frequency components (zero-crossing rate < 0.05) corresponding to rotational frequency harmonics.

**Misalignment**: Detected via specific harmonic patterns in the 1500-3000 Hz range indicating axial or angular shaft misalignment.

---

## Implementation Details

### Data Processing Pipeline

**Step 1: Audio Loading and Normalization**

Audio files are loaded using librosa with preservation of native sampling rates. Amplitude normalization ensures consistent feature extraction across recordings with varying input gains.

**Step 2: Feature Computation**

For each audio sample:
- 13 MFCC coefficients computed with default hop length
- Spectral features extracted using librosa's spectral analysis functions
- Mean and standard deviation calculated across time frames
- Features concatenated into 34-dimensional vectors

**Step 3: Dataset Preparation**

Features are compiled into training and validation sets with:
- 80/20 train-validation split
- Random shuffling to prevent temporal bias
- Standardization using training set statistics

### Neural Network Architecture

**Design Rationale**

The two-layer fully connected architecture balances model capacity with computational efficiency. This configuration provides sufficient non-linear transformation capability for the feature space while maintaining rapid inference suitable for edge deployment.

**Training Configuration**

- Optimizer: Adam with learning rate 0.001
- Loss function: Binary cross-entropy
- Batch size: 32 samples
- Epochs: 50 with early stopping capability
- Validation monitoring: Loss and accuracy tracked per epoch

### Health Score Calculation

The health score H is computed as:

```
H = (1 - P_anomaly) × 100
```

Where P_anomaly represents the neural network's output probability of anomalous condition. This provides an intuitive 0-100 scale where:
- 90-100: Excellent condition
- 70-89: Good condition, monitoring recommended
- 50-69: Degraded condition, maintenance planning required
- Below 50: Critical condition, immediate intervention needed

---

## Results and Performance

### Model Performance Metrics

**Training Convergence**: The model achieved stable convergence within 50 epochs, demonstrating effective learning from the feature representations.

**Classification Capability**: The binary classification framework successfully distinguishes between normal and anomalous acoustic signatures with clear decision boundaries.

### Visualization and Analysis

**Waveform Analysis**: Time-domain plots reveal amplitude patterns and transient characteristics associated with different operational states.

**Spectrogram Representation**: Time-frequency decomposition visualizes how spectral content evolves, highlighting anomalous frequency components.

**Feature Distribution**: Statistical analysis of extracted features shows separability between normal and fault conditions across multiple feature dimensions.

### Limitations and Constraints

**Data Availability**: Due to lack of access to industrial vibration monitoring equipment, the current implementation was validated using synthetic data and publicly available audio samples. This represents the primary limitation in establishing real-world performance benchmarks.

**Generalization**: The model requires retraining or transfer learning when deployed on new equipment types or operational environments due to domain-specific acoustic characteristics.

**Environmental Noise**: While preprocessing includes normalization, extreme background noise conditions may require additional filtering or adaptive thresholding mechanisms.

---

## Deployment Considerations

### Edge Device Optimization

**Model Compression Pathway**

For production deployment, the following optimization strategy is recommended:

1. Post-training quantization to reduce model size by 4x
2. Conversion to TensorFlow Lite format for mobile/embedded inference
3. Hardware-specific compilation for target processors

**Expected Performance Targets**

- Model size: Under 100 KB after quantization
- Inference latency: Under 50ms on ARM Cortex-M7 class processors
- Power consumption: Compatible with battery-operated sensor nodes

### Real-Time Monitoring Architecture

**Continuous Operation Mode**

For deployed systems, audio should be captured in overlapping windows (e.g., 5-second segments with 2-second overlap) to ensure no transient events are missed between analysis intervals.

**Alert Generation Logic**

Health scores should be aggregated over sliding windows with configurable threshold settings to balance sensitivity and false alarm rates based on operational requirements.

### Integration Considerations

**Standard Interfaces**

The system can integrate with existing industrial control systems through:
- MQTT messaging for IoT platforms
- REST API for SCADA integration
- Modbus/TCP for PLC connectivity
- OPC-UA for industrial automation standards

---

## Future Enhancements

### Short-Term Improvements

**Enhanced Noise Robustness**

Implement adaptive filtering and spectral subtraction techniques to improve performance in high-noise industrial environments. Augment training data with controlled noise injection at various SNR levels.

**Expanded Failure Mode Library**

Incorporate additional failure classifications including:
- Gear tooth wear and pitting
- Lubrication degradation
- Cavitation in pumps
- Belt slippage and wear

**Multi-Sensor Fusion**

Combine acoustic analysis with temperature, current, and vibration acceleration data for comprehensive equipment health assessment.

### Long-Term Development Roadmap

**Unsupervised Anomaly Detection**

Transition to autoencoder-based architectures that learn normal operational patterns without requiring labeled failure data, enabling deployment on equipment where failure examples are scarce.

**Temporal Pattern Recognition**

Implement LSTM or Transformer-based models to capture long-term degradation trends and provide predictive remaining useful life estimates.

**Explainable AI Integration**

Incorporate SHAP values or attention mechanisms to provide interpretable explanations of which acoustic features drive specific anomaly classifications.

**Transfer Learning Framework**

Develop pre-trained foundation models on diverse machinery types that can be fine-tuned for specific equipment with minimal additional training data.

---

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Audio input capability (microphone or sensor interface)

### Setup Instructions

**1. Clone Repository**

```bash
git clone https://github.com/yourusername/device-health-monitoring.git
cd device-health-monitoring
```

**2. Create Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### Basic Usage Example

**Training the Model**

```python
from health_monitor import HealthMonitor

# Initialize system
monitor = HealthMonitor()

# Load training data
monitor.load_data('path/to/audio/files')

# Train model
monitor.train(epochs=50, batch_size=32)

# Save trained model
monitor.save_model('models/health_monitor_v1.h5')
```

**Analyzing Equipment**

```python
# Load trained model
monitor.load_model('models/health_monitor_v1.h5')

# Analyze audio file
result = monitor.analyze('equipment_recording.wav')

print(f"Health Score: {result['health_score']}")
print(f"Status: {result['status']}")
print(f"Failure Mode: {result['failure_mode']}")
```

### Google Colab Notebook

For interactive exploration and demonstration, access the complete implementation:

[Device Health Monitoring - Google Colab](https://colab.research.google.com/drive/17mx-zdVmwZHIBaHRMSMEa2agxR8PpnW8?usp=sharing)

---

## Technical Dependencies

### Core Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.x | Neural network framework |
| Keras | 2.x | High-level API for model building |
| NumPy | 1.21+ | Numerical computations |
| Librosa | 0.9+ | Audio processing and feature extraction |
| Matplotlib | 3.5+ | Visualization and plotting |
| Scikit-learn | 1.0+ | Data preprocessing and utilities |

### Optional Dependencies

- **sounddevice**: Real-time audio capture from microphones
- **pyaudio**: Alternative audio interface library
- **TensorFlow Lite**: Model conversion for embedded deployment
- **ONNX**: Model export for cross-platform compatibility

---

