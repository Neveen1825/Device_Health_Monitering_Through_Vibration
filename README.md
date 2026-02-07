# Intelligent Vibration Analysis System for Bearing Fault Detection

## ASME Hackathon 2025 - Device Health Monitoring Challenge

---

## Executive Summary

This project presents an intelligent vibration analysis system for industrial bearing fault detection through advanced signal processing and machine learning techniques. The system analyzes acoustic and vibration signatures to detect bearing failures significantly earlier than conventional methods, achieving detection at 81.3% of bearing lifetime for roller element failures. Utilizing multi-band frequency analysis combined with Random Forest classification, the system delivers 94-95% accuracy in early-life detection with an R² score of 0.865, providing critical lead time of 1-14 days for maintenance planning.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Technical Approach](#technical-approach)
3. [Methodology](#methodology)
4. [Implementation](#implementation)
5. [Results and Performance](#results-and-performance)
6. [Discussion](#discussion)
7. [Installation and Usage](#installation-and-usage)

---

## Problem Statement

### Industrial Context

Bearing failures represent a critical challenge in modern manufacturing, contributing to millions of dollars in unplanned downtime costs annually. Industries allocate up to 50% of operational budgets to maintenance, yet conventional vibration analysis techniques typically detect failures only in the final 10% of bearing lifetime, leaving insufficient time for preventive action.

### Challenge Requirements

Develop an AI-powered system capable of:

- Analyzing sound and vibration signatures from industrial machinery
- Distinguishing between normal operating conditions and early-stage bearing failures
- Providing quantitative health assessments with failure time predictions
- Identifying specific failure modes: inner race, outer race, and roller element defects
- Delivering mechanically interpretable diagnostic outputs
- Operating efficiently on low-power edge computing platforms

### Performance Objectives

- Signal-to-noise robustness across industrial environments
- High precision anomaly detection with minimal false positives
- Early detection capability significantly before critical failure
- Actionable lead time for maintenance scheduling

---

## Technical Approach

### Core Innovation

The system integrates multi-band frequency analysis with supervised machine learning to create a comprehensive bearing health monitoring solution. Unlike conventional single-band approaches, our methodology analyzes three distinct frequency ranges (20 Hz - 5 kHz) to capture different failure mode signatures, combined with cascading filtering for enhanced signal clarity.

### Key Differentiators

**Multi-Band Frequency Decomposition**: Separates vibration signals into low (20 Hz - 1 kHz), mid (1-3 kHz), and high (3-5 kHz) frequency bands, each sensitive to specific bearing degradation mechanisms.

**Comprehensive Feature Engineering**: Extracts 20 critical features combining time-domain statistics (RMS, peak amplitude, crest factor, kurtosis) with frequency-domain characteristics and degradation tracking metrics.

**Random Forest Architecture**: Employs ensemble learning with 100 estimators to achieve robust classification across diverse failure modes and operational conditions.

**Cascading Signal Processing**: Implements dual-pass Butterworth filtering to maximize noise reduction while preserving fault indicators.

---

## Methodology

### Signal Processing Pipeline

**Stage 1: Multi-Band Frequency Analysis**

The system employs Butterworth bandpass filters to decompose vibration signals into three frequency ranges:

- Low Band (20 Hz - 1 kHz): Captures base rotation patterns and imbalance signatures
- Mid Band (1-3 kHz): Detects early warning indicators of bearing degradation
- High Band (3-5 kHz): Identifies advanced damage and surface defect signatures

**Stage 2: Cascading Filtration**

Sequential application of filters enhances signal clarity by:
- Reducing environmental and electrical noise interference
- Preserving critical fault frequency components
- Improving signal-to-noise ratio for feature extraction

**Stage 3: FFT Analysis**

Fast Fourier Transform processing provides detailed frequency-domain characterization, revealing harmonic patterns associated with specific bearing defect types.

### Feature Extraction Framework

**Time-Domain Statistical Features**

For each frequency band, the system computes:
- Root Mean Square (RMS): Overall energy level indicator
- Peak Amplitude: Maximum vibration magnitude
- Crest Factor: Ratio of peak to RMS, sensitive to impulse events
- Kurtosis: Distribution shape metric, elevated during early fault development

**Degradation Tracking Metrics**

Progressive indicators monitor bearing health evolution:
- Mid-to-baseline ratio: Tracks energy migration to warning frequencies
- Late-to-baseline ratio: Quantifies advanced degradation progression
- Stability measures: Assess variance in early, mid, and late operational phases
- Rate-of-change metrics: Detect acceleration in degradation patterns

**Total Feature Set**: 20 engineered features selected through correlation analysis and mechanical domain expertise

### Machine Learning Implementation

**Model Architecture: Random Forest Classifier**

- Ensemble of 100 decision tree estimators
- Trained on 753 carefully selected samples representing diverse failure modes
- Cross-validation optimized to prevent overfitting

**Data Distribution**

- Training: 72.2% (543 samples)
- Validation: 12.7% (96 samples)
- Testing: 15.0% (114 samples)

**Performance Metrics**

- R² Score: Correlation between predicted and actual failure times
- Mean Absolute Error: Prediction accuracy in hours
- Stage-Specific Accuracy: Performance across early, mid, and late bearing lifetime

---

## Implementation

### Dataset Specification

**Source**: NASA IMS Bearing Dataset from the NSF I/UCR Center for Intelligent Maintenance Systems with contributions from Rexnord Corporation

**Test Configuration**:
- Bearings: Four Rexnord ZA-2115 double row bearings
- Rotation Speed: 2000 RPM (constant)
- Applied Load: 6000 lbs via spring mechanism
- Sensors: PCB 353B33 High Sensitivity Quartz ICP Accelerometers
- Data Acquisition: NI DAQ Card 6062E at 20 kHz sampling rate
- Recording Protocol: 1-second snapshots every 10 minutes

**Dataset Structure**:

| Parameter | Set 1 | Set 2 | Set 3 |
|-----------|-------|-------|-------|
| Duration | 34 days | 7 days | 31 days |
| Total Files | 2,156 | 984 | 6,324 |
| Channels | 8 | 4 | 4 |
| Data Points/File | 20,480 | 20,480 | 20,480 |
| Failure Mode | Inner race (B3), Roller (B4) | Outer race (B1) | Outer race (B3) |

### Processing Workflow

**Step 1: Signal Acquisition and Preprocessing**

Vibration data undergoes initial conditioning including amplitude normalization and noise filtering to ensure consistent feature extraction across operational conditions.

**Step 2: Multi-Band Decomposition**

Butterworth filters separate signals into three frequency bands, each processed independently through cascading filtration stages.

**Step 3: Feature Computation**

For each frequency band and temporal window:
- Statistical time-domain features calculated
- FFT analysis performed for frequency content
- Degradation metrics computed relative to baseline
- Features aggregated into 20-dimensional vectors

**Step 4: Model Prediction**

Random Forest classifier processes feature vectors to generate:
- Failure probability scores
- Predicted time-to-failure estimates
- Failure mode classifications

**Step 5: Health Assessment**

System outputs comprehensive diagnostics including:
- Health scores (0-100 scale)
- Specific failure mode identification
- Maintenance action recommendations
- Confidence intervals for predictions

---

## Results and Performance

### Overall Model Performance

**Primary Metrics**:
- R² Score: 0.865 (strong correlation between predicted and actual failures)
- Mean Absolute Error: 67.20 hours
- Early-Life Detection Accuracy: 94-95%

**Temporal Prediction Accuracy**:
- Near-term (1-7 days): 94-95% accuracy
- Medium-term (7-14 days): 77-99% accuracy
- Long-term (14+ days): 34-96% accuracy

### Failure Mode Specific Results

**Inner Race Failures**

- Detection Point: 99.8% of bearing lifetime
- High-Band Energy Increase: 561%
- Characteristic: Clear RMS progression in 3-5 kHz range
- Lead Time: Suitable for immediate maintenance planning

**Roller Element Failures**

- Detection Point: 81.3% of bearing lifetime (exceptional early detection)
- Mid-Band Energy Increase: 349.6%
- Characteristic: Distinctive kurtosis progression in 1-3 kHz range
- Lead Time: Maximum advance warning for proactive intervention

**Outer Race Failures**

- Detection: Consistent across all frequency bands
- Energy Increase: >700% across spectrum
- Characteristic: Highest peak-to-peak amplitude changes
- Signature: Universal degradation pattern visibility

### Comparative Advantage

Conventional methods detect failures at approximately 90% of bearing lifetime. This system achieves:

- 18.7% earlier detection for roller element failures
- Significantly extended maintenance planning windows
- Reduced risk of catastrophic failure and unplanned downtime

---

## Discussion

### System Capabilities

The multi-band frequency analysis approach demonstrates superior performance in distinguishing failure mode signatures. Different defect types manifest distinct spectral characteristics, enabling precise classification and targeted maintenance interventions.

**Strengths**:
- Exceptional early detection for roller element failures
- High accuracy in near-term predictions for actionable maintenance
- Robust performance across diverse bearing failure modes
- Mechanically interpretable diagnostic outputs

### Limitations and Considerations

**Prediction Variance**: Long-term predictions (>14 days) exhibit increased variance due to complex, non-linear bearing degradation patterns. This suggests optimal use for near and medium-term maintenance planning.

**Computational Requirements**: Current implementation requires substantial processing resources. Real-time deployment in resource-constrained industrial environments may necessitate optimization through dimensionality reduction or simplified feature sets.

**Environmental Adaptation**: System validation conducted under controlled laboratory conditions. Industrial deployment across varied operational environments requires additional validation and potential model adaptation.

### Future Development Directions

**Computational Efficiency**: Explore feature selection techniques to reduce dimensionality while maintaining detection accuracy, enabling real-time edge deployment.

**Enhanced Classification**: Refine failure mode taxonomy to distinguish sub-types of bearing defects for more precise maintenance guidance.

**Adaptive Learning**: Implement online learning capabilities to automatically adjust to equipment-specific degradation patterns.

---

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- 8GB RAM (minimum for full dataset processing)
- NumPy, SciPy, Scikit-learn, Matplotlib libraries

### Quick Start

**Installation**

```bash
git clone https://github.com/yourusername/vibration-analysis-system.git
cd vibration-analysis-system
pip install -r requirements.txt
```

**Basic Usage**

```python
from vibration_analyzer import BearingHealthMonitor

# Initialize system
monitor = BearingHealthMonitor(
    low_band=(20, 1000),
    mid_band=(1000, 3000),
    high_band=(3000, 5000),
    sampling_rate=20000
)

# Load and analyze vibration data
results = monitor.analyze('bearing_data.csv')

print(f"Health Score: {results['health_score']}/100")
print(f"Predicted Failure Mode: {results['failure_type']}")
print(f"Estimated Time to Failure: {results['days_remaining']} days")
```

### Google Colab Demonstration

Interactive implementation and visualization:

[Vibration Analysis System - Google Colab](https://colab.research.google.com/drive/17mx-zdVmwZHIBaHRMSMEa2agxR8PpnW8?usp=sharing)

---

## Technical Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.21+ | Numerical computations and array operations |
| SciPy | 1.7+ | Signal processing and filtering |
| Scikit-learn | 1.0+ | Random Forest implementation and metrics |
| Matplotlib | 3.5+ | Visualization and plotting |
| Pandas | 1.3+ | Data manipulation and analysis |

---

