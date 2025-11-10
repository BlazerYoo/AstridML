# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AstridML** is a machine learning pipeline designed to help female athletes optimize their health and performance by analyzing menstrual cycle symptoms and wearable training data. The project addresses the research gap in women's sports science (less than 6% of sports science research focuses exclusively on female athletes).

### Core Objective
Build a pipeline that collects and streams health data from wearable devices into an ML pipeline that processes data and provides nutritional, recovery, and performance recommendations.

## Architecture

The system consists of three main components:

### 1. Synthetic Data Generator (SDG)
- Simulates wearable training data and menstrual cycle symptoms
- Used for testing and development before integrating real wearable device streams
- Future goal: Direct streaming from wearables/Apple HealthKit

### 2. Machine Learning Pipeline
- **Data Preprocessing Module (DPM)**: Cleans and reformats data, engineers features for ML models
- **ML Model Training**: Predicts symptoms and athletic performance; prescribes recommendations
- **Model Evaluation**: Validates prediction accuracy against thresholds

### 3. API Endpoint
- Receives wearable data (simulated by SDG initially)
- Routes data to DPM in the ML pipeline

## Technology Stack

- **Language**: Python
- **ML/Data Libraries**: NumPy, pandas, SciPy, scikit-learn, TensorFlow
- **GPU Computing**: CUDA
- **Testing**: pytest (functional and property-based testing at unit and integration levels)
- **Package Management**: uv
- **CI/CD**: GitHub Actions
- **Version Control**: Git/GitHub

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_<module>.py

# Run with verbose output
pytest -v

# Run property-based tests
pytest -v tests/test_properties.py
```

### Package Management
```bash
# Build package with uv
uv build

# Publish package
uv publish
```

## Testing Strategy

The project uses **pytest** with both functional and property-based testing:

- **SDG Tests**: Verify realistic random data generation
- **DPM Tests**: Validate correct input/output shape and dimensions
- **ML Model Tests**: Ensure correct tensor dimensions and data formats
- **API Tests**: Verify requests are correctly routed to DPM/ML pipeline
- **Test Scopes**: Both unit and integration tests

## Key Design Considerations

### Data Flow
1. Synthetic data (or real wearable data) → API endpoint
2. API → Data Preprocessing Module (DPM)
3. DPM → ML Model (training/inference)
4. ML Model → Recommendations output

### Model Requirements
- Predict symptoms and athletic performance based on historical data
- Meet accuracy thresholds for predictions
- Generate actionable nutritional, recovery, and performance recommendations

### GPU Utilization
When implementing ML components, ensure CUDA is properly configured for TensorFlow operations to leverage GPU acceleration during training and inference.

## Module Organization

Expected module structure (to be implemented):
- **sdg/**: Synthetic data generation
- **dpm/**: Data preprocessing and feature engineering
- **models/**: ML model definitions and training code
- **api/**: API endpoint implementation
- **tests/**: Comprehensive test suite
