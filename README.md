# Sinapsis Pytesseract OCR Agent

## Project Overview
This project implements a custom OCR Agent for the **Sinapsis AI Framework**, leveraging **Google's Tesseract engine**. It is engineered to solve specific challenges found in student ID cards (UDLAP Credentials), such as holographic noise, low contrast, and geometric distortion.

The package is fully modular and integrates with the official [Sinapsis OCR Repository](https://github.com/Sinapsis-AI/sinapsis-ocr).

### Package Structure
To ensure the agent runs correctly on any device, the file structure must be preserved as follows:

```text
packages/sinapsis_pytesseract/
├── pyproject.toml              # Dependency definitions
├── README.md                   # This documentation
└── src/
    └── sinapsis_pytesseract/
        ├── configs/
        │   └── pytesseract_textIdInference.yaml  # MASTER CONFIGURATION (Run this)
        ├── helpers/
        │   ├── __init__.py
        │   └── tags.py         # Tag definitions for the UI
        └── templates/
            ├── __init__.py     # Template registration logic
            └── pytesseract_ocr.py  # Main logic (Auto-Scan + OCR)
```

## System Requirements (Prerequisites)
Before running the Python code, Tesseract must be installed at the system level.

### macOS (Homebrew)
```bash
brew install tesseract
brew install tesseract-lang
```

### Windows
1. Download the installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer. **CRITICAL:** During installation, expand "Additional Language Data" and select **Spanish**.
3. Add the installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your System Environment Variables (PATH).

### Linux (Ubuntu / Debian)
```bash
sudo apt-get update
# Install engine and dev tools
sudo apt-get install tesseract-ocr libtesseract-dev
# CRITICAL: Install Spanish language pack
sudo apt-get install tesseract-ocr-spa
```

## Installation
This package is designed to be installed as a plugin within the Sinapsis environment.

1. Clone or navigate to the root of the `sinapsis-ocr` repository.
2. Place the `sinapsis_pytesseract` folder inside `packages/`.
3. Install the package in editable mode:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Install the package
pip install -e packages/sinapsis_pytesseract
```
> **Note:** This command reads the `pyproject.toml` file and installs all necessary Python libraries (pytesseract, opencv-python, etc.) automatically.

## Usage
To execute the agent with the full preprocessing pipeline (Grayscale -> Blur -> Invert -> Auto-Scan), run:

```bash
sinapsis run packages/sinapsis_pytesseract/src/sinapsis_pytesseract/configs/pytesseract_textIdInference.yaml
```
The results (images with bounding boxes) will be saved in the dataset folder defined in the YAML configuration.

## 🔧 Configuration & Customization
The `PytesseractOCR` template can be configured in the YAML file. Below are the available parameters:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `tesseract_params.lang` | OCR Language (requires system pack). | "spa" |
| `tesseract_params.psm` | Page Segmentation Mode. (3=Auto, 11=Sparse). | 3 |
| `perspective_params.enable` | Activates the Auto-Scan (Crop & Straighten). | True |
| `min_confidence` | Minimum score (0.0-1.0) to filter garbage text. | 0.4 |

**Example YAML snippet:**
```yaml
- template_name: PytesseractOCR
  class_name: PytesseractOCR
  attributes:
    tesseract_params:
      lang: spa
      psm: 3
    perspective_params:
      enable: True
    min_confidence: 0.4
```

## Author
Roberto Benitez
Sinapsis AI Internship Project
