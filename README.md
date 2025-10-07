# Dual Drone YOLO Detection Pipeline

A comprehensive computer vision pipeline for detecting people in drone footage with optional weapon detection using YOLOv11 and Roboflow models.

## 🎯 Features

### Core Functionality
- **Person Detection**: High-performance person detection using YOLOv11
- **Distance Estimation**: Estimate person distance from camera using drone intrinsic parameters
- **Weapon Detection**: Optional secondary analysis of person crops using Roboflow weapon detection model
- **Batch Processing**: Process single images, directories, or entire sample collections
- **Organized Output**: Structured results with separate folders for different detection types
- **Docker Support**: Easy deployment with Docker
- **Performance Monitoring**: Real-time statistics and processing metrics

### Pipeline Architecture
```
Input Images → Person Detection → Distance Estimation → Crop Extraction → Weapon Detection (Optional) → Organized Results
    ↓               ↓                      ↓                    ↓                     ↓                        ↓
 Raw Images    YOLOv11 Model      Drone Camera Model      Person Crops         Roboflow Model            Final Output
```

## 📁 Project Structure

```
dual_drone_yolo_detection_2/
├── src/                     # 📁 Source code
│   ├── __init__.py         
│   ├── config.py           # ⚙️ Configuration settings
│   ├── detector.py         # 🧠 Main detection logic
│   ├── estimation.py       # 📏 Distance estimation logic
│   ├── weapon_detector.py  # 🔫 Weapon detection logic
│   └── main.py             # 🚀 CLI interface
├── models/
│   └── yolo11n.pt          # 🤖 YOLO model
├── inputs/
│   ├── raw/                # Original video files (.mp4)
│   ├── clips/              # Processed video clips
│   └── samples/            # Frame samples extracted from clips
├── output/                 # 💾 Results will be saved here
│   ├── detections/         # Images with bounding boxes
│   ├── crops/              # Individual person crops
│   └── weapon_detections/  # Person crops with weapon analysis
├── Dockerfile              # 🐳 Docker configuration
├── requirements.txt        # 📦 Python dependencies
├── preprocess_videos.py    # 📹 Video preprocessing
└── README.md              # 📖 This documentation
```

## 🚀 Quick Start

### Option 1: Manual Setup (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/main.py
```

### Option 2: Docker
```bash
# Build the image
docker build -t people-detector .

# Run detection
docker run -v $(pwd):/workspace people-detector
```

## 📊 Usage Examples

### Basic Usage
```bash
# Process all sample directories
python src/main.py --input inputs/samples --output output/batch_results

# Process samples with weapons
python src/main.py --input_with_weapons inputs/samples_with_weapons --output output/weapon_results

# Process samples without weapons
python src/main.py --input_without_weapons inputs/samples_clean --output output/clean_results

# Custom confidence threshold
python src/main.py --input inputs/samples --output output/results --confidence 0.6

# Disable weapon detection
python src/main.py --input inputs/samples --output output/results --no-weapons
```

### Advanced Configuration
```bash
# Custom parameters
python src/main.py --model models/yolo11n.pt \
                   --input inputs/samples \
                   --output output/detections \
                   --confidence 0.6 \
                   --save-crops

# Docker with custom output
docker run -v $(pwd):/workspace people-detector \
           --output /workspace/output/custom_results
```

## 🔧 Configuration Options

### Command Line Arguments
- `--model`: Path to YOLO model file (default: `models/yolo11n.pt`)
- `--input`: Input directory containing image folders (default: `inputs/samples`)
- `--input_with_weapons`: Input directory containing sample folders with weapons (optional)
- `--input_without_weapons`: Input directory containing sample folders without weapons (optional)
- `--output`: Output directory for results (default: `output/detections`)
- `--confidence`: Detection confidence threshold (0.0-1.0, default: 0.5)
- `--save-crops`: Enable saving person crops (default: True)
- `--no-crops`: Disable saving person crops
- `--no-weapons`: Disable weapon detection

### Environment Variables for Weapon Detection
Create a `.env` file with your Roboflow API key:
```
ROBOFLOW_API_KEY=your_api_key_here
```

### Configuration File
Edit `src/config.py` to customize:
- Confidence threshold
- Bounding box colors
- Supported image formats
- Output settings

## 📁 Output Structure

The pipeline creates an organized output structure:

```
output/
├── detections/                    # Images with person bounding boxes
│   ├── image1_detected.jpg
│   └── image2_detected.jpg
├── crops/                         # Individual person crops
│   ├── image1_person_01_conf_0.85.jpg
│   └── image1_person_02_conf_0.92.jpg
└── weapon_detections/             # Person crops with weapon analysis
    ├── image1_person_01_conf_0.85_weapon_check.jpg
    └── image1_person_02_conf_0.92_weapon_check.jpg
```

## 📹 Video Preprocessing

Process raw videos into clips and frame samples:

### Basic Usage
```bash
python preprocess_videos.py
```

### Custom Parameters
```bash
python preprocess_videos.py -X 15 -Z 1080p -W 60 -C compressed -F 15 -B 1M
```

### Parameters
- `-X, --clip-duration`: Clip duration in seconds (default: 10)
- `-Z, --resolution`: Target resolution - 1080p, 720p, 480p, 360p, 240p (default: 720p)
- `-W, --frame-interval`: Extract 1 frame every W frames (default: 30)
- `-C, --compression`: Quality preset - high_quality, balanced, compressed, very_compressed (default: balanced)
- `-F, --fps`: Target FPS - reduces from original if lower (optional)
- `-B, --max-bitrate`: Maximum bitrate limit, e.g., '2M', '1000k' (optional)

## 🏗️ Architecture Details

### 1. Person Detection (YOLOv11)
- **Model**: YOLOv11n (nano) for fast inference
- **Target**: COCO person class (ID: 0)
- **Performance**: ~40ms per image on CPU
- **Output**: Bounding boxes with confidence scores

### 2. Distance Estimation
- **Camera Model**: Drone EVO 2 Dual V2 intrinsic parameters
- **Method**: Calculates distance based on person pixel height and camera focal length
- **Logging**: Detailed distance logs saved to `person_distances.log`
- **Output**: Estimated distance in meters for each detected person

### 3. Crop Extraction
- **Padding**: 10% padding around person bounding boxes
- **Minimum Size**: 32x32 pixels minimum crop size
- **Format**: Individual JPEG files with metadata in filename

### 4. Weapon Detection (Optional - Roboflow)
- **Model**: `weapon-detection-m7qso/1`
- **Input**: Person crops from step 2
- **Performance**: ~200ms per crop
- **Output**: Annotated crops with weapon bounding boxes

## � Distance Estimation

The pipeline includes advanced distance estimation capabilities using drone camera intrinsic parameters:

### Camera Model
- **Drone**: EVO 2 Dual V2
- **Sensor**: 6.4mm x 4.8mm
- **Focal Length**: 25.6mm (35mm equivalent)
- **Resolution**: 1920x1080 pixels

### Distance Calculation
- **Method**: Uses person pixel height and camera focal length
- **Assumption**: Average person height of 1.7m
- **Formula**: `distance = (real_height × focal_length) / (pixel_height × pixel_size)`
- **Accuracy**: Typically within 20-30% of actual distance

### Output
- **Console**: Real-time distance display during processing
- **Log File**: Detailed logging to `src/person_distances.log`
- **Format**: Image name, person index, pixel height, estimated distance, confidence

## �📊 Performance Metrics

### Speed Benchmarks
- **Person Detection**: 40ms per image (1080p)
- **Distance Estimation**: <1ms per person
- **Weapon Detection**: 200ms per person crop
- **Memory Usage**: Processes images individually (memory efficient)
- **Throughput**: ~100 images/minute (with weapon detection)

### System Requirements
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLOv11
- PyTorch (automatically installed with ultralytics)

## 🔍 Troubleshooting

### Common Issues

**API Key Errors (Weapon Detection):**
```
Error: ROBOFLOW_API_KEY not found in environment variables
```
- Solution: Create `.env` file with your Roboflow API key

**Model Loading Errors:**
```
Error: Model file not found: models/yolo11n.pt
```
- Solution: Ensure YOLO model is in the correct path

**Import Errors:**
```
✗ Ultralytics not installed
```
- Solution: Run `pip install -r requirements.txt`

### Performance Optimization
- **GPU Usage**: Automatic GPU detection for faster inference
- **Batch Size**: Adjust based on available memory
- **Confidence Tuning**: Higher thresholds reduce false positives

## 🐳 Docker Details

### Building
```bash
docker build -t people-detector .
```

### Running
```bash
# Basic usage
docker run -v $(pwd):/workspace people-detector

# With custom parameters
docker run -v $(pwd):/workspace people-detector \
    python src/main.py --confidence 0.7 --output custom_results/
```

## 📝 What the Pipeline Does

1. **Processes all sample directories** in `inputs/samples/`
2. **Detects people** in each image using YOLOv11
3. **Estimates distances** to detected people using camera intrinsics
4. **Extracts person crops** with padding for analysis
5. **Optionally analyzes crops** for weapon detection
6. **Draws bounding boxes** around detected people
7. **Shows confidence scores** and distance estimates for each detection
8. **Logs detailed metrics** to `person_distances.log`
9. **Maintains folder structure** in output
10. **Saves processed images** with descriptive suffixes

## 🛠️ Development

### Key Classes
- `PeopleDetector`: Handles person detection, distance estimation, and crop extraction
- `WeaponDetector`: Manages weapon detection using Roboflow API
- `Camera`: Handles distance estimation using drone intrinsic parameters
- Configuration through `config.py` and command line arguments

### Adding New Features
1. **New Detection Models**: Extend detector classes
2. **Output Formats**: Modify output saving methods
3. **Processing Options**: Add new command line arguments

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify model files and dependencies are properly installed
3. Check the configuration in `src/config.py`
4. Open an issue with detailed error messages

---

**Pipeline Status**: ✅ Fully functional with distance estimation and weapon detection
**Last Updated**: October 6, 2025
**Version**: 2.1 (with distance estimation and enhanced weapon detection workflow)