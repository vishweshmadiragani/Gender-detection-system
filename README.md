# Gender Detection ML Project

A real-time gender detection system using deep learning and computer vision. This project can detect and classify faces as male or female in real-time using your webcam.

DOWNLOAD THE MODELS FROM THIS LINK: https://www.dropbox.com/scl/fo/8z77zddt6c1iearscw7oi/ALcX7UVJ6Fc0hTjHnIHSXaY?rlkey=jrxwwoom0771xgg9bph6u6sg2&st=8fy8unaz&dl=0

## 🌟 Features

- **Real-time Detection**: Live gender detection using webcam feed
- **Face Detection**: Automatic face detection and cropping
- **High Accuracy**: Trained CNN model with data augmentation
- **Easy to Use**: Simple Python script for real-time detection
- **Cross-platform**: Works on Windows, macOS, and Linux

## 📁 Project Structure

```
MALE_FEMALE/
├── gender_dataset_face/          # Training dataset
│   ├── man/                     # Male face images
│   └── woman/                   # Female face images
├── gender_detection.keras        # Trained model file
├── gender_detection.model        # Alternative model format
├── gender.py                     # Real-time detection script
├── train.py                      # Model training script
├── requirements.txt              # Python dependencies
├── plot.png                      # Training metrics visualization
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam
- Sufficient lighting for face detection

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd MALE_FEMALE
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run real-time detection**
   ```bash
   python gender.py
   ```

### Usage

1. **Start the application**:
   ```bash
   python gender.py
   ```

2. **Position yourself in front of the webcam**

3. **View the results**:
   - Green rectangles around detected faces
   - Gender prediction with confidence percentage
   - Example: "man: 95.67%" or "woman: 87.23%"

4. **Exit the application**:
   - Press 'Q' to quit

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: 96x96x3 RGB images
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Batch Normalization**: For training stability
- **MaxPooling**: For dimensionality reduction
- **Dropout**: For regularization (0.25-0.5)
- **Dense Layers**: Fully connected layers for classification
- **Output**: Binary classification (Male/Female)

### Training Details

- **Epochs**: 100
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Data Augmentation**: Rotation, shifts, zoom, horizontal flip

## 📊 Dataset

The model is trained on a custom dataset containing:
- **Male faces**: 731 images
- **Female faces**: 1,131 images
- **Total**: 1,862 face images

### Data Preprocessing

- Images resized to 96x96 pixels
- Normalized pixel values (0-1)
- Data augmentation for better generalization
- Train/validation split (80/20)

## 🔧 Training Your Own Model

If you want to train the model with your own dataset:

1. **Prepare your dataset**:
   ```
   gender_dataset_face/
   ├── man/
   │   ├── face_1.jpg
   │   ├── face_2.jpg
   │   └── ...
   └── woman/
       ├── face_1.jpg
       ├── face_2.jpg
       └── ...
   ```

2. **Update the dataset path** in `train.py`:
   ```python
   image_files = [f for f in glob.glob(r'path/to/your/dataset' + "/**/*", recursive=True) if not os.path.isdir(f)]
   ```

3. **Run training**:
   ```bash
   python train.py
   ```

4. **Monitor training**:
   - Training progress will be displayed
   - Loss and accuracy plots saved as `plot.png`
   - Model saved as `gender_detection.keras`

## 📋 Dependencies

- **tensorflow>=2.0.0**: Deep learning framework
- **opencv-python>=4.5.0**: Computer vision library
- **numpy>=1.19.0**: Numerical computing
- **scikit-learn>=0.24.0**: Machine learning utilities
- **matplotlib>=3.3.0**: Plotting and visualization
- **cvlib>=0.2.0**: Face detection library

## 🎯 Performance

- **Real-time Processing**: ~30 FPS on modern hardware
- **Accuracy**: Varies based on lighting, angle, and face quality
- **Face Detection**: Uses cvlib for robust face detection
- **Gender Classification**: CNN-based binary classification

## 🔍 How It Works

1. **Face Detection**: Uses cvlib to detect faces in each frame
2. **Face Cropping**: Extracts and crops detected face regions
3. **Preprocessing**: Resizes to 96x96 and normalizes pixel values
4. **Gender Prediction**: CNN model predicts male/female with confidence
5. **Visualization**: Draws bounding boxes and labels on the frame

## 🛠️ Troubleshooting

### Common Issues

1. **Webcam not working**:
   - Check if webcam is connected and not in use by other applications
   - Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

2. **Poor detection accuracy**:
   - Ensure good lighting conditions
   - Face should be clearly visible and not too far from camera
   - Avoid extreme angles

3. **Model not loading**:
   - Ensure `gender_detection.keras` file exists in the project directory
   - Check if TensorFlow version is compatible

4. **Dependencies issues**:
   - Create a virtual environment: `python -m venv venv`
   - Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
   - Install requirements: `pip install -r requirements.txt`

## 📈 Future Improvements

- [ ] Add age detection
- [ ] Support for multiple faces simultaneously
- [ ] Mobile app integration
- [ ] API endpoint for web applications
- [ ] Improved model architecture (ResNet, EfficientNet)
- [ ] Better data augmentation techniques
- [ ] Confidence threshold adjustment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created for ES-IOT Assignment 2 - Gender Detection using Machine Learning

---

**Note**: This project is for educational purposes. Always respect privacy and obtain consent when using face detection technologies. 
