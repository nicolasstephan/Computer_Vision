## Parking Lot Occupancy Classifier
This Jupyter Notebook trains and evaluates a binary image classifier to determine whether a parking spot is occupied or free using video input. The pipeline uses TensorFlow with MobileNetV2 via transfer learning and fine-tuning. It also includes video frame extraction, contour-based segmentation, and frame-by-frame classification visualization.


![frame_01_annotated](https://github.com/user-attachments/assets/4a7b58cc-9047-4815-806d-546d369d847e)


### Features
- Preprocessing of labeled image data (empty vs not_empty)
- Image augmentation for improved generalization
- Model training using MobileNetV2 with early stopping and learning rate scheduling
- Fine-tuning of the top layers of the pre-trained model
- Evaluation with accuracy, loss, and a classification report
- Frame extraction from video using a binary mask of parking spots
- Cropping, resizing, and classification of each parking spot in selected frames
- Visualization and overlay of predictions on original video frames

### Requirements
- Python 3.8+
- TensorFlow 2.19+
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Usage
#### Prepare the Dataset
Structure your dataset as:

clf-data/

  empty/

  not_empty/

#### Train the Model
Run the notebook cells up to training and fine-tuning the model.

#### Evaluate Performance
Accuracy, loss, confusion matrix, and a classification report are generated to evaluate the model.

#### Analyze a Video

Provide a parking lot video and a corresponding binary mask.
The notebook extracts frames, crops individual spots, classifies each, and overlays predictions.

#### Output
A video with drawn bounding boxes for each spot labeled as either free (green) or taken (red) is generated.

### Notes

CUDA and cuDNN setup is optional but recommended for faster training.
This notebook is designed for educational and prototyping purposes. Real-world deployment may require additional calibration, robust detection pipelines, and environmental variability handling.
