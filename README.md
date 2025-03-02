# DL-Assignment
160122737159
# Fashion MNIST Feedforward Neural Network

## Requirements
- Ensure Python 3.x is installed.
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```
- Run the model training with:
  ```bash
  python train.py
  ```
- Evaluate the model:
  ```bash
  python evaluate.py
  ```

## Dataset
- The Fashion MNIST dataset is automatically downloaded using TensorFlow/Keras.
- It contains 70,000 grayscale images of size 28x28 pixels, divided into 10 classes.
- The dataset is preprocessed by normalizing pixel values to the range [0,1] to improve model performance.
- **Normalization:** Suppose pixel values range from 0 to 255, dividing by 255 scales the values to [0,1]. This helps the model process the data more efficiently and speeds up training. Normalization ensures that variations in pixel intensities do not negatively impact learning, making convergence faster and more stable.

## Hyperparameter Tuning
The model supports flexible hyperparameter selection, including:
- Learning rates: `1e-3, 1e-4`
- Batch sizes: `16, 32, 64`
- Number of epochs: `10, 20, 50`
- Optimizers: `SGD, Adam, RMSprop`
- Activation functions: `ReLU, Softmax (for final layer)`
- Regularization: `L2 (weight decay 0.001)`
- Hidden layer sizes: `64, 64`

## Findings from Experiments
### Best Performance:
- **Batch Size:** 64
- **Weight Decay:** 0.001 (L2 Regularization)
- **Hidden Layer Sizes:** 64, 64
- **Optimizer:** Adam
- **Activation Functions:** ReLU (hidden layers), Softmax (output layer)
- **Learning Rate:** `1e-3`
- **Training Accuracy:** 80%
- **Test Accuracy:** 84%
- **Validation Accuracy:** 85%

### Configurations That Did Not Work:
- **High Learning Rate:** Led to instability and poor convergence.
- **Low Batch Size (<32):** Increased training time and led to high variance.

### Recommendations:
1. **Training Duration:** Training the model for longer improves performance, but overfitting must be avoided.
2. **Batch Size Selection:** Choose a batch size that balances stability and performance. Larger batch sizes (>128) should be avoided.
3. **Epochs Based on Model Size:** The number of epochs should be selected considering the model's complexity to prevent underfitting or overfitting.
4. **Regularization Usage:** Applying L2 regularization improves model robustness and prevents overfitting.

## Evaluation Results
- The final trained model achieves high validation accuracy.
- The confusion matrix indicates strong classification performance across all Fashion MNIST categories.
- Model training and evaluation steps are clearly defined and easy to reproduce.

