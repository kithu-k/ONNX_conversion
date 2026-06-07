# ONNX_conversion

## Repository Structure

### 1. Notebooks
- `Resnet50.ipynb`: Deep residual feature extraction and transfer learning pipeline for complex flood terrain classification.
- `CNN_LSTM.ipynb`: Spatiotemporal sequence modeling using **ConvLSTM2D** to track and predict flood progression lines over sequential timesteps.
- `CNN_FCNN_UNet.ipynb`: A unified multi-model benchmark pipeline evaluating static segmentors under identical datasets, handling spatial pixel imbalances with custom Combined Losses (BCE + Dice Loss).

### 2. Models (`/models`)
- `CNN_LSTM.onnx` 
- `fcnn.onnx` 
- `simple_cnn.onnx` 
- `u-net.onnx` 
