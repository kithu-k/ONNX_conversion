import os
import sys

try:
    import tensorflow as tf
    import tf2onnx
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

def automate_unified_conversion(input_folder="saved_models", output_folder="models"):
    os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        print(f"Error: Directory '{input_folder}' not found.")
        sys.exit(1)

    model_files = os.listdir(input_folder)
    
    for file_name in model_files:
        model_path = os.path.join(input_folder, file_name)
        model_name, ext = os.path.splitext(file_name)
        output_onnx_path = os.path.join(output_folder, f"{model_name}.onnx")
        
        # ---------------------------------------------------------
        # TENSORFLOW / KERAS CONVERSION (.keras, .h5, .pb)
        # ---------------------------------------------------------
        if ext in ['.keras', '.h5', '.pb']:
            if not TF_AVAILABLE:
                print(f"[Skip] TensorFlow not installed, skipping {file_name}")
                continue
                
            print(f"\n[TF/Keras] Converting {file_name}...")
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                input_signature = [tf.TensorSpec(shape=model.inputs[0].shape, dtype=model.inputs[0].dtype)]
                
                onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
                with open(output_onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                print(f"[Success] Saved: {output_onnx_path}")
            except Exception as e:
                print(f"[Error] Failed to convert TF model {file_name}: {e}")

        # ---------------------------------------------------------
        # PYTORCH CONVERSION (.pth, .pt)
        # ---------------------------------------------------------
        elif ext in ['.pth', '.pt']:
            if not TORCH_AVAILABLE:
                print(f"[Skip] PyTorch not installed, skipping {file_name}")
                continue
                
            print(f"\n[PyTorch] Converting {file_name}...")
            try:
                # 1. Attempt to load as a standalone TorchScript model first
                try:
                    model = torch.jit.load(model_path, map_location='cpu')
                    is_torchscript = True
                except Exception:
                    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
                    is_torchscript = False
                
                model.eval()
                # 2. Extract expected input channels
                if is_torchscript:
                    detected_channels = 3 
                else:
                    detected_channels = next(model.parameters()).shape[1]
                
                # 3. Dynamically build the dummy tracer tensor
                dummy_input_shape = (1, detected_channels, 256, 256) 
                dummy_tensor = torch.randn(*dummy_input_shape)
                
                # 4. Export to ONNX
                # Force dynamo=False to utilize the classic stable exporter layout
                try:
                    torch.onnx.export(
                        model, 
                        dummy_tensor, 
                        output_onnx_path, 
                        export_params=True, 
                        opset_version=13, 
                        do_constant_folding=True, 
                        input_names=['input'], 
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                        dynamo=False 
                    )
                except TypeError:
                    # Backward-compatibility fallback if running on an older environment 
                    torch.onnx.export(
                        model, 
                        dummy_tensor, 
                        output_onnx_path, 
                        export_params=True, 
                        opset_version=13, 
                        do_constant_folding=True, 
                        input_names=['input'], 
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                print(f"[Success] Saved: {output_onnx_path}")          
            except Exception as e:
                print(f"[Error] Failed to convert PyTorch model {file_name}. Error: {e}")

        # ---------------------------------------------------------
        # CLASSICAL ML CONVERSION (.pkl, .pickle, .joblib)
        # ---------------------------------------------------------
        elif ext in ['.pkl', '.pickle', '.joblib']:
            if not SKL2ONNX_AVAILABLE:
                print(f"[Skip] 'skl2onnx' not installed, skipping {file_name}")
                continue
                
            print(f"\n[Classical ML] Processing {file_name}...")
            try:
                # 1. Load the model using joblib (best for scikit-learn numpy structures)
                try:
                    import joblib
                    model = joblib.load(model_path)
                except ImportError:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                
                # 2. Automatically discover the input feature count
                if hasattr(model, 'n_features_in_'):
                    num_features = model.n_features_in_
                elif hasattr(model, 'coef_'):
                    # Fallback configuration for basic linear models 
                    num_features = model.coef_.shape[-1] if len(model.coef_.shape) > 1 else model.coef_.shape[0]
                else:
                    raise ValueError("Could not automatically determine the number of input features. "
                                     "Ensure the model was fully trained before saving.")
                
                print(f"[Detect] Classical ML Model found. Features detected: {num_features}")
                
                # 3. Setup the initial structural blueprint (Batch size 'None', Feature Count)
                initial_type = [('float_input', FloatTensorType([None, num_features]))]
                
                # 4. Convert using skl2onnx extension
                options = {id(model): {'zipmap': False}} if hasattr(model, 'predict_proba') else None
                
                onnx_ml_model = convert_sklearn(
                    model, 
                    initial_types=initial_type, 
                    target_opset=13,
                    options=options
                )

                with open(output_onnx_path, "wb") as f:
                    f.write(onnx_ml_model.SerializeToString()) 
                print(f"[Success] Successfully converted to: {output_onnx_path}")
                
            except Exception as e:
                print(f"[Error] Failed to convert Classical ML model {file_name}. Error: {e}")

if __name__ == "__main__":
    automate_unified_conversion()