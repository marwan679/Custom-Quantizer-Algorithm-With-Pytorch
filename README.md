# 🚀 PyTorch Custom Quantizer Algorithm

A cutting-edge implementation of a custom weight quantization algorithm for PyTorch models, enabling efficient 8-bit weight storage with 16-bit activation (W8A16) to reduce model size and inference time while maintaining performance.

## ✨ Features

- **Custom W8A16 Linear Layer**: Implements efficient 8-bit weight quantization with 16-bit activations
- **Model Agnostic**: Works with any PyTorch model containing Linear layers
- **Flexible Configuration**: Exclude specific layers (e.g., lm_head) from quantization
- **Preserved Performance**: Maintains model accuracy through careful scaling factor computation
- **Easy Integration**: Simple API to quantize existing models with minimal code changes

## 🏗️ Architecture

The quantizer replaces standard `nn.Linear` layers with custom `W8A16LinearLayer` modules that:

- Store weights as 8-bit integers (`int8`)
- Maintain scaling factors to preserve numerical precision
- Support optional bias terms
- Use custom forward pass for efficient computation

## 🚦 Quick Start

### Installation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### Basic Usage

```python
# Create your model
model = MyCustomModel()

# Apply quantization (excluding specific layers)
Quantizer(model, W8A16LinearLayer, ["lm_head"])

# Use as normal
output = model(input_data)
```

### Creating Quantized Layers

```python
# Manual layer creation
quant_layer = W8A16LinearLayer(in_features=512, out_features=256, bias=True)

# Quantize existing weights
quant_layer.quantize(original_weights)
```

## 🧪 Examples

### Transformers Integration

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pre-trained model
model_id = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Apply quantization (preserve lm_head)
Quantizer(model, W8A16LinearLayer, ["lm_head"])

# Use quantized model
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

### Object Detection Models

```python
from transformers import DetrForObjectDetection, DetrImageProcessor

# Load detection model
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# Apply quantization
Quantizer(model, W8A16LinearLayer, [])

# Significant memory reduction!
previous_memory = model.get_memory_footprint()  # Check savings
```

## 📊 Performance Benefits

- **Reduced Memory Footprint**: 8-bit weights instead of 32-bit (4x compression)
- **Faster Inference**: Integer operations are faster on many hardware platforms
- **Maintained Accuracy**: Careful scaling preserves model performance

## 🎯 How It Works

1. **Weight Quantization**: Convert FP32 weights to INT8 using per-channel scaling factors
2. **Scale Calculation**: `scales = abs(weights).max(dim=-1).values / 127`
3. **Integer Conversion**: `int8_weights = round(weights / scales.unsqueeze(1))`
4. **Efficient Forward**: Dequantize during computation: `output = linear(input, int8_weights) * scales`

## 🔧 Advanced Configuration

### Custom Exclusion Patterns

```python
# Exclude multiple layers
excluded_layers = ["lm_head"]
Quantizer(model, W8A16LinearLayer, excluded_layers)
```

### Manual Quantization Control

```python
# Access quantized layers directly
for name, module in model.named_modules():
    if isinstance(module, W8A16LinearLayer):
        print(f"Layer {name}: scales={module.scales.shape}")
```

## 📈 Validation

The implementation includes comprehensive testing with:

- 🤖 Transformer-based language models
- 👁️ Computer vision models (object detection)
- 🧪 Custom architectures
- ✅ Comparison against original FP32 models

## 🚨 Limitations

- Currently supports only Linear layers
- May require fine-tuning for optimal performance
- Hardware-specific optimizations not included

## 🔮 Future Enhancements

- [ ] Support for more layer types (Conv, LSTM, etc.)
- [ ] Automatic calibration for optimal scaling factors
- [ ] Hardware-aware quantization schemes
- [ ] Quantization-aware training support
- [ ] ONNX export support

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues, feature requests, and pull requests.
