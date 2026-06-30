# SonoInstruct

It's a GitHub repository of paper "A Multimodal Instruction Dataset and Benchmark for Ultrasound Understanding".

## 📄 Paper

**A Multimodal Instruction Dataset and Benchmark for Ultrasound Understanding**

📌 Paper: upcoming  



## 📊 Dataset

📌 Hugging Face Dataset: [https://huggingface.co/datasets/Ssdaizi/SonoInstruct](https://huggingface.co/datasets/Ssdaizi/SonoInstruct) 



## 🤖 Model

📌 Hugging Face Model: https://huggingface.co/Ssdaizi/Qwen3-VL-2B-Sono


## 🚀 Quick Start

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
model_path = "Ssdaizi/Qwen3-VL-2B-Sono" 

# Load the model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test1.png",
            },
            {
                "type": "text",
                "text": "Is this a benign lesion or a malignant lesion?",
            }
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Generate response
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    top_p=0.8,
    top_k=20,
    temperature=0.7,
    repetition_penalty=1.0
)

# Remove input tokens from generated output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text[0])
```
