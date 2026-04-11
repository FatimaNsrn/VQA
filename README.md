# Visual Question Answering — BLIP Fine-tuned

Ask natural language questions about any image using a fine-tuned BLIP model served through a Flask web app.

## Demo
Upload an image → type a question → get an answer.

## Project Structure

├── app.py                  # Flask backend
├── model_loader.py         # Model loading and inference
├── templates/
│   └── index.html          # Frontend UI
└── blip_vqa_finetuned/     # Fine-tuned model weights (download separately)

## How It Works
- **Model**: BLIP (blip-vqa-base) fine-tuned on 5,000 samples from VQA v2
- **Backend**: Flask handles image uploads and routes questions to the model
- **Inference**: model_loader.py loads the model once at startup and runs generation on each request
- **Frontend**: HTML/CSS interface with image preview and answer display

  ## Fine-tuning Details

**Dataset**: VQA v2 via HuggingFace streaming (lmms-lab/vqav2)
- 5,000 training / 1,000 validation / 1,000 test samples

**Model**: Salesforce/blip-vqa-base (~250M parameters), loaded in bfloat16

**Optimizer**: AdamW — lr=3e-5, weight_decay=0.01, eps=1e-8

**Scheduler**: Linear warmup for first 10% of steps, then linear decay

**Training**: 3 epochs, batch size 8, gradient clipping at 1.0, bfloat16 mixed precision via torch.amp.autocast

**Loss per epoch**:
| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 5.0420 | 0.1592 |
| 2 | 0.0934 | 0.1215 |
| 3 | 0.0523 | 0.1233 |

## Setup

```bash
pip install flask transformers torch pillow
```

Download the fine-tuned model and place the `blip_vqa_finetuned/` folder in the project root, then:

```bash
python app.py
```

Open `http://localhost:5000`

