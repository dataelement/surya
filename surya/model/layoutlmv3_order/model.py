import torch
from transformers import LayoutLMv3ForTokenClassification

from surya.settings import settings


def load_model(
    checkpoint=settings.ORDER_LAYOUTLMV3_MODEL_CHECKPOINT,
    device=settings.TORCH_DEVICE_MODEL,
    dtype=settings.MODEL_DTYPE,
):
    if torch.cuda.is_available():
        device = device
        if torch.cuda.is_bf16_supported():
            supports_bfloat16 = True
        else:
            supports_bfloat16 = False
    else:
        device = torch.device('cpu')
        supports_bfloat16 = False

    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint, torch_dtype=dtype)
    if supports_bfloat16:
        model.bfloat16()
    model.to(device).eval()

    if settings.LAYOUT_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 1
        torch._dynamo.config.suppress_errors = False

        print(f"Compiling layout model {checkpoint} on device {device} with dtype {dtype}")
        model = torch.compile(model)

    print(f"Loaded layout model {checkpoint} on device {device} with dtype {dtype}")
    return model
