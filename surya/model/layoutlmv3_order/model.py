import torch
from transformers import LayoutLMv3ForTokenClassification

from surya.settings import settings


def load_model(
    checkpoint=settings.ORDER_LAYOUTLMV3_MODEL_CHECKPOINT,
    device=settings.TORCH_DEVICE_MODEL,
    dtype=settings.MODEL_DTYPE,
):
    model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint, torch_dtype=dtype)
    model = model.to(device)
    model = model.eval()

    if settings.LAYOUT_STATIC_CACHE:
        torch.set_float32_matmul_precision('high')
        torch._dynamo.config.cache_size_limit = 1
        torch._dynamo.config.suppress_errors = False

        print(f"Compiling layout model {checkpoint} on device {device} with dtype {dtype}")
        model = torch.compile(model)

    print(f"Loaded layout model {checkpoint} on device {device} with dtype {dtype}")
    return model
