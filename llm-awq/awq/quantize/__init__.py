try:
    from .w8a8_linear import *
except ModuleNotFoundError:
    # awq_inference_engine is llm-awq's compiled CUDA extension (built later,
    # see Task 7 of the DVD AWQ-quant plan). W8A8 static-scale linears are an
    # unrelated feature (int8 activation+weight) not used by the W4A16
    # weight-only auto_scale/auto_clip search this repo needs, so degrade
    # gracefully instead of blocking every import of awq.quantize.
    pass
from .smooth import *
