<!--Copyright 2026-present The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Bi-dimensional Weight-Decomposed Low-Rank Adaptation (BoRA)

> [!NOTE]
> This is a variant of LoRA and therefore everything that is possible with LoRA is valid for this method except otherwise stated on this page.

BoRA extends [DoRA](lora_variant_dora). DoRA learns one magnitude per row of the weight matrix (i.e. per output feature). BoRA additionally learns one magnitude per column (i.e. per input feature), which makes the weight decomposition symmetric along both dimensions of the matrix. For more information on BoRA, see https://huggingface.co/papers/2412.06441.

```py
from peft import LoraConfig

config = LoraConfig(use_bora=True, ...)
```

At initialization, both magnitude vectors are set to the row and column norms of the base weight, so that the adapted model produces the same outputs as the base model before training.

Current limitations:

- BoRA only supports linear layers (`torch.nn.Linear` and `transformers.pytorch_utils.Conv1D`).
- BoRA cannot be combined with `use_dora=True`.
- The norms used to rescale the weight are treated as constants during backpropagation (as is done for DoRA).
- Unlike DoRA, the forward pass cannot reuse the output of the base layer, since the column magnitudes rescale the input of the layer. BoRA is therefore somewhat slower than DoRA.
