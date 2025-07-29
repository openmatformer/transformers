<!--Copyright 2025 Dustin Loring. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Minigpt4o

## Overview

Minigpt4o is a multimodal model based on the Gemma 3n architecture, but with a vocabulary size of 200,000 and codebase authored and adapted by Dustin Loring. Unlike Gemma 3n, all code for Minigpt4o is self-contained and does not import from other models; instead, the code was copied and renamed for clarity and independence.

Minigpt4o supports text, vision, and audio modalities, and is designed for research and experimentation with large-vocab, multimodal architectures. The model is intended as a template for further research and extension.

> [!TIP]
> Click on the Minigpt4o models in the right sidebar for more examples of how to apply Minigpt4o to different vision, audio, and language tasks.

The example below demonstrates how to generate text based on an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-text-to-text",
    model="dustinloring/minigpt4o-base",
    device=0,
    torch_dtype=torch.bfloat16
)
pipeline(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    text="<start_of_image> What is shown in this image?"
)
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoProcessor, Minigpt4oForConditionalGeneration

model = Minigpt4oForConditionalGeneration.from_pretrained(
    "dustinloring/minigpt4o-base",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
processor = AutoProcessor.from_pretrained(
    "dustinloring/minigpt4o-base",
    padding_side="left"
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
print(processor.decode(output[0], skip_special_tokens=True))
```

</hfoption>
<hfoption id="transformers CLI">

```bash
echo -e "Plants create energy through a process known as" | transformers run --task text-generation --model dustinloring/minigpt4o-base --device 0
```

</hfoption>
</hfoptions>

## Notes

-   Use [`Minigpt4oForConditionalGeneration`] for image-audio-and-text, image-and-text, image-and-audio, audio-and-text, image-only and audio-only inputs.
-   Minigpt4o supports multiple images per input, but make sure the images are correctly batched before passing them to the processor. Each batch should be a list of one or more images.

    ```py
    url_cow = "https://media.istockphoto.com/id/1192867753/photo/cow-in-berchida-beach-siniscola.jpg?s=612x612&w=0&k=20&c=v0hjjniwsMNfJSuKWZuIn8pssmD5h5bSN1peBd1CmH4="
    url_cat = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"

    messages =[
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url_cow},
                {"type": "image", "url": url_cat},
                {"type": "text", "text": "Which image is cuter?"},
            ]
        },
    ]
    ```
-   Text passed to the processor should have a `<image_soft_token>` token wherever an image should be inserted.
-   Minigpt4o accepts at most one target audio clip per input, though multiple audio clips can be provided in few-shot prompts, for example.
-   Text passed to the processor should have a `<audio_soft_token>` token wherever an audio clip should be inserted.
-   The processor has its own [`~ProcessorMixin.apply_chat_template`] method to convert chat messages to model inputs.

## Minigpt4oAudioFeatureExtractor

[[autodoc]] Minigpt4oAudioFeatureExtractor

## Minigpt4oProcessor

[[autodoc]] Minigpt4oProcessor

## Minigpt4oTextConfig

[[autodoc]] Minigpt4oTextConfig

## Minigpt4oVisionConfig

[[autodoc]] Minigpt4oVisionConfig

## Minigpt4oAudioConfig

[[autodoc]] Minigpt4oAudioConfig

## Minigpt4oConfig

[[autodoc]] Minigpt4oConfig

## Minigpt4oTextModel

[[autodoc]] Minigpt4oTextModel
    - forward

## Minigpt4oModel

[[autodoc]] Minigpt4oModel
    - forward

## Minigpt4oForCausalLM

[[autodoc]] Minigpt4oForCausalLM
    - forward

## Minigpt4oForConditionalGeneration

[[autodoc]] Minigpt4oForConditionalGeneration
    - forward 