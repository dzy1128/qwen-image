import io
import json
import os
import base64
import logging

import numpy as np
import torch
from PIL import Image
import requests

logger = logging.getLogger("QwenImage")

NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，"
    "人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)

SIZE_OPTIONS = {
    "1:1  (768×768)":    "768*768",
    "1:1  (1024×1024)":  "1024*1024",
    "3:2  (1152×768)":    "1152*768",
    "3:2  (1560×1040)":  "1560*1040",
    "2:3  (768×1152)":  "768*1152",
    "2:3  (1040×1560)":  "1040*1560",
    "4:3  (1536×1024)":   "1536*1024",
    "3:4  (1024×1536)":  "1024*1536",
    "16:9 (1920×1080)":  "1920*1080",
    "9:16 (1080×1920)":  "1080*1920",
    "16:10 (1920×1200)":  "1920*1200",
    "16:10 (1536×2560)":  "1536*2560",
    "10:16 (1200×1920)":  "1200*1920",
    "10:16 (1536×2560)":  "1536*2560",
}


def _tensor_to_base64(tensor: torch.Tensor) -> str:
    if tensor.dim() == 4:
        tensor = tensor[0]
    img_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _download_image(url: str, timeout: int = 180):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "unknown")
    file_size = len(resp.content)
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    fmt = img.format or content_type.split("/")[-1].upper()
    img_np = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np)
    return tensor, fmt, file_size


class QwenImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    ["qwen-image-2.0", "qwen-image-2.0-pro"],
                    {"default": "qwen-image-2.0"},
                ),
                "size": (list(SIZE_OPTIONS.keys()), {"default": "1:1  (1024×1024)"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1}),
                "prompt_extend": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "generate"
    CATEGORY = "Qwen"

    def generate(
        self,
        model: str,
        text: str,
        n: int,
        prompt_extend: bool,
        watermark: bool,
        seed: int,
        size: str,
        image_1=None,
        image_2=None,
        image_3=None,
    ):
        import dashscope
        from dashscope import MultiModalConversation

        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY 环境变量，请先设置后再使用本节点。"
            )

        content = []
        input_image_count = 0
        for img_tensor in (image_1, image_2, image_3):
            if img_tensor is not None:
                content.append({"image": _tensor_to_base64(img_tensor)})
                input_image_count += 1

        if text.strip():
            content.append({"text": text.strip()})

        if not content:
            raise ValueError("请至少提供一张输入图像或输入文本提示词。")

        messages = [{"role": "user", "content": content}]
        size_value = SIZE_OPTIONS[size]

        logger.info(
            "Qwen Image API 调用 | 模型=%s | 输入图像=%d | n=%d | size=%s",
            model,
            input_image_count,
            n,
            size_value,
        )

        response = MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            stream=False,
            n=n,
            watermark=watermark,
            negative_prompt=NEGATIVE_PROMPT,
            prompt_extend=prompt_extend,
            size=size_value,
            seed=seed,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"API 调用失败\n"
                f"HTTP 返回码: {response.status_code}\n"
                f"错误码: {response.code}\n"
                f"错误信息: {response.message}\n"
                f"参考文档: https://help.aliyun.com/zh/model-studio/error-code"
            )

        output_contents = response.output.choices[0].message.content

        log_lines = [
            f"模型名称: {model}",
            f"提示词: {text}",
            f"输入图像数量: {input_image_count}",
            f"输出图像数量: {len(output_contents)}",
            f"请求尺寸: {size_value}",
            "─" * 40,
        ]

        image_tensors = []
        for i, item in enumerate(output_contents, start=1):
            url = item.get("image", "")
            log_lines.append(f"图像 {i} URL: {url}")

            tensor, fmt, file_size = _download_image(url)
            h, w = tensor.shape[0], tensor.shape[1]

            log_lines.append(f"图像 {i} 分辨率: {w}×{h}")
            log_lines.append(f"图像 {i} 格式: {fmt}")
            log_lines.append(f"图像 {i} 文件大小: {file_size / 1024:.1f} KB")
            log_lines.append("")

            image_tensors.append(tensor)

        batch = torch.stack(image_tensors, dim=0)

        log_text = "\n".join(log_lines)
        logger.info("\n%s", log_text)

        return (batch, log_text)


NODE_CLASS_MAPPINGS = {
    "QwenImage": QwenImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage": "Qwen Image Generation",
}
