import io
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

# ---------------------------------------------------------------------------
# Qwen Image size options
# ---------------------------------------------------------------------------

SIZE_OPTIONS = {
    "1:1  (768×768)":    "768*768",
    "1:1  (1024×1024)":  "1024*1024",
    "3:2  (1152×768)":    "1152*768",
    "3:2  (1560×1040)":  "1560*1040",
    "2:3  (768×1152)":  "768*1152",
    "2:3  (1040×1560)":  "1040*1560",
    "2:3  (1672×2508)":  "1672*2508",
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


# ---------------------------------------------------------------------------
# Wan 2.6 Image size options
# ---------------------------------------------------------------------------

WAN_SIZE_OPTIONS = {
    "1K (跟随输入图比例)":  "1K",
    "2K (跟随输入图比例)":  "2K",
    "1:1  (1280×1280)":    "1280*1280",
    "2:3  (800×1200)":     "800*1200",
    "3:2  (1200×800)":     "1200*800",
    "3:4  (960×1280)":     "960*1280",
    "4:3  (1280×960)":     "1280*960",
    "9:16 (720×1280)":     "720*1280",
    "16:9 (1280×720)":     "1280*720",
    "21:9 (1344×576)":     "1344*576",
}


class WanImageNode:
    """Wan 2.6 Image – 图像编辑 & 图文混排"""

    WAN_MODELS = ["wan2.6-image", "wan2.7-image", "wan2.7-image-pro"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (cls.WAN_MODELS, {"default": "wan2.6-image"}),
                "enable_interleave": (
                    "BOOLEAN",
                    {"default": False, "label_on": "图文混排", "label_off": "图像编辑"},
                ),
                "size": (list(WAN_SIZE_OPTIONS.keys()), {"default": "1K (跟随输入图比例)"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    FUNCTION = "generate"
    CATEGORY = "Qwen"

    # ------------------------------------------------------------------

    def _build_content(self, text, images):
        content = []
        if text.strip():
            content.append({"text": text.strip()})
        for img_tensor in images:
            if img_tensor is not None:
                content.append({"image": _tensor_to_base64(img_tensor)})
        return content

    # ------------------------------------------------------------------

    def _call_edit_mode(self, api_key, model, content, size_value, n,
                        prompt_extend, watermark, seed):
        """enable_interleave=False  →  图像编辑模式（同步调用）"""
        from dashscope.aigc.image_generation import ImageGeneration
        from dashscope.api_entities.dashscope_response import Message

        message = Message(role="user", content=content)

        kwargs = dict(
            model=model,
            api_key=api_key,
            messages=[message],
            negative_prompt=NEGATIVE_PROMPT,
            prompt_extend=prompt_extend,
            watermark=watermark,
            n=n,
            enable_interleave=False,
            size=size_value,
        )
        if seed > 0:
            kwargs["seed"] = seed

        response = ImageGeneration.call(**kwargs)

        if response.status_code != 200:
            raise RuntimeError(
                f"Wan API 调用失败\n"
                f"HTTP 返回码: {response.status_code}\n"
                f"错误码: {getattr(response, 'code', 'N/A')}\n"
                f"错误信息: {getattr(response, 'message', 'N/A')}\n"
                f"参考文档: https://help.aliyun.com/zh/model-studio/error-code"
            )

        image_urls = []
        for choice in response.output.choices:
            for item in choice.message.content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_urls.append(item["image"])
        return image_urls, None

    # ------------------------------------------------------------------

    def _call_interleave_mode(self, api_key, model, content, size_value,
                              max_images, watermark, seed):
        """enable_interleave=True  →  图文混排模式（异步调用 + 等待）"""
        from dashscope.aigc.image_generation import ImageGeneration
        from dashscope.api_entities.dashscope_response import Message

        message = Message(role="user", content=content)

        kwargs = dict(
            model=model,
            api_key=api_key,
            messages=[message],
            negative_prompt=NEGATIVE_PROMPT,
            enable_interleave=True,
            max_images=max_images,
            watermark=watermark,
            size=size_value,
        )
        if seed > 0:
            kwargs["seed"] = seed

        task = ImageGeneration.async_call(**kwargs)
        if task.status_code != 200:
            raise RuntimeError(
                f"Wan 任务创建失败\n"
                f"HTTP 返回码: {task.status_code}\n"
                f"错误码: {getattr(task, 'code', 'N/A')}\n"
                f"错误信息: {getattr(task, 'message', 'N/A')}"
            )

        logger.info("Wan 异步任务已创建, task_id=%s, 等待完成...",
                     task.output.task_id)

        result = ImageGeneration.wait(task=task, api_key=api_key)
        if result.output.task_status != "SUCCEEDED":
            raise RuntimeError(
                f"Wan 任务失败, 状态: {result.output.task_status}\n"
                f"错误码: {getattr(result, 'code', 'N/A')}\n"
                f"错误信息: {getattr(result, 'message', 'N/A')}"
            )

        image_urls = []
        text_parts = []
        for choice in result.output.choices:
            for item in choice.message.content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        image_urls.append(item["image"])
                    elif item.get("type") == "text":
                        text_parts.append(item["text"])

        generated_text = "\n".join(text_parts) if text_parts else None
        return image_urls, generated_text

    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        model: str,
        enable_interleave: bool,
        size: str,
        n: int,
        max_images: int,
        prompt_extend: bool,
        watermark: bool,
        seed: int,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ):
        import dashscope
        dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "未检测到 DASHSCOPE_API_KEY 环境变量，请先设置后再使用本节点。"
            )

        images = [img for img in (image_1, image_2, image_3, image_4)
                  if img is not None]
        input_image_count = len(images)

        if not enable_interleave and input_image_count == 0 and not text.strip():
            raise ValueError("图像编辑模式下请至少提供 1 张输入图像和文本提示词。")
        if enable_interleave and input_image_count > 1:
            raise ValueError("图文混排模式下最多只能输入 1 张参考图像。")

        content = self._build_content(text, images)
        size_value = WAN_SIZE_OPTIONS[size]

        mode_name = "图文混排" if enable_interleave else "图像编辑"
        logger.info(
            "Wan API 调用 | 模型=%s | 模式=%s | 输入图像=%d | size=%s",
            model, mode_name, input_image_count, size_value,
        )

        if enable_interleave:
            image_urls, generated_text = self._call_interleave_mode(
                api_key, model, content, size_value, max_images, watermark,
                seed,
            )
        else:
            image_urls, generated_text = self._call_edit_mode(
                api_key, model, content, size_value, n, prompt_extend,
                watermark, seed,
            )

        if not image_urls:
            raise RuntimeError("API 未返回任何图像。")

        log_lines = [
            f"模型名称: {model}",
            f"模式: {mode_name}",
            f"提示词: {text}",
            f"输入图像数量: {input_image_count}",
            f"输出图像数量: {len(image_urls)}",
            f"请求尺寸: {size_value}",
            "─" * 40,
        ]

        if generated_text:
            log_lines.append("生成文本:")
            log_lines.append(generated_text)
            log_lines.append("─" * 40)

        image_tensors = []
        for i, url in enumerate(image_urls, start=1):
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


# ===========================================================================
# Node registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "QwenImage": QwenImageNode,
    "WanImage": WanImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage": "Qwen Image Generation",
    "WanImage": "Wan 2.6 Image Generation",
}
