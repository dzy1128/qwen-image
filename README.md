# Qwen Image Generation - ComfyUI 自定义节点

基于阿里云通义万相（Qwen Image）API 的 ComfyUI 图像生成节点。

## 功能

- 支持 `qwen-image-2.0` 和 `qwen-image-2.0-pro` 模型
- 支持最多 3 张可选参考图像输入
- 支持单张/批量图像输出（最多 6 张）
- 多种常用分辨率和宽高比可选
- 输出详细日志信息（模型、提示词、图像地址、分辨率、格式、文件大小等）

## 依赖

```
dashscope>=1.20.0
requests>=2.28.0
Pillow>=9.0.0
numpy>=1.23.0
torch>=2.0.0
```

> `torch`、`numpy`、`Pillow` 通常已随 ComfyUI 安装，一般只需额外安装 `dashscope`。

### 快速安装

```bash
pip install dashscope
```

## 配置

使用前需设置阿里云百炼平台的 API Key 环境变量：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

API Key 获取地址：<https://help.aliyun.com/zh/model-studio/get-api-key>

## 使用说明

### 输入参数

| 参数 | 类型 | 说明 |
|---|---|---|
| model | 下拉选择 | `qwen-image-2.0`（默认）/ `qwen-image-2.0-pro` |
| text | 文本框 | 图像生成提示词，可为空 |
| n | 整数 | 输出图像数量，1-6，默认 1 |
| prompt_extend | 布尔 | 是否启用提示词扩展，默认关闭 |
| watermark | 布尔 | 是否添加水印，默认关闭 |
| seed | 整数 | 随机种子，范围 0-2147483647 |
| size | 下拉选择 | 输出图像尺寸，提供多种常用宽高比 |
| image_1 | 图像（可选） | 参考图像 1 |
| image_2 | 图像（可选） | 参考图像 2 |
| image_3 | 图像（可选） | 参考图像 3 |

### 可选尺寸

| 宽高比 | 可选分辨率 |
|---|---|
| 1:1 | 512×512, 768×768, 1024×1024, 1536×1536, 2048×2048 |
| 3:2 / 2:3 | 768×512, 1536×1024 / 512×768, 1024×1536 |
| 4:3 / 3:4 | 1024×768, 1536×1152 / 768×1024, 1152×1536 |
| 16:9 / 9:16 | 1024×576, 1536×864, 1920×1080 / 576×1024, 864×1536, 1080×1920 |

### 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| images | IMAGE | 生成的图像（支持批量） |
| log | STRING | 详细日志（模型、提示词、图像 URL、分辨率、格式、文件大小） |

## 节点位置

在 ComfyUI 中添加节点：`Qwen` → `Qwen Image Generation`
