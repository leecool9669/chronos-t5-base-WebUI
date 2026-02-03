# -*- coding: utf-8 -*-
"""Chronos-T5 Base 时间序列预测与可视化 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr
import numpy as np


def fake_load_model():
    """模拟加载 Chronos-T5 Base 模型，仅用于界面演示。"""
    return "模型状态：Chronos-T5 Base 已就绪（演示模式，未加载真实权重）"


def fake_predict(history_text: str, prediction_length: int, num_samples: int):
    """模拟预测结果与可视化描述。"""
    try:
        plen = max(1, min(64, int(prediction_length)) if prediction_length is not None else 12)
    except (TypeError, ValueError):
        plen = 12
    try:
        nsamp = max(1, min(100, int(num_samples)) if num_samples is not None else 20)
    except (TypeError, ValueError):
        nsamp = 20
    if not (history_text or "").strip():
        return (
            "请输入或粘贴历史时间序列数据（每行一个数值，或逗号/空格分隔）。\n"
            "示例：1.2, 1.5, 1.8, 2.0, 1.9, 2.1"
        ), None
    lines = [
        "[演示] 已对历史序列进行 Chronos-T5 预测（未加载真实模型）。",
        f"历史长度：{len([x for x in history_text.replace(',', ' ').split() if x.strip()])} 点",
        f"预测长度：{plen} 步，采样数：{nsamp}",
        "",
        "说明：加载真实 Chronos-T5 Base 模型后，将在此显示预测中位数、80% 预测区间及可视化图表。",
    ]
    text_out = "\n".join(lines)
    try:
        parts = [float(x.strip()) for x in history_text.replace(",", " ").split() if x.strip()]
    except ValueError:
        parts = [1.0, 1.2, 1.5, 1.8, 2.0, 1.9, 2.1]
    if len(parts) < 2:
        parts = [1.0, 1.2, 1.5, 1.8, 2.0]
    x_hist = np.arange(len(parts))
    x_pred = np.arange(len(parts), len(parts) + plen)
    median = np.linspace(parts[-1], parts[-1] + 0.3, plen)
    low, high = median - 0.2, median + 0.2
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(x_hist, parts, color="royalblue", label="历史数据")
        plt.plot(x_pred, median, color="tomato", label="预测中位数（演示）")
        plt.fill_between(x_pred, low, high, color="tomato", alpha=0.3, label="80% 预测区间（演示）")
        plt.legend(); plt.grid(True, alpha=0.3); plt.xlabel("时间步"); plt.ylabel("数值"); plt.tight_layout()
        fig = plt.gcf()
    except Exception:
        pass
    return text_out, fig


def build_ui():
    with gr.Blocks(title="Chronos-T5 Base WebUI") as demo:
        gr.Markdown("## Chronos-T5 Base · 时间序列预测与可视化 WebUI 演示")
        gr.Markdown("本界面以交互方式展示 Chronos-T5 Base 的典型使用流程：模型加载、历史序列输入、预测长度与采样数设置、预测结果及可视化（演示模式，未加载真实模型）。")
        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)
        with gr.Tabs():
            with gr.Tab("预测"):
                gr.Markdown("输入或粘贴历史时间序列（每行一个数值，或逗号/空格分隔），设置预测步长与采样数。")
                history_in = gr.Textbox(label="历史时间序列", placeholder="例如：1.2, 1.5, 1.8, 2.0, 1.9, 2.1, 2.2", lines=4)
                with gr.Row():
                    pred_len = gr.Slider(1, 64, value=12, step=1, label="预测长度")
                    num_samp = gr.Slider(1, 50, value=20, step=1, label="采样数")
                pred_btn = gr.Button("预测（演示）")
                text_out = gr.Textbox(label="预测结果说明", lines=8, interactive=False)
                plot_out = gr.Plot(label="预测可视化（演示）")
                pred_btn.click(fn=fake_predict, inputs=[history_in, pred_len, num_samp], outputs=[text_out, plot_out])
            with gr.Tab("关于"):
                gr.Markdown("Chronos-T5 Base 为基于 T5 架构的预训练时间序列预测模型，约 200M 参数。将时间序列量化为词元后使用交叉熵训练，推理时自回归采样得到概率预测。")
        gr.Markdown("---\n*说明：当前为轻量级演示界面，未实际下载与加载 Chronos-T5 Base 模型参数。*")
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=18760)
