# SASP-MPDocVQA

基于自注意力评分机制的多页文档视觉问答系统。

## 简介

冻结Pix2Struct预训练权重，引入轻量化SASP（Self-Attention Scoring Page）页面选择模块作为可学习的中间决策机制。采用单层Transformer编码器判别页面-问题相关性，通过显式概率输出与注意力权重可视化实现可解释性与可控性的统一。与Moonshot大模型API集成，实现多页PDF解析、页面相关性可视化、关键页智能筛选与智能问答的完整功能闭环。

## 核心特性

- 轻量化设计：冻结基座模型，仅训练单层Transformer选页模块
- 可解释决策：显式概率输出 + 注意力权重可视化
- 联合优化：MSE损失与对比损失协同训练
- 混合架构：本地SASP选页 + Moonshot API问答生成
- Web演示：支持PDF上传、评分可视化、交互问答

## 环境依赖

| 包名 | 版本 | 功能定位 |
|:---|:---|:---|
| torch | 2.7.1+cu128 | 深度学习计算图构建与自动微分 |
| torchvision | 0.22.1+cu128 | 图像变换与可视化工具 |
| transformers | 4.49.0 | 预训练模型加载与分词器管理 |
| accelerate | 0.34.4 | 混合精度训练与设备管理 |
| datasets | 3.3.2 | 数据集流式加载 |
| tokenizers | 0.21.0 | 高速分词流水线 |
| sentencepiece | 0.2.1 | Pix2Struct采用的Unigram分词后端 |
| pillow | 11.1.0 | 文档图像解码与预处理 |
| opencv-python | 4.13.0.92 | 图像增强与可视化渲染 |
| pandas | 2.3.3 | 实验指标记录与表格序列化 |
| matplotlib | 3.9.2 | 训练曲线与注意力热力图绘制 |
| openpyxl | 3.1.5 | Excel格式训练日志导出 |
| tqdm | 4.67.1 | 批次迭代进度监控 |
| editdistance | 0.8.1 | 答案生成评估的编辑距离计算 |
| safetensors | 0.5.3 | 模型权重安全序列化格式 |
| huggingface_hub | 0.34.4 | 模型仓库交互与缓存管理 |
| flask | 3.1.2 | Web服务接口部署 |
| flask-cors | 6.0.2 | 跨域资源共享支持 |
| pdf2image | 1.17.0 | PDF文档页面提取 |
| aiohttp | 3.12.15 | 异步HTTP客户端（API调用） |
| openai | 2.29.0 | Moonshot API兼容接口 |

# 下载Pix2Struct权重
python download_model.py

# 启动Web服务（自动检查依赖与模型路径）
python web/start_server.py

核心文件
| 路径                             | 功能                         |
| :----------------------------- | :------------------------- |
| web/start\_server.py           | 依赖检查与模型路径自动查找启动            |
| web/app.py                     | Web服务后端，SASP选页+Moonshot问答  |
| web/inference.py               | SASP选页+Moonshot推理程序        |
| templates/index.html           | 前端交互界面（上传/问答/可视化）          |
| prob\_model.py                 | SASP概率评分模块（Transformer/池化） |
| train.py                       | SASP模块训练主程序                |
| dataset.py                     | MP-DocVQA数据集加载与预处理         |
| metrics.py                     | ANLS/准确率等评估指标计算            |
| util\_log.py                   | 训练日志写入与模型权重保存              |
| seed.py                        | 设置随机种子保证实验可复现              |
| download\_model.py             | 多镜像下载Pix2Struct模型文件        |
| create\_extreme\_valid\_npy.py | 生成全页验证集NPY数据文件             |

目录说明
| 目录       | 用途          |
| :------- | :---------- |
| models/  | 基础模型存放处     |
| dataset/ | 数据集存放处      |
| results/ | 训练结果日志存放处   |
| uploads/ | 图片/PDF上传保存处 |
| fonts/   | 字体存放处       |

# 数据集
MP-DocVQA（Multi-Page Document Visual Question Answering, 多页文档视觉问答）数据集进行模型训练与性能评估，该数据集由西班牙巴塞罗那自治大学计算机视觉中心（Computer Vision Center, Universitat Autònoma de Barcelona）构建并维护
https://rrc.cvc.uab.es/?ch=17&com=tasks

