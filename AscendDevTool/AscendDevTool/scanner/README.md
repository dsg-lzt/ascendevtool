# 昇腾模型迁移分析工具 (CANN ms_fmk_transplt) 操作指南

本模块负责扫描分析待迁移的 PyTorch 模型，提取其中由于兼容性问题不被 Ascend NPU 支持的算子或 API，并输出包含 Excel 报表在内的分析结果。
由于 CANN 自带了非常成熟的扫描分析工具，**我们在该阶段不需要额外开发代码**，请严格根据下方指引直接调用系统底层工具。

## 1. 环境准备与前提

确保目标机器或开发容器中已安装 CANN（如版本 9.0.0.beta2 社区版或类似）。
默认安装路径一般为：`/usr/local/Ascend/`（如使用非 root 用户或自定义路径安装，请将下文该路径替换为您实际的 `${INSTALL_DIR}`）。

我们使用的子工具处于安装路径中的如下层级：
`/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/`

## 2. 启动分析任务

工具依赖一个命令行脚本 `pytorch_analyse.sh` 进行静态或动态 API 扫描分析。

**使用步骤：**

1. 进入分析工具目录
   ```bash
   cd /usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/
   ```

2. 运行脚本进行扫描 (请将 `<YOUR_WORKSPACE_PATH>` 替换为你本地拉取代码的绝对路径)
   ```bash
   ./pytorch_analyse.sh -i <YOUR_WORKSPACE_PATH>/target_models \
                        -o <YOUR_WORKSPACE_PATH>/scanner/reports \
                        -v 2.1.0 \
                        -m torch_apis
   ```

**参数解析：**
* `-i`: 待分析模型脚本的绝对路径（本工程中即为您存放原始 PyTorch 训练/推理代码的 `target_models/`）。
* `-o`: 扫描结果报告的输出路径，建议固定指向本工程的 `scanner/reports/` 目录下。
* `-v`: 您待分析的原始 PyTorch 脚本所对应的框架版本（如 2.1.0）。
* `-m`: 分析模式。这里填入 `torch_apis` 表示仅对 Pytorch API 的兼容性进行分析即可满足我们生成不支持算子列表的需求。

## 3. 分析输出

工具执行成功后，请回到本项目 `scanner/reports/` 目录查看结果。
其中的 Excel 表格或 txt 报告罗列了所有明确报出**不受支持**（Not Supported）的 PyTorch 原生算子，这些算子的名字，就是您接下来进行 **Patcher (Monkey Patch)** 及 **OpBuilder (Ascend C底座开发)** 补齐的目标。