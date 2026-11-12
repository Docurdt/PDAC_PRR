# PDAC PRR 使用与上传说明

这套代码依据现有结果对应的原始实现和权重整理，包含 ESC、PRRpath、
PRRpath+text、PRRfull 的推理定义及输入预处理工具。技术参数以
[模型说明](docs/MODEL_CARD.md)和权重清单为准；论文文字正在另行统一。

## 上传到 GitHub

1. 解压 `PDAC_PRR_source_v0.1.0_complete_20261008.zip`。
2. 将其中 `PDAC_PRR` 目录里的文件上传到
   [Docurdt/PDAC_PRR](https://github.com/Docurdt/PDAC_PRR) 仓库根目录，
   使 `README.md`、`pdac_prr/` 和 `model_manifest.json` 位于根目录。
3. 保留 `.gitignore` 和 `.streamlit/config.toml` 等隐藏文件。使用浏览器上传
   时如隐藏文件未被选中，补充创建这些文件；使用 Git 可完整保留目录。
4. 模型 ZIP 放在 GitHub Releases 的附件区。安装工具会按清单逐文件验证
   SHA-256；模型二进制不进入普通 Git 提交。

四类模型权重均已补齐，每类25个检查点，总计100个。ESC、PRRpath 各一个 ZIP；
PRRpath+text、PRRfull 各两个 ZIP，分别包含第1–3折的15个检查点和第4–5折的10个
检查点。所有附件均低于 GitHub Release 的单文件大小限制。

两个分包是可独立安装的 ZIP，请逐个安装，不需要拼接成一个文件：

```sh
pdac-prr-weights --bundle . --install PDAC_PRR_prrpath_text_weights_v0.1.0_part1.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrpath_text_weights_v0.1.0_part2.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrfull_weights_v0.1.0_part1.zip
pdac-prr-weights --bundle . --install PDAC_PRR_prrfull_weights_v0.1.0_part2.zip
pdac-prr-weights --bundle . --model prrpath_text prrfull
```

安装完整的某类权重后，再运行该模型。发布说明见
[RELEASE_NOTES.md](docs/RELEASE_NOTES.md)。源码许可证先按 MIT 准备；权重发布许可
需作者单独确定，外部基础模型按各自条款获取。

## 运行示例

使用 Python 3.10–3.12，在仓库目录中执行：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
pdac-prr-weights --bundle . --install /模型ZIP所在目录/PDAC_PRR_prrpath_weights_v0.1.0.zip
pdac-prr --bundle . --model prrpath --input examples/synthetic_path_features.npz --output predictions/example.json
```

示例输入为随机合成特征，没有患者数据。输出为研究风险分数、五折分数及
检查点数量，不输出未经验证的个人生存时间或高低风险诊断。

原始 WSI 先由固定版本的 TRIDENT 提取四类基础模型特征，再用
`scripts/prepare_inputs.py` 生成推理输入；具体步骤见英文 README。
病理报告使用固定版本的中文 BERT。临床编码保留原流程的队列内归一化，
需要明确提供预处理队列上下文，未自行增加单患者固定归一化参数。

本地上传界面可通过以下命令运行，接受已准备的 NPZ 特征文件：

```sh
python -m pip install -e '.[ui]'
streamlit run app.py --server.address 127.0.0.1
```

GitHub 仅分发代码和模型附件，网页界面在使用者电脑运行。验证范围和尚未
完成的原始 WSI 流程验证见 [VALIDATION.md](docs/VALIDATION.md)。
