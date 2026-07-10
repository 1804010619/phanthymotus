# ASR 模型 CER 测试记录

## 测试日期
2026-07-07

## 模型列表

| 模型 | 大小 | 训练数据 | Epoch | 学习率 | 说明 |
|------|------|----------|-------|--------|------|
| Legacy 20260109 | 81MB | WenetSpeech | 20 | 2e-4 | 早期微调版本 |
| EP150 微调 | 81MB | Aishell4 + aireport | 150 | 2e-3 | 后期微调版本 |
| 官方原版 | 79MB | 无（ModelScope 预训练） | - | - | sherpa-onnx 转换版 |
| Zipformer CTC small zh | 61MB | 无（官方预训练） | - | - | CTC 解码 |

---

## 测试集 1：test_aireport/zh（翻译设备场景）

- **来源**：`/mnt/disk1/zengzhitao/asr/channel/data/test_aireport/zh/manifest.jsonl`
- **样本数**：3747（过滤 >80% 中文字）
- **场景**：翻译设备实时语音识别，口语对话
- **音频格式**：PCM16 WAV，16kHz

| 模型 | CER |
|------|-----|
| 官方原版 | **0.1956** |
| Zipformer CTC | 0.2357 |
| Legacy 20260109 | 0.3433 |
| EP150 微调 | 0.3507 |

**结论**：官方版本最佳，两个微调版本明显变差。翻译设备场景下微调数据分布不匹配。

---

## 测试集 2：ai_report_20251229_20260305（微调时验证集）

- **来源**：两个模型的 `cer_test_results.json`
- **样本数**：6570
- **场景**：翻译设备日志数据
- **注意**：音频文件已丢失（`asr_offline/` 目录迁移后缺失）
- **数据来源**：仅从 CER 报告文件读取

| 模型 | CER |
|------|-----|
| EP150 微调 | **0.1227** |
| Legacy 20260109 | 0.1932 |

**结论**：在此数据集上 EP150 优于 Legacy，但该数据集与 aireport 测试集分布不同。

---

## 测试集 3：Fleurs zh（Google 多语言语音数据集）

- **来源**：`/mnt/disk1/zengzhitao/data/fleurs/zh/test.tsv`
- **样本数**：787（过滤 >80% 中文字）
- **场景**：标准中文朗读句，部分中英文混读
- **音频格式**：IEEE float32 WAV，16kHz（需手动解析 WAV 头）

| 模型 | CER |
|------|-----|
| Legacy 20260109 | **0.1088** |
| EP150 微调 | 0.1353 |
| Zipformer CTC | 0.1371 |
| 官方原版 | 0.1502 |

**结论**：Fleurs 场景下 Legacy 微调最佳，两个微调版本均优于官方原版。

---

## 跨数据集对比总结

| 模型 | aireport CER | Fleurs CER |
|------|-------------|------------|
| Legacy 20260109 | 0.3433 | **0.1088** |
| EP150 微调 | 0.3507 | 0.1353 |
| 官方原版 | **0.1956** | 0.1502 |
| Zipformer CTC | 0.2357 | 0.1371 |

**关键发现**：
1. 两个测试集分布差异大，模型排名完全相反
2. aireport（翻译设备口语）上官方版本最好，微调反而差
3. Fleurs（标准朗读）上 Legacy 微调最好
4. EP150 在 aireport 上过拟合严重（验证集 0.12 → 测试集 0.35）
5. Legacy 20 epoch 泛化更稳健

---

## 技术备注

### WAV 格式
- aireport：PCM16（format code 1），Python `wave` 库可直接读取
- Fleurs：IEEE float32（format code 3），Python `wave` 库不支持，需手动解析 WAV 头

### CER 计算
- 使用 Levenshtein distance / len(ref)
- `autojunk=False`，避免频繁中文字被误判为 junk

### 测试脚本
- `/mnt/disk1/zengzhitao/embodied-ai/test_asr_models.py`
