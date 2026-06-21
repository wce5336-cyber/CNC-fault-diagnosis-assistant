# 数据目录

## external_knowledge/

| 子目录 | 说明 |
|--------|------|
| `raw/` | 原始报警手册文本（FANUC / Siemens） |
| `parsed/` | 结构化 CSV（`fanuc_alarms.csv`, `siemens_alarms.csv`） |
| `manuals/` | 中文维修场景 Markdown |

重新拉取外部数据：

```bash
python -m src.fetch_sources
python -m src.process_cnc_datasets
```
