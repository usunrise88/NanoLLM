# M1 8 ГиБ: потолок prefill, нагрев, FA на коротких батчах, длина драфта и MLX (2026-10-22, вечер)

Разбор и выводы: `docs/AUTOTUNE.md`, раздел «Metal на M1: потолок prefill, нагрев, FA на коротких батчах, MLX». Здесь только
числа; ответы модели содержат стенограмму владельца и в репозиторий не кладутся (локально `.cache/gemma-json/`).

MacBook Air M1 (`MacBookAir10,1`, GPU 7 ядер, 8 ГиБ), macOS 27.0.1, сеть 45 Вт, библиотеки `b11600-n8`, модель
`gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (все матрицы Q4_0), `nanollm serve` с `uma-4g`. Каждый замер, кроме
`llama-bench-pp4096-no-pauses.md`, начат при тепловом давлении 0 плюс 30 с покоя (`scripts/`).

| Файл | Что |
| --- | --- |
| `raw/llama-bench-pp4096-no-pauses.md` | FA × n_ubatch без пауз: падение по порядку запусков, это нагрев |
| `raw/llama-bench-pp4096-sustained.json`, `raw/thermal-pressure-sustained.log` | pp4096 двенадцать раз подряд и тепловое давление раз в 2 с |
| `raw/llama-bench-fa-ubatch-cooled.txt` | FA on/off × n_ubatch 256/512 с паузами |
| `raw/llama-bench-small-batch-and-d4096.txt` | `-fa small_batch` (инструменты `darwin_arm64` его не принимают) и tg64 на глубине 4096 |
| `raw/gemma-fa-{on,small}-r{1,2}.log`, `raw/gemma-fa-abba-summary.txt` | `gemma.json`, A-B-B-A, `flash_attention` on / small, MTP n_max 2 |
| `raw/gemma-draft-{none,1,2,3,4}-r{1,2}.log`, `raw/gemma-draft-sweep-summary.txt` | `gemma.json`, `flash_attention: small`, без MTP и n_max 1–4, туда и обратно |
| `raw/gemma-mlx-*.log` | `gemma.json` через mlx-lm 0.32.2 (`scripts/mlx_steps.py`, PLE из файла) |
