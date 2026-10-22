# M1 8 ГиБ: KLD Metal против CPU на 64 чанках (2026-10-22, 23:47 — 2026-10-23, 00:10)

Разбор и выводы: `docs/AUTOTUNE.md`, раздел «Metal на M1: golden, conformance, KLD на 64 чанках».

MacBook Air M1 (`MacBookAir10,1`, GPU 7 ядер, 8 ГиБ), macOS 27.0.1, сеть, библиотеки `b11600-n8`, модель
`gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`. Эталон: CPU (`-dev none -ngl 0`), 4 потока, KV f16, flash attention, 64 чанка по
512 токенов корпуса `.cache/kld/corpus.txt` (sha256 `fbece42a…ade1b5`, тот же, что у прогона на 8 чанках), PPL 11,8289,
логиты 8,56 ГБ в `~/nanollm-data/kld/base-cpu-m1-t4-c64.bin` (в репозиторий не кладутся), 9 мин 17 с. Кандидаты на
Metal (`-ngl 99`), по одному.

| Файл | Конфигурация | KLD среднее ± SE | медиана | p99 | top-1, % | max |
| --- | --- | --- | --- | --- | --- | --- |
| `kld-metal-c64.json` | `-fa on` (умолчание до этого замера) | 1,682e-3 ± 9,7e-5 | 3,83e-4 | 1,844e-2 | 98,725 ± 0,088 | 1,128 |
| `kld-metal-fa-small-c64.json` | FA на батчах до 8 токенов (патч 0007, через `scripts/fa_small.c`) | 1,684e-3 ± 9,8e-5 | 3,86e-4 | 1,855e-2 | 98,725 ± 0,088 | 1,126 |
| `kld-metal-fa-off-c64.json` | `-fa off` | 1,687e-3 ± 9,9e-5 | 3,84e-4 | 1,838e-2 | 98,750 ± 0,087 | 1,127 |

`llama-perplexity` из архива `darwin_arm64` не принимает `-fa small_batch` (его `libllama-common` без части патча 0007
для инструментов). `scripts/fa_small.c` подменяет `llama_init_from_model` через `DYLD_INSERT_LIBRARIES` и ставит
`flash_attn_type` = 2; лог прогона это подтверждает (`fa_small: flash_attn_type 1 -> 2 (small_batch)`).
`scripts/llama-perplexity` — обёртка, которую получает `optimize -tools`, `scripts/candidates.sh` — прогон всех трёх.
В отчётах `flags` у варианта small те же, что у FA on: режим задаёт прослойка, а не флаг.
