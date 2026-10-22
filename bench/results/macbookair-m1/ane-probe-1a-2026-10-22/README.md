# Проба Apple Neural Engine, этап 1a (MacBook Air M1 8 ГиБ, macOS 27.0.1, 2026-10-22)

Чанк из слоёв с формами Gemma 4 E4B (d = 2560, FFN 10240, 8 голов, 2 KV-головы, голова 256 у слоёв со скользящим
окном и 512 у полного слоя) и **случайными весами**: проба отвечает про память, размещение и время, про качество не
говорит ничего. Инструменты: `tools/coreml/probe_chunk.py` (сборка, `MLComputePlan`), `tools/coreml/predict_bench.m`
(нативное время и память без Python), coremltools 9.0, Python 3.11. Итоги и выводы — в `docs/AUTOTUNE.md`,
раздел «Metal на Apple M1 8 ГиБ … и правила гейта для Apple Neural Engine».

- `raw/probe-fp16-ane.log`, `raw/probe-variants.log` — Python-прогоны (время включает копирование KV-входов, около 9 МБ на вызов).
- `raw/native.log`, `raw/seven.log`, `raw/full-token.json` — нативный `predict_bench`: один чанк, семь чанков и полный токен (7 + 4 части головы).
- `raw/plan_inspect.py` — разбор `MLComputePlan` по операциям.
- `energy/` — замер `powermetrics` (`tools/coreml/energy_suite.sh`): простой, ANE (pal4, 11 моделей) и Metal (`llama-bench`); итог и пересчёт в мДж/токен в `docs/AUTOTUNE.md`, раздел «Энергия».

**Оговорка про файлы.** Перезагрузка Mac во время работы (см. AUTOTUNE, «Падение при сборке головы») очистила
`/private/tmp`, где лежали исходные логи. `probe-variants.log`, `probe-fp16-ane.log`, `native.log`, `seven.log` и
`prefill64.log` восстановлены из консольных выводов той же сессии: значения те же, строки перепечатаны. Только
`full-token.json` записан после перезагрузки и подлинный. Любую строку можно переснять: `tools/coreml/probe_chunk.py`
и `predict_bench` с теми же переменными (`PROBE_IDX`, `PROBE_Q`, `PROBE_SEED`).
