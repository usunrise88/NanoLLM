# Apple Neural Engine через Core ML: план, пробы и вердикт (2026-10-22)

Статус: **ANE остановлен решением владельца (2026-10-22: «выглядит как проблемный путь»), пробы 0, 1a и 1b и замер энергии выполнены, бэкенд не написан.** Числа и сырые файлы — в [AUTOTUNE.md](../AUTOTUNE.md)
(«Metal на Apple M1 …», «Проба Apple Neural Engine, этап 1a», «Проба 1b») и в
`bench/results/macbookair-m1/`. Этот файл хранит решения, подход и то, что осталось от плана.

## Зачем и что решил владелец

NPU в роудмапе — это v2.0 (VitisAI, OpenVINO через ORT GenAI), железа под них пока нет. ANE единственный NPU, который
можно проверить сейчас. Решения владельца (2026-10-22):

- рантайм Core ML (Core AI позже, за тем же интерфейсом в Go);
- гейт продолжения: качество в пороге **и** одно из двух: (а) энергия на токен ≥ 1,5× от Metal при decode ≥ 0,7× Metal;
  (б) decode или TTFT быстрее Metal;
- измерять на MacBook Air M1 8 ГиБ (Mac с 16 ГиБ нет), `sudo` для `powermetrics` есть, Xcode не обязателен;
- артефакт, если до него дойдёт, хранится в HF-репозитории владельца под именем `<модель>-coreml-v<N>` (по имени модели и
  версии; пространство имён HF не названо), в каталоге закрепляется коммитом, размером и sha256;
- ANE в роудмапе — третий кандидат NPU v2.0;
- порог качества и минимальная macOS (15) выбраны мной по поручению владельца и записаны в AUTOTUNE до первого замера ANE.

## Выбранный подход (если бы гейт прошёл)

Core ML, вызываемый из Go через `purego/objc` (CGO_ENABLED=0, без нового нативного артефакта); запасной путь тонкий
Obj-C-мост с C ABI в `libs-build.yml`. Конвертер строит чанки Core ML из того же GGUF; токенизатор, сэмплер и грамматика
остаются в libllama (`vocab_only`, `llama_sampler_apply` по чужим логитам); строки `token_embd` и PLE читает хост;
KV держит хост. Отклонено: ORT GenAI + CoreML EP (E4B не экспортируется, `MatMulNBits` и `GroupQueryAttention`
уходят на CPU), Core AI сейчас (macOS 27+, Swift, сырой), свой dylib сразу, MLX (не использует ANE, вопрос O5).

## Что выяснили

| Вопрос | Ответ |
| --- | --- |
| Metal на M1 (база) | pp 301, tg 17,6–18,1, MTP 27,5 ток/с, KLD против CPU 1,05e-3 (гейт 4,0e-3) |
| Что принимает ANE | палитризация (LUT) и int8 по каналу; fp16-программа ≥ ~1 ГБ отвергается целиком; **блочное квантование отвергает**, значит точный Q4_0 на нём невозможен |
| Память | веса остаются сжатыми (pal4: +2,2 ГиБ wired на всю модель при 167 МиБ в процессе), раскрытия в fp16 нет |
| Скорость pal4 | decode 87 мс на токен (0,65× Metal), prefill ~770 ток/с (2,6× Metal) |
| Качество кодирований (KLD, носитель Q8_0, шум носителя 1,58e-3) | int8: 3,56e-3 (проходит); LUT6 в FFN: 5,6e-2; LUT6 всё: 7,9e-2; LUT4 всё: 0,39 (не проходят) |
| Скорость int8 | decode ~91 мс на токен (~0,6× Metal), prefill ~515 ток/с (1,7× Metal) |
| Цена int8 | около 4,5 ГиБ wired из 8: на этой машине остаётся около 2 ГиБ под ОС и приложения |
| Энергия на токен (`powermetrics`, рельсы CPU+GPU+ANE) | ANE pal4 196 мДж (2291 мВт при 11,7 ток/с) против Metal 412 мДж (7887 мВт при 19,15 ток/с): в 2,1× экономнее; для int8 оценка 1,6× |

**Падение Mac при сборке головы** (14:28): мой первый вариант пробы вырос до 4,6 ГиБ на 8 ГиБ и почти полном диске,
`watchdogd` не получил процессорного времени 93 с. С тех пор у пробы и конвертеров есть проверка доступной памяти и места
и ограничитель пика; подробности в AUTOTUNE.

## Вердикт и что дальше

Бэкенд `coreml` **пока не писать**: годен только int8 (4,5 ГиБ), decode на нём медленнее Metal на 40 %, преимущество
в prefill и в энергии (около 2× экономнее Metal на токен). Остаются открытыми, и без них гейт не закрыть:

1. энергия int8 на ANE: измерена только для `pal4` (не проходит по качеству), для int8 это оценка; настоящий замер нужен
   на Mac с ≥ 16 ГиБ, где int8 можно загрузить целиком (`sudo -v && tools/coreml/energy_suite.sh`);
2. этап 1c: арифметика fp16 на ANE и граф Core ML с нормами, RoPE, PLE и общим KV (ошибка графа складывается с ошибкой
   весов, а Gemma склонна к переполнению в fp16);
3. полная загрузка int8 на ANE (4,5 ГиБ wired) на Mac с ≥ 16 ГиБ, не на этом;
4. только после этого этапы 2–6 исходного плана: мост, бэкенд `coreml` и проводка (`internal/backend/coreml`, планировщик
   фиксированной формы, реестр с архивом артефакта), бенчмарк и KLD-инструменты для не-llama.cpp бэкендов, n-gram и MTP
   через функцию проверки K+1 токенов (на ANE проверка пачки почти не дороже одного токена), документы и `verified[]`.

## Как повторить

```bash
scripts/fetch-libs.sh --tools "" darwin_arm64              # библиотеки и llama-perplexity закреплённого тега
go run ./tools/kldcorpus -manifest testdata/kld/corpus.json -out .cache/kld/corpus.txt   # на 2026-10-22 три статьи
                                                           # изменили рендер: хэши в копии манифеста, не в репозитории
go build -o .cache/bin/optimize ./tools/optimize
.cache/bin/optimize base -tools dist/darwin_arm64/tools -model GGUF -corpus .cache/kld/corpus.txt \
  -logits ~/nanollm-data/kld/base-cpu-m1-t4-c8.bin -- -t 4 -fa on -ngl 0 --chunks 8
uv venv --python 3.11 .cache/coreml-venv && uv pip install --python .cache/coreml-venv/bin/python coremltools numpy gguf
tools/coreml/quality_suite.sh control i8c mix pal6 pal4    # носители и KLD (около часа, 7 ГБ диска на носитель)
.cache/coreml-venv/bin/python -I tools/coreml/probe_chunk.py build pal4 .cache/coreml/chunk.mlpackage
clang -fobjc-arc -O2 -framework CoreML -framework Foundation -o .cache/coreml/predict_bench tools/coreml/predict_bench.m
.cache/coreml/predict_bench .cache/coreml/chunk.mlpackage ane 60
```

Все тяжёлые шаги проверяют доступную память и свободное место перед стартом и останавливаются сами; на Mac с 8 ГиБ
запускайте их при закрытых браузере и тяжёлых приложениях.
