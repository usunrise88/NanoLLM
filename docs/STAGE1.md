# NanoLLM — план реализации этапа 1

Этот документ раскрывает этап 1 из [ROADMAP.md](ROADMAP.md) до уровня задач.
Этап 1 идёт на VM без GPU: 4 vCPU, 15 GB RAM, около 25 GB свободного диска.
Пункт 1.1 закрыт: llama.cpp закреплён на `b11600`, каталог — на Gemma 4 E4B
`UD-Q4_K_XL` с int4-драфтером. `libllama` берётся из релизных архивов тега;
сборка из исходников — запасной вариант. Дизайн публичного бенчмарка (пункт 1.11) —
в [BENCHMARKS.md](BENCHMARKS.md). Номера `§N` ссылаются на NanoLLM
Development Specification.

## Факты закреплённого тега, которые определяют план

Факты собраны по `llama.h` и исходникам `b11600` (коммит `ace20f6`), purego
v0.11.2 и NanoASR.

| Факт (b11600, purego v0.11.2) | Следствие |
| --- | --- |
| purego v0.11.2 передаёт и возвращает структуры по значению на linux, darwin и windows (amd64/arm64). Ограничения: на SysV amd64 аргумент занимает ≤ 26 слов (`llama_context_params` 168 B — 21 слово); колбэки на Windows без float; колбэков не больше 2000, они не освобождаются | `llama_model_load_from_file` и `llama_init_from_model` вызываются напрямую, шим не нужен. Версия purego закрепляется. bindgen проверяет размер структур ≤ 208 B. `progress_callback(float)` не используется: прогресс загрузки берётся из другого источника. Колбэки (лог) регистрируются один раз на процесс |
| В libllama нет `llama_build_number`. Есть `llama_version()` и `ggml_commit()` | Проверка сборки сверяет `ggml_commit()` с `ace20f6`. ldflags берут его из нового `scripts/llama.cpp.commit`. `version` и `/healthz` показывают тег и коммит |
| `use_mmap`/`use_mlock` заменены на `llama_load_mode` (none, mmap, mlock, mmap+mlock, dio). Отдельно есть `lazy_mode` (off, auto — только тензоры > 4 GiB, on; auto и on требуют mmap). Repack включает `use_extra_bufts` | `LoadSpec` и `Plan` получают load_mode и lazy_mode. Правило: lazy несовместим с dio |
| `per_layer_token_embd` относится к `LLM_TENSOR_LAYER_INPUT` и поэтому всегда лежит в CPU-буфере, при любом `gpu_layers` | **M2 — штатное поведение upstream:** lazy off + PLE закреплён в RAM. M3/M7 = lazy on + mmap. Проверка на lavapipe только подтверждает размещение буферов |
| Tied embeddings: если выходной слой на GPU, `token_embd` копируется в VRAM | Планировщик учитывает вторую копию G0 в VRAM |
| KV (точно, по `llama-context.cpp:377` и `llama-kv-cache-iswa.cpp:73`): `n_ctx` → pad256; `n_ctx_seq = pad256(n_ctx/n_seq_max)` без unified; full-слои — `n_ctx_seq` ячеек на поток; SWA-слои — `pad256(min(size_base, n_swa·(unified ? n_seq_max : 1) + n_ubatch))` или `size_base` при `swa_full`. KV есть у `block_count − shared_kv_layers` слоёв; head dim свой у SWA и full; без FA V транспонирован с max `n_embd_v_gqa`; квантованный V требует FA. `swa_full=true` — дефолт C API, но `common` и llama-server ставят `false`. Драфтер своей KV не держит | Формула в 1.4 точная. Golden-тест сверяет её со строкой лога «SWA KV cache … cells» на сетке (unified, n_seq, n_ubatch). Планировщик требует FA для `q8_0` V |
| При `swa_full=false` откат дальше ~`n_swa` ломает SWA-слои (llama-server спасается чекпоинтами) | Prompt cache и `swa_full` — явное решение 1.4/1.7 |
| Цикл MTP есть только в `common/speculative.cpp` (~450 строк, путь `is_mem_shared`). Функции `llama_{set,get}_embeddings_nextn*` и `llama_get_ctx_other` экспортируются с C++-декорацией (Itanium на Linux/macOS, MSVC на Windows — релиз собран clang-cl), заголовок помечен staging. Скрытое состояние (`pending_h`) не входит в `llama_state_seq_*` | MTP — отдельный подэтап 1.5.M с конечным автоматом. bindgen содержит мини-декоратор имён, имена сверяются с таблицами экспорта (`debug/elf`, `debug/pe`, `debug/macho`). Upstream предложить `extern "C"` |
| `llama_chat_apply_template` не знает Gemma 4 (возвращает −1); Jinja есть только в `common` | Шаблон и парсер tool calls пишутся на Go (1.7) |
| `json_schema_to_grammar` есть только в C++; в тестах 81 эталонный случай | Порт на Go, эти случаи — корпус |
| `GGML_ABORT` → `abort()`. Исключение C++ из любой точки входа → `std::terminate` | Входы проверяются в Go до вызова, сервис перезапускает супервизор. Бенчмарк гоняет каждый пресет в отдельном процессе |
| Linux-архивы зависят от `libgomp.so.1` и `libvulkan.so.1`, которых нет в составе. На macOS нет `GGML_BACKEND_DL`, поэтому `load_all_from_path` там ничего не делает | Установщик проверяет эти библиотеки. `devices` показывает ошибку dlopen каждого бэкенда. В Docker-образ ставятся `libgomp1` и `libvulkan1` |
| В архивах релиза есть `llama-cli`, `llama-perplexity`, `llama-tokenize` (`LLAMA_BUILD_TOOLS=ON`) | Golden-тесты, KLD и эталон MTP берут инструменты того же тега через `fetch-libs --tools` |
| В HF есть крошечные случайные Gemma 4 E (`tiny-random/gemma-4-e`), конвертер знает `Gemma4AssistantForCausalLM`. Готового крошечного драфтера нет, а драфтер обязан совпадать по `n_embd_out` с целью | Крошечный драфтер генерируется случайной инициализацией `transformers` из уменьшенного конфига, согласованного с крошечной целью; seed и скрипт сохраняются. Крошечные модели проверяют только механику и эквивалентность, acceptance > 0 проверяется только на E4B |

## Порядок и зависимости

```
1.2 Реестр ───────────────────────────────┐
1.3 GGUF ──> 1.4 План ──┬─> 1.5 Биндинг ──┼─> 1.5.M MTP ──> golden, real conformance
   └─> 1.6 Range-lock   │                 │
1.0 Основа (fake, Step*) ─> 1.7 Сервер ───┼─> 1.8 UI ──> 1.9 Поставка
1.11a benchfmt, hwinfo (сразу)            ├─> 1.10 Оценка (нужен и golden 1.5)
                         1.11b раннер, клиент (ядро 1.7, план 1.4; smoke E4B — 1.5)
                         1.11c сервис (после 1-C)
```

`*` Интерфейс `Model.Step` предварительный. Он замораживается после спайка
1.5.0 и прототипа MTP на крошечной модели; до этого 1.7 строится на фейке.

Контрольные точки:

- **1-A «Инвентарь»** (1.2, 1.3, 1.4): размер PLE известен, `plan` работает
  для 19 пресетов.
- **1-B «Движок»** (1.5, 1.5.M, 1.6): E4B с MTP через биндинг на CPU VM,
  lavapipe.
- **1-C «Сервер»** (1.0, 1.7): conformance зелёный на фейке.
- **1-D «Продукт»** (1.8, 1.9, 1.11).
- **1-E «Оценка»** (1.10).

## 1.0 Основа

- **`go.mod`:**
  - добавить `github.com/ebitengine/purego v0.11.2` (закреплённая версия) и
    `golang.org/x/sys`;
  - `modernc.org/sqlite` добавить вместе с очередью;
  - `go mod tidy` уберёт ложное `// indirect` у yaml.
- **`backend.Model` (предварительно).** Основной метод — `Step(ctx,
  []SeqStep) ([]SeqOut, error)`.
  - **Батч.** Prefill-чанки ≤ `n_batch` и decode всех слотов собираются в
    один батч. Сэмплинг, драфт и проверка MTP делаются внутри бэкенда; на
    выходе 1..n+1 токенов на слот.
  - **Отмена.** `ctx` живёт столько же, сколько цикл модели. Слот отменяется
    на границе шага. На `abort_callback` через purego не полагаемся.
  - **Бюджет батча.** `Capabilities().MaxDraft` и `SeqStep.MaxDraft` на слот
    (0, когда бюджет батча или грамматика не позволяют драфт). Проверка
    драфта стоит `n_draft+1` токенов слота.
  - **Грамматика и penalties.** Проверка идёт через цепочку сэмплера слота
    по одной позиции. В грамматику и penalties попадают только принятые
    токены, как `common_sampler_sample_and_accept_n`.
  - **Stop/EOG внутри принятого отрезка.** Бэкенд возвращает точку обреза и
    делает `seq_rm` хвоста.
  - **Logprobs** — log-softmax сырых логитов до сэмплера. Флаг вывода
    ставится только слотам, которые их запросили, на каждой позиции
    проверки.
  - **Prompt cache.** После попадания в кэш история penalties заполняется
    заново, перенос MTP сбрасывается.
  - **Прочее.** `Capabilities()` и `MemoryReport()`. `Decode`/`Sample`
    становятся внутренними для `llamacpp`.
  - **`LoadSpec`:** load_mode, lazy_mode, переопределения буферов,
    `swa_full`, `kv_unified`, драфтер.
- **`internal/backend/fake`:**
  - побайтовый токенизатор со спецтокенами Gemma 4;
  - сценарии: эхо, текст, tool call, JSON, бесконечный вывод;
  - **несколько токенов за шаг и имитация acceptance**;
  - скорости, ошибки, «память»;
  - включается явно через `backends.enabled: [fake]`, только для
    разработки; `make run-fake` для UI.
- **Фикстуры** (`tools/fixtures/`):
  - крошечные `gemma4` и согласованный `gemma4-assistant` (случайные веса,
    seed);
  - конвертируются `convert_hf_to_gguf.py` того же тега;
  - публикуются ассетом релиза `test-fixtures`, sha256 лежат в `testdata`;
  - E4B берётся из `NANOLLM_TEST_MODELS`; без переменной тест пропускается
    и пишет причину.
- **CI.** Новый job `native` на матрице ubuntu, ubuntu-arm, windows,
  windows-arm, macos: `fetch-libs` для хоста, фикстуры, тесты с тегом
  `native`.

## 1.2 Реестр

- **Загрузчик:** порт `nanoasr/internal/registry/download.go` и `remote.go`.
  - Докачка через `Range`; частичный файл перехэшируется.
  - Зеркала — HF-совместимые базы (`<base>/{repo}/resolve/{rev}/{path}`).
  - Редиректы только на https. 403/404/410 и несовпадение хэша — постоянные
    ошибки.
  - Лимит размера из манифеста. Таймаут на простой, а не на весь запрос.
  - Single-flight на модель, семафор. `strict_license` отклоняет
    `unconfirmed`.
- **Раскладка:** `<data_dir>/models/<id>/<revision>/<file>`, рядом
  `manifest.yaml` и штамп проверки (sha, size, mtime). При загрузке модели
  проверяются размер и штамп; `pull` пересчитывает хэш.
- **`catalog_url`:** файл или https с той же схемой, только закреплённые
  записи.
- **CLI:**
  - `models list` (complete/partial/corrupt), `pull <id> [-configured]` с
    прогрессом, `rm`;
  - `models catalog` уже есть;
  - `init` скачивает E4B с драфтером, если не указан `-no-download`.
- **HTTP:**
  - `POST /api/v1/models/{id}/download` (admin): SSE через порт
    `httpx/sse.go`;
  - события `progress {file, downloaded, total, percent}`, `done`, `error`
    (problem);
  - повторный вызов присоединяется к идущей загрузке; уже установленная
    модель сразу получает `done`;
  - `GET /api/v1/models`.
- **Тесты:** TLS-сервер с `ServeContent`, как в NanoASR. Проверяются докачка,
  зеркала, отказ http, 404, отмена, несовпадение хэша, strict_license,
  single-flight.
- **На VM:** настоящий `pull` E4B с прерыванием и докачкой.

## 1.3 GGUF-инвентарь

- **`internal/gguf`: парсер заголовка.**
  - magic, v3, KV (включая массивы), тензоры, `general.alignment` (u32),
    смещение данных;
  - таблица типов ggml, включая Q1_0, Q2_0, MXFP4, NVFP4;
  - проверка кумулятивных смещений;
  - массивы токенизатора не держатся в памяти;
  - fuzz и тестовый writer.
- **Группы `gemma4` / `gemma4-assistant`:**

  | Группа | Тензоры |
  | --- | --- |
  | G0 | `token_embd`, `output` (если нет — tied, копия на GPU учитывается в 1.4) |
  | G1 | `per_layer_token_embd` |
  | G2 | `attn_{q,k,v,qkv,output}` |
  | G2 / `ple_proj` | `per_layer_model_proj`, `inp_gate`, `proj` |
  | G3 | `ffn_{gate,up,down}` |
  | G4 | нормы, `layer_output_scale`, `rope_freqs`, `per_layer_proj_norm`, `post_norm` |
  | G6 | весь драфтер |

  - G5 и G7 считает планировщик.
  - Неизвестный тензор — ошибка при сборке каталога, предупреждение в
    `inspect`.
- **`models inspect <path|id> [--tensors] [--json]`:**
  - метаданные и суммы по группам;
  - инвентарь хранится в `internal/registry/tensors/<id>.json` с привязкой
    к sha256 и подгружается в `Manifest.Tensors`;
  - гиперпараметры KV: `block_count`, `head_count_kv[]`, `key_length[_swa]`,
    `sliding_window[_pattern]`, `shared_kv_layers`,
    `embedding_length_per_layer_input`;
  - `--probe` — после 1.5.
- **Готово:**
  - инвентари E4B и драфтера закоммичены;
  - сумма с паддингом совпадает с размером области данных;
  - **размер PLE записан**, провизорная карта §15 пересчитана.
- **Измерено** (`internal/registry/tensors/gemma-4-e4b-qat.json`, заголовок
  файла; сумма с паддингом равна области данных у обоих файлов):

  | Группа | Байт | MiB | Доля цели |
  | --- | --- | --- | --- |
  | G0 `token_embd` (tied, `output` нет) | 377 487 360 | 360,0 | 9,0 % |
  | **G1 PLE** `per_layer_token_embd` | **1 585 446 912** | **1512,0** | **37,7 %** |
  | G2 внимание | 330 301 440 | 315,0 | 7,9 % |
  | G2 / `ple_proj` | 46 448 640 | 44,3 | 1,1 % |
  | G3 FFN | 1 857 945 600 | 1771,9 | 44,2 % |
  | G4 нормы и мелочь | 2 241 704 | 2,1 | 0,1 % |
  | G6 драфтер (весь файл) | 43 891 728 | 41,9 | — |

  - Ядро без PLE — 2 614 424 744 байта (2,43 GiB); весь файл цели —
    4 199 871 656 байт тензоров. Веса цели — `q4_0` и `f32`, K-квантов нет.
  - KV: 42 слоя, KV у первых 24 (20 SWA по 2×256×2 элемента, 4 full по
    2×512×2), остальные 18 переиспользуют. f16, FA, `n_ubatch` 512, один
    слот, `swa_full=false`: `n_ctx` 4096 → 104 MiB (64 full + 40 SWA),
    32768 → 552 MiB (512 + 40). Без FA SWA-V дополняется до 1024: +20 MiB.
    `--swa-full` при 4096 — 224 MiB. Совпадает с логом `llama-completion`
    b11600 на этой VM, в том числе `-np 4` (4 потока, SWA 1024 ячейки) и
    `q8_0`.
  - G7 на CPU: 563 MiB при `n_ubatch` 512, из них 512 MiB — логиты
    (262 144 × 512 × 4 Б, `n_outputs_max` по умолчанию равен `n_batch`).
    Остальное ≈ 94 КиБ на токен ubatch плюс маска KQ.
  - Repack (буфер AMX на этой VM) — 2491,17 MiB: ровно `q4_0`-веса слоёв
    (2131,17) плюс копия tied output (360,00). Размер repack не меняет, но
    копия выходного `token_embd` становится анонимной. Входные
    `token_embd` и PLE не репакуются и остаются в `CPU_Mapped` (1872 MiB).
  - При `n_outputs_max = slots·(1 + n_draft)`, как ставит llama-server,
    логиты в G7 занимают `n_vocab·4` на выход, а не на токен ubatch.

## 1.4 Планировщик

- **`plan.Plan`:**
  - preset, mode, `gpu_layers`, N;
  - байты по группам и местам: VRAM, RAM-locked, file-backed, disk;
  - KV type, `swa_full`, `n_ctx`, slots, `n_batch`/`n_ubatch`, FA;
  - load_mode, lazy_mode, repack, переопределения, диапазоны для range-lock;
  - MTP с причиной;
  - предупреждения; отказ с недостачей по каждому режиму.
- **Входы:**
  - инвентарь и гиперпараметры, бюджет, устройства;
  - таблица доказанных режимов: M7/M8 выключены до 1.6; M2 разрешён после
    подтверждения 1.5.
- **G5** — точная формула из таблицы фактов. Правила: FA обязателен для
  `q8_0` V; драфтер добавляет 0 KV.
- **`swa_full` и prompt cache (решение).**
  - Если полный SWA-кэш помещается в бюджет после пола `n_ctx`/slots,
    планировщик ставит `swa_full=true`, и prompt cache переиспользует
    префикс любой длины.
  - Иначе `swa_full=false`: переиспользование только в пределах окна, слот
    без подходящего префикса делает prefill заново, план пишет причину.
  - Чекпоинты как в llama-server откладываются.
- **G7.** Оценка по `n_ubatch`, `n_embd`, `n_vocab` с коэффициентами по
  бэкендам. Калибровка по логу `compute buffer size` на CPU VM; для GPU
  `calibrated: false`.
- **Tied output на GPU** — плюс копия G0 в VRAM.
- **Алгоритм:**
  1. Перебор `PlannerOrder`, для M4 — наибольшее N.
  2. Рост `n_ctx`/slots к целям §15 с полом 4096 и одним слотом.
  3. Иначе следующий режим.
  4. Если ничего не подошло — отказ с недостачей.
- **Решение по пробелу спецификации:** RAM хоста у VRAM-only пресетов не
  бюджетируется, а измеряется и показывается в плане.
- **Выход:**
  - `nanollm plan <model> --preset <name> [--device] [--json]`;
  - план в `GET /api/v1/models`;
  - golden-планы 19 пресетов в `testdata/plan/` (`-update`);
  - отдельная проверка арифметики `uma-*`.

## 1.5 Биндинг к libllama

- **1.5.0 Спайк ABI (первым делом).** CI на шести целях:
  - возврат структур `*_default_params`;
  - загрузка крошечной модели (80 B по значению);
  - поиск декорированных `nextn`-символов.

  План Б — C-шим, собранный `zig cc`.
- **`tools/bindgen`:**
  - парсер подмножества C (`llama.h`, `ggml-backend.h`, `gguf.h`);
  - генерирует `zz_generated.go`: структуры с паддингом, enum, allowlist из
    ~80 функций;
  - мини-декоратор имён C++ (Itanium и MSVC для указателя на структуру,
    `bool`, `int32`, `float*`);
  - проверка размеров структур ≤ 208 B.
- **Тест раскладки:** bindgen генерирует `_Static_assert(sizeof/offsetof ==
  N)`, а `zig cc -target <triple> -fsyntax-only` проверяет их для всех шести
  целей в одном Linux-job.
- **Загрузка:**
  - порядок `ggml-base` → `ggml` → `llama`, только абсолютные пути;
  - Windows: `SetDefaultDllDirectories` + `AddDllDirectory`;
  - macOS: `@loader_path`, проверка через `debug/macho`; бэкенды там
    вкомпилированы;
  - затем `ggml_backend_load_all_from_path(lib)` до `llama_backend_init`;
  - проверка `ggml_commit()`;
  - лог → slog, колбэк регистрируется один раз.
- **Устройства:**
  - `ggml_backend_dev_*` и props; IGPU → `Unified`; селектор
    `llamacpp:vulkan0`;
  - пропавший бэкенд — предупреждение с ошибкой dlopen;
  - `devices` показывает `verified[]`.
- **Модель и контекст из плана:**
  - devices, `n_gpu_layers`, load_mode, lazy_mode, `tensor_buft_overrides`
    (`ggml_backend_dev_buffer_type` / `ggml_backend_cpu_buffer_type`),
    `use_extra_bufts`;
  - `n_seq_max`, типы KV, FA, `swa_full`;
  - `llama_batch_ext` + `llama_process`.
- **Сэмплер:**
  - цепочка на слот (logit_bias, penalties, top_k/p, min_p, temp,
    dist/greedy);
  - `llama_sampler_init_grammar` и lazy-вариант;
  - logprobs в Go по запросу.
- **Состояние:** `llama_state_seq_*` (`_ext`), откат через
  `llama_memory_seq_rm`.
- **Тесты:**
  - **CI на шести целях** с крошечной моделью: токенизация, greedy,
    round-trip, grammar, отказ при чужом коммите, перечисление CPU.
  - **На VM** с E4B на CPU: smoke, KV из лога совпадает с формулой 1.4,
    калибровка G7.
- **Golden:**
  - `fetch-libs --tools` достаёт инструменты тега;
  - `testdata/golden/b11600/gemma-4-e4b-qat/cpu.json`: промпт, текст, id и
    CPU-вариант ggml;
  - генерируется на VM один раз.
- **lavapipe:**
  - `mesa-vulkan-drivers` и `libvulkan1`, архив ubuntu-vulkan;
  - E4B с `gpu_layers = all`;
  - по логу проверить PLE в CPU-буфере, копию tied output и строку prefetch
    при lazy;
  - итог записать.
- **`nanollm run --prompt`** — минимальный one-shot для спайка v0.1.

## 1.5.M MTP (подэтап)

- **Автомат на слот:** idle → prefill (запомнить `pending_h`) → draft (до
  `n_max`, `p_min`, top-k 10, одна позиция с state-строкой) → verify (батч на
  цели, принять префикс, `seq_rm` хвоста, `pending_h` = строка проверки) →
  draft. Любой trim, restore или попадание в prompt cache сбрасывает перенос
  и пропускает драфт на один шаг.
- **Контекст драфтера:** `ctx_type=MTP`, `ctx_other`, те же `n_ctx`,
  `n_seq_max`, типы KV, FA, `kv_unified` (гарантирует план).
  `set_embeddings_nextn` на обоих контекстах.
- **Сверка с эталоном:** `llama-cli --spec-type draft-mtp` того же тега на
  крошечной модели под greedy: число драфтов и принятых совпадает.
- **Эквивалентность на E4B (VM):**
  - greedy с драфтером и без на 5 промптах;
  - цель — точное совпадение; при расхождении отчёт даёт отрыв top-1/top-2
    логита в первой точке расхождения (шум GEMM/GEMV или баг);
  - acceptance > 0.
- **Чеклист обновления тега:** staging API, имена символов, эталон
  `llama-cli`.

## 1.6 Range-lock (механика)

- **`internal/memlock`:**
  - read-only shared mmap;
  - диапазоны из плана, выровненные по странице;
  - **Linux/macOS:** `mlock`; `RLIMIT_MEMLOCK` поднимается soft → hard,
    иначе ошибка с недостачей; `LimitMEMLOCK=infinity` в unit;
  - **Windows:** `VirtualLock` после `SetProcessWorkingSetSize`;
  - `madvise(WILLNEED)` / `PrefetchVirtualMemory`.
- **Доказательство покрытия маппинга libllama** через резидентность (поле
  `Locked:` в smaps показывает только наш VMA):
  - **Linux:** две проекции синтетического файла. На незакреплённой —
    `madvise(MADV_PAGEOUT)`, проверка `mincore`: закреплённые диапазоны
    остаются, остальные уходят.
  - **Windows:** `EmptyWorkingSet` + `QueryWorkingSetEx`.
  - **macOS:** только измерение, пока не доказано.

  Тесты в CI.
- **VM с E4B:** mmap, ядро закреплено. После `MADV_PAGEOUT` маппинга
  libllama закреплённые диапазоны резидентны.
- **Время жизни пина** равно времени жизни модели. Пиковые locked и page
  cache идут в `bench`.

## 1.7 Сервер v1.0 на фейковом бэкенде

1. **Ядро `core.Service`** (`internal/engine`):
   - стадии с таймингами;
   - валидация (`n > 1`, картинки → 400);
   - сэмплинг по умолчанию из манифеста;
   - `context_overflow`, `max_tokens_cap`;
   - стоп-последовательности с удержанием хвоста;
   - `max_reasoning_tokens` закрывает thinking;
   - usage, cached_tokens, warnings.
2. **Шаблон Gemma 4 на Go** (`internal/chat`):
   - system/`<|think|>`, `model`, tools, вызовы и ответы, пустой канал мысли;
   - побайтовая проверка против рендера оригинального шаблона из GGUF через
     `jinja2` на корпусе (tools, multi-turn, thinking, параллельные вызовы);
     фикстуры генерирует `tools/chatfixtures/gen.py`;
   - Jinja на Go откладывается до второй модели — открытый вопрос §14 этим
     закрыт;
   - потоковый парсер tool calls (порт `common/parsers/gemma4.cpp`);
   - разделение reasoning.
3. **Грамматика** (`internal/grammar`):
   - порт `json_schema_to_grammar`, 81 случай → `testdata`;
   - каждая грамматика проверяется через `llama_sampler_init_grammar` на
     крошечной модели;
   - `json_object`, `tool_choice: required`, lazy для `auto`.
4. **Пул:**
   - ключ (model, revision, backend, device, mode, preset, n_ctx);
   - refcount, LRU, pin;
   - hot swap: новые запросы идут в новый экземпляр, старый дорабатывает.
5. **Слоты и батчинг** (`internal/sched`):
   - цикл на модель → `Step`; вход между шагами; отмена;
   - `session_id`;
   - prompt cache: слот с самым длинным префиксом, с учётом правила
     `swa_full` из 1.4;
   - governor — порт NanoASR; приоритет interactive;
   - `-race`-тесты: справедливость, вход посреди батча, отмена, hit rate,
     hot swap.
6. **Очередь:** порт job, store/sqlite и webhook из NanoASR.
   - JSONL, статусы, курсор, продолжение после рестарта;
   - SSE, отмена;
   - `max_pending` → 503;
   - HMAC по схеме NanoASR, `queue.webhook_secret`.
7. **Ключи и лимиты:**
   - `key list/issue/remove [-admin -interactive -rps -hash]` через порт
     `config/edit.go`;
   - rate limiter → 429;
   - LimitBody, RequestID, SecurityHeaders.
8. **Маршруты:**
   - 18 `native` + 4 `openai`;
   - SSE (chunks + `[DONE]`), logprobs, tools, `response_format`;
   - problem+json и конверт OpenAI;
   - настоящий `/readyz`.
9. **Conformance:** `tests/conformance/` — pytest + пакет `openai` против
   `serve` на фейке, job в CI. Переключатель env запускает тот же набор на
   реальной модели.
10. **`nanollm run`:** REPL и `--prompt`, `--device`, `--preset`, `--think`.
11. **Если останется время:** `ollama`, `/v1/embeddings`.
12. **Новые ключи конфига:** `registry.models_dir`, `queue.webhook_secret`,
    `server.max_body`, `bench.*`. Каждый — в схеме, примере, валидации и
    env.

## 1.8 Web UI

- **Стек NanoASR:** React 19, Vite, TanStack, Tailwind 4, i18next ru/en,
  vitest, playwright. Каркас из `nanoasr/web`.
- **Экраны:**
  - Chat;
  - Jobs;
  - Models (загрузка по SSE, load/unload с планом, pin, hot swap);
  - Devices;
  - Settings;
  - **Bench**: на фейке на этапе 1, на реальном железе — на этапе 3.
- **Правила:** только публичный API, ключ при первом 401.
- **Сборка:** `make web`, job `web`, playwright e2e против `serve` на
  фейке, `TestHasSPA`.

## 1.9 Сборка и поставка

- **`scripts/dist.sh`** (порт NanoASR):
  - архивы по §10, включая `-noui`;
  - `lib/`, `nanollm.yaml`, unit-файлы, LICENSE и NOTICE (MIT llama.cpp/ggml);
  - CUDA add-on;
  - darwin universal бинарь через `llvm-lipo`.
- **`release.yml`:**
  1. web;
  2. сборка на одном ubuntu с `GOWORK=off`;
  3. smoke на нативных раннерах (`version`, `devices`, `run --prompt` на
     крошечной модели);
  4. `sha256sums.txt` и релиз;
  5. сводка бенчмарка в release notes.
- **Установщики:**
  - `install.sh`: systemd с `LimitMEMLOCK=infinity`, **launchd пишется с
    нуля**;
  - `install.ps1`: Windows service через `x/sys/windows/svc`;
  - оба проверяют `libgomp`/`libvulkan`;
  - `--cuda` распаковывает add-on в `lib/`;
  - `--uninstall`, `--purge`, обновление на месте, `--archive` для тестов.
- **Управление службой:** `nanollm service install|uninstall|start|stop|restart|status`
  плюс алиасы `--install` и другие, как в NanoASR.
- **Docker:** CPU/Vulkan и CUDA, с `libgomp1` и `libvulkan1`, веса в volume.
- **Подпись macOS:** codesign каждой dylib с hardened runtime, notarytool;
  включается при наличии секретов.
- **Smoke установщиков** на трёх ОС из локального архива.

## 1.10 Инструменты оценки (без выводов)

- **`tools/optimize/`:**
  - обёртка `llama-perplexity --kl-divergence-base/--kl-divergence` того же
    тега;
  - логиты эталонной конфигурации;
  - отчёт `eval.v1`.
- **`nanollm eval <model> --suite <name>`** работает через `core.Service`.
- **Формат набора:** JSONL `{id, lang, capability, messages, tools?,
  response_format?, checker, thinking}`.
- **Наборы:**
  - русские инструкции;
  - JSON/tools;
  - длинный контекст до 16k;
  - Эра (данные владельца);
  - 50 промптов для оценки человеком.
- **Корпус:** манифест sha256; хранится в HF dataset или ассете релиза.
- **`nanollm eval rate <dir>`:** оценка 1–5 / «годится для звонка» и A/B с
  клавиатуры, метки скрыты.
- **Smoke на VM:** 5 элементов на E4B CPU.

## 1.11 Бенчмарк

Дизайн — в [BENCHMARKS.md](BENCHMARKS.md).

- **1.11a, сразу:**
  - модуль `benchfmt` и JSON Schema;
  - `internal/hwinfo` для трёх ОС (код и тесты на раннерах CI; модель
    ноутбука на реальном железе — этап 3).
- **1.11b, после 1.4 и ядра 1.7:**
  - оркестратор и `bench worker` с отдельным процессом на пресет;
  - наборы `standard` и `dev`;
  - клиент отправки;
  - smoke на E4B CPU после биндинга 1.5 (цифры VM не публикуются).
- **1.11c, после 1-C:**
  - сервис `services/bench`: API, страницы, образ;
  - e2e в CI;
  - staging-деплой, только если владелец даст хостинг.

## Выход из этапа 1

- **План.** `plan` работает для всех пресетов по измеренному инвентарю.
  Размер PLE и статус M2 записаны.
- **Биндинг.**
  - шесть целей CI на крошечной модели;
  - на VM: E4B с драфтером, эквивалентность MTP и golden CPU записаны;
  - автомат MTP сверен с `llama-cli`.
- **Сервер.** Сервер, UI, очередь и реестр работают на фейке.
  Conformance и `go test -race ./...` зелёные.
- **Поставка.** Release workflow собирает все архивы, установщики проходят
  smoke на трёх ОС.
- **Бенчмарк.** Цепочка `bench --suite standard` → отправка → сервис →
  страница зелёная в e2e.
