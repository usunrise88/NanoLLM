# NanoLLM внутри приложения: `nanollm managed`

Это руководство для того, кто встраивает NanoLLM в приложение (например, AI-софтфон) и запускает его в фоне.
Оно описывает весь путь от начала до конца:

- откуда брать бинарники и куда их класть;
- как запускать процесс, как с ним разговаривать и как его останавливать;
- как пройти первую настройку и что делает автонастройка;
- где NanoLLM хранит модели, профили, очередь и журнал.

Документ рассчитан на то, чтобы по нему мог работать агент без доступа к исходникам NanoLLM.

---

## 0. Коротко

1. Положите содержимое архива `nanollm-<версия>-<ос>-<арх>-noui` из релиза рядом с приложением целиком:
   бинарник вместе с каталогом `lib/`.
2. Запускайте `nanollm managed -data-dir <каталог> -parent-pid <pid приложения>` как дочерний процесс.
   - stdin — канал, который приложение держит открытым и в который ничего не пишет.
   - stdout — канал, из которого приложение читает строки JSON.
3. Первая строка stdout — `{"event":"ready","url":"http://127.0.0.1:<порт>","token":"…",…}`. Дальше это
   обычный HTTP API: `Authorization: Bearer <token>`, адрес строго `url` из этой строки.
4. Если в `missing_models` есть модель, скачайте её: `POST /api/v1/models/{id}/download`.
5. Как только модель на диске, NanoLLM сам подбирает настройку под машину (автонастройка, раздел 10) и
   применяет её. Это занимает минуту-две, о начале и конце сообщают события `tune_started` и `tune_done`.
   Пока идёт замер, запросы к модели получают `503` с `Retry-After`.
6. Чат — `POST /v1/chat/completions`, как у OpenAI.
7. Чтобы остановить процесс, закройте его stdin. Если приложение упадёт, ОС закроет stdin сама, и NanoLLM
   завершится.

Права администратора не нужны ни на одном шаге. Всё работает на Windows, Linux и macOS.

---

## 1. Как это устроено

```
┌──────────────── приложение ─────────────┐
│                                         │   stdin  (pipe, никогда не пишем)
│  QProcess / CreateProcess / posix_spawn ├──────────────────────────────┐
│                                         │   stdout (pipe, строки JSON)  │
│  HTTP-клиент ──── Bearer <token> ───┐   │◄─────────────────────────┐   │
└─────────────────────────────────────┼───┘                          │   ▼
                                      │                 ┌─────────────┴─────────┐
                                      └── 127.0.0.1:N ─►│  nanollm managed       │
                                                        │  (дочерний процесс)    │
                                                        └──────────┬────────────┘
                                                                   │
                                     <data-dir>/ settings.json, models/, tune/, nanollm.db, logs/
```

- **Один процесс** отвечает на чат, ведёт очередь заданий, скачивает модели и сам себя настраивает.
- **Порт выбирает ОС** при каждом запуске. Слушается только `127.0.0.1`, поэтому брандмауэр Windows и macOS
  ничего не спрашивает.
- **Токен выдумывается при каждом запуске** и сообщается только в stdout. Он не попадает ни в аргументы
  командной строки, ни на диск.
- **Настройки** (модель, режим автонастройки) хранит сам NanoLLM в `<data-dir>/settings.json`. Приложение
  меняет их через API, и они применяются сразу, без перезапуска.
- **Время жизни** процесса привязано к приложению: через stdin и через `-parent-pid`.
- **Приоритет** процесса ниже обычного, чтобы приложение не теряло CPU. В тихом режиме на Windows процесс ещё
  и в классе EcoQoS: ОС держит его на экономных ядрах и частотах.
- **Конфиги и окружение игнорируются.** Файл `nanollm.yaml` и переменные `NANOLLM_*` в этом режиме не
  читаются: вся конфигурация берётся из флагов и `settings.json`.
- **Служба не нужна и не мешает.** `nanollm managed` не трогает службу `nanollm serve`, если она установлена
  на той же машине, и не делит с ней каталог данных.

---

## 2. Откуда брать бинарники

### 2.1. Релизы GitHub

Релизы лежат на <https://github.com/usunrise88/nanollm/releases>. Для встраивания нужны архивы **без веб-UI**
(`-noui`):

| Платформа | Архив |
|---|---|
| Windows x64 | `nanollm-<tag>-windows-amd64-noui.zip` |
| Windows ARM64 | `nanollm-<tag>-windows-arm64-noui.zip` |
| Linux x64 | `nanollm-<tag>-linux-amd64-noui.tar.gz` |
| Linux ARM64 | `nanollm-<tag>-linux-arm64-noui.tar.gz` |
| macOS (Apple silicon и Intel, один бинарник) | `nanollm-<tag>-darwin-universal-noui.tar.gz` |

Дополнения CUDA для видеокарт NVIDIA содержат только каталог `lib/` и распаковываются поверх основного
архива:

| Архив | Для чего |
|---|---|
| `nanollm-<tag>-windows-amd64-cuda13.zip` | CUDA 13.4: Turing (RTX 20xx) и новее, драйвер 580+. Брать его, если подходит. |
| `nanollm-<tag>-windows-amd64-cuda.zip` | CUDA 12.x: для более старых драйверов. |
| `nanollm-<tag>-linux-amd64-cuda.tar.gz` | CUDA 12.x на Linux. |

Без дополнения NanoLLM работает на CPU и через Vulkan на любой видеокарте с драйвером Vulkan, на Mac — через
Metal.

- **Контрольные суммы.** Рядом лежит `sha256sums.txt`. Сверяйте с ним загруженные архивы в сборочном
  конвейере приложения.
- **Версия.** Зафиксируйте конкретный тег, а не «последний». Проверить версию: `nanollm version`. Её же
  сообщает событие `ready` (`version`).
- **Совместимость протокола.** За неё отвечает поле `api_version` в событии `ready`, сейчас `1`. Если оно
  больше, чем умеет приложение, запуск лучше считать несовместимым.

### 2.2. Сборка из исходников

Если нужного архива нет, его можно собрать (нужны Go 1.27+ и bash):

```bash
git clone https://github.com/usunrise88/nanollm && cd nanollm
git checkout v<версия>
UI=0 VERSION=v<версия> ./scripts/dist.sh                    # архив для этой машины
UI=0 VERSION=v<версия> ./scripts/dist.sh windows-amd64      # или явно: linux-amd64, darwin-universal, …
UI=0 VERSION=v<версия> ./scripts/dist.sh windows-amd64-cuda13   # дополнение CUDA
```

Скрипт печатает имена готовых архивов. Нативные библиотеки он берёт из закреплённого релиза llama.cpp
(`scripts/fetch-libs.sh`), C-компилятор не нужен.

### 2.3. Что внутри архива

| Файл | Нужен ли приложению |
|---|---|
| `nanollm` / `nanollm.exe` | Да. |
| `lib/` | Да, целиком: llama.cpp и ggml (`llama.dll`, `ggml*.dll` на Windows; `.so` на Linux; `.dylib` на macOS). Набор файлов зависит от платформы. |
| `nanollm.yaml`, `README.md`, `LICENSE`, `NOTICE`, `nanollm.service`, `nanollm.plist` | Нет. `LICENSE` и `NOTICE` положите туда, где приложение хранит лицензии сторонних компонентов. |

---

## 3. Куда класть

Правило одно: **бинарник лежит рядом со своим каталогом `lib/`**. NanoLLM ищет библиотеки в `lib/` рядом с
настоящим путём бинарника (символические ссылки разворачиваются). Переименовывать библиотеки, разносить их по
разным папкам или класть DLL рядом с `.exe` вместо `lib\` нельзя.

### Windows

```
<каталог установки приложения>\
    Softphone.exe
    nanollm\
        nanollm.exe
        lib\
            llama.dll
            ggml.dll
            ggml-base.dll
            ggml-cpu-*.dll
            ggml-vulkan.dll
            ggml-cuda.dll, cudart64_13.dll, cublas*.dll   (из дополнения CUDA)
            …
```

- **Без админа.** Установка на пользователя в `%LOCALAPPDATA%\Programs\<Softphone>` прав администратора не
  требует.
- **Отдельная папка.** Держите `nanollm\` отдельной папкой, чтобы её DLL не смешивались с DLL приложения.
- **Подпись.** Подпишите `nanollm.exe` и DLL своим сертификатом Authenticode, как и остальные файлы
  приложения. Иначе SmartScreen и антивирусы будут относиться к ним настороженно.

### Linux

```
<каталог приложения>/
    softphone
    nanollm/
        nanollm            (chmod +x)
        lib/
            libllama.so
            libggml*.so
            …
```

- **Зависимости.** Для CPU нужна `libgomp1`, для видеокарт через Vulkan — `libvulkan1`. В deb- или
  rpm-пакете приложения объявите их зависимостями. `nanollm devices -lib-dir <путь к lib>` скажет, какая
  библиотека не загрузилась и почему.
- **Форматы пакетов.** Подходит для AppImage, deb, rpm и tar.gz.
- **Flatpak.** Каталог данных должен быть внутри песочницы, например `$XDG_DATA_HOME` приложения:
  `~/.var/app/<id>/data`.

### macOS

```
Softphone.app/
  Contents/
    MacOS/Softphone
    Helpers/
      nanollm/
        nanollm
        lib/
          libllama.dylib
          libggml*.dylib
          …
```

**Подпись и нотаризация.** Каждый Mach-O подписывается отдельно, своим Developer ID и с hardened runtime.
Сначала подписываются библиотеки, затем бинарник, затем само `.app`:

```bash
H="Softphone.app/Contents/Helpers/nanollm"
for f in "$H"/lib/*.dylib "$H"/nanollm; do
  codesign --force --timestamp --options runtime --sign "Developer ID Application: …" "$f"
done
codesign --force --timestamp --options runtime --sign "Developer ID Application: …" Softphone.app
```

- Особых entitlements nanollm не нужно: все библиотеки подписаны той же командой, так что library validation
  проходит.
- `--deep` не используйте: подписывайте по одному файлу, как выше.

**App Sandbox** (только если приложение распространяется через Mac App Store). Дочерний процесс наследует
песочницу. Тогда нужно:

- приложению — `com.apple.security.network.server` и `com.apple.security.network.client`;
- `nanollm` — `com.apple.security.inherit`;
- каталог данных — внутри контейнера приложения.

---

## 4. Где NanoLLM хранит данные

Всё, что пишет NanoLLM, лежит в одном **каталоге данных**. Он задаётся флагом `-data-dir`. Рекомендуем всегда
передавать его явно, внутри каталога данных самого приложения:

| ОС | Рекомендуемый `-data-dir` | Каталог по умолчанию, если флаг не передан |
|---|---|---|
| Windows | `%LOCALAPPDATA%\<Softphone>\nanollm` | `%LOCALAPPDATA%\NanoLLM` |
| macOS | `~/Library/Application Support/<Softphone>/nanollm` | `~/Library/Application Support/NanoLLM/managed` |
| Linux | `$XDG_DATA_HOME/<softphone>/nanollm` (обычно `~/.local/share/…`) | `$XDG_DATA_HOME/nanollm`, иначе `~/.local/share/nanollm` |

Каталог должен быть на локальном диске, лучше на SSD: модель весит больше 4 ГБ и читается при каждой
загрузке. На Windows это Local, а не Roaming: модель не должна ездить с профилем пользователя.

Что в нём лежит:

| Путь | Что это | Можно ли удалять |
|---|---|---|
| `settings.json` | Выбор приложения: модель, режим, автонастройка. Есть с момента первой настройки. | Да: следующий запуск снова будет первым (`configured: false`). |
| `nanollm.lock` | Блокировка «каталог занят». ОС снимает её, когда процесс завершается, как угодно. | Не нужно трогать. |
| `models/<id>/<ревизия>/` | Скачанная модель: файлы GGUF и отметки проверки sha256. | Только при остановленном процессе. Модель придётся скачать заново. |
| `tune/<модель>.<режим>.json` (+ `.prev`) | Профили автонастройки: что выбрано для этой машины и почему. | Да: следующий запуск измерит заново. |
| `tune/<модель>.<режим>.failed.json` | Запись о неудачном замере (раздел 10.4). | Да: следующий запуск попробует снова. |
| `nanollm.db` (+ `-wal`, `-shm`) | SQLite: очередь заданий и их результаты (30 дней). | Только при остановленном процессе. Пропадёт очередь. |
| `promptcache/` | Снимки кэша промптов на диске, если он включён. | Да, при остановленном процессе. |
| `logs/nanollm.log` (+ `nanollm.log.1`) | Журнал в JSON, по строке на событие. При 64 МБ ротируется, хранится одно предыдущее поколение. | Да. |
| `settings.json.invalid` | Повреждённый файл настроек, отодвинутый в сторону (см. событие `warning`). | Да. |

- **Удаление приложения.** Удалите весь каталог данных.
- **Обновление NanoLLM.** Каталог данных совместим между версиями: модели, очередь и настройки сохраняются.
  Профили автонастройки привязаны к версии библиотек: после обновления, которое их сменило, процесс измерит
  заново сам (раздел 10).

---

## 5. Запуск

### 5.1. Командная строка

```
nanollm managed -data-dir <каталог> -parent-pid <pid приложения>
```

| Флаг | По умолчанию | Смысл |
|---|---|---|
| `-data-dir DIR` | см. раздел 4 | Каталог данных. Создаётся, если его нет. |
| `-parent-pid PID` | `0` (не следить) | Завершиться, когда процесс PID завершится. Передавайте PID приложения. |
| `-watch-stdin=BOOL` | `true` | Завершиться, когда закроется stdin. Не выключайте без причины. |
| `-low-priority=BOOL` | `true` | Работать с пониженным приоритетом в любом режиме: nice 10 на Linux и macOS, BELOW_NORMAL на Windows. С `false` пониженный приоритет остаётся только у тихого режима. |
| `-ui` | выкл. | Отдать веб-интерфейс на `{url}/ui`. Только для сборок с UI, не для `-noui`. Для отладки. |

Рабочий каталог процесса значения не имеет.

### 5.2. Стандартные потоки

- **stdin.** Обязательно канал (pipe), который приложение держит открытым всё время жизни процесса и **в
  который ничего не пишет**. Когда приложение закрывает его конец или завершается, NanoLLM видит конец потока и
  штатно останавливается.

  > Если запустить NanoLLM с stdin, подключённым к `/dev/null` или `NUL`, он сразу увидит конец потока и
  > завершится. Это сделано намеренно: процесс без хозяина не должен оставаться работать.
- **stdout.** Канал. NanoLLM пишет туда **только** события, по одной JSON-строке на событие (раздел 6).
  Читайте построчно всё время работы процесса, иначе буфер канала переполнится.
- **stderr.** Пишется туда только при аварии рантайма (паника Go, падение в нативной библиотеке) и одна строка
  с причиной, если запуск не удался. Либо читайте его в лог приложения, либо перенаправьте в файл или «в
  никуда». Не оставляйте канал непрочитанным.

### 5.3. Особенности ОС

**Windows**

- Создавайте процесс с флагом `CREATE_NO_WINDOW`, чтобы не мелькала консоль. Свои дочерние процессы
  (замеры автонастройки, `nvidia-smi`) NanoLLM тогда тоже запускает без окна.
- Желательно, хотя и не обязательно: поместите процесс в Job Object с `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`.
  Тогда ОС убьёт его вместе с процессами замера даже в том редком случае, когда stdin и `-parent-pid` не
  сработали. Прав администратора не нужно.
- Сигналов в духе SIGTERM у процесса без консоли нет. Штатная остановка — закрыть stdin (раздел 8).

**macOS и Linux.** Ничего особенного. SIGTERM и SIGINT тоже работают как штатная остановка.

### 5.4. Пример: Qt (C++)

```cpp
// nanollmprocess.h — минимальная обвязка
#include <QProcess>
#include <QJsonDocument>
#include <QJsonObject>
#include <QCoreApplication>
#include <QStandardPaths>
#include <QDir>
#ifdef Q_OS_WIN
#include <windows.h>   // CREATE_NO_WINDOW
#endif

class NanoLlmProcess : public QObject {
    Q_OBJECT
public:
    QString url, token;

    void start() {
        const QString appDir = QCoreApplication::applicationDirPath();
#if defined(Q_OS_WIN)
        const QString exe = appDir + "/nanollm/nanollm.exe";
#elif defined(Q_OS_MACOS)
        const QString exe = appDir + "/../Helpers/nanollm/nanollm";
#else
        const QString exe = appDir + "/nanollm/nanollm";
#endif
        const QString dataDir =
            QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/nanollm";

        p.setProgram(exe);
        p.setArguments({"managed", "-data-dir", QDir::toNativeSeparators(dataDir),
                        "-parent-pid", QString::number(QCoreApplication::applicationPid())});
        p.setProcessChannelMode(QProcess::SeparateChannels);
#if defined(Q_OS_WIN)
        p.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments *a) {
            a->flags |= CREATE_NO_WINDOW;
        });
#endif
        connect(&p, &QProcess::readyReadStandardOutput, this, &NanoLlmProcess::onStdout);
        connect(&p, &QProcess::readyReadStandardError, this, [this] {
            qWarning() << "nanollm stderr:" << p.readAllStandardError();
        });
        connect(&p, &QProcess::finished, this, &NanoLlmProcess::onFinished);
        p.start(QIODevice::ReadWrite);   // stdin остаётся открытым — в него ничего не пишем
    }

    void stop() {                         // штатная остановка
        p.closeWriteChannel();            // закрыть stdin
        if (!p.waitForFinished(15000)) p.kill();
    }

signals:
    void ready(QString url, QString token, bool configured, QStringList missingModels);
    void tuning(QString mode, int estimateSeconds);   // замер начался: чат подождёт
    void tuned(QString mode, QString profile);        // замер закончился и применён
    void warning(QString code, QString message);
    void fatal(QString code, QString message);

private:
    QProcess p;

    void onStdout() {
        while (p.canReadLine()) {
            const QJsonObject ev = QJsonDocument::fromJson(p.readLine()).object();
            const QString name = ev["event"].toString();
            if (name == "ready") {
                url = ev["url"].toString();
                token = ev["token"].toString();
                QStringList missing;
                for (const auto &m : ev["missing_models"].toArray()) missing << m.toString();
                emit ready(url, token, ev["configured"].toBool(), missing);
            } else if (name == "tune_started") {
                emit tuning(ev["mode"].toString(), ev["estimate_s"].toInt());
            } else if (name == "tune_done") {
                emit tuned(ev["mode"].toString(), ev["profile"].toString());
            } else if (name == "warning") {
                emit warning(ev["code"].toString(), ev["message"].toString());
            } else if (name == "fatal") {
                emit fatal(ev["code"].toString(), ev["message"].toString());
            }
        }
    }
    void onFinished(int code, QProcess::ExitStatus st) { /* см. раздел 8: перезапуск с паузой */ }
};
```

HTTP-запросы делайте через `QNetworkAccessManager`:

- адрес — ровно `url` из события `ready`;
- заголовок — `Authorization: Bearer <token>`.

Для SSE-потоков (прогресс загрузки модели, ход замера, события задания) и для потокового чата
(`"stream": true`) читайте ответ построчно в `readyRead`.

### 5.5. Пример: Python (для проверки руками)

```python
import json, os, subprocess, urllib.request

p = subprocess.Popen(["./nanollm/nanollm", "managed", "-data-dir", "/tmp/nanollm-data",
                      "-parent-pid", str(os.getpid())],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
events = (json.loads(line) for line in p.stdout)
ready = next(events)
auth = {"Authorization": "Bearer " + ready["token"], "Content-Type": "application/json"}

def call(method, path, body=None):
    req = urllib.request.Request(ready["url"] + path, method=method, headers=auth,
                                 data=None if body is None else json.dumps(body).encode())
    return json.load(urllib.request.urlopen(req))

print(call("GET", "/api/v1/managed")["features"])
for ev in events:                  # модель на диске: при первом запуске идёт автонастройка
    print(ev)
    if ev["event"] in ("tune_done", "warning"):
        break
answer = call("POST", "/v1/chat/completions",
              {"messages": [{"role": "user", "content": "Привет!"}], "max_tokens": 64})
print(answer["choices"][0]["message"]["content"])
p.stdin.close()                    # остановка
print(list(events), p.wait())
```

---

## 6. События на stdout

Каждая строка — один JSON-объект, первое поле всегда `event`. Незнакомые события и поля игнорируйте: в будущих
версиях их может стать больше.

### `ready` — процесс принимает запросы

```json
{"event":"ready","url":"http://127.0.0.1:53124","port":53124,
 "token":"sk-nanollm-…","pid":4242,"version":"v0.2.0","api_version":1,
 "data_dir":"C:\\Users\\u\\AppData\\Local\\Softphone\\nanollm",
 "configured":true,"restart_required":false,"missing_models":[],
 "tune":{"state":"ready","mode":"silent"}}
```

| Поле | Смысл |
|---|---|
| `url` | Базовый адрес API. Используйте его как есть, с `127.0.0.1`. |
| `token` | Ключ на этот запуск. После перезапуска будет другой. |
| `configured` | `false` — первый запуск, настроек ещё нет (раздел 9). Процесс работает и так: с моделью по умолчанию в тихом режиме. |
| `restart_required` | Сразу после старта всегда `false`. |
| `missing_models` | Модели, которые нужны по текущим настройкам, но не скачаны. |
| `tune` | Состояние автонастройки на старте (`state` как в разделе 10.3) и режим. `missing` или `stale` означают, что сейчас начнётся замер. |

Событие приходит быстро, за секунду-две: модель загружается в память уже в фоне, или сначала идёт замер. Запрос,
пришедший раньше, дождётся загрузки модели, а во время замера получит `503` с `Retry-After`. Разумный таймаут
ожидания `ready` — 30 секунд.

### `tune_started` — начался замер автонастройки

```json
{"event":"tune_started","run":"5f0c9a1e2b3d4c6f","mode":"silent","model":"gemma-4-e4b-qat",
 "reason":"missing","estimate_s":60}
```

| `reason` | Почему |
|---|---|
| `missing` | При старте: для режима и модели на этой машине профиля нет (первый запуск). |
| `stale` | При старте: профиль есть, но машина изменилась (драйвер, библиотеки, видеокарта, память, модель). |
| `download` | Только что скачалась модель по умолчанию. |
| `settings` | Приложение сменило режим или модель, и для новой пары профиля нет. |
| `requested` | Приложение попросило: `POST /api/v1/managed/tune`. |
| `api` | Приложение само запустило замер через `POST /api/v1/optimize` (раздел 11.5). |

`estimate_s` — примерная длительность в секундах. Ход замера по шагам виден в
`GET {url}/api/v1/optimize/{run}/events` (SSE). Пока замер идёт, запросы к модели получают `503` с
`Retry-After`, а очередь заданий ждёт.

### `tune_done` — замер закончился, настройка применена

```json
{"event":"tune_done","run":"5f0c9a1e2b3d4c6f","mode":"silent","model":"gemma-4-e4b-qat",
 "applied":true,"profile":"silent@9b1f…","device":"llamacpp:cpu","plan_mode":"M9 mmap",
 "answer_s":391.0,"measured_bytes":672004416}
```

| Поле | Смысл |
|---|---|
| `applied` | Профиль уже действует, модель перезагружена по нему. `false` только у замера, запущенного через `POST /api/v1/optimize`: его применяет `POST /api/v1/optimize/{run}/apply`. |
| `profile` | Имя профиля: режим и хэш выбора. |
| `device` | Где работает модель: `llamacpp:cpu`, `llamacpp:cuda0`, `llamacpp:vulkan0`, … |
| `answer_s` | Сколько секунд на этой машине занимает эталонный ответ: промпт 4096 токенов и 300 токенов ответа. |
| `measured_bytes` | Сколько памяти (своя память процесса плюс видеопамять) занимает модель с этой настройкой. |

### `warning`

```json
{"event":"warning","code":"tune_failed","message":"…","run":"…","mode":"silent","model":"gemma-4-e4b-qat"}
```

| `code` | Что случилось |
|---|---|
| `settings_reset` | `settings.json` не удалось прочитать. Файл отодвинут в `settings.json.invalid`, и запуск считается первым (`configured: false`). |
| `tune_failed` | Замер не удался. Модель работает по ограничениям режима без профиля. Сам по себе замер больше не повторится, пока что-то не изменится (раздел 10.4). |
| `tune_skipped` | При старте замер нужен, но прошлый на этой же машине и этой же версии не удался (поля `error` и `failed_at`). Модель работает по ограничениям режима. |

### `restart_required` — следующий запуск будет работать иначе

```json
{"event":"restart_required","reasons":["priority"]}
```

Почти всё применяется на ходу, поэтому событие редкое. Сейчас причина одна: `priority`. На Linux и macOS с
`-low-priority=false` процесс в тихом режиме работает с nice 10, а поднять свой приоритет обратно ОС процессу
не даёт. После смены режима на `optimal` или `powerful` процесс работает в новом режиме, но с пониженным
приоритетом до перезапуска. С `-low-priority=true` (по умолчанию) этого не бывает.

### `stopping` — процесс останавливается

```json
{"event":"stopping","reason":"stdin closed"}
```

`reason` бывает `stdin closed`, `parent exited` или `signal`.

### `fatal` — процесс не может работать и сейчас завершится с кодом 1

```json
{"event":"fatal","code":"data_dir_locked","message":"…"}
```

| `code` | Что случилось | Что делать |
|---|---|---|
| `data_dir_locked` | В этом каталоге данных уже работает другой NanoLLM. | Не перезапускать в цикле. Найти и остановить другой экземпляр (второе окно приложения?), либо дать каждому свой `-data-dir`. |
| `data_dir_unavailable` | Каталог данных нельзя создать или в него нельзя писать. | Проверить путь и права. |
| `invalid_arguments` | Неизвестный флаг или неверное значение. | Исправить командную строку. |
| `settings_invalid` | Сохранённые настройки не превращаются в рабочую конфигурацию. | Исправить через API или удалить `settings.json`. |
| `listen_failed` | Не удалось занять порт на 127.0.0.1. | Перезапустить. Если повторяется, проверить защитное ПО. |
| `parent_gone` | Процесса из `-parent-pid` уже нет. | Передать правильный PID. |
| `server_failed` | Сервер не собрался или упал. | Подробности в `message` и в `logs/nanollm.log`. |
| `internal` | Не удалось сделать токен (нет источника случайности). | Перезапустить; повторяется — смотреть журнал. |

Код выхода процесса: `0` — штатная остановка, `1` — ошибка (перед ней было событие `fatal`, если процесс успел
его написать).

---

## 7. Авторизация и безопасность

- **Токен.** Каждый запрос, кроме `/healthz` и `/readyz`, требует `Authorization: Bearer <token>`. Токен из
  `ready` административный: с ним можно всё, включая управление моделями и автонастройку. Не пишите его в логи
  приложения и не сохраняйте на диск. После перезапуска он всё равно другой.
- **Заголовок Host.** NanoLLM отвечает только на запросы с `Host` равным `127.0.0.1:<порт>`, `localhost:<порт>`
  или `[::1]:<порт>`. Любой другой получает `421`. Это защита от DNS rebinding: веб-страница в браузере
  пользователя не сможет добраться до API. Обычный HTTP-клиент, который ходит по `url` из `ready`, ставит
  правильный `Host` сам.
- **Сеть.** Процесс ходит в сеть только за моделями, по явной команде, и только по HTTPS. Промпты и ответы
  машину не покидают.
- **Ошибки** приходят в формате `application/problem+json`:

  ```json
  {"type":"about:blank","title":"Bad Request","status":400,
   "detail":"mode must be silent, optimal or powerful","code":"invalid_request","param":"mode"}
  ```

  Ориентируйтесь на `code` и `param`. `detail` написан для человека.

  Маршруты OpenAI (`/v1/…`) отвечают ошибкой в формате OpenAI, с теми же кодами:
  `{"error":{"message":"…","type":"server_error","param":"","code":"model_unavailable"}}`.

---

## 8. Жизненный цикл

### Старт

1. Запустить процесс (раздел 5).
2. Дождаться события `ready`; таймаут 30 секунд.
   - Если пришло `fatal` или процесс завершился раньше, показать ошибку (раздел 6).
3. Если `missing_models` не пуст, предложить пользователю скачать модель (раздел 9.3).
4. Если пришло `tune_started`, показать, что модель настраивается под машину и будет готова примерно через
   `estimate_s` секунд. Чат в это время отвечает `503` с `Retry-After`.

### Штатная остановка

1. Закрыть stdin процесса. В Qt это `closeWriteChannel()`.
2. NanoLLM пишет `stopping`, перестаёт принимать запросы и даёт текущим до 5 секунд. Задания очереди, которые не
   успели, остаются в базе и продолжатся при следующем запуске. Замер автонастройки, если он шёл, прерывается
   (его процессы завершаются) и при следующем запуске начнётся заново. Неудачей он не считается.
3. Ждать выхода до 15 секунд, затем убить процесс. Ничего не сломается: база SQLite в режиме WAL переживает и
   это.

### Перезапуск (после `restart_required`)

Остановить процесс, как описано выше, дождаться выхода, запустить снова.

- Не запускайте новый процесс, пока не вышел старый: он получит `data_dir_locked`.
- Перезапуск занимает секунду-две плюс загрузку модели в фоне.

### Аварийный выход

Если процесс завершился без события `stopping` или с кодом ≠ 0 и без `fatal`, перезапускайте с нарастающей
паузой: 1, 2, 4, 8, … до 60 секунд. После минуты стабильной работы паузу можно сбросить. При
`fatal: data_dir_locked` повторять не надо, а `fatal: data_dir_unavailable` повторять бессмысленно.

### Если упало приложение

Делать ничего не нужно. ОС закрывает канал stdin, NanoLLM видит это и завершается. Если по какой-то причине не
увидит, его остановит `-parent-pid`, а на Windows ещё и Job Object. Сирот не остаётся.

### Два окна приложения

Один каталог данных — один процесс NanoLLM. Второй экземпляр приложения либо пользуется процессом первого (это
задача самого приложения, через свой IPC), либо запускает свой NanoLLM со своим `-data-dir`. Помните, что
каждый процесс держит в памяти свою копию модели.

---

## 9. Первый запуск и настройка

Первый запуск отличается полем `"configured": false` в `ready`. Процесс уже работает с настройками по
умолчанию: модель `gemma-4-e4b-qat`, тихий режим, автонастройка включена. Настраивать что-то обязательно
только если приложению нужно другое. Скачать модель нужно обязательно.

### 9.1. Каталог моделей

```
GET {url}/api/v1/catalog
```

Сейчас в каталоге одна модель:

| `id` | Что это | Скачивать |
|---|---|---|
| `gemma-4-e4b-qat` | Gemma 4 E4B (Unsloth QAT, `UD-Q4_K_XL`) с драфтером MTP для ускорения ответа | 4,22 ГБ + 60 МБ |

`GET {url}/api/v1/models` показывает, что из каталога на диске (`disk`: `absent`, `partial`, `complete`,
`corrupt`) и что загружено в память.

### 9.2. Настройки

```
PATCH {url}/api/v1/managed/settings
Content-Type: application/json

{"mode":"optimal"}
```

Передаются только поля, которые надо изменить. Остальные остаются как были. Неизвестное поле — ошибка `400`
(опечатки не проглатываются). Чтобы вернуть значение по умолчанию, передайте `""`.

| Поле | Тип | По умолчанию | Смысл |
|---|---|---|---|
| `default_model` | string | `""` (= `gemma-4-e4b-qat`) | Модель для запросов, в которых модель не указана. Должна быть в каталоге. |
| `mode` | string | `""` (= `silent`) | Режим автонастройки: `silent`, `optimal` или `powerful` (раздел 10.1). |
| `autotune` | string | `""` (= `auto`) | `auto` — мерить самому, когда для режима нет актуального профиля; `off` — работать по ограничениям режима и мерить только по просьбе приложения. |
| `thinking` | bool | `false` | Разрешить модели «думать» перед ответом в запросах, где это не указано. Ответ точнее на сложных задачах, но ждать его дольше. |
| `mirrors` | string[] | `[]` | HTTPS-зеркала Hugging Face для скачивания моделей, по порядку после основного адреса. |
| `log_level` | string | `""` (= `info`) | `debug`, `info`, `warn` или `error`. |

Ответ — статус целиком (раздел 9.4). Настройки сохраняются и применяются сразу:

- новая модель становится моделью по умолчанию;
- новый режим ставит свои ограничения и свой профиль и перезагружает модель по ним;
- если для нового режима или модели профиля на этой машине нет, начинается замер (`tune_started` с
  `reason: settings`).

### 9.3. Скачать модель

Список того, что нужно скачать, — в `missing_models` события `ready` и ответа `GET /api/v1/managed`.

```
POST {url}/api/v1/models/{id}/download
```

Ответ — поток Server-Sent Events:

```
event: progress
data: {"file":"gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf","downloaded":104857600,"total":4215695776,"percent":2.5}

…

event: done
data: {"id":"gemma-4-e4b-qat"}
```

- **Успех и ошибка.** Успех — событие `done`. Ошибка — событие `error` с объектом problem+json (раздел 7).
- **Поток и загрузка связаны.** Если закрыть поток, загрузка остановится. Скачанное не пропадает: повторный
  `POST` продолжит с того же места. Держите поток открытым, пока идёт загрузка, и показывайте прогресс.
- **Повторный запрос** на ту же модель, пока она качается, присоединяется к идущей загрузке.
- **Проверка.** Каждый файл проверяется по sha256 из каталога. Повреждённое не устанавливается.
- **Сеть.** Модели скачиваются с Hugging Face по HTTPS. Если он недоступен, задайте `mirrors`.
- **Что дальше.** Как только модель на диске, начинается автонастройка (`tune_started` с
  `reason: download`), а за ней модель загружается в память.

Модель **никогда не скачивается сама**. Запрос к модели, которой нет на диске, получит `503 model_unavailable`
с подсказкой, что её нужно скачать.

### 9.4. Статус

```
GET {url}/api/v1/managed
```

```json
{
  "version":"v0.2.0","api_version":1,
  "configured":true,"restart_required":false,
  "data_dir":"…","models_dir":"…/models",
  "settings":{"default_model":"","mode":"optimal","autotune":"","thinking":false,"mirrors":[],"log_level":""},
  "features":{
    "chat":{"state":"ready","models":["gemma-4-e4b-qat"]},
    "autotune":{"state":"ready","mode":"optimal","model":"gemma-4-e4b-qat",
                "profile":{"id":"optimal@3c2a…","created":"2026-10-24T10:15:02Z","device":"llamacpp:cuda0",
                           "plan_mode":"M3 vram-ple-disk","n_ctx":16384,"mtp":{"enabled":true,"draft_max":3},
                           "answer_s":2.8,"measured_bytes":4135727104}}
  },
  "missing_models":[]
}
```

| `features.chat.state` | Смысл |
|---|---|
| `missing` | Модели нет на диске (она в `missing`). Скачайте её. |
| `loading` | Модель на диске, но ещё не в памяти: загружается или ждёт конца замера. |
| `ready` | Модель в памяти, чат отвечает сразу. |

`restart_required` и `restart_reasons` описаны в разделе 6 (`restart_required`).

### 9.5. Вся первая настройка — одна последовательность

```
start → ready{configured:false, missing_models:["gemma-4-e4b-qat"], tune:{state:"waiting_model"}}
PATCH /api/v1/managed/settings {"mode":"optimal"}   → (по желанию)
POST /api/v1/models/gemma-4-e4b-qat/download (SSE)  → прогресс в UI
                                                     ← tune_started{reason:"download", mode:"optimal"}
                                                     ← tune_done{applied:true}
POST /v1/chat/completions                            → ответ
```

Перезапуск в этой последовательности не нужен.

---

## 10. Автонастройка

### 10.1. Режимы

Режим выбирает приложение (`mode` в настройках). Внутри режима NanoLLM сам выбирает лучшее для машины:
устройство (CPU, видеокарта, встроенная графика), раскладку модели в памяти, длину контекста, ускорение MTP,
число потоков.

| | `silent` (по умолчанию) | `optimal` | `powerful` |
|---|---|---|---|
| Устройства | только CPU | любые: видеокарта, встроенная графика, CPU | любые |
| Память (своя память процесса плюс видеопамять) | минимальная, при которой ответ идёт не медленнее 1 токена в секунду | до 4 ГиБ | вся свободная |
| Потоки CPU | 2 | половина физических ядер | все |
| Приоритет | ниже обычного; на Windows ещё EcoQoS | ниже обычного (с `-low-priority=false` — обычный) | так же |
| Для чего | Машиной пользуются, пока модель отвечает, и не замечают её: не шумят вентиляторы видеокарты, не тормозят другие программы. Ответы самые медленные. | Быстрый ответ при умеренной цене: видеокарта, если она есть, но не больше 4 ГиБ памяти. | Самый быстрый ответ, всё остальное подождёт. |

Если в `powerful` свободной памяти меньше 6 ГиБ, режим работает с ограничениями `optimal`, но на всех ядрах, и
говорит об этом в `features.autotune.problems` (`mode_degraded`).

### 10.2. Когда идёт замер

Замер идёт, если:

- `autotune` не `off`;
- модель на диске;
- для пары «модель, режим» на этой машине нет актуального профиля. Профиль устаревает, когда меняется что-то
  из того, на чём он измерен: видеокарта или её драйвер, процессор, объём памяти, библиотеки NanoLLM, файл
  модели;
- прошлый замер этой пары на этой же машине и этой же версии NanoLLM не провалился (раздел 10.4).

Тогда замер начинается сам:

- при старте, сразу после `ready`;
- когда скачалась модель по умолчанию;
- после `PATCH /api/v1/managed/settings`, сменившего режим или модель.

Замер длится минуту-две: NanoLLM по очереди пробует несколько вариантов в отдельных процессах и меряет
скорость ответа и занятую память. Во время замера:

- запросы к модели получают `503` с `Retry-After` (до 10 секунд), очередь заданий ждёт;
- модель выгружена из памяти: замер загружает свои копии;
- машина занята сильнее, чем обычно в этом режиме. Тихий режим и меряется тихо: на CPU, с пониженным
  приоритетом.

После замера профиль сохраняется в `tune/`, применяется, и модель загружается по нему. Следующие запуски
берут профиль с диска и не меряют.

Если режим или модель сменили во время замера, текущий замер доводится до конца, а затем применяется новая
настройка: с замером, если для неё нет профиля.

### 10.3. Что видно в статусе

`features.autotune` в `GET /api/v1/managed`:

| `state` | Смысл |
|---|---|
| `running` | Идёт замер (`run` — его идентификатор). |
| `ready` | Профиль режима для модели действует (`profile`). |
| `waiting_model` | Модели нет на диске; замер начнётся после её скачивания. |
| `missing` | Профиля нет, замер вот-вот начнётся. |
| `stale` | Профиль устарел (`changes` — что изменилось), замер вот-вот начнётся. |
| `failed` | Замер не удался (`message` — почему). Модель работает по ограничениям режима. |
| `off` | `autotune: off`, профиля нет. Модель работает по ограничениям режима. |

`problems` — то, что мешает режиму прямо сейчас: например, видеопамять занята другими программами
(`vram_busy`) или для `powerful` мало свободной памяти (`mode_degraded`).

### 10.4. Неудачный замер

Если ни один вариант не удалось измерить (не загрузилась библиотека, не хватило памяти, упал драйвер),
приходит `warning` с кодом `tune_failed`. Модель всё равно работает: по ограничениям режима, без профиля.

NanoLLM записывает неудачу вместе с «отпечатком» машины и версией. Следующие запуски на той же машине не
меряют снова, а сообщают `warning` с кодом `tune_skipped`. Снова мерить NanoLLM начнёт сам, когда изменится
машина, драйвер, библиотеки, модель, режим или версия NanoLLM.

Попросить замер можно в любой момент, в том числе после неудачи:

```
POST {url}/api/v1/managed/tune
```

Ответ — `202` и `{"run":"…","estimate_s":60}`. Если модели нет на диске — `409`. Если замер уже идёт — `409`.

### 10.5. Подробности замера

`GET {url}/api/v1/optimize/{run}/events` — SSE с ходом замера: `survey` (что за машина), `problem`,
`candidate` (варианты), `probe_start` и `probe` (каждая проба с результатом), `decision`, `done` или `error`.
`GET {url}/api/v1/optimize` — режим, профиль и устарел ли он. Эти маршруты — те же, что у службы
`nanollm serve` (docs/OPTIMIZE.md, §4.2).

---

## 11. Работа с API

### 11.1. Чат (совместим с OpenAI)

```
POST {url}/v1/chat/completions
Content-Type: application/json

{"messages":[{"role":"system","content":"Ты помощник оператора колл-центра."},
             {"role":"user","content":"Кратко перескажи разговор: …"}],
 "max_tokens":300, "stream":true}
```

- Поле `model` можно не передавать: берётся модель из настроек.
- `stream: true` — ответ потоком SSE по токенам, как у OpenAI. Без него — один JSON.
- Поддерживаются `temperature`, `top_p`, `stop`, `seed`, `tools`, `response_format` (в том числе JSON Schema).
- Если в приложении уже есть клиент OpenAI, достаточно поменять базовый URL на `{url}/v1` и ключ на токен.
- `GET {url}/v1/models` — модели в формате OpenAI.

Собственный диалект NanoLLM (`POST {url}/api/v1/chat`) отдаёт больше подробностей: время по стадиям,
использованный кэш промпта, статистику ускорения.

### 11.2. Задания в фоне (очередь)

Для пачки запросов, которые не нужны прямо сейчас (например, пересказы звонков за день):

1. Поставить задание: `POST {url}/api/v1/jobs` с телом JSONL — по строке на запрос, каждая строка — запрос
   `/api/v1/chat` с необязательным `custom_id`. Ответ — `202` и описание задания с `id`.
2. Следить за ним: `GET {url}/api/v1/jobs/{id}/events` (SSE).
3. Забрать результат: `GET {url}/api/v1/jobs/{id}`.
4. Отменить при необходимости: `DELETE {url}/api/v1/jobs/{id}`.

Задания уступают интерактивным запросам: чат пользователя не ждёт очередь. Задания хранятся в `nanollm.db`
30 дней и переживают перезапуски.

### 11.3. Управление моделью

```
GET  {url}/api/v1/models                    # что на диске и что в памяти
POST {url}/api/v1/models/{id}/load          # загрузить в память сейчас
POST {url}/api/v1/models/{id}/unload        # выгрузить из памяти
GET  {url}/api/v1/devices                   # устройства, которые видит NanoLLM
```

### 11.4. Здоровье

- `GET {url}/healthz` → `200` и версии. Без токена.
- `GET {url}/readyz` → `200`, когда модель по умолчанию в памяти; `503` с причиной, когда нет. Без токена.

### 11.5. Замер вручную

`POST {url}/api/v1/optimize` с `{"mode":"optimal"}` запускает замер, не применяя его; `{"dry_run":true}`
только осматривает машину. Применить результат — `POST {url}/api/v1/optimize/{run}/apply`. Режим применённого
профиля записывается в настройки (`mode`). Обычному приложению это не нужно: автонастройка делает то же
сама.

---

## 12. Ресурсы

- **Диск.** Модель — 4,3 ГБ. Профили — килобайты. База очереди и журнал — мегабайты.
- **Память и скорость.** Зависят от режима (раздел 10.1) и машины; точные числа — `measured_bytes` и
  `answer_s` в `tune_done` и в профиле. Для `gemma-4-e4b-qat` на ноутбуке с Core i7-12700H, 32 ГБ и RTX 3070
  Ti Laptop 8 ГБ:

  | Режим | Устройство | Память | Эталонный ответ (промпт 4096 токенов, ответ 300) |
  |---|---|---|---|
  | `silent` | CPU, 2 потока | 0,67 ГБ | 391 с |
  | `optimal` | CUDA, раскладка M3 | 4,1 ГБ | 2,8 с |
  | `powerful` | CUDA, вся модель в видеопамяти | 6,4 ГБ | 2,9 с |

  Короткий вопрос без длинного контекста тихий режим обрабатывает намного быстрее эталона: основное время
  уходит на чтение длинного промпта.

  Кроме того, ОС держит в страничном кэше прочитанные части файла модели. Эту память она отдаёт другим
  программам сама, и в лимиты режима она не входит.
- **CPU.** Приоритет ниже обычного. Потоков — по режиму. Между запросами процесс CPU не ест.
- **Видеокарта.** Только в `optimal` и `powerful`, и только если NanoLLM её видит (`GET /api/v1/devices`). Для
  NVIDIA на Windows и Linux нужно дополнение CUDA (раздел 2.1), иначе работает Vulkan.
- **Сеть.** Только скачивание моделей по явной команде.

---

## 13. Обновление NanoLLM вместе с приложением

1. Остановить процесс (раздел 8).
2. Заменить папку `nanollm/` целиком: бинарник и `lib/` должны быть из одного архива (и дополнение CUDA той же
   версии).
3. Запустить. Каталог данных подхватится как есть. Если обновление сменило библиотеки, профиль устареет, и
   процесс сам измерит заново (`tune_started` с `reason: stale`).
4. Проверить `api_version` в `ready`: протокол меняется, только если это число растёт.

---

## 14. Диагностика

- **Журнал:** `<data-dir>/logs/nanollm.log`, по JSON-строке на запись.
  - Первая строка каждого запуска — `starting nanollm managed` с версией, режимом и моделью.
  - Затем открытие библиотек и устройств, ход автонастройки, загрузки модели и по строке на каждый HTTP-запрос:
    путь, статус, время. Тексты запросов и ответов в журнал не пишутся.
  - Подробнее — `log_level: debug`.
- **Версия:** `nanollm version` — версии NanoLLM и llama.cpp.
- **Устройства и библиотеки:** `nanollm devices -lib-dir <путь к lib>` — какие библиотеки загрузились, какие
  устройства видны и почему не загрузилась остальная часть.

| Симптом | Причина |
|---|---|
| Процесс сразу выходит, `stopping` с `stdin closed` | stdin не канал (подключён к `/dev/null`/`NUL`) или его закрыли. |
| `fatal: data_dir_locked` | Второй процесс на том же каталоге данных; старый ещё не вышел. |
| `401` | Нет заголовка `Authorization` или токен от прошлого запуска. |
| `421` | Запрос не на `127.0.0.1:<порт>` (другое имя хоста, прокси). Обращайтесь строго по `url`. |
| `503` с `Retry-After` на чат | Идёт замер автонастройки (`features.autotune.state: running`). Повторите через `Retry-After` секунд. |
| `503 model_unavailable` без `Retry-After` | Модели нет на диске (`features.chat.state: missing`) или она не поместилась в память. |
| `warning: tune_failed` | Ни один вариант не удалось измерить; подробности в журнале. Модель работает по ограничениям режима. |
| Модель работает на CPU в `optimal` | Видеокарты не видно: нет дополнения CUDA или драйвера Vulkan, либо видеопамять занята (`problems`). |
| На Windows мелькает консоль | Процесс запущен без `CREATE_NO_WINDOW`. |
| На Linux: в `nanollm devices` нет CPU | Нет `libgomp1`. |
| На macOS: «library not loaded: @rpath/libllama.dylib» | Нет `lib/` рядом с бинарником или подпись сломала одну из библиотек. |

---

## 15. Чек-лист интеграции

- [ ] Архив нужной версии и платформы (и дополнение CUDA, если нужно) сверен с `sha256sums.txt`; папка
      `nanollm/` лежит рядом с приложением целиком, вместе с `lib/`.
- [ ] Windows: `nanollm.exe` и DLL подписаны. macOS: каждый Mach-O подписан с `--options runtime`, приложение
      нотаризовано. Linux: зависимости `libgomp1` и `libvulkan1`.
- [ ] Запуск: `managed -data-dir <свой каталог> -parent-pid <свой pid>`; stdin — открытый канал без записи;
      stdout читается построчно; stderr читается или отброшен; на Windows `CREATE_NO_WINDOW` (и Job Object).
- [ ] Ожидание `ready` с таймаутом; обработка `fatal` по коду.
- [ ] Все запросы на `url` из `ready` с `Bearer <token>`; токен нигде не сохраняется.
- [ ] `missing_models` → загрузка модели с прогрессом, поток держится открытым до `done`.
- [ ] `tune_started` / `tune_done` → пользователь видит, что модель настраивается; `503` с `Retry-After`
      на чат во время замера обрабатывается повтором.
- [ ] Выбор режима (`silent`, `optimal`, `powerful`) в настройках приложения → `PATCH /api/v1/managed/settings`.
- [ ] `warning: tune_failed` / `tune_skipped` → запись в лог приложения; по желанию кнопка «Настроить
      заново» → `POST /api/v1/managed/tune`.
- [ ] Остановка: закрыть stdin, ждать до 15 с, затем убить.
- [ ] Аварийный выход → перезапуск с нарастающей паузой; при `data_dir_locked` — без повторов.
- [ ] Удаление приложения удаляет каталог данных.
