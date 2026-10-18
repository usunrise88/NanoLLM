package web

import (
	"net/http"
	"strings"
)

// Languages of the pages.
const (
	LangRU = "ru"
	LangEN = "en"
)

var messages = map[string][2]string{ // key → {ru, en}
	"site":             {"NanoLLM — бенчмарк", "NanoLLM benchmark"},
	"nav.results":      {"Результаты", "Results"},
	"nav.howto":        {"Как запустить", "How to run"},
	"nav.privacy":      {"Приватность", "Privacy"},
	"nav.export":       {"Данные", "Data"},
	"title.benchmarks": {"Бенчмарк NanoLLM", "NanoLLM benchmark"},
	"lead":             {"Скорость пресетов NanoLLM на железе пользователей: медиана по установкам, каждая установка и каждая машина считаются один раз.", "How NanoLLM's presets run on people's hardware: medians over installations, each installation and each machine counted once."},
	"f.model":          {"Модель", "Model"},
	"f.suite":          {"Набор", "Suite"},
	"f.class":          {"Класс устройства", "Device class"},
	"f.cpu":            {"Процессор", "CPU"},
	"f.gpu":            {"Видеокарта", "GPU"},
	"f.system":         {"Модель ноутбука/ПК", "Laptop/PC model"},
	"f.os":             {"ОС", "OS"},
	"f.preset":         {"Пресет", "Preset"},
	"f.any":            {"любой", "any"},
	"f.apply":          {"Показать", "Show"},
	"f.reset":          {"Сбросить", "Reset"},
	"c.system":         {"Система", "System"},
	"c.preset":         {"Пресет", "Preset"},
	"c.tg":             {"Генерация, ток/с", "Generation, tok/s"},
	"c.tgn":            {"На глубине, ток/с", "At depth, tok/s"},
	"c.pp":             {"Prefill, ток/с", "Prefill, tok/s"},
	"c.chat":           {"Чат, ток/с", "Chat, tok/s"},
	"c.ttft":           {"TTFT, мс", "TTFT, ms"},
	"c.mtp":            {"Ускорение MTP", "MTP speed-up"},
	"c.installs":       {"Установок", "Installs"},
	"c.load":           {"Загрузка, мс", "Load, ms"},
	"c.status":         {"Статус", "Status"},
	"c.mode":           {"Режим", "Mode"},
	"c.backend":        {"Бэкенд", "Backend"},
	"c.check":          {"Проверка", "Check"},
	"c.output":         {"Вывод", "Output"},
	"empty":            {"Пока нет результатов под эти фильтры.", "No results for these filters yet."},
	"note.range":       {"В скобках — p25–p75.", "In brackets: p25–p75."},
	"badge.verified":   {"проверено вендором", "vendor-verified"},
	"ff.laptop":        {"ноутбук", "laptop"},
	"ff.desktop":       {"ПК", "desktop"},
	"ff.server":        {"сервер", "server"},
	"ff.vm":            {"виртуальная машина", "virtual machine"},
	"ff.unknown":       {"неизвестно", "unknown"},
	"kind.system":      {"Система", "System"},
	"kind.cpu":         {"Процессор", "CPU"},
	"kind.gpu":         {"Видеокарта", "GPU"},
	"s.title":          {"Отправка", "Submission"},
	"s.machine":        {"Машина", "Machine"},
	"s.client":         {"Клиент", "Client"},
	"s.model":          {"Модель", "Model"},
	"s.date":           {"Дата прогона", "Run date"},
	"s.runs":           {"Пресеты", "Presets"},
	"s.raw":            {"Отправленный JSON (без ключа установки)", "Submitted JSON (without the install key)"},
	"s.queued":         {"Отправка ждёт модерации и пока не показывается.", "This submission is awaiting moderation and is not shown yet."},
	"s.gone":           {"Отправка удалена её автором.", "This submission was withdrawn by its author."},
	"s.withdraw":       {"Удалить свою отправку: nanollm bench withdraw", "Withdraw your submission: nanollm bench withdraw"},
	"p.ok":             {"в пределах потолка", "within the ceiling"},
	"p.unknown":        {"железо неизвестно", "unknown hardware"},
	"p.implausible":    {"выше потолка", "above the ceiling"},
	"o.none":           {"не с чем сравнить", "nothing to compare"},
	"o.agree":          {"совпадает с когортой", "agrees with its cohort"},
	"o.prefix":         {"общий префикс, токенов:", "shared prefix, tokens:"},
	"gpu.none":         {"без видеокарты", "no GPU"},
	"outlier":          {"выброс", "outlier"},
	"notfound":         {"Не найдено.", "Not found."},
	"draft":            {"Черновик. Текст ждёт юридической проверки и может измениться.", "Draft. This text is pending legal review and may change."},
	"footer.license":   {"Данные открыты под лицензией", "Data are open under"},
	"export.csv":       {"CSV", "CSV"},
	"export.json":      {"JSON", "JSON"},
	"export.runs":      {"все прогоны", "all runs"},
	"recent":           {"Недавние отправки", "Recent submissions"},
	"title.howto":      {"Как запустить бенчмарк", "How to run the benchmark"},
	"title.privacy":    {"Политика приватности бенчмарка", "Benchmark privacy policy"},
	"lang.other":       {"English", "Русский"},
	"status.ok":        {"ok", "ok"},
	"status.skipped":   {"пропущен", "skipped"},
	"status.refused":   {"не помещается", "refused"},
	"status.failed":    {"ошибка", "failed"},
}

// T is the message for a key in a language; an unknown key shows itself.
func T(lang, key string) string {
	m, ok := messages[key]
	if !ok {
		return key
	}
	if lang == LangEN {
		return m[1]
	}
	return m[0]
}

// lang picks the page language: ?lang= (remembered in a cookie, the
// service's only cookie), the cookie, then Accept-Language.
func lang(w http.ResponseWriter, r *http.Request) string {
	if l := r.URL.Query().Get("lang"); l == LangRU || l == LangEN {
		http.SetCookie(w, &http.Cookie{Name: "lang", Value: l, Path: "/", MaxAge: 365 * 24 * 3600, SameSite: http.SameSiteLaxMode, HttpOnly: true})
		return l
	}
	if c, err := r.Cookie("lang"); err == nil && (c.Value == LangRU || c.Value == LangEN) {
		return c.Value
	}
	for _, part := range strings.Split(r.Header.Get("Accept-Language"), ",") {
		tag := strings.ToLower(strings.TrimSpace(strings.SplitN(part, ";", 2)[0]))
		switch {
		case strings.HasPrefix(tag, "ru"):
			return LangRU
		case strings.HasPrefix(tag, "en"):
			return LangEN
		}
	}
	return LangEN
}
