#!/usr/bin/env python3
# Generate internal/eval/suites/reasoning.jsonl: problems that need several steps of
# reasoning, with answers the script computes (never typed by hand).
#
#     python3 -I tools/reasoningcases.py > internal/eval/suites/reasoning.jsonl
#
# The other suites are saturated (53/53 without thinking, docs/QUALITY-STAGE2.md 4):
# they cannot say whether thinking helps. These are six kinds of problem, four per
# kind and language, 48 items, built from a fixed seed so the suite is the same
# on every run. Every item asks for the answer on a last line "ANSWER: <value>"; the
# checker is a regular expression for exactly that value, so a wrong value, a
# missing line and an answer cut short by the token budget all fail.
import itertools
import json
import math
import random
import re
from datetime import date

SEED = 20261023
rng = random.Random(SEED)

SUFFIX = {
    "ru": "\n\nЗакончи ответ одной строкой вида `ANSWER: <значение>`: только число или слово, без единиц измерения.",
    "en": "\n\nEnd your reply with one line of the form `ANSWER: <value>`: only the number or the word, no units.",
}
NAMES = {
    "ru": ["Аня", "Борис", "Вера", "Глеб", "Дина", "Егор", "Жанна", "Игорь"],
    "en": ["Ann", "Boris", "Clara", "Dmitri", "Eva", "Felix", "Greta", "Hugo"],
}
WOMEN = {"ru": ["Аня", "Вера", "Дина", "Жанна"], "en": ["Ann", "Clara", "Eva", "Greta"]}
WEEKDAYS = {
    "ru": ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
MONTHS = {
    "ru": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"],
    "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
}

items = []


def add(kind, lang, n, question, value):
    value = str(value)
    pat = r"(?mi)^[\s*`]*ANSWER:\s*[*`]*" + re.escape(value) + r"[*`]*\.?\s*$"
    items.append({
        "id": f"reason-{kind}-{lang}-{n}", "lang": lang, "capability": f"reasoning-{lang}",
        "messages": [{"role": "user", "content": question + SUFFIX[lang]}],
        "checker": {"type": "regex", "pattern": pat},
    })


def arithmetic(lang, n):
    while True:
        p, q = rng.randrange(40, 90, 5), rng.randrange(5, 30, 5)
        a, b = rng.randrange(3, 9), rng.randrange(4, 12)
        cost = a * p + b * (p - q)
        paid = (cost // 500 + 1) * 500
        if paid - cost > 0:
            break
    name = rng.choice(WOMEN[lang])
    if lang == "ru":
        return (f"В магазине тетрадь стоит {p} рублей, а ручка на {q} рублей дешевле тетради. "
                f"{name} купила тетради ({a} шт.) и ручки ({b} шт.) и расплатилась одной купюрой в {paid} рублей. "
                "Сколько рублей сдачи ей дали?"), paid - cost
    return (f"A notebook costs {p} dollars and a pen costs {q} dollars less than a notebook. "
            f"{name} buys {a} notebooks and {b} pens and pays with a single note of {paid} dollars. "
            "How many dollars of change does she get?"), paid - cost


def ordering(lang, n):
    people = rng.sample(NAMES[lang], 5)  # people[0] is the tallest
    pairs = [(people[i], people[i + 1]) for i in range(4)]  # (taller, shorter)
    rng.shuffle(pairs)
    stmts = []
    for taller, shorter in pairs:
        if rng.random() < 0.5:
            stmts.append((f"{taller} выше, чем {shorter}." if lang == "ru" else f"{taller} is taller than {shorter}."))
        else:
            stmts.append((f"{shorter} ниже, чем {taller}." if lang == "ru" else f"{shorter} is shorter than {taller}."))
    k = rng.choice([2, 3, 4])
    if lang == "ru":
        q = " ".join(stmts) + f" Кто {k}-й по росту, если считать от самого высокого?"
    else:
        q = " ".join(stmts) + f" Counting from the tallest, who is number {k} in height?"
    return q, people[k - 1]


def calendar(lang, n):
    if n < 2:  # the weekday some days ahead
        start = rng.randrange(7)
        days = rng.randrange(40, 400)
        ans = WEEKDAYS[lang][(start + days) % 7]
        if lang == "ru":
            return (f"Сегодня {WEEKDAYS['ru'][start]}. Какой день недели будет через {days} дней? "
                    "Ответь названием дня недели с маленькой буквы."), ans.lower()
        return f"Today is {WEEKDAYS['en'][start]}. What day of the week is it {days} days from today?", ans
    # days between two dates of a year that is not a leap year
    while True:
        d1 = date(2025, rng.randrange(1, 7), rng.randrange(1, 28))
        d2 = date(2025, rng.randrange(7, 13), rng.randrange(1, 28))
        if (d2 - d1).days > 60:
            break
    if lang == "ru":
        return (f"Сколько дней прошло от {d1.day} {MONTHS['ru'][d1.month - 1]} до {d2.day} {MONTHS['ru'][d2.month - 1]} "
                "того же года (год не високосный)? Сам начальный день не считай, конечный считай."), (d2 - d1).days
    return (f"How many days are there from {MONTHS['en'][d1.month - 1]} {d1.day} to {MONTHS['en'][d2.month - 1]} {d2.day} "
            "of the same year (not a leap year)? Do not count the first day, count the last one."), (d2 - d1).days


def rates(lang, n):
    if n % 2 == 0:  # a price that rose and then fell
        while True:
            orig = rng.randrange(100, 1000, 10)
            up, down = rng.choice([(20, 25), (50, 40), (30, 20), (25, 40)])
            final = orig * (100 + up) * (100 - down)
            if final % 10000 == 0:
                final //= 10000
                break
        if lang == "ru":
            return (f"Цена товара сначала выросла на {up} %, а потом снизилась на {down} % от новой цены, и стала равна {final} руб. "
                    "Сколько рублей стоил товар вначале?"), orig
        return (f"A price first rose by {up} % and then fell by {down} % of the new price, and became {final} dollars in all. "
                "How many dollars was it at the start?"), orig
    while True:  # a mean speed
        v1, v2 = rng.randrange(40, 100, 10), rng.randrange(40, 100, 10)
        t1, t2 = rng.randrange(1, 6), rng.randrange(1, 6)
        dist, time = v1 * t1 + v2 * t2, t1 + t2
        if v1 != v2 and dist % time == 0:
            break
    if lang == "ru":
        return (f"Поезд ехал {t1} ч со скоростью {v1} км/ч, потом {t2} ч со скоростью {v2} км/ч. "
                "Какова его средняя скорость на всём пути в км/ч?"), dist // time
    return (f"A train ran {t1} h at {v1} km/h and then {t2} h at {v2} km/h. "
            "What is its mean speed over the whole way, in km/h?"), dist // time


def counting(lang, n):
    if n == 0:
        people = rng.randrange(8, 20)
        return (f"На встрече {people} человек, и каждый пожал руку каждому другому ровно один раз. Сколько было рукопожатий?" if lang == "ru"
                else f"{people} people meet and each shakes hands with every other person exactly once. How many handshakes are there?"), people * (people - 1) // 2
    if n == 1:
        total, pick = rng.randrange(7, 12), rng.randrange(3, 5)
        return (f"Из {total} кандидатов нужно выбрать комиссию из {pick} человек без должностей. Сколькими способами это можно сделать?" if lang == "ru"
                else f"A committee of {pick} with no distinct roles is chosen from {total} candidates. In how many ways can this be done?"), math.comb(total, pick)
    if n == 2:
        top = rng.randrange(100, 400)
        a, b = rng.choice([(3, 5), (4, 6), (3, 7), (2, 9)])
        both = top // math.lcm(a, b)
        cnt = top // a + top // b - both
        return (f"Сколько натуральных чисел от 1 до {top} включительно делятся на {a} или на {b} (хотя бы на одно из них)?" if lang == "ru"
                else f"How many natural numbers from 1 to {top} inclusive are divisible by {a} or by {b} (by at least one of them)?"), cnt
    digits = rng.sample(range(1, 9), 4)
    cnt = sum(1 for p in itertools.permutations(digits, 3) if p[2] % 2 == 0)
    ds = ", ".join(map(str, sorted(digits)))
    return (f"Сколько различных трёхзначных чисел с разными цифрами можно составить из цифр {ds}, чтобы число было чётным?" if lang == "ru"
            else f"How many different three-digit numbers with distinct digits can be made from the digits {ds} so that the number is even?"), cnt


def knights(lang, n):
    # Three islanders; a knight always tells the truth, a knave always lies.
    # The statements are drawn until exactly one assignment is consistent.
    people = ["A", "B", "C"]
    while True:
        spec = []
        for i in range(3):
            kind = rng.choice(["knight", "knave", "oneknave", "bothsame"])
            other = rng.choice([j for j in range(3) if j != i])
            spec.append((kind, other))

        def holds(kind, other, i, v):
            if kind == "knight":
                return v[other]
            if kind == "knave":
                return not v[other]
            if kind == "oneknave":
                return sum(not x for x in v) == 1
            return v[i] == v[other]

        sols = [v for v in itertools.product([True, False], repeat=3)
                if all(holds(k, o, i, v) == v[i] for i, (k, o) in enumerate(spec))]
        if len(sols) == 1:
            break
    words = {
        "ru": {"knight": "{o} — рыцарь", "knave": "{o} — лжец", "oneknave": "среди нас ровно один лжец", "bothsame": "мы с {o} одного типа"},
        "en": {"knight": "{o} is a knight", "knave": "{o} is a knave", "oneknave": "exactly one of us is a knave", "bothsame": "{o} and I are of the same kind"},
    }[lang]
    lines = []
    for i, (kind, other) in enumerate(spec):
        said = words[kind].format(o=people[other])
        lines.append(f"{people[i]}: «{said}»." if lang == "ru" else f'{people[i]} says: "{said}".')
    if lang == "ru":
        q = ("На острове рыцари всегда говорят правду, а лжецы всегда лгут. Трое островитян A, B и C сказали: "
             + " ".join(lines) + " Сколько среди них рыцарей?")
    else:
        q = ("On an island knights always tell the truth and knaves always lie. Three islanders A, B and C said: "
             + " ".join(lines) + " How many of them are knights?")
    return q, sum(sols[0])


KINDS = [("arith", arithmetic), ("order", ordering), ("calendar", calendar), ("rates", rates), ("count", counting), ("knights", knights)]

for lang in ("ru", "en"):
    for kind, fn in KINDS:
        for n in range(4):
            q, v = fn(lang, n)
            add(kind, lang, n + 1, q, v)

print("// Multi-step reasoning, ru and en: six kinds of problem, four each. Generated by tools/reasoningcases.py (seed %d); every answer is computed, the last line is ANSWER: <value>." % SEED)
for it in items:
    print(json.dumps(it, ensure_ascii=False))
