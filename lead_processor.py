# ОБЪЕДИНЁННЫЙ СКРИПТ: СУММАРАЙЗ + ОЦЕНКА + СООБЩЕНИЯ
# Использует 8 API ключей параллельно для ускорения
# pip install google-generativeai pandas openpyxl python-dotenv

import pandas as pd
import json
import time
import re
import threading
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# ============ НАСТРОЙКИ ============
# Загрузка API ключей из переменных окружения
# ВАЖНО: Некоторые API ключи могут быть скомпрометированы (статус 403 "leaked")
# Мы используем только рабочие ключи для избежания 403 ошибок
API_KEYS_ALL = []
for i in range(1, 9):
    key = os.getenv(f"GOOGLE_API_KEY_{i}")
    if key:
        API_KEYS_ALL.append((i, key))

# Функция для проверки валидности API ключа
def _test_api_key(api_key):
    """Проверить, работает ли API ключ (не в статусе 403 leaked)"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        model.generate_content("test")
        return True
    except Exception as e:
        if "403" in str(e) and "leaked" in str(e).lower():
            return False
        # Если это другая ошибка, считаем ключ валидным (может быть квота или сеть)
        return True

# Фильтруем рабочие ключи
API_KEYS = []
print("🔍 Проверка API ключей...")
for key_num, key in API_KEYS_ALL:
    if _test_api_key(key):
        API_KEYS.append(key)
        print(f"  ✅ Ключ #{key_num}: OK")
    else:
        print(f"  ❌ Ключ #{key_num}: СКОМПРОМЕТИРОВАН (403 leaked)")

if not API_KEYS:
    raise ValueError("❌ Нет рабочих API ключей! Все ключи скомпрометированы или отсутствуют. Добавьте новые ключи в .env файл.")

# ============ КОНФИГУРАЦИЯ МОДЕЛЕЙ ============
# Модель 1: gemma-3-27b-it (основная - большие лимиты)
MODEL_PRIMARY = "gemma-3-27b-it"
MODEL_PRIMARY_RPM = 30  # Requests Per Minute
MODEL_PRIMARY_RPD = 15000  # Requests Per Day

# Модель 2: gemini-2.5-flash-lite (fallback - лимиты исчерпаны на сегодня)
MODEL_FALLBACK = "gemini-2.5-flash-lite"
MODEL_FALLBACK_RPM = 10
MODEL_FALLBACK_RPD = 20

BATCH_SIZE = 150  # Пользователей в одном батче для оценки
MAX_WORKERS = min(8, len(API_KEYS))  # Параллельных потоков

# Входной файл
INPUT_FILE = 'users_copy.xlsx'
OUTPUT_FILE = 'leads_processed.xlsx'

# Блокировка для потокобезопасности
lock = threading.Lock()
api_key_index = 0

# ============ УПРАВЛЕНИЕ ЛИМИТАМИ API ============
class RateLimitTracker:
    """Отслеживает использование лимитов API для каждой модели"""
    def __init__(self):
        self.requests_today = {}  # {model: count}
        self.requests_this_minute = {}  # {model: [timestamps]}
        self.last_minute_reset = {}  # {model: timestamp}
        self.day_start = datetime.now()

    def _reset_minute_if_needed(self, model):
        """Сбросить счётчик минуты если прошла минута"""
        now = datetime.now()
        if model not in self.last_minute_reset:
            self.last_minute_reset[model] = now

        time_elapsed = (now - self.last_minute_reset[model]).total_seconds()
        if time_elapsed >= 60:
            self.requests_this_minute[model] = []
            self.last_minute_reset[model] = now

    def _reset_day_if_needed(self):
        """Сбросить счётчик дня если прошли сутки"""
        now = datetime.now()
        time_elapsed = (now - self.day_start).total_seconds()
        if time_elapsed >= 86400:  # 24 часа
            self.requests_today = {}
            self.day_start = now

    def can_use_model(self, model, rpm_limit, rpd_limit):
        """Проверить может ли модель использоваться"""
        self._reset_minute_if_needed(model)
        self._reset_day_if_needed()

        # Проверка RPM (requests per minute)
        if model not in self.requests_this_minute:
            self.requests_this_minute[model] = []

        # Очистить старые запросы (старше минуты)
        now = datetime.now()
        self.requests_this_minute[model] = [
            ts for ts in self.requests_this_minute[model]
            if (now - ts).total_seconds() < 60
        ]

        if len(self.requests_this_minute[model]) >= rpm_limit:
            return False, f"Превышен лимит RPM ({rpm_limit})"

        # Проверка RPD (requests per day)
        if model not in self.requests_today:
            self.requests_today[model] = 0

        if self.requests_today[model] >= rpd_limit:
            return False, f"Превышен лимит RPD ({rpd_limit})"

        return True, "OK"

    def record_request(self, model):
        """Записать запрос для модели"""
        self._reset_minute_if_needed(model)
        self._reset_day_if_needed()

        if model not in self.requests_this_minute:
            self.requests_this_minute[model] = []
        if model not in self.requests_today:
            self.requests_today[model] = 0

        self.requests_this_minute[model].append(datetime.now())
        self.requests_today[model] += 1

    def get_status(self):
        """Получить статус использования лимитов"""
        self._reset_day_if_needed()
        status = {
            "timestamp": datetime.now().isoformat(),
            "models": {}
        }

        for model, count in self.requests_today.items():
            status["models"][model] = {
                "requests_today": count,
                "minute_requests": len(self.requests_this_minute.get(model, []))
            }

        return status

rate_limiter = RateLimitTracker()

# Инициализация genai один раз при старте с первым доступным API ключом
def _init_genai():
    """Инициализировать Google Generative AI один раз"""
    if API_KEYS:
        try:
            genai.configure(api_key=API_KEYS[0])
            return True
        except Exception as e:
            print(f"⚠️  Ошибка при инициализации genai: {str(e)[:50]}")
            return False
    return False

_genai_initialized = _init_genai()

def sanitize_input(text, max_length=500):
    """Санитизировать входные данные для использования в промптах"""
    if not text:
        return ""
    # Преобразовать в строку и обрезать
    text = str(text).strip()
    if len(text) > max_length:
        text = text[:max_length]
    # Убрать потенциально опасные символы (но не переделывать полностью)
    # Оставляем текст как есть, так как LLM обычно безопасен
    return text

def is_empty_value(value):
    """Проверка, является ли значение пустым"""
    if value is None:
        return True
    if pd.isna(value):
        return True
    str_val = str(value).strip()
    if str_val == '':
        return True
    # Проверяем только строковые представления NaN, но не буквальное "nan" в данных
    try:
        if pd.isna(float(str_val)):
            return True
    except (ValueError, TypeError):
        pass
    str_val_lower = str_val.lower()
    if str_val_lower in ['none', 'null']:
        return True
    return False

def get_next_api_key():
    """Получить следующий API ключ по кругу"""
    global api_key_index
    with lock:
        if not API_KEYS:
            raise ValueError("API ключи не инициализированы!")
        if api_key_index is None:
            api_key_index = 0
        key = API_KEYS[api_key_index % len(API_KEYS)]
        api_key_index += 1
        return key

def get_model_with_fallback():
    """Получить модель с fallback при исчерпании лимитов"""
    max_attempts = 10  # Максимум 10 попыток = 60 сек
    attempts = 0

    while attempts < max_attempts:
        # Сначала пробуем использовать primary модель
        can_use, reason = rate_limiter.can_use_model(
            MODEL_PRIMARY,
            MODEL_PRIMARY_RPM,
            MODEL_PRIMARY_RPD
        )

        if can_use:
            return MODEL_PRIMARY

        # Если primary исчерпана, пробуем fallback
        can_use_fallback, reason_fb = rate_limiter.can_use_model(
            MODEL_FALLBACK,
            MODEL_FALLBACK_RPM,
            MODEL_FALLBACK_RPD
        )

        if can_use_fallback:
            print(f"  ⚠️  {reason}, переключаемся на {MODEL_FALLBACK}")
            return MODEL_FALLBACK

        # Если обе модели исчерпаны, ждём и пробуем снова
        print(f"  ⏳ {reason}, {reason_fb} - ожидание 6 сек... (попытка {attempts + 1}/{max_attempts})")
        time.sleep(6)
        attempts += 1

    # Если превышено максимальное количество попыток
    print(f"  ❌ КРИТИЧЕСКАЯ ОШИБКА: Оба лимита API исчерпаны! Невозможно продолжить.")
    raise RuntimeError("Лимиты всех моделей исчерпаны, невозможно получить модель")

# ============ СУММАРАЙЗ ============
def summarize_profile(row, api_key):
    """Создаёт суммарное описание деятельности"""
    name = str(row.get('Имя', '') or '').strip()
    surname = str(row.get('Фамилия', '') or '').strip()
    description = str(row.get('Описание профиля', '') or '').strip()

    if is_empty_value(name) and is_empty_value(surname) and is_empty_value(description):
        return "Деятельность не указана"

    info_parts = []
    if not is_empty_value(name):
        info_parts.append(f"Имя: {name}")
    if not is_empty_value(surname):
        info_parts.append(f"Фамилия: {surname}")
    if not is_empty_value(description):
        info_parts.append(f"Описание: {description}")

    if not info_parts:
        return "Деятельность не указана"

    info_text = "\n".join(info_parts)

    prompt = f"""Проанализируй информацию и создай краткое описание деятельности:

{info_text}

Напиши сухое и лаконичное описание (2-3 предложения) в формате: имя, фамилия, чем занимается, название компании/бизнеса.

Правила:
- Пиши факты напрямую, без фраз "что указывает", "что говорит о"
- Используй прямой стиль: "Имя Фамилия занимается [деятельность]. Компания [название] специализируется на [услуги]"
- Будь конкретным и информативным

Ответ:"""

    try:
        # Выбираем модель с fallback логикой
        current_model = get_model_with_fallback()

        model = genai.GenerativeModel(current_model)
        response = model.generate_content(prompt, generation_config={"temperature": 0.7, "max_output_tokens": 500})

        # Записываем использованный запрос
        rate_limiter.record_request(current_model)

        if hasattr(response, 'text'):
            result = response.text.strip()
            # Очистка
            prefixes = ["Ответ:", "Описание:", "Деятельность:"]
            for prefix in prefixes:
                if result.lower().startswith(prefix.lower()):
                    result = result[len(prefix):].strip()
            result = re.sub(r'\s+', ' ', result).strip()
            if len(result) > 20:
                return result
        return "Деятельность не определена"
    except Exception as e:
        error_str = str(e).lower()
        if "429" in str(e) or "quota" in error_str:
            time.sleep(2)
        return "Ошибка API"

# ============ БАТЧ ОЦЕНКА ============
def score_batch(batch_data, api_key, batch_num):
    """Оценивает батч пользователей"""
    batch_size = len(batch_data)

    users_text = ""
    for i, user in enumerate(batch_data, 1):
        name = str(user.get('Имя', '')).strip()
        surname = str(user.get('Фамилия', '')).strip()
        desc = str(user.get('Суммарное описание', '')).strip()
        if not desc or desc.lower() in ['nan', 'none', '']:
            desc = "Нет описания"
        users_text += f"{i}. {name} {surname}\n   {desc}\n\n"

    prompt = f"""Проанализируй {batch_size} пользователей и присуди каждому скор интереса для веб-агентства CodexAI от 1 до 100.

МЫ ИЩЕМ: владельцев бизнеса, предпринимателей, экспертов которым НУЖЕН сайт или улучшение сайта.

ВЫСОКИЙ СКОР (70-100):
- Владельцы бизнеса, предприниматели, директора компаний
- Офлайн-бизнесы: рестораны, салоны, клиники, магазины, услуги
- Эксперты, коучи, консультанты без упоминания своего сайта
- Начинающие бизнесы которым нужен первый сайт
- Компании со старым/плохим сайтом

СРЕДНИЙ СКОР (30-69):
- Маркетологи, SMM-специалисты (могут рекомендовать нас клиентам)
- Менеджеры в компаниях
- Фрилансеры не из IT сферы

НИЗКИЙ СКОР (1-29):
- РАЗРАБОТЧИКИ, программисты, веб-дизайнеры = 5-15 (наши конкуренты!)
- IT-специалисты, DevOps, тестировщики = 5-15
- Веб-агентства, digital-студии = 1-10 (прямые конкуренты)
- Студенты, безработные = 5-20
- Личные аккаунты без бизнеса = 10-25

КЛЮЧЕВОЕ: Если человек сам делает сайты или работает в IT = НИЗКИЙ СКОР!

СПИСОК:
{users_text}

ОТВЕТ: JSON массив ТОЛЬКО index и score:
[{{"index": 1, "score": 85}}, ...]"""

    try:
        # Выбираем модель с fallback логикой
        current_model = get_model_with_fallback()

        model = genai.GenerativeModel(current_model)
        response = model.generate_content(prompt)

        # Записываем использованный запрос
        rate_limiter.record_request(current_model)

        if hasattr(response, 'text') and response.text:
            text = response.text.strip()
            # Использовать неполадочное выражение для правильного парсинга JSON
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                try:
                    scores_array = json.loads(json_match.group(0))
                    if not isinstance(scores_array, list):
                        print(f"  Батч #{batch_num} ошибка: JSON не является массивом")
                        return None

                    scores_dict = {}
                    for item in scores_array:
                        if not isinstance(item, dict):
                            continue
                        idx = item.get('index')
                        score = item.get('score')
                        if idx is not None and score is not None:
                            try:
                                score_int = int(float(score))
                                if 0 <= score_int <= 100:
                                    # idx - 1 потому что индексы в JSON начинаются с 1
                                    # Проверка границ: индекс должен быть в диапазоне батча
                                    if 0 <= idx - 1 < batch_size:
                                        scores_dict[idx - 1] = score_int
                                    else:
                                        print(f"    ⚠️  Индекс {idx} вне диапазона батча ({batch_size})")
                            except (ValueError, TypeError, OverflowError) as e:
                                print(f"    ⚠️  Ошибка конвертирования score '{score}': {str(e)[:30]}")
                                pass

                    # Пустой результат - валидное состояние (может быть пустой батч)
                    if scores_dict:
                        print(f"  Батч #{batch_num}: {len(scores_dict)} оценок")
                    else:
                        print(f"  Батч #{batch_num}: оценки не найдены (пустой результат)")
                    return scores_dict  # Возвращаем пустой dict вместо None
                except json.JSONDecodeError as e:
                    print(f"  Батч #{batch_num} ошибка JSON: {str(e)[:50]}")
                    return None
            else:
                print(f"  Батч #{batch_num} ошибка: JSON не найден в ответе")
                return None
        else:
            print(f"  Батч #{batch_num} ошибка: пустой ответ от API")
            return None
    except Exception as e:
        print(f"  Батч #{batch_num} ошибка: {str(e)[:50]}")
        return None

# ============ СЛОВАРЬ РУСИФИКАЦИИ ИМЁН ============
NAMES_TO_CYRILLIC = {
    # Мужские имена
    'artem': 'Артём', 'artemiy': 'Артемий', 'alexander': 'Александр', 'alex': 'Алекс',
    'alexey': 'Алексей', 'aleksey': 'Алексей', 'andrey': 'Андрей', 'andrei': 'Андрей',
    'andrew': 'Андрей', 'anton': 'Антон', 'boris': 'Борис', 'denis': 'Денис',
    'dmitry': 'Дмитрий', 'dmitri': 'Дмитрий', 'dima': 'Дима', 'eugene': 'Евгений',
    'evgeny': 'Евгений', 'evgeniy': 'Евгений', 'fedor': 'Фёдор', 'fyodor': 'Фёдор',
    'grigory': 'Григорий', 'igor': 'Игорь', 'ilya': 'Илья', 'ivan': 'Иван',
    'kirill': 'Кирилл', 'konstantin': 'Константин', 'leonid': 'Леонид', 'maxim': 'Максим',
    'max': 'Макс', 'maksim': 'Максим', 'mikhail': 'Михаил', 'michael': 'Михаил',
    'misha': 'Миша', 'nikita': 'Никита', 'nikolay': 'Николай', 'nikolai': 'Николай',
    'nick': 'Николай', 'oleg': 'Олег', 'pavel': 'Павел', 'paul': 'Павел',
    'peter': 'Пётр', 'petr': 'Пётр', 'roman': 'Роман', 'ruslan': 'Руслан',
    'sergey': 'Сергей', 'sergei': 'Сергей', 'stanislav': 'Станислав', 'stas': 'Стас',
    'timur': 'Тимур', 'vadim': 'Вадим', 'valery': 'Валерий', 'viktor': 'Виктор',
    'victor': 'Виктор', 'vitaly': 'Виталий', 'vladimir': 'Владимир', 'vlad': 'Влад',
    'vladislav': 'Владислав', 'yaroslav': 'Ярослав', 'yuri': 'Юрий', 'yury': 'Юрий',
    'george': 'Георгий', 'gena': 'Гена', 'gleb': 'Глеб', 'egor': 'Егор',
    'arseny': 'Арсений', 'arseniy': 'Арсений', 'daniil': 'Даниил', 'daniel': 'Даниил',
    'timofey': 'Тимофей', 'semyon': 'Семён', 'simon': 'Симон', 'matvey': 'Матвей',
    'stepan': 'Степан', 'steven': 'Степан', 'vasily': 'Василий',
    # Женские имена
    'anna': 'Анна', 'anastasia': 'Анастасия', 'nastya': 'Настя', 'alexandra': 'Александра',
    'alina': 'Алина', 'daria': 'Дарья', 'darya': 'Дарья', 'dasha': 'Даша',
    'ekaterina': 'Екатерина', 'kate': 'Катя', 'katya': 'Катя', 'elena': 'Елена',
    'helen': 'Елена', 'lena': 'Лена', 'eva': 'Ева', 'evgenia': 'Евгения',
    'irina': 'Ирина', 'julia': 'Юлия', 'yulia': 'Юлия', 'kristina': 'Кристина',
    'ksenia': 'Ксения', 'kseniya': 'Ксения', 'larisa': 'Лариса', 'lyudmila': 'Людмила',
    'maria': 'Мария', 'masha': 'Маша', 'marina': 'Марина', 'natalya': 'Наталья',
    'natalia': 'Наталья', 'natasha': 'Наташа', 'nina': 'Нина', 'olga': 'Ольга',
    'polina': 'Полина', 'svetlana': 'Светлана', 'sveta': 'Света', 'tatiana': 'Татьяна',
    'tanya': 'Таня', 'valentina': 'Валентина', 'valeria': 'Валерия', 'vera': 'Вера',
    'victoria': 'Виктория', 'vika': 'Вика', 'yana': 'Яна', 'alena': 'Алёна',
    'alyona': 'Алёна', 'diana': 'Диана', 'elizaveta': 'Елизавета', 'liza': 'Лиза',
    'galina': 'Галина', 'karina': 'Карина', 'lyubov': 'Любовь', 'margarita': 'Маргарита',
    'nadezhda': 'Надежда', 'sofia': 'София', 'sonya': 'Соня', 'tamara': 'Тамара',
    'veronika': 'Вероника', 'zhanna': 'Жанна', 'zoya': 'Зоя',
}

def russify_name(name):
    """Русифицирует латинское имя в кириллицу (Artem → Артём)"""
    if not name:
        return name

    name = str(name).strip()
    if not name:
        return ""

    # Если уже на кириллице - возвращаем как есть
    if any('\u0400' <= c <= '\u04FF' for c in name):
        return name

    # Ищем в словаре (регистронезависимо)
    name_lower = name.lower()
    if name_lower in NAMES_TO_CYRILLIC:
        return NAMES_TO_CYRILLIC[name_lower]

    # Проверяем составные имена (типа "Artem Ignatev" - берём только имя)
    parts = name.split()
    if len(parts) > 1:
        first_part = parts[0].lower()
        if first_part in NAMES_TO_CYRILLIC:
            return NAMES_TO_CYRILLIC[first_part]

    # Если не нашли в словаре - возвращаем оригинал
    return name

# ============ ГЕНЕРАЦИЯ СООБЩЕНИЙ ============
def generate_messages(row, api_key):
    """Генерирует 2 сообщения для лида"""
    name = str(row.get('Имя', '') or '').strip()
    surname = str(row.get('Фамилия', '') or '').strip()
    summary = str(row.get('Суммарное описание', '') or '').strip()

    if not name or name.lower() in ['nan', 'none']:
        name = ""

    # Русифицируем имя (Artem → Артём)
    name = russify_name(name)

    # Сообщение 1: Приветственное ТОЛЬКО с именем (без фамилии)
    if name:
        msg1 = f"Добрый день, {name}!\n\nМы - веб-агентство CodexAI. Посмотрите наши кейсы: codexai.pro"
    else:
        msg1 = f"Добрый день!\n\nМы - веб-агентство CodexAI. Посмотрите наши кейсы: codexai.pro"

    # Сообщение 2: Персонализированное и интересное
    prompt_msg2 = f"""Напиши ВТОРОЕ сообщение для Telegram (2-3 предложения). Первое сообщение уже отправлено с приветствием.

КОНТЕКСТ ЛИДА:
{summary if summary else 'Информация о деятельности не указана'}

МЫ: веб-агентство CodexAI, делаем сайты, лендинги, веб-приложения.
НАШИ КЕЙСЫ: codexai.pro

ЗАДАЧА: Напиши персонализированное сообщение, которое:
1) Показывает, что мы понимаем их сферу деятельности
2) Предлагает конкретную пользу (сайт поможет привлечь клиентов / увеличить продажи / показать экспертность)
3) Приглашает посмотреть релевантные кейсы на codexai.pro
4) НЕ используй "Добрый день" - это уже было в первом сообщении

СТИЛЬ: дружелюбный, без пустых фраз, конкретно про их бизнес

Ответ (только текст сообщения, без кавычек):"""

    try:
        # Выбираем модель с fallback логикой
        current_model = get_model_with_fallback()

        model = genai.GenerativeModel(current_model)
        resp = model.generate_content(prompt_msg2, generation_config={"temperature": 0.8, "max_output_tokens": 300})

        # Записываем использованный запрос
        rate_limiter.record_request(current_model)

        if hasattr(resp, 'text') and resp.text:
            msg2 = resp.text.strip()
            # Удаляем возможные префиксы
            prefixes = ["Ответ:", "Сообщение:"]
            for prefix in prefixes:
                if msg2.lower().startswith(prefix.lower()):
                    msg2 = msg2[len(prefix):].strip()
            # Ограничиваем длину
            if len(msg2) > 500:
                msg2 = msg2[:500].rsplit(' ', 1)[0] + "..."
        else:
            msg2 = "Посмотрите наши кейсы и примеры работ на codexai.pro. Мы помогаем компаниям создавать современные сайты и веб-приложения."
    except Exception as e:
        print(f"    Ошибка при генерации сообщения: {str(e)[:50]}")
        msg2 = "Посмотрите наши кейсы и примеры работ на codexai.pro. Мы помогаем компаниям создавать современные сайты и веб-приложения."

    return msg1, msg2

# ============ ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА ============
def process_summarize_parallel(df, indices_to_process):
    """Параллельный суммарайз"""
    results = {}
    results_lock = threading.Lock()

    def process_one(idx):
        try:
            # Проверка границ индекса перед доступом
            if idx >= len(df) or idx < 0:
                print(f"    ⚠️  Индекс {idx} вне диапазона DataFrame ({len(df)} строк)")
                return idx, "Ошибка индекса"
            row = df.iloc[idx]
            api_key = get_next_api_key()
            result = summarize_profile(row, api_key)
            time.sleep(0.3)  # Небольшая пауза
            return idx, result
        except IndexError:
            print(f"    ❌ IndexError при обработке строки {idx}")
            return idx, "Ошибка индекса"
        except Exception as e:
            print(f"    ❌ Ошибка при обработке строки {idx}: {str(e)[:50]}")
            return idx, "Ошибка обработки"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in indices_to_process}
        done = 0
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                with results_lock:
                    results[idx] = result
                done += 1
                if done % 10 == 0:
                    print(f"  Суммарайз: {done}/{len(indices_to_process)}")
            except Exception as e:
                print(f"    Ошибка в потоке: {str(e)[:50]}")

    return results

def process_messages_parallel(df, indices):
    """Параллельная генерация сообщений"""
    results = {}
    results_lock = threading.Lock()

    def process_one(idx):
        try:
            row = df.iloc[idx]
            api_key = get_next_api_key()
            msg1, msg2 = generate_messages(row, api_key)
            time.sleep(0.3)
            return idx, msg1, msg2
        except Exception as e:
            print(f"    Ошибка при генерации сообщений для строки {idx}: {str(e)[:50]}")
            return idx, "Ошибка", "Ошибка"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in indices}
        done = 0
        for future in as_completed(futures):
            try:
                idx, msg1, msg2 = future.result()
                with results_lock:
                    results[idx] = (msg1, msg2)
                done += 1
                if done % 10 == 0:
                    print(f"  Сообщения: {done}/{len(indices)}")
            except Exception as e:
                print(f"    Ошибка в потоке: {str(e)[:50]}")

    return results

# ============ MAIN ============
def main():
    print("=" * 70)
    print("LEAD PROCESSOR: СУММАРАЙЗ + ОЦЕНКА + СООБЩЕНИЯ")
    print(f"API ключей: {len(API_KEYS)} | Параллельных потоков: {MAX_WORKERS}")
    print("=" * 70)
    print("\n📊 КОНФИГУРАЦИЯ МОДЕЛЕЙ:")
    print(f"  Основная: {MODEL_PRIMARY} (RPM: {MODEL_PRIMARY_RPM}, RPD: {MODEL_PRIMARY_RPD})")
    print(f"  Fallback: {MODEL_FALLBACK} (RPM: {MODEL_FALLBACK_RPM}, RPD: {MODEL_FALLBACK_RPD})")
    print("=" * 70 + "\n")

    # Загрузка
    print(f"Загрузка {INPUT_FILE}...")
    try:
        # Проверка размера файла перед загрузкой
        file_size_mb = os.path.getsize(INPUT_FILE) / (1024 * 1024)
        if file_size_mb > 100:  # Предупреждение если больше 100 MB
            print(f"⚠️  ВНИМАНИЕ: большой файл ({file_size_mb:.1f} MB). Обработка может быть медленной.")

        df = pd.read_excel(INPUT_FILE)
        num_rows = len(df)

        if num_rows > 100000:  # Предупреждение если более 100K строк
            print(f"⚠️  ВНИМАНИЕ: большой файл ({num_rows} строк). Обработка может быть медленной.")

        print(f"Загружено: {num_rows} пользователей\n")
    except FileNotFoundError:
        print(f"❌ Ошибка: файл '{INPUT_FILE}' не найден!")
        return
    except MemoryError:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: недостаточно памяти для загрузки файла!")
        return
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {str(e)}")
        return

    # Инициализация колонок
    if 'Суммарное описание' not in df.columns:
        df['Суммарное описание'] = None
    if 'Интерес' not in df.columns:
        df['Интерес'] = None
    if 'Сообщение 1' not in df.columns:
        df['Сообщение 1'] = None
    if 'Сообщение 2' not in df.columns:
        df['Сообщение 2'] = None

    start_time = datetime.now()

    # ===== ШАГ 1: СУММАРАЙЗ =====
    print("=" * 70)
    print("ШАГ 1: СУММАРАЙЗ")
    print("=" * 70)

    # Находим строки без суммарайза
    needs_summary = []
    for idx in range(len(df)):
        val = df.at[idx, 'Суммарное описание']
        if is_empty_value(val):
            needs_summary.append(idx)

    print(f"Требуется суммарайз: {len(needs_summary)} из {len(df)}")

    if needs_summary:
        summary_results = process_summarize_parallel(df, needs_summary)
        for idx, result in summary_results.items():
            df.at[idx, 'Суммарное описание'] = result
        print(f"Суммарайз завершён: {len(summary_results)}\n")
    else:
        print("Все суммарайзы уже есть\n")

    # Промежуточное сохранение
    df.to_excel(OUTPUT_FILE, index=False)

    # ===== ШАГ 2: ОЦЕНКА =====
    print("=" * 70)
    print("ШАГ 2: ОЦЕНКА ЛИДОВ")
    print("=" * 70)

    # Находим неоценённые (используем is_empty_value для консистентности)
    needs_score = [idx for idx in range(len(df)) if is_empty_value(df.at[idx, 'Интерес'])]
    print(f"Требуется оценка: {len(needs_score)} из {len(df)}")

    if needs_score:
        total_batches = (len(needs_score) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Батчей: {total_batches}")

        for batch_num in range(total_batches):
            start_idx = batch_num * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(needs_score))
            batch_indices = needs_score[start_idx:end_idx]

            # Проверка на пустой батч
            if not batch_indices:
                print(f"  Батч #{batch_num + 1}: пропущен (пуст)")
                continue

            # Проверка наличия требуемых колонок
            required_cols = ['Имя', 'Фамилия', 'Суммарное описание']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ❌ Батч #{batch_num + 1} ошибка: отсутствуют колонки: {missing_cols}")
                continue

            try:
                batch_data = df.iloc[batch_indices][required_cols].to_dict('records')
            except (KeyError, IndexError) as e:
                print(f"  ❌ Батч #{batch_num + 1} ошибка при получении данных: {str(e)[:50]}")
                continue
            api_key = get_next_api_key()

            scores = score_batch(batch_data, api_key, batch_num + 1)

            if scores:
                # Потокобезопасное обновление DataFrame
                # Все проверки и обновления внутри lock для атомарности
                for rel_idx, score in scores.items():
                    with lock:
                        try:
                            # Проверка границ ВНУТРИ lock
                            if 0 <= rel_idx < len(batch_indices):
                                orig_idx = batch_indices[rel_idx]
                                # Дополнительная проверка индекса DataFrame
                                if 0 <= orig_idx < len(df):
                                    df.at[orig_idx, 'Интерес'] = score
                                else:
                                    print(f"    ⚠️  Индекс {orig_idx} вне диапазона DataFrame")
                            else:
                                print(f"    ⚠️  Индекс {rel_idx} вне диапазона батча ({len(batch_indices)})")
                        except (IndexError, KeyError, TypeError) as e:
                            print(f"    ❌ Ошибка при обновлении строки {rel_idx}: {str(e)[:30]}")

            time.sleep(0.5)

        print(f"Оценка завершена\n")
    else:
        print("Все оценки уже есть\n")

    # Промежуточное сохранение
    df.to_excel(OUTPUT_FILE, index=False)

    # ===== ШАГ 3: СООБЩЕНИЯ =====
    print("=" * 70)
    print("ШАГ 3: ГЕНЕРАЦИЯ СООБЩЕНИЙ")
    print("=" * 70)

    # Генерируем сообщения только для интересных лидов (скор >= 50)
    needs_messages = []
    for idx in range(len(df)):
        score = df.at[idx, 'Интерес']
        msg1 = df.at[idx, 'Сообщение 1']
        try:
            # Безопасное преобразование скора в float с обработкой ошибок
            if pd.notna(score):
                score_float = float(score)
                if score_float >= 50:
                    if is_empty_value(msg1):
                        needs_messages.append(idx)
        except (ValueError, TypeError):
            # Пропустить строку если скор не можно преобразовать
            continue

    print(f"Генерация сообщений для лидов (скор >= 50): {len(needs_messages)}")

    if needs_messages:
        msg_results = process_messages_parallel(df, needs_messages)
        for idx, (msg1, msg2) in msg_results.items():
            df.at[idx, 'Сообщение 1'] = msg1
            df.at[idx, 'Сообщение 2'] = msg2
        print(f"Сообщения сгенерированы: {len(msg_results)}\n")
    else:
        print("Все сообщения уже есть\n")

    # ===== ФИНАЛЬНОЕ СОХРАНЕНИЕ =====
    print("=" * 70)
    print("СОХРАНЕНИЕ")
    print("=" * 70)

    # Сортируем по интересу
    # Убедитьсяч что колонка имеет правильный тип перед сортировкой
    try:
        df['Интерес'] = pd.to_numeric(df['Интерес'], errors='coerce')
    except Exception:
        pass  # Если ошибка, пропускаем преобразование

    df = df.sort_values('Интерес', ascending=False, na_position='last').reset_index(drop=True)

    # Сохраняем ТОЛЬКО Excel
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Файл сохранён: {OUTPUT_FILE}")

    # ===== СТАТИСТИКА =====
    elapsed = (datetime.now() - start_time).total_seconds()
    scored = df[df['Интерес'].notna()]

    print("\n" + "=" * 70)
    print("СТАТИСТИКА")
    print("=" * 70)
    print(f"Время: {elapsed/60:.1f} мин")
    print(f"Обработано: {len(scored)}")

    if len(scored) > 0:
        hot = len(scored[scored['Интерес'] >= 80])
        warm = len(scored[(scored['Интерес'] >= 50) & (scored['Интерес'] < 80)])
        cold = len(scored[(scored['Интерес'] >= 20) & (scored['Интерес'] < 50)])

        print(f"\nГОРЯЧИЕ (80-100): {hot}")
        print(f"ТЁПЛЫЕ (50-79):  {warm}")
        print(f"ХОЛОДНЫЕ (<50):  {cold}")

    # ===== СТАТИСТИКА ИСПОЛЬЗОВАНИЯ API =====
    api_stats = rate_limiter.get_status()
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ИСПОЛЬЗОВАНИЯ API")
    print("=" * 70)
    for model, stats in api_stats.get("models", {}).items():
        print(f"{model}:")
        print(f"  Запросов сегодня: {stats['requests_today']}")
        print(f"  Запросов за минуту: {stats['minute_requests']}")

    print("\n" + "=" * 70)
    print("ТОП-10 ЛИДОВ")
    print("=" * 70)

    top = df[df['Интерес'] >= 50].head(10)
    for i, (_, row) in enumerate(top.iterrows(), 1):
        name = str(row['Имя']) if pd.notna(row['Имя']) else ''
        surname = str(row['Фамилия']) if pd.notna(row['Фамилия']) else ''
        score = row['Интерес']
        print(f"{i:2d}. [{score:3.0f}] {name} {surname}")

    print(f"\nГотово! Результаты в {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
