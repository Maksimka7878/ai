# Для работы требуется установить библиотеку: pip install google-generativeai
import pandas as pd
import time
import re
from datetime import datetime
import google.generativeai as genai

# Настройки Gemini API
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_1")  # Загружаем из .env
MODEL = "gemma-3-27b-it"  # Изменено на flash-lite для лучшей производительности и лимитов
genai.configure(api_key=GEMINI_API_KEY)
MAX_REQUESTS_PER_MINUTE = 29  # безопасный лимит, можешь поднять до 25–28 если всё ок
START_INDEX = 5054

print(f"Используется Gemini API с моделью: {MODEL}")

# Счетчик запросов для отслеживания лимитов
request_counter = {
    'count': 0,
    'start_time': time.time(),
    'total_requests': 0
}

# Функция для запроса к Gemini API
def ask_gemini(prompt):
    """Отправляет запрос к Gemini API и возвращает ответ с учётом лимита RPS/ RPM"""
    global request_counter
    
    max_retries = 5
    base_delay = 5  # сейчас не используется, но можно задействовать для других ошибок

    for attempt in range(max_retries):
        try:
            now = time.time()
            elapsed = now - request_counter['start_time']

            # Если прошла минута — обнуляем окно
            if elapsed >= 60:
                request_counter['count'] = 0
                request_counter['start_time'] = now
                elapsed = 0

            # Если за текущую минуту уже достигли лимит — ждём до конца окна
            if request_counter['count'] >= MAX_REQUESTS_PER_MINUTE:
                wait_time = 60 - elapsed + 0.5  # +0.5 сек запас
                print(f"Достигнут лимит {MAX_REQUESTS_PER_MINUTE}/мин, пауза {wait_time:.1f}с...")
                time.sleep(wait_time)
                # после ожидания открываем новое окно
                request_counter['count'] = 0
                request_counter['start_time'] = time.time()

            # === сам запрос к модели ===
            model = genai.GenerativeModel(MODEL)
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 2000,
            }

            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Проверяем блокировки безопасности
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason:
                    print(f"Запрос заблокирован: {response.prompt_feedback.block_reason}")
                    return None

            # Достаём текст ответа
            content = None
            if hasattr(response, 'text'):
                content = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        content = candidate.content.parts[0].text.strip()
                    elif hasattr(candidate.content, 'text'):
                        content = candidate.content.text.strip()

            if content:
                # Увеличиваем счётчики только при успешном ответе
                request_counter['count'] += 1
                request_counter['total_requests'] += 1
                return content
            else:
                print("Пустой ответ от API")
                return None

        except Exception as e:
            error_str = str(e)

            # Rate limit от самого API — сразу выходим
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                request_counter['count'] = 0
                print(f"Rate limit! Превышен лимит запросов. Пропускаем запрос.")
                return None

            print(f"Ошибка запроса: {error_str}")
            return None

    return None


# Загрузка файла Excel
file_path = 'chat_users_error_20251210_023434.xlsx'
print(f"Загрузка файла: {file_path}")
df = pd.read_excel(file_path)
print(f"Загружено строк: {len(df)}")

# Функция для создания суммарного описания деятельности
def summarize_profile_with_nlp(row):
    name = str(row.get('Имя', '') or '').strip()
    surname = str(row.get('Фамилия', '') or '').strip()
    description = str(row.get('Описание профиля', '') or '').strip()
    
    def is_empty(value):
        if not value:
            return True
        if value.lower() in ['nan', 'none', 'null']:
            return True
        return False
    
    if is_empty(name) and is_empty(surname) and is_empty(description):
        return "Деятельность не указана"
    
    info_parts = []
    if not is_empty(name):
        info_parts.append(f"Имя: {name}")
    if not is_empty(surname):
        info_parts.append(f"Фамилия: {surname}")
    if not is_empty(description):
        info_parts.append(f"Описание: {description}")
    
    if not info_parts:
        return "Деятельность не указана"
    
    info_text = "\n".join(info_parts)
    
    # Промпт для анализа деятельности
    prompt = f"""Проанализируй информацию и создай краткое описание деятельности:

{info_text}

Напиши сухое и лаконичное описание (3-5 предложений) в формате: имя, фамилия, чем занимается, название компании/бизнеса.

Правила:
- Пиши факты напрямую, без фраз "что указывает", "что говорит о", "что свидетельствует", "что означает"
- Используй прямой стиль: "Имя Фамилия занимается [деятельность]. Компания [название] специализируется на [услуги]"
- Избегай лишних вводных слов и объяснений
- Будь конкретным и информативным

Ответ:"""

    result = ask_gemini(prompt)
    
    if result is None:
        return "Ошибка API"
    
    result = result.strip().strip('"').strip("'").strip()
    
    # Удаляем префиксы из начала
    prefixes_to_remove = [
        "Ответ:", "Описание:", "Деятельность:", 
        "На основе данных:", "Судя по информации:",
        "Этот человек", "Данный человек"
    ]
    for prefix in prefixes_to_remove:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
    
    # Убираем лишние фразы из всего текста
    phrases_to_remove = [
        " что указывает на ",
        " что указывает ",
        " что говорит ",
        " что свидетельствует ",
        ", что указывает на ",
        ", что означает "
    ]
    for phrase in phrases_to_remove:
        result = result.replace(phrase, " ")
        result = result.replace(phrase.capitalize(), " ")
    
    # Очищаем множественные пробелы
    result = re.sub(r'\s+', ' ', result).strip()
    
    if not result or len(result) < 20:
        return "Деятельность не определена"
    
    unclear_responses = [
        "недостаточно данных",
        "не определена",
        "информации недостаточно",
        "нет информации",
        "не указано"
    ]
    if any(phrase in result.lower() for phrase in unclear_responses):
        return "Деятельность не определена"
    
    if len(result) > 1500:
        result = result[:1500]
        last_period = result.rfind('.')
        if last_period > 500:
            result = result[:last_period + 1].strip()
    
    return result

# Обработка данных
print("\n" + "="*60)
print("НАЧАЛО ОБРАБОТКИ")
print("="*60)
total = len(df)
results = []
output_file = 'chat_users_error_20251210_023434_processed.xlsx'

# Тестовый запрос
print("\n=== ТЕСТОВЫЙ ЗАПРОС ===")
test_row = df.iloc[0] if len(df) > 0 else None
if test_row is not None:
    print(f"Имя: {test_row.get('Имя', 'N/A')}")
    print(f"Фамилия: {test_row.get('Фамилия', 'N/A')}")
    print(f"Описание: {str(test_row.get('Описание профиля', 'N/A'))[:100]}")
    print(f"\nОтправка запроса к Gemini API...")
    
    start_time = time.time()
    test_result = summarize_profile_with_nlp(test_row)
    elapsed = time.time() - start_time
    
    print(f"Результат ({elapsed:.1f}с): {test_result}")
    print("="*60 + "\n")
    
    if test_result in ["Деятельность не определена", "Ошибка API", "Деятельность не указана"]:
        print("ВНИМАНИЕ: Тестовый запрос не дал нормального результата!")
        print("Продолжаем обработку первых 5 записей...")

try:
    start_processing = datetime.now()
    
    # Обработка с 1 пользователя до исчерпания лимита API
    START_INDEX = 5054  # Начинаем с 1-го пользователя (индекс 0)
    
    # Инициализируем результаты для всех записей
    if 'Суммарное описание' not in df.columns:
        df['Суммарное описание'] = [None] * len(df)
    
    print(f"Обработка начинается с 1-го пользователя")
    print(f"Будет обрабатываться до исчерпания лимита API")
    print(f"Начало: {start_processing.strftime('%H:%M:%S')}\n")
    
    position = 0
    api_limit_reached = False
    
    # Обрабатываем всех пользователей начиная с первого
    for idx in range(START_INDEX, len(df)):
        if api_limit_reached:
            print(f"\nДостигнут лимит API. Остановка обработки.")
            break
            
        position += 1
        row = df.iloc[idx]
        
        if position % 5 == 0 or position == 1:
            percentage = position * 100 // total if total > 0 else 0
            elapsed_time = (datetime.now() - start_processing).total_seconds()
            avg_time = elapsed_time / position if position > 0 else 0
            remaining = (total - position) * avg_time if avg_time > 0 else 0
            
            print(f"[{position}/{total}] ({percentage}%) | "
                  f"Осталось: ~{int(remaining/60)}мин {int(remaining%60)}сек | "
                  f"Запросов: {request_counter['total_requests']}")
        
        result = summarize_profile_with_nlp(row)
        
        # Проверяем, не достигнут ли лимит API
        if result == "Ошибка API":
            # Проверяем, была ли это ошибка rate limit
            api_limit_reached = True
            print(f"\nДостигнут лимит API на пользователе {position}. Остановка.")
            break
        
        # Сохраняем результат в правильную позицию исходного датафрейма
        df.at[idx, 'Суммарное описание'] = result
        results.append(result)
        
        if position <= 3:
            print(f"  → {result[:150]}{'...' if len(result) > 150 else ''}")
        
        # Сохранение каждые 10 записей
        if position % 10 == 0:
            df.to_excel(output_file, index=False)
            print(f"  Сохранено")
        
    print("\nОбработка завершена!")
    
except KeyboardInterrupt:
    print("\nПрервано пользователем. Сохранение...")
    df.to_excel(output_file, index=False)
    print(f"Сохранено {len(results)} обработанных строк")
    raise
    
except Exception as e:
    print(f"\nОшибка: {str(e)}")
    print("Сохранение промежуточных результатов...")
    df.to_excel(output_file, index=False)
    print(f"Сохранено {len(results)} обработанных строк")
    raise

# Финальное сохранение
print(f"\nСохранение в: {output_file}")
df.to_excel(output_file, index=False)
print("Файл сохранен!")

# Статистика
print("\n" + "="*60)
print("СТАТИСТИКА")
print("="*60)
print(f"Всего записей: {len(df)}")
print(f"Всего запросов к API: {request_counter['total_requests']}")

successful = sum(1 for r in results if r and r not in [
    "Деятельность не определена", 
    "Деятельность не указана", 
    "Ошибка API"
])
not_defined = sum(1 for r in results if r == "Деятельность не определена")
not_specified = sum(1 for r in results if r == "Деятельность не указана")
errors = sum(1 for r in results if r == "Ошибка API")

print(f"\nУспешно: {successful} ({successful*100//len(df) if len(df) > 0 else 0}%)")
print(f"Не определено: {not_defined}")
print(f"Не указано: {not_specified}")
print(f"Ошибки: {errors}")

print("\n" + "="*60)
print("ПРИМЕРЫ РЕЗУЛЬТАТОВ")
print("="*60)
print(df[['Имя', 'Фамилия', 'Описание профиля', 'Суммарное описание']].head(10).to_string())