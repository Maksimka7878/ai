# Простой тест на 1 пользователе
import pandas as pd
import time
from datetime import datetime
import google.generativeai as genai

# Настройки
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_1")  # Загружаем из .env
MODEL = "gemini-2.5-flash"
genai.configure(api_key=GEMINI_API_KEY)

print(f"Используется Gemini API с моделью: {MODEL}")

# Загрузка файла
file_path = 'chat_users_error_20251210_023434.xlsx'
print(f"\nЗагрузка файла: {file_path}")
df = pd.read_excel(file_path)
print(f"Загружено строк: {len(df)}")

# Берем первого пользователя
test_row = df.iloc[0]
print(f"\nТестируем пользователя:")
print(f"Имя: {test_row.get('Имя', 'N/A')}")
print(f"Фамилия: {test_row.get('Фамилия', 'N/A')}")
print(f"Описание: {str(test_row.get('Описание профиля', 'N/A'))[:100]}")

# Подготовка данных
name = str(test_row.get('Имя', '') or '').strip()
surname = str(test_row.get('Фамилия', '') or '').strip()
description = str(test_row.get('Описание профиля', '') or '').strip()

info_parts = []
if name and name.lower() not in ['nan', 'none', 'null']:
    info_parts.append(f"Имя: {name}")
if surname and surname.lower() not in ['nan', 'none', 'null']:
    info_parts.append(f"Фамилия: {surname}")
if description and description.lower() not in ['nan', 'none', 'null']:
    info_parts.append(f"Описание: {description}")

info_text = "\n".join(info_parts)

# Промпт
prompt = f"""Проанализируй информацию и опиши деятельность человека:

{info_text}

Напиши 2-3 предложения о профессии, бизнесе или услугах. Будь конкретным.

Ответ:"""

print(f"\nОтправка запроса к API...")
start_time = time.time()

try:
    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 1000,
        }
    )
    
    if hasattr(response, 'text'):
        result = response.text.strip()
    elif hasattr(response, 'candidates') and response.candidates:
        result = response.candidates[0].content.parts[0].text.strip()
    else:
        result = "Ошибка: не удалось получить ответ"
    
    elapsed = time.time() - start_time
    
    # Очистка результата
    prefixes_to_remove = ["Ответ:", "Описание:", "Деятельность:"]
    for prefix in prefixes_to_remove:
        if result.lower().startswith(prefix.lower()):
            result = result[len(prefix):].strip()
    
    print(f"\nРезультат ({elapsed:.1f}с):")
    print(f"{result}")
    
    # Сохранение в файл
    df_result = df.copy()
    df_result['Суммарное описание'] = [result] + [None] * (len(df) - 1)
    output_file = 'chat_users_error_20251210_023434_processed.xlsx'
    df_result.to_excel(output_file, index=False)
    
    print(f"\n✓ Файл сохранен: {output_file}")
    print(f"✓ Обработано: 1 из {len(df)} записей")
    
except Exception as e:
    print(f"\n✗ Ошибка: {str(e)}")
    import traceback
    traceback.print_exc()





