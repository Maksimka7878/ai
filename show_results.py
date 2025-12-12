import pandas as pd

# Загружаем обработанный файл
df = pd.read_excel('chat_users_error_20251210_023434_processed.xlsx')

print('='*80)
print('ПЕРВЫЕ 5 ОБРАБОТАННЫХ ПОЛЬЗОВАТЕЛЕЙ')
print('='*80)

for i in range(min(5, len(df))):
    print(f'\n--- Пользователь {i+1} ---')
    print(f'Имя: {df.iloc[i].get("Имя", "N/A")}')
    print(f'Фамилия: {df.iloc[i].get("Фамилия", "N/A")}')
    desc = str(df.iloc[i].get("Описание профиля", "N/A"))
    print(f'Описание профиля: {desc[:200]}')
    summary = df.iloc[i].get("Суммарное описание", "N/A")
    print(f'\nСуммарное описание (сгенерировано AI):')
    print(f'{summary}')
    print('-'*80)

