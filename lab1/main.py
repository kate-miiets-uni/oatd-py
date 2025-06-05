import requests
import re

url = 'https://www.bbc.com/ukrainian/articles/cqj77vv07x2o'

# Надсилаємо запит на отримання тексту
response = requests.get(url)

# Якщо запит вдалий
if response.status_code == 200:
    text = response.text

    # Очищення тексту від зайвих пробілів та пунктуації
    clean_text = re.sub(r'[^\w\s.]', '', text)

    # Розбиття тексту за крапками на окремі речення
    sentences = clean_text.split('.')

    # Видаляємо пусті речення
    sentences = [s.strip() for s in sentences if s.strip()]

    # Визначаємо загальну кількість речень
    sentence_count = len(sentences)

    print(f"Загальна кількість речень у тексті: {sentence_count}")
# Якщо запит невдалий
else:
    print("Не вдалося завантажити сторінку.")