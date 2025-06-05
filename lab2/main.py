import requests
import re

url = 'https://www.bbc.com/ukrainian/articles/c249qgylz5eo'

# Надсилаємо запит на отримання тексту
response = requests.get(url)

# Якщо запит вдалий
if response.status_code == 200:
    text = response.text

    # Очищуємо текст від пунктуації
    clean_text = re.sub(r'\W+', '', text)

    # Знаходимо телефонні номери в тексті
    phone_numbers = re.findall( r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', clean_text)

    # Виводимо очищений текст та кількість знайдених номерів
    print(f"Очищений текст: {clean_text[:100]}")
    print(f"Знайдено телефонних номерів :{len(phone_numbers)}")

# Якщо запит невдалий
else:
    print("Не вдалося завантажити сторінку.")