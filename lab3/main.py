import requests
import nltk


url = 'https://www.unian.ua/world/karti-groshi-p-yatero-kapibar-13017267.html'

# Надсилаємо запит на отримання тексту
response = requests.get(url)

if response.status_code == 200:
    text = response.text

    # Токенізація тексту
    nltk.download('punkt')
    tokens = nltk.word_tokenize(text.lower()) # Приводимо до нижнього регістру для єдності

    # Формування біграм
    bigram_list = list(nltk.bigrams(tokens))

    # Визначення частоти появи біграм
    freq_dist = nltk.FreqDist(bigram_list)

    # Вивід 10 найпоширеніших біграм
    print("10 найпоширеніших біграм: ")
    for bigram, count in freq_dist.most_common(10):
        print(f"{bigram}: {count}")

# Якщо запит невдалий
else:
    print("Не вдалося завантажити сторінку.")