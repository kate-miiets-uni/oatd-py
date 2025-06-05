import cloudscraper
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


url = 'https://nv.ua/ukr/world/countries/mask-pokidaye-politiku-cherez-rozcharuvannya-ta-pislya-hvili-kritiki-50516685.html'
# Надсилаємо запит на отримання тексту за посиланням
scraper = cloudscraper.create_scraper()
response = scraper.get(url)

if response.status_code == 200:
    text = response.text

    # Векторизація тексту
    vectorizer = TfidfVectorizer(stop_words=["та", "і", "й", "у", "в", "на", "це", "що", "не", "а", "щоб", "про"])  # Виключаємо стоп-слова
    tfidf_matrix = vectorizer.fit_transform([text])

    # Створення DataFrame для зберігання матриці у вигляді таблиці
    df = pd.DataFrame(tfidf_matrix[0].T.todense(),
                      index=vectorizer.get_feature_names_out(), columns=["TF-IDF"])

    # Сортування слів за спаданням
    df = df.sort_values('TF-IDF', ascending=False)

    # Виведення 10-ти найвагоміших слів
    print(df[:10])

# Якщо запит невдалий
else:
    print("Запит невдалий.")