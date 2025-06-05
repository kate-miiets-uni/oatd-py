import requests
import re
from bs4 import BeautifulSoup
from gensim.corpora import Dictionary
from gensim.models import LdaModel


# Функція приводить текст до нижнього регістру, очищає його від пунктуації та стоп-слів
def clean_text(text):
    # Сет стоп-слів
    stop_words = {"щоб", "про", "але", "для", "від", "через", "також", "якщо", "вже", "він", "вона", "вони", "однак", "його", "може", "який", "яка", "які", }
    # Приводимо до нижнього регістру для уніфікації слів
    text = text.lower()
    # Видаляємо усі символи окрім літер, цифр та _
    text = re.sub(r'\W+', ' ', text)
    # Розділяємо текст на слова для видалення стоп-слів
    words = text.split()
    # Видаляємо стоп-слова та короткі слова (менше 3-х символів)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return words  # Повертаємо список слів, а не об'єднаний рядок


urls = [
    'https://www.bbc.com/ukrainian/articles/cn7zr18rx46o',
    'https://www.bbc.com/ukrainian/articles/cjwvx7783xvo',
    'https://www.bbc.com/ukrainian/articles/czd30mqlv2lo',
    'https://www.bbc.com/ukrainian/articles/c0r55z299ggo',
    'https://www.bbc.com/ukrainian/articles/cvgqld6mld9o'
]

# Список для зберігання документів з текстами новин
documents = []

# Для кожного посилання зі списку посилань
for url in urls:
    # Надсилаємо запит на сервер про отримання тексту новини
    response = requests.get(url)
    # Якщо запит вдалий
    if response.status_code == 200:
        # Видаляємо з текста новини усі теги
        soup = BeautifulSoup(response.text, 'html.parser')
        # Видобуваємо текст з тегів <p>
        text = ' '.join([p.text for p in soup.find_all('p')])  # Отримуємо текст із тегів <p>
        # Додаємо текст у список з документами
        documents.append(text)

print(documents[0][:100])

cleaned_documents = [clean_text(doc) for doc in documents]
print(cleaned_documents[0][:100])

# Створення словника (Dictionary) з очищених документів
dictionary = Dictionary(cleaned_documents)
print(f"\nКількість унікальних токенів у словнику: {len(dictionary)}")

# Створення корпусу документів у форматі BoW (Bag of Words)
corpus = [dictionary.doc2bow(doc) for doc in cleaned_documents]
print("\nПриклад елемента корпусу (BoW для першого документа):")
print(corpus[0][:10]) # Виводимо перші 10 пар (id слова, частота) для першого документа

# Побудова LDA-моделі
num_topics = 4
lda_model = LdaModel(corpus=corpus,
                     num_topics=num_topics,
                     id2word=dictionary,
                     random_state=100, # Для відтворюваності результатів
                     passes=10,
                     alpha='auto') # Автоматичне визначення параметрів alpha

# Виведення отриманих тем та їх ключових слів
print(f"\nLDA-модель з {num_topics} темами:")
for idx, topic in lda_model.print_topics(-1):
    print(f"Тема {idx}: {topic}")




