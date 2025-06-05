import requests
import re
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import precision_score, recall_score




# Функція приводить текст до нижнього регістру, очищає його від пунктуації та стоп-слів
def clean_text(text):
    # Сет стоп-слів
    stop_words = {"щоб", "про", "але", "для", "від", "через", "також", "якщо", "вже", "він", "вона", "вони", "однак",
                  "його", "може", "який", "яка", "які", }
    # Приводимо до нижнього регістру для уніфікації слів
    text = text.lower()
    # Видаляємо усі символи окрім літер, цифр та _
    text = re.sub(r'\W+', ' ', text)
    # Розділяємо текст на слова для видалення стоп-слів
    words = text.split()
    # Видаляємо стоп-слова та короткі слова (менше 3-х символів)
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)


# Список новин (5 спортивних, 5 культурних)
urls = [
    'https://www.bbc.com/ukrainian/articles/c1k4x2xp124o',
    'https://www.bbc.com/ukrainian/articles/ckgyqkrn5n6o',
    'https://www.bbc.com/ukrainian/articles/crlrkpk0ywro',
    'https://www.bbc.com/ukrainian/articles/c3w66qz4j8yo',
    'https://www.bbc.com/ukrainian/articles/cl7yyw8xwg9o',
    'https://www.bbc.com/ukrainian/articles/cqj77vv07x2o',
    'https://www.bbc.com/ukrainian/articles/c4grdnxg0l6o',
    'https://www.bbc.com/ukrainian/articles/cwy630j7ed1o',
    'https://www.bbc.com/ukrainian/articles/cp8kyze7zm3o',
    'https://www.bbc.com/ukrainian/articles/cy9derrnlzzo',
]

# Список для зберігання документів з текстами новин
documents = []
# Список для зберігання міток класів (0 для спорту, 1 для культури)
labels = []

# Для кожного посилання зі списку посилань
for i, url in enumerate(urls):
    # Визначаємо мітку класу: перші 5 - спорт (0), наступні 5 - культура (1)
    label = 0 if i < 5 else 1
    labels.append(label)

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

# Токенізація текстів
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(cleaned_documents)
sequences = tokenizer.texts_to_sequences(cleaned_documents)

# Padding (вирівнювання довжини послідовностей)
max_len = max([len(x) for x in sequences]) if sequences else 0
if max_len < 100:
    max_len = 100

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Перетворюємо мітки на масив NumPy
labels = np.array(labels)

print(f"\nКількість унікальних токенів у словнику: {len(tokenizer.word_index)}")
print(f"Максимальна довжина послідовності: {max_len}")
print(f"Форма підготовлених даних (padded_sequences): {padded_sequences.shape}")
print(f"Форма міток (labels): {labels.shape}")

# Розділення даних на тренувальний та тестовий набори
# 80% для навчання, 20% для тестування
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels)

print(f"\nРозмір тренувального набору: {len(X_train)} документів")
print(f"Розмір тестового набору: {len(X_test)} документів")

# Побудова та навчання LSTM-моделі
vocab_size = len(tokenizer.word_index) + 1 # Розмір словника (+1 для <unk> токена)
embedding_dim = 100 # Розмірність векторів вбудовування слів
lstm_units = 128 # Кількість одиниць у шарі LSTM

model = Sequential([
    # Шари моделі
    # Embedding Layer: Перетворює індекси слів у щільні вектори
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    # LSTM Layer: Обробляє послідовності, захоплюючи залежності на великих відстанях
    LSTM(units=lstm_units),
    # Dense Output Layer: Бінарна класифікація з сигмоїдною активацією
    Dense(units=1, activation='sigmoid')
])

# Компіляція моделі
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Процес навчання моделі ---")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

#  Оцінка моделі на тестовому наборі
y_pred_proba = model.predict(X_test) # Отримуємо ймовірності належності до позитивного класу
y_pred = (y_pred_proba > 0.5).astype(int) # Перетворюємо ймовірності на бінарні класи (0 або 1)

# Обчислюємо precision (точність) та recall (повноту)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n--- Результати класифікації ---")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")