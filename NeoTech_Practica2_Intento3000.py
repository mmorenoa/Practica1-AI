''' Hace falta instalar con pip el `matplotlib` y descargar manualmente el Glove (no iba de la otra manera)'''

import os.path
import requests
from shutil import unpack_archive
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize': (20, 6),'figure.dpi': 64})

# Modificar al vuestro 
glove_file = 'D:/.UNIVERSIDAD/AI/glove/glove.6B.50d.txt'
glove_dir = 'D:/.UNIVERSIDAD/AI/glove/'

with open(glove_file, 'r', encoding='utf-8') as f:
    words = []
    embeddings = []
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        embedding = np.array([float(val) for val in parts[1:]])
        words.append(word)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)
    word_to_index = {word: index for index, word in enumerate(words)}

print('\nOK (abrir el fichero Glove)\n')

# Cuántas dimensiones tienen nuestros word vectors (50, 100, 200 o 300)
EMBEDDING_DIM = 50
# El tamaño máximo de nuestro vocabulario (se escogerán las más frecuentes)
MAX_VOCAB_SIZE = 10000
# El tamaño de la frase más larga con la que alimentar el modelo
MAX_SEQUENCE_LENGTH = 150

print(f'Loading GloVe {EMBEDDING_DIM}-d embedding... ', end='')
print("\n")
word2vec = {}
with open(os.path.join(glove_dir, f'glove.6B.{EMBEDDING_DIM}d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print(f'done ({len(word2vec)} word vectors loaded)')
print("\n")







print('Loading absract training dataset... ', end='')
df = pd.read_csv('D:/.UNIVERSIDAD/AI/data/papers.csv') # Modificar al vuestro
abstracts = df['abstract'].values


# HACER LO DE LAS KEYWORDS
keywords = ["Hola que tal", "me llamo", "Sofia"] #keywords = df['keywords'].values
combined_keywords = " ".join(keywords)
print("keywords:", keywords)
# Eliminar palabras repetidas
unique_keywords = list(set(keywords))
# Crear diccionario
diccionario = {word: i for i, word in enumerate(unique_keywords)}
# Crear matriz de ceros
print("LENGTH",len(keywords))
print(unique_keywords)
matrizKeywords = np.zeros((len(keywords), len(unique_keywords)))
# Rellenar matriz 
for i, word in enumerate(unique_keywords):
    for token in word.split():
        matrizKeywords[i][diccionario[token]] = 1
print(matrizKeywords.astype(int).tolist())



print(f'done ({len(abstracts)} articulos loaded)')
print(f'Biggest abstracgt length:  {max(len(s) for s in abstracts)}')
print(f'Smallest comment length: {min(len(s) for s in abstracts)}')
print(f'Avg. comment length:     {np.mean([len(s) for s in abstracts])}')
print(f'Median comment length:   {sorted(len(s) for s in abstracts)[len(abstracts) // 2]}')
print('-' * 20)
print(f'Example comment: {abstracts[9]}')
print(f'Example targets: {keywords[5]}')
print("\n")


tokenizer =   tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(abstracts)
sequences =   tokenizer.texts_to_sequences(abstracts)
word_index =  tokenizer.word_index
print(f'Biggest index: {max(max(seq) for seq in sequences if len(seq) > 0)}')
print(f'Unique tokens: {len(word_index)}')
print('-' * 20)
print(f'Example comment: {abstracts[9]}: {sequences[9]}')
print("\n")

tokenizerTargets =   tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizerTargets.fit_on_texts(targets)
sequencesTargets =   tokenizerTargets.texts_to_sequences(targets)
word_index =  tokenizerTargets.word_index
print(f'Biggest index: {max(max(seq) for seq in sequencesTargets if len(seq) > 0)}')
print(f'Unique tokens: {len(word_index)}')
print('-' * 20)
print(f'Example comment: {targets[9]}: {sequencesTargets[9]}')
print("\n")

data = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    value=0,
    maxlen=MAX_SEQUENCE_LENGTH
)
print(f'Data tensor shape: {data.shape}')
data[0]

targets2 = tf.keras.preprocessing.sequence.pad_sequences(
    sequencesTargets,
    value=0,
    maxlen=MAX_SEQUENCE_LENGTH
)

print('Loading embedding with GloVe vectors... ', end='')
# Cargamos sólo las palabras elegidas de nuestro conjunto de datos
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Creamos la capa de embedding
embedding_layer = tf.keras.layers.Embedding(
  input_dim=num_words,
  output_dim=EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=MAX_SEQUENCE_LENGTH,
  trainable=False,
)
print('done')
print("\n")


#----------------------------------------------------------------------------------------------------------------


input_ = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,)) 
x = embedding_layer(input_) 
x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(x) 
x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x) 
x = tf.keras.layers.MaxPooling1D(pool_size=(2), strides=(2))(x) 
x = tf.keras.layers.Flatten()(x) 
x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')(x) 
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')(x) 
x = tf.keras.layers.Dropout(0.3)(x) 
output = tf.keras.layers.Dense(len(keywords), activation='sigmoid')(x)  
 
model = tf.keras.Model(input_, output) 
model.compile( 
  loss='binary_crossentropy', 
  optimizer='adam', 
  metrics=['binary_accuracy'], 
) 
model.summary() 

print("-----Data------")
print(data[0])
X = np.asarray(data).astype(np.float32)
print("-----X------")
print(X)
print("-----Keyword------")
print(keywords[0])
print(keywords[1])
print(keywords[2])
print("-----targets-------")
print(targets[0])
print(targets[1])
print(targets[2])
Y = np.asarray(targets2).astype(np.float32)
print("-----Y-------")
print(Y)

history = model.fit(X, Y, epochs=30, validation_split=0.1, batch_size=4096)


# Probamos
plt.subplot(1, 2, 1) 
plt.plot(history.history['loss'], label='Training') 
plt.plot(history.history['val_loss'], label='Validation') 
plt.xlabel('Epoch') 
plt.ylabel('Loss') 
plt.title(f'Training: {history.history["loss"][-1]:.2f}, validation: {history.history["val_loss"][-1]:.2f}') 
plt.legend() 
 
plt.subplot(1, 2, 2) 
plt.plot(history.history['binary_accuracy'], label='Training') 
plt.plot(history.history['val_binary_accuracy'], label='Validation') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.title(f'Training: {history.history["binary_accuracy"][-1]:.2f}, validation: {history.history["val_binary_accuracy"][-1]:.2f}') 
plt.legend() 
 
plt.tight_layout() 
plt.show()



