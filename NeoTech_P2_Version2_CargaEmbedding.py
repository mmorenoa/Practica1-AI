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
glove_file = 'Desktop/AI/glove/glove.6B.50d.txt'
glove_dir = 'Desktop/AI/glove/'

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
MAX_SEQUENCE_LENGTH = 50

print(f'Loading GloVe {EMBEDDING_DIM}-d embedding... ', end='')
print("\n")
word2vec = {}
with open(os.path.join(glove_dir, f'glove.6B.{EMBEDDING_DIM}d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print(f'done ({len(word2vec)} word vectors loaded)')
print("\n")


#Esto ha sido modificado no se si está bien lode las keywords
print('Loading absract training dataset... ', end='')
df = pd.read_csv('Desktop/AI/papers.csv') # Modificar al vuestro
abstracts = df['abstract'].values
keywords = df['keywords'].apply(lambda x: x.split(','))  
keywords = keywords.apply(lambda x: [' '.join(x)]) # une las palabras con espacios
# df['top_keywords'] = df[['keyword1'. 'keyword2', 'keyword3', 'keyword4', 'keyword5']].apply(extract_keywords, axis=1)      
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

data = tf.keras.preprocessing.sequence.pad_sequences(
    sequences,
    value=0,
    maxlen=MAX_SEQUENCE_LENGTH
)
print(f'Data tensor shape: {data.shape}')
data[0]

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