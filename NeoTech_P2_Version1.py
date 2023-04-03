''' Hace falta instalar con pip el `matplotlib`'''

import os.path
import requests
from shutil import unpack_archive
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams.update({'figure.figsize': (20, 6),'figure.dpi': 64})


GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'
GLOVE_FILE = 'tmp/glove.6B.zip'
GLOVE_DIR = 'tmp/'

if not os.path.isdir(GLOVE_DIR):
    os.makedirs(GLOVE_DIR)

# Descargamos el dataset comprimido de GloVe (si no lo tenemos ya)
if not os.path.exists(GLOVE_FILE):
    print('Downloading ...', end='')
    with open(GLOVE_FILE, 'wb') as f:
        r = requests.get(GLOVE_URL, allow_redirects=True)
        f.write(r.content)
    print('OK')

# Lo descomprimimos en el directorio 'glove'
print('Unpacking ...', end='')
unpack_archive(GLOVE_FILE, GLOVE_DIR)
print('OK')

# Cuántas dimensiones tienen nuestros word vectors (50, 100, 200 o 300)
EMBEDDING_DIM = 50
# El tamaño máximo de nuestro vocabulario (se escogerán las más frecuentes)
MAX_VOCAB_SIZE = 10000
# El tamaño de la frase más larga con la que alimentar el modelo
MAX_SEQUENCE_LENGTH = 50

print(f'Loading GloVe {EMBEDDING_DIM}-d embedding... ', end='')
word2vec = {}
with open(os.path.join(GLOVE_DIR, f'glove.6B.{EMBEDDING_DIM}d.txt')) as f:
    for line in f:
        values = line.split()
        word2vec[values[0]] = np.asarray(values[1:], dtype='float32')
print(f'done ({len(word2vec)} word vectors loaded)')


#Esto ha sido modificado no se si está bien
print('Loading absract training dataset... ', end='')
df = pd.read_csv('papers.csv')
abstracts = df['abstract'].values
keywords = df[['keyword1', 'keyword2', 'keyword3', 'keyword4', 'keyword5']].values
print(f'done ({len(abstracts)} articulos loaded)')
print(f'Biggest abstracgt length:  {max(len(s) for s in abstracts)}')
print(f'Smallest comment length: {min(len(s) for s in abstracts)}')
print(f'Avg. comment length:     {np.mean([len(s) for s in abstracts])}')
print(f'Median comment length:   {sorted(len(s) for s in abstracts)[len(abstracts) // 2]}')
print('-' * 20)
print(f'Example comment: {abstracts[9]}')
print(f'Example targets: {keywords[9]}')


tokenizer =   tf.keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(abstracts)
sequences =   tokenizer.texts_to_sequences(abstracts)
word_index =  tokenizer.word_index
print(f'Biggest index: {max(max(seq) for seq in sequences if len(seq) > 0)}')
print(f'Unique tokens: {len(word_index)}')
print('-' * 20)
print(f'Example comment: {abstracts[9]}: {sequences[9]}')

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


#----------------------------------------------------------------------------------------------------------------