#!/usr/bin/env python
# coding: utf-8

# In[1]:


corpus = [
    "I like to learn deep learning",
    "Deep learning is interesting",
    "I enjoy studying deep learning"
]


# In[2]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams

# Tokenize the corpus
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}

# Generate training data
vocab_size = len(word2idx) + 1
target_words, context_words = [], []
for sentence in corpus:
    tokenized = tokenizer.texts_to_sequences([sentence])[0]
    pairs, _ = skipgrams(tokenized, vocabulary_size=vocab_size, window_size=1, negative_samples=0)
    target, context = zip(*pairs)
    target_words.extend(target)
    context_words.extend(context)


# In[3]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Reshape

embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1))
model.add(Reshape((embedding_dim,)))
model.add(Dense(units=vocab_size, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

# Train the model
model.fit(x=target_words, y=context_words, epochs=50)


# In[4]:


word_to_lookup = "deep"
word_idx = word2idx[word_to_lookup]
word_embedding = model.layers[0].get_weights()[0][word_idx]

print(f"Embedding for '{word_to_lookup}': {word_embedding}")


# In[ ]:




