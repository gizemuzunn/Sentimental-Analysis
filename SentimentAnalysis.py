import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, CuDNNGRU
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('comments.csv')
print(dataset)

target =  dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()

cutoff = int(len(data) * 0.80)
x_train,x_text = data[:cutoff],data[cutoff:]
y_train,y_test = target[:cutoff],target[cutoff:]

num_words = 10000

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)

x_train_tokens = tokenizer.texts_to_sequences(x_train)

x_test_tokens = tokenizer.texts_to_sequences(x_text)

num_tokens = [len(tokens)for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2* np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)

print(np.sum(num_tokens < max_tokens) / len(num_tokens))

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens)

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(),idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token !=0]
    text = ' '.join(words)
    return text

model = Sequential()
embedding_size = 50

model.add(Embedding(input_dim=num_words,
                    output_dim = embedding_size,
                    input_length = max_tokens,
                    name='embedding_layer'))
model.add(GRU(units=16,return_sequences=True))
model.add(GRU(units=8,return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1,activation='sigmoid'))

optimizer = Adam(lr = 1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

model.fit(x_train_pad,y_train,epochs=1,batch_size=256)

result = model.evaluate(x_test_pad,y_test)
print(result[1])

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred ])
cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

len(incorrect)

idx = incorrect[0]
print(idx)

text = x_text[idx]
print(text)
print(y_pred[idx])
print(cls_true[idx])

text1 = "Ürün Mükemmel"
text2 = "Rezalet"
text3 = "Güzel Ama Kargo Yavaş"

texts= [text1,text2,text3]

tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens,maxlen=max_tokens)
print(tokens_pad.shape)

print(model.predict(tokens_pad))