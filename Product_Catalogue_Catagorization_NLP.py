# Import File in Python
import pandas as pd

df = pd.read_csv(r'....\amazon_co_ecommerce_sample.csv')

df.head(10) # displays first 10 rows
df.shape[0] # displays number of rows
df.shape[1] # displays number of columns

print(df.dtypes) # check the data type of each column

# understand the target distribution (row count)
df[['uniq_id', 'amazon_category_and_sub_category']].groupby('amazon_category_and_sub_category').count()

# Data Cleaning

# add a new column to represent the first-level categories
df['L1_category'] = df.apply (lambda row: row['amazon_category_and_sub_category'].split('>')[0].strip(), axis=1)

# Does not work due to some malformed values in 'amazon_category_and_sub_category' field

# Try this new approach :

def extract_l1_category(full_category: str) -> str:
    if not isinstance(full_category, str):
        return None
    
    if not full_category:
        return None
    
    return full_category.split('>')[0].strip()

df['L1_category'] = df.apply (lambda row: extract_l1_category(row['amazon_category_and_sub_category']), axis=1)

df.head(10)

# see the distribution of values in new L1 category 

df[['uniq_id', 'L1_category']].groupby('L1_category').count()

# filter possibly mal-formed rows
clean_df = df.dropna(subset=['L1_category'])
print(clean_df.shape[0])
# 9,310

# L1_category value counts in Descending order 
l1_category_cnt = clean_df['L1_category'].value_counts()

# There is sparse distribution for some categories in L1 category variable. Hence, they can be grouped in one subgroup
Here , we see what are the categories with sparse distribution

infreq_categories = l1_category_cnt.loc[l1_category_cnt < 10].index.values
print(f'infreq_categories: {infreq_categories}')

# replace infrequent L1 categories by `OTHERS`
clean_df['L1_category'].replace(to_replace=infreq_categories, value='OTHERS', inplace=True)

# check the distribution of these reassigned L1 categories
category_cnt_df = clean_df[['uniq_id', 'L1_category']].groupby('L1_category', as_index=False).count().sort_values(['uniq_id'], ascending=False).rename(index=str, columns={"uniq_id": "cnt"})
print(f'unique L1 categories now: {category_cnt_df.shape[0]}')
#unique L1 categories now: 17

################### Modeling with just 1 variable : product name ########################
# Using Natural Language processing to obtain features from product_name : Bag-of-words model with Tf-IDF weighting to represent product names.
# Feature extraction: Bag-of-words model with Tf-IDF weighting to represent product names.
# Model: Logistic Regression / Naive Bayes

# Data Preparation

# Sampling 

from sklearn.model_selection import train_test_split
y = clean_df['L1_category'].values
product_names = clean_df['product_name'].values

# split the data set to be 80% training and 20% validatoin
product_names_train, product_names_val, y_train, y_val = train_test_split(product_names, y, random_state=0, stratify=y)
print(product_names_train.shape, product_names_val.shape, y_train.shape, y_val.shape)
# (6982,) (2328,) (6982,) (2328,)

# Feature Extraction

from sklearn.feature_extraction.text import TfidfVectorizer

# perform feature extraction: bag-of-words with TF-IDF weighting on training data
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(product_names_train)
# for word, idx in vectorizer.vocabulary_.items():
#     print(word, idx)
print(f'X_train.shape: {X_train.shape}')
X_train.shape: (6982, 10000)

# To see the weights assigned to tokens (words)
print(X_train[0])

idx2word = {idx: word for (word, idx) in vectorizer.vocabulary_.items()}

# check the word corresponding column idx=3714
print(idx2word[3429])
# cut
print(product_names_train[0])
# Die Cut Découpage Sheet - Christmas Glitter Princesses

# Shows the vocabulary used to assign weights by the TF-IDF

print(vectorizer.vocabulary_)

# perform feature extraction: bag-of-words with TFIDF weighting on validation data
X_val = vectorizer.transform(product_names_val)
print(f'X_val.shape: {X_val.shape}')
# X_val.shape: (2328, 10000)

################ 1. Trying Logistic regression to see model resuts ######################

from sklearn.linear_model import LogisticRegression
import time

model = LogisticRegression(class_weight='balanced')

start = time.time()
print(f'Fitting model on {X_train.shape[0]} samples...')
# fit model
model.fit(X_train, y_train)
end = time.time()

print('Finished model training in %.3f seconds.' % (end-start))
# Finished model training in 67.474 seconds.

# perform your trained model on test data to get predictions
y_preds = model.predict(X_val)

# Comparing predicted product category with actual category
for i, pred in enumerate(y_preds):
    print(f'{i} pred: {pred}, true: {y_val[i]}')

# Evaluating the model performance
def print_evaluation(y_val, y_preds):
    from sklearn.metrics import classification_report, accuracy_score
    print(classification_report(y_true=y_val, y_pred=y_preds))
    acc = accuracy_score(y_true=y_val, y_pred=y_preds)
    print(f'accuracy: {round(acc, 3)}')

print_evaluation(y_val=y_val, y_preds=y_preds)

                           precision    recall  f1-score   support

            Arts & Crafts       0.84      0.86      0.85       201
      Baby & Toddler Toys       0.41      0.58      0.48        26
      Characters & Brands       0.86      0.65      0.74       240
  Die-Cast & Toy Vehicles       0.85      0.88      0.86       305
      Dolls & Accessories       0.93      0.86      0.89        93
              Fancy Dress       0.95      0.91      0.93       146
       Figures & Playsets       0.74      0.84      0.78       278
                    Games       0.86      0.83      0.85       235
                  Hobbies       0.80      0.82      0.81       366
        Jigsaws & Puzzles       0.79      0.86      0.82        69
  Musical Toy Instruments       0.75      0.60      0.67         5
    Novelty & Special Use       1.00      1.00      1.00         3
                   OTHERS       0.20      0.12      0.15        16
           Party Supplies       0.94      0.93      0.94       174
             Pretend Play       0.46      0.55      0.50        11
Puppets & Puppet Theatres       0.96      0.99      0.97        67
    Sports Toys & Outdoor       0.81      0.81      0.81        93

                micro avg       0.83      0.83      0.83      2328
                macro avg       0.77      0.77      0.77      2328
             weighted avg       0.83      0.83      0.83      2328

accuracy: 0.83

# To check how far apart were the incoorectly labelled product categories by the model, error analysis done as below :

def error_analysis(y_preds, y_val):
    for i, y_pred in enumerate(y_preds):
        y_true = y_val[i]
        if y_pred != y_true:
            print(f'{product_names_val[i]}\npred: {y_pred}\ntrue: {y_true}\n')

error_analysis(y_preds, y_val)

##################### 2.Trying Naive Bayes classification ########################

from sklearn.naive_bayes import MultinomialNB

# fit model
model = MultinomialNB()

start = time.time()
print(f'Fitting model on {X_train.shape[0]} samples...')
model.fit(X_train, y_train)

end = time.time()
print('Finished model training in %.3f seconds.' % (end-start))

# Finished model training in 32.965 seconds.

y_preds = model.predict(X_val)
print_evaluation(y_val=y_val, y_preds=y_preds)

                           precision    recall  f1-score   support

            Arts & Crafts       0.83      0.83      0.83       201
      Baby & Toddler Toys       0.00      0.00      0.00        26
      Characters & Brands       0.73      0.70      0.71       240
  Die-Cast & Toy Vehicles       0.81      0.90      0.85       305
      Dolls & Accessories       1.00      0.73      0.84        93
              Fancy Dress       0.97      0.88      0.92       146
       Figures & Playsets       0.73      0.85      0.78       278
                    Games       0.81      0.83      0.82       235
                  Hobbies       0.66      0.87      0.75       366
        Jigsaws & Puzzles       0.92      0.49      0.64        69
  Musical Toy Instruments       0.00      0.00      0.00         5
    Novelty & Special Use       0.00      0.00      0.00         3
                   OTHERS       0.00      0.00      0.00        16
           Party Supplies       0.93      0.94      0.93       174
             Pretend Play       0.00      0.00      0.00        11
Puppets & Puppet Theatres       0.97      0.90      0.93        67
    Sports Toys & Outdoor       1.00      0.42      0.59        93

                micro avg       0.79      0.79      0.79      2328
                macro avg       0.61      0.55      0.57      2328
             weighted avg       0.79      0.79      0.78      2328

accuracy: 0.794

################### Adding Product Description to add more information to model to predict ######################

Training/validaton examples: product_name and product_description as input to predict L1_category.
Feature extraction: the concatenation of the following two vectors:
- `product_names` vector: Bag-of-words model with Tf-IDF weighting to represent product names.
- `product_description` vector: Bag-of-words model with Tf-IDF weighting to represent product descriptions.
Model: Logistic Regression

# Feature extraction - multiple fields as input

products = clean_df[['product_name', 'product_description']]

# split the data set to be 80% training and 20% validatoin
products_train, products_val, y_train, y_val = train_test_split(products, y, random_state=0, stratify=y)
print(products_train.shape, products_val.shape, y_train.shape, y_val.shape)

# Vector representation - Product names
# perform feature extraction: bag-of-words with TF-IDF weighting on product names in the training data
names_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_names_train = names_vectorizer.fit_transform(products_train['product_name'].values)
print(f'X_names_train.shape: {X_names_train.shape}')

# X_names_train.shape: (6982, 10000)

# Vector representation - Product description

# perform feature extraction: bag-of-words with TF-IDF weighting on product titles in the training data
desc_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
products_train['product_description'].fillna('', inplace=True)
X_desc_train = desc_vectorizer.fit_transform(products_train['product_description'].values)
print(f'X_desc_train.shape: {X_desc_train.shape}')

# X_desc_train.shape: (6982, 10000)

# concatenating the Derived features from prouduct names and product description in a single data frame

from scipy.sparse import hstack
X_train = hstack([X_names_train, X_desc_train])
print(f'X_train.shape: {X_train.shape}')

# X_train.shape: (6982, 20000)

#################### 3. Trying Logistic regression again with this new set of derived features #####################

# fit model
model = LogisticRegression(class_weight='balanced')

start = time.time()
print(f'Fitting model on {X_train.shape[0]} samples...')
model.fit(X_train, y_train)

end = time.time()
print('Finished model training in %.3f seconds.' % (end-start))

# Finished model training in 16.002 seconds

# Validating model performance

# perform feature extraction: bag-of-words with TF-IDF weighting on product names in the validation data
X_names_val = names_vectorizer.transform(products_val['product_name'].values)
print(f'X_names_val.shape: {X_names_val.shape}')

# X_names_val.shape: (2328, 10000)

# perform feature extraction: bag-of-words with TF-IDF weighting on product descriptions in the validation data
products_val['product_description'].fillna('', inplace=True)
X_desc_val = desc_vectorizer.transform(products_val['product_description'].values)
print(f'X_desc_val.shape: {X_desc_val.shape}')

# X_desc_val.shape: (2328, 10000)

# Concatenating the derived features from product names and product descriptions 


X_val = hstack([X_names_val, X_desc_val])
print(f'X_val.shape: {X_val.shape}')
X_val.shape: (2328, 20000)

y_preds = model.predict(X_val)
print_evaluation(y_val=y_val, y_preds=y_preds)

                           precision    recall  f1-score   support

            Arts & Crafts       0.87      0.94      0.90       201
      Baby & Toddler Toys       0.65      0.65      0.65        26
      Characters & Brands       0.86      0.69      0.77       240
  Die-Cast & Toy Vehicles       0.89      0.90      0.90       305
      Dolls & Accessories       0.91      0.87      0.89        93
              Fancy Dress       0.96      0.92      0.94       146
       Figures & Playsets       0.75      0.86      0.80       278
                    Games       0.88      0.91      0.90       235
                  Hobbies       0.83      0.83      0.83       366
        Jigsaws & Puzzles       0.82      0.90      0.86        69
  Musical Toy Instruments       0.50      0.20      0.29         5
    Novelty & Special Use       1.00      1.00      1.00         3
                   OTHERS       0.50      0.06      0.11        16
           Party Supplies       0.94      0.96      0.95       174
             Pretend Play       0.78      0.64      0.70        11
Puppets & Puppet Theatres       0.99      0.99      0.99        67
    Sports Toys & Outdoor       0.87      0.87      0.87        93

                micro avg       0.86      0.86      0.86      2328
                macro avg       0.82      0.78      0.78      2328
             weighted avg       0.86      0.86      0.86      2328

accuracy: 0.862

# To see how far apart are the predicted product categories from the actual ones
error_analysis(y_preds, y_val)


############################ 4. Using Word Embeddings with Convolutional Neural Networks #############################

# Training/validaton examples: product_name only as input to predict L1_category.
# Feature extraction: Each product name is represented by a sequence of words, each of which is encoded 
# as a dense vector (word embedding vector), concatenated with bag-of-words representation.
# Model: A neural network
# Note: due to the small size of this data set, using word embeddings alone actually yields much inferrior performance 
# than the traditional bag-of-words model. Therefore, we would demonstrate a hybrid model by combining word embeddings 
# and bag-of-words representations in a unified neural network.

# Data Preparation

# Step 1: Convert each product name into a sequence of integer, each integer represents the index of the 
# corresponding word in the vocabulary

from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from string import punctuation, digits

stopwords = set(stopwords.words('english'))

exclude = set(punctuation).union(digits)

tokenizer = Tokenizer(num_words=10000, oov_token='OOV')
def preprocess(text):
    def clean_token(token):
        # remove punctuations and digits in a token
        return ''.join([c for c in token if c not in exclude])
    
    # remove stopwords and punctuations/digits in the title
    tokens = text.strip().lower().split()
    text = [clean_token(token) for token in tokens if clean_token(token) not in stopwords]
    return " ".join(text)

preprocessed_product_names_train = [preprocess(product_name) for product_name in product_names_train]
preprocessed_product_names_val = [preprocess(product_name) for product_name in product_names_val]
Using TensorFlow backend.

# To see what the processed words look like after removal of punctuation (caps converted to small) and special charatcters ( Ex: - , digits)

print(f'raw product name: "{product_names_train[0]}"')
print(f'processed product name: "{preprocessed_product_names_train[0]}"')

tokenizer.fit_on_texts(preprocessed_product_names_train + preprocessed_product_names_val)
docs_train = tokenizer.texts_to_sequences(preprocessed_product_names_train)
docs_val = tokenizer.texts_to_sequences(preprocessed_product_names_val)
vocab = tokenizer.word_counts

print(docs_train[0])
# [74, 324, 2713, 492, 94, 92, 3512]

from keras.preprocessing.sequence import pad_sequences
# pad documents to a max length of 30 words
max_length = 30
padded_docs_train = pad_sequences(docs_train, maxlen=max_length, padding='post')
padded_docs_val = pad_sequences(docs_val, maxlen=max_length, padding='post')

print(padded_docs_train[0])
print(padded_docs_val[0])



# Step 2: Convert categorical multi-class labels into one-hot encoded vectors

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
labels_train = encoder.fit_transform(y_train)
for i, label in enumerate(labels_train[:10]):
    print(f'{y_train[i]}\t{label}')

Characters & Brands	[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Hobbies	[0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
Die-Cast & Toy Vehicles	[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
Jigsaws & Puzzles	[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
Figures & Playsets	[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
Jigsaws & Puzzles	[0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
Characters & Brands	[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Figures & Playsets	[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
Characters & Brands	[0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
Die-Cast & Toy Vehicles	[0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]

labels_val = encoder.transform(y_val)
for i, label in enumerate(labels_val[:10]):
    print(f'{y_val[i]}\t{label}')

Step 3: Load pre-trained word embeddings (GloVe Embeddings)

from numpy import asarray
from tqdm import tqdm

import zipfile
zip_ref = zipfile.ZipFile('F:\Installation\Python\GloVe-1.2\glove.6B.zip', 'r')
zip_ref.extractall()

# load the whole embedding into memory
embeddings_index = {}
embedding_dim = 50
for line in tqdm(open(f'glove.6B.{embedding_dim}d.txt', encoding="utf8")):
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print(f'Loaded {len(embeddings_index)} word vectors.')

print(embeddings_index['cat'])
[ 0.45281  -0.50108  -0.53714  -0.015697  0.22191   0.54602  -0.67301
 -0.6891    0.63493  -0.19726   0.33685   0.7735    0.90094   0.38488
  0.38367   0.2657   -0.08057   0.61089  -1.2894   -0.22313  -0.61578
  0.21697   0.35614   0.44499   0.60885  -1.1633   -1.1579    0.36118
  0.10466  -0.78325   1.4352    0.18629  -0.26112   0.83275  -0.23123
  0.32481   0.14485  -0.44552   0.33497  -0.95946  -0.097479  0.48138
 -0.43352   0.69455   0.91043  -0.28173   0.41637  -1.2609    0.71278
  0.23782 ]


# create a weight matrix for words in training docs
from numpy import zeros

vocab_size = len(vocab)
print(f'vocab_size: {vocab_size}')
embedding_matrix = zeros((vocab_size, embedding_dim))
not_found = 0
for word, i in vocab.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        not_found += 1
#         print(word)

vocab_size
10979        
print(f'not_found: {not_found}')
# not found: 2183

print(embedding_matrix.shape)
(10979,50)

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Embedding, Dropout, Conv1D, MaxPool1D, Flatten, Concatenate

embedding_input = Input(shape=(max_length, ), dtype='int32', name='embedding_input')
x = Embedding(output_dim=embedding_dim, input_dim=len(vocab), weights=[embedding_matrix],
              input_length=max_length, trainable=True)(embedding_input)
x = Conv1D(64, 3, padding='valid', activation='relu', strides=1)(x)
x = MaxPool1D(pool_size=3)(x)
x = Conv1D(32, 3, padding='valid', activation='relu',strides=1)(x)
x = MaxPool1D(pool_size=2)(x)
x = Flatten()(x)
bow_input = Input(shape=(X_train.shape[1], ), name='bow')
x = Concatenate()([x, bow_input])

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(len(encoder.classes_), activation='softmax', name='output')(x)

model = Model(inputs=[embedding_input, bow_input], outputs=[output])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit([padded_docs_train, X_train.toarray()], [labels_train], epochs=10)

# Model Evaluation 

# Step 1: Generate the raw predictions (as in arrays)

import numpy as np

raw_preds = model.predict([padded_docs_val, X_val.toarray()])
pred_classes = np.argmax(raw_preds, axis=1)
y_preds = []
for i, class_idx in enumerate(pred_classes):
    pred_label = encoder.classes_[class_idx]
    y_preds.append(pred_label)

print(y_preds[0])
# Dolls & Accessories

print(raw_preds[0])

print_evaluation(y_preds, y_val)

#Epochs			10	
#Precision(Wgtd)	77
#FI Score (Wgtd)	75

##############################################################################


Epochs = 10

                           precision    recall  f1-score   support

            Arts & Crafts       0.75      0.82      0.78       184
      Baby & Toddler Toys       0.27      0.64      0.38        11
      Characters & Brands       0.82      0.43      0.56       459
  Die-Cast & Toy Vehicles       0.75      0.86      0.80       265
      Dolls & Accessories       0.81      0.93      0.86        81
              Fancy Dress       0.81      0.96      0.88       123
       Figures & Playsets       0.58      0.74      0.65       217
                    Games       0.76      0.91      0.83       197
                  Hobbies       0.79      0.74      0.77       388
        Jigsaws & Puzzles       0.77      0.87      0.82        61
  Musical Toy Instruments       0.00      0.00      0.00         0
    Novelty & Special Use       0.00      0.00      0.00         0
                   OTHERS       0.12      0.17      0.14        12
           Party Supplies       0.93      0.90      0.91       178
             Pretend Play       0.45      0.56      0.50         9
Puppets & Puppet Theatres       0.97      0.94      0.96        69
    Sports Toys & Outdoor       0.68      0.85      0.75        74

                micro avg       0.75      0.75      0.75      2328
                macro avg       0.60      0.67      0.62      2328
             weighted avg       0.77      0.75      0.75      2328

accuracy: 0.753
