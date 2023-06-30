# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 09:41:13 2023

@author: Jens Bender
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text  # prerequisite for using the BERT preprocessing layer of TensorFlow Hub

# Download data
# Create a "data" folder and download the "Ethos_Dataset_Binary.csv" file from: 
# https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/tree/master/ethos/ethos_data

# Load data
df = pd.read_csv("./data/Ethos_Dataset_Binary.csv", sep=";")

# Convert pandas dataframe to numpy array
data = df.to_numpy()

# Extract comments from column 0
comments = data[:, 0]

# Extract labels from column 1 
labels = data[:, 1]


#==============================================================================    
#========================== Exploratoy Data Analysis ==========================
#============================================================================== 

# Histogram of labels
plt.hist(labels, color="steelblue", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Hate Speech Labels")
plt.show()

# Dichotomize labels (hate speech: 0 = no, 1 = yes) 
# Decision criterion: Hate speech if minimum 50% of reviewers rated the comment as hate speech 
labels[labels >= 0.5] = 1
labels[labels < 0.5] = 0
labels = labels.astype(int)

# Histogram of binary labels
plt.hist(labels, color="steelblue", edgecolor="black")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Hate Speech Binary Labels")
plt.show()

# Label frequencies
num_hate_comments = len(labels[labels == 1])
num_no_hate_comments = len(labels[labels == 0])
percent_hate_comments = np.round(num_hate_comments/len(labels)*100,2)
percent_no_hate_comments = np.round(num_no_hate_comments/len(labels)*100,2)
print("Number of hate speech comments")
print(f"Hate speech: {num_hate_comments} comments ({percent_hate_comments}%)")
print(f"No Hate Speech: {num_no_hate_comments} comments ({percent_no_hate_comments}%)")
print("="*20)

# Comment length
comment_length = [len(comment) for comment in comments]
# Histogram of comment length
plt.hist(comment_length, bins=200)
plt.xlabel("Number of words per comment")
plt.ylabel("Counts")
plt.show()
# Histogram of comment length from 0 to 700 words
plt.hist(comment_length, bins=200)
plt.xlabel("Number of words per comment")
plt.ylabel("Counts")
plt.xlim(0, 700)
plt.show()
# Mean, Std, Min, Max
print("Comment Length")
print(f"Mean: {np.round(np.array(comment_length).mean())} words")
print(f"Std: {np.round(np.array(comment_length).std())} words")
print(f"Min: {np.round(np.array(comment_length).min())} words")
print(f"Max: {np.round(np.array(comment_length).max())} words")
print("="*20)

# Display first 10 comments
print("Here are 10 example comments.")
for i in range(10):
    print(f"Comment {i+1}: {comments[i]}")
print("="*20) 


#==============================================================================    
#=============================== Model Building ===============================
#============================================================================== 

# Split dataset into training and test data 
comments_train, comments_test, labels_train, labels_test = train_test_split(
    comments, labels, test_size=0.3, random_state=42)

# Initialize instance for early stopping (used in all models)
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=20,
    restore_best_weights=True)

# Specify dropout rate (used in all models) 
dropout_rate = 0.5


#==============================================================================    
#============================= Model 1: SimpleRNN =============================
#==============================================================================    

# Initialize tokenizer 
num_words = 5000  # maximum number of unique words in the dictionary of the tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=num_words,
    filters='"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=" ", char_level=False, oov_token=None)

# Fit tokenizer to training data  
tokenizer.fit_on_texts(comments_train)

# Save dictionary
word_index = tokenizer.word_index

# Display first 100 words of the dictionary
count = 0
for key, value in word_index.items():
    print(f"{key}: {value}")
    count += 1
    if count == 100:
        break
    
# Apply tokenizer to training and test data 
sequences_train = tokenizer.texts_to_sequences(comments_train)
sequences_test = tokenizer.texts_to_sequences(comments_test)

# Display first 5 sequences
print(sequences_train[:5])  

# Apply sequence padding to obtain consistent sequence length 
max_length = 15
padded_sequences_train = keras.preprocessing.sequence.pad_sequences(
    sequences_train, 
    maxlen=max_length, 
    truncating="post", 
    padding="post"
    )
padded_sequences_test = keras.preprocessing.sequence.pad_sequences(
    sequences_test, 
    maxlen=max_length, 
    truncating="post", 
    padding="post"
    )

# Display first 5 padded sequences
print(padded_sequences_train[:5])

# Specify the number of dimensions of the word vectors in the embedding layer
word_vector_dim = 50

# Specify model
model1 = keras.models.Sequential()
model1.add(keras.layers.Embedding(num_words+1,  # number of words in tokenizer +1 for the "0" used for padding 
                                  word_vector_dim,
                                  input_length=max_length, 
                                  mask_zero=True))
model1.add(keras.layers.SimpleRNN(128, 
                                  return_sequences=True,
                                  dropout=dropout_rate,
                                  recurrent_dropout=dropout_rate))
model1.add(keras.layers.SimpleRNN(128, 
                                  dropout=dropout_rate, 
                                  recurrent_dropout=dropout_rate))
model1.add(keras.layers.Dense(64, activation="relu"))
model1.add(keras.layers.Dense(1, activation="sigmoid"))

# Summarize model 
model1.summary()

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model1.compile(optimizer=optimizer, 
               loss="binary_crossentropy",
               metrics=["accuracy"])

# Fit model
model1_history = model1.fit(padded_sequences_train, labels_train,
                            epochs=100, batch_size=8,
                            validation_data=(padded_sequences_test, labels_test),
                            callbacks=early_stopping)  

# Save model    
model1.save("saved_models/model1")

# Load model
model1 = keras.models.load_model("saved_models/model1")

# Learning curve: Loss
plt.plot(model1_history.history["loss"], label="Train Loss")
plt.plot(model1_history.history["val_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
    
# Learning curve: Accuracy
plt.plot(model1_history.history["accuracy"], label="Train Accuracy")
plt.plot(model1_history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate model: Accuracy for training and test data
train_score_model1 = model1.evaluate(padded_sequences_train, labels_train)
test_score_model1 = model1.evaluate(padded_sequences_test, labels_test)
print("Accuracy Train data: ", train_score_model1[1])
print("Accuracy Test data: ", test_score_model1[1])

# Predicted labels for test data 
labels_pred_prob_model1 = model1.predict(padded_sequences_test)
labels_pred_model1 = labels_pred_prob_model1.copy()
labels_pred_model1[labels_pred_model1 >= 0.5] = 1
labels_pred_model1[labels_pred_model1 < 0.5] = 0

# Evaluate model: Classification report for test data
print("Classification Report: Model 1 (SimpleRNN)")
print(classification_report(labels_test, labels_pred_model1))

# Evaluate model: Confusion matrix for test data
cm = confusion_matrix(labels_test, labels_pred_model1)
cm_disp = ConfusionMatrixDisplay(cm)
cm_disp.plot()

# Illustrative examples: True vs. predicted labels for clear and obvious hate speech 
for i in [236, 207, 15, 29, 183]:
    print(f"Comment: {comments_test[i]}")
    print(f"True label: {labels_test[i]}")
    print(f"Predicted label: {int(labels_pred_model1[i][0])}")
    print("===")

    
#==============================================================================    
#=============================== Model 2: LSTM ================================
#==============================================================================    

# Initialize tokenizer 
num_words = 5000  # maximum number of unique words in the dictionary of the tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(
    num_words=num_words,
    filters='"!#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=" ", char_level=False, oov_token=None)

# Fit tokenizer to training data  
tokenizer.fit_on_texts(comments_train)

# Apply tokenizer to training and test data 
sequences_train = tokenizer.texts_to_sequences(comments_train)
sequences_test = tokenizer.texts_to_sequences(comments_test)  

# Apply sequence padding to obtain consistent sequence length 
max_length = 150
padded_sequences_train = keras.preprocessing.sequence.pad_sequences(
    sequences_train, 
    maxlen=max_length, 
    truncating="post", 
    padding="post"
    )
padded_sequences_test = keras.preprocessing.sequence.pad_sequences(
    sequences_test, 
    maxlen=max_length, 
    truncating="post", 
    padding="post"
    )

# Specify the number of dimensions of the word vectors in the embedding layer
word_vector_dim = 50

# Specify model
model2 = keras.models.Sequential()
model2.add(keras.layers.Embedding(num_words+1,  # number of words in tokenizer +1 for the "0" used for padding 
                                  word_vector_dim,
                                  input_length=max_length, 
                                  mask_zero=True))
model2.add(keras.layers.LSTM(128, 
                             return_sequences=True,
                             dropout=dropout_rate, 
                             recurrent_dropout=dropout_rate))
model2.add(keras.layers.LSTM(128, 
                             dropout=dropout_rate, 
                             recurrent_dropout=dropout_rate))
model2.add(keras.layers.Dense(64, activation="relu"))
model2.add(keras.layers.Dense(1, activation="sigmoid"))

# Summarize model
model2.summary()

# Compile model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=optimizer, 
               loss="binary_crossentropy",
               metrics=["accuracy"])

# Fit model
model2_history = model2.fit(padded_sequences_train, labels_train,
                            epochs=100, batch_size=32,
                            validation_data=(padded_sequences_test, labels_test),
                            callbacks=early_stopping)

# Save model    
model2.save("saved_models/model2")

# Load model
model2 = keras.models.load_model("saved_models/model2")

# Learning curve: Loss
plt.plot(model2_history.history["loss"], label="Train Loss")
plt.plot(model2_history.history["val_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
    
# Learning curve: Accuracy
plt.plot(model2_history.history["accuracy"], label="Train Accuracy")
plt.plot(model2_history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate model: Accuracy for training and test data
train_score_model2 = model2.evaluate(padded_sequences_train, labels_train)
test_score_model2 = model2.evaluate(padded_sequences_test, labels_test)
print("Accuracy Train data: ", train_score_model2[1])
print("Accuracy Test data: ", test_score_model2[1])

# Predicted labels for test data 
labels_pred_prob_model2 = model2.predict(padded_sequences_test)
labels_pred_model2 = labels_pred_prob_model2.copy()
labels_pred_model2[labels_pred_model2 >= 0.5] = 1
labels_pred_model2[labels_pred_model2 < 0.5] = 0

# Evaluate model: Classification report for test data
print("Classification Report: Model 2 (LSTM)")
print(classification_report(labels_test, labels_pred_model2))

# Evaluate model: Confusion matrix for test data
cm = confusion_matrix(labels_test, labels_pred_model2)
cm_disp = ConfusionMatrixDisplay(cm)
cm_disp.plot()

# Illustrative examples: True vs. predicted labels for clear and obvious hate speech 
for i in [236, 207, 15, 29, 183]:
    print(f"Comment: {comments_test[i]}")
    print(f"True label: {labels_test[i]}")
    print(f"Predicted label: {int(labels_pred_model2[i][0])}")
    print("===")
    

#==============================================================================    
#========================== Model 3: Fine-tuned BERT ==========================
#==============================================================================    

# Specify the input layer
text_input = keras.layers.Input(shape=(), dtype=tf.string, name="text")

# Preprocess the text inputs using the BERT preprocessing layer
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
                              name="preprocessing")
preprocessed_inputs = preprocessor(text_input)  # sequence length of preprocessed inputs is 128 tokens

# Load the BERT model
bert = hub.KerasLayer("https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
                      trainable=True, 
                      name="BERT")

# Pass the preprocessed text inputs through the BERT model
bert_outputs = bert(preprocessed_inputs)

# Extract the pooled output from the BERT outputs
pooled_bert_output = bert_outputs["pooled_output"]

# Apply dropout regularization
pooled_bert_output = keras.layers.Dropout(dropout_rate)(pooled_bert_output)

# Fine-tune the BERT model with dense layers for the hate speech detection task
dense = keras.layers.Dense(128, activation="relu")(pooled_bert_output)
outputs = keras.layers.Dense(1, activation="sigmoid", name="classifier")(dense)

# Create model
model3 = keras.Model(inputs=text_input, outputs=outputs)

# Summarize model 
model3.summary()

# Compile model 
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model3.compile(optimizer=optimizer, 
               loss="binary_crossentropy",
               metrics=["accuracy"])

# Fit model
model3_history = model3.fit(comments_train, labels_train,
                            epochs=100, batch_size=8,
                            validation_data=(comments_test, labels_test),
                            callbacks=early_stopping)  

# Save model    
model3.save("saved_models/model3")

# Load model
model3 = keras.models.load_model("saved_models/model3")

# Learning curve: Loss
plt.plot(model3_history.history["loss"], label="Train Loss")
plt.plot(model3_history.history["val_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
    
# Learning curve: Accuracy
plt.plot(model3_history.history["accuracy"], label="Train Accuracy")
plt.plot(model3_history.history["val_accuracy"], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Evaluate model: Accuracy for training and test data
train_score_model3 = model3.evaluate(comments_train, labels_train)
test_score_model3 = model3.evaluate(comments_test, labels_test)
print("Accuracy Train data: ", train_score_model3[1])
print("Accuracy Test data: ", test_score_model3[1])

# Predicted labels for test data 
labels_pred_prob_model3 = model3.predict(comments_test)
labels_pred_model3 = labels_pred_prob_model3.copy()
labels_pred_model3[labels_pred_model3 >= 0.5] = 1
labels_pred_model3[labels_pred_model3 < 0.5] = 0

# Evalute model: Classification report for test data
print("Classification Report: Model 3 (Bert)")
print(classification_report(labels_test, labels_pred_model3))

# Evaluate model: Confusion matrix for test data
cm = confusion_matrix(labels_test, labels_pred_model3)
cm_disp = ConfusionMatrixDisplay(cm)
cm_disp.plot()
 
# Illustrative examples: True vs. predicted labels for clear and obvious hate speech 
for i in [236, 207, 15, 29, 183]:
    print(f"Comment: {comments_test[i]}")
    print(f"True label: {labels_test[i]}")
    print(f"Predicted label: {int(labels_pred_model3[i][0])}")
    print("===")
  
# Illustrative examples: Comparing SimpleRNN, LSTM and fine-tuned BERT 
for i in [236, 207, 15, 29, 183]:
    print(f"Comment: {comments_test[i]}")
    print(f"True label: {labels_test[i]}")
    print(f"SimpleRNN: {int(labels_pred_model1[i][0])}")
    print(f"LSTM: {int(labels_pred_model2[i][0])}")
    print(f"Fine-Tuned BERT: {int(labels_pred_model3[i][0])}")
    print("===")