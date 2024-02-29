import io
import re
import string
import tqdm
import numpy as np
from collections import Counter
import random
from datetime import datetime
import statistics
import pandas as pd

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import tensorflow as tf
from tensorflow import keras
from keras import layers

from multiprocessing import Process, Manager


import plotly
import plotly.graph_objs as go
from sklearn.decomposition import PCA

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
global word
word = {'target' : [], 'context' : []}

#Tensor buffer var
BATCH_SIZE = 1024
BUFFER_SIZE = 10000


# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels



##########################
##  MAin Code #######
#########################

def callMyWord2Vec(lines, window_size, embedding_dim, num_ns, sequence_length, vocab_size, optimizer):

  text_ds = tf.data.Dataset.from_tensor_slices(lines)
  # text_ds = tf.data.Dataset.from_tensor_slices(lines).filter(lambda x: tf.cast(tf.strings.length(x), bool))

  # Now, create a custom standardization function to lowercase the text and
  # remove punctuation.
  def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation), '')


  # Use the `TextVectorization` layer to normalize, split, and map strings to
  # integers. Set the `output_sequence_length` length to pad all samples to the
  # same length.
  vectorize_layer = layers.TextVectorization(
      standardize=custom_standardization,
      max_tokens=vocab_size,
      output_mode='int',
      output_sequence_length=sequence_length)


  vectorize_layer.adapt(text_ds.batch(1024))

  # Save the created vocabulary for reference.
  inverse_vocab = vectorize_layer.get_vocabulary()
  print(inverse_vocab[:20])

  # Vectorize the data in text_ds.
  text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
  text_vector_ds = text_vector_ds.shuffle(BUFFER_SIZE)

  sequences = list(text_vector_ds.as_numpy_iterator())
  print(len(sequences))
  # print(sequences)
  
  # for seq in sequences[:5]:
    # print(f"{seq} => {[inverse_vocab[i] for i in seq]}")


  targets, contexts, labels = generate_training_data(
      sequences=sequences,
      window_size=window_size,
      num_ns=num_ns,
      vocab_size=vocab_size,
      seed=SEED)

  targets = np.array(targets)
  contexts = np.array(contexts)
  labels = np.array(labels)

  print('\n')
  print(f"targets.shape: {targets.shape}")
  print(f"contexts.shape: {contexts.shape}")
  print(f"labels.shape: {labels.shape}")


  #Shuffling dataset
  input_data = list(zip(targets, contexts, labels))
  random.shuffle(input_data)
  targets, contexts, labels = list(zip(*input_data))
  targets = np.array(targets)
  contexts = np.array(contexts)
  labels = np.array(labels)

  #Split into train and test
  l = len(targets)
  print('Length of train data', int(0.8*l))
  print('Length of test data', (l - int(0.8*l)))

  targets_train = targets[:int(0.8*l)]
  contexts_train = contexts[:int(0.8*l)]
  labels_train = labels[:int(0.8*l)]

  targets_test = targets[int(0.8*l):]
  contexts_test = contexts[int(0.8*l):]
  labels_test = labels[int(0.8*l):]

  #Joins all the variables target, context and labels to make a numpy array
  dataset = tf.data.Dataset.from_tensor_slices(((targets_train, contexts_train), labels_train))
  dataset_test = tf.data.Dataset.from_tensor_slices(((targets_test, contexts_test), labels_test))

  #Batch groups the data into batch size and the remainder rows which do not divide evenly by batch size are dropped to 
  # keep even number of elements in each batch
  #Shuffle randomly shuffles the dataset. Buffer size should be greater than number of training examples.
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
  dataset_test = dataset_test.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


  #Cache the dataset
  #  number of elements that will be buffered when prefetching
  dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
  dataset_test = dataset_test.cache().prefetch(buffer_size=AUTOTUNE)


  class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
      super(Word2Vec, self).__init__()
      self.target_embedding = layers.Embedding(vocab_size,
                                        embedding_dim,
                                        input_length=1,
                                        name="w2v_embedding")
      self.context_embedding = layers.Embedding(vocab_size,
                                         embedding_dim,
                                         input_length=num_ns+1)
      self.dotlist = []

    def call(self, pair):
      target, context = pair
      # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
      # context: (batch, context)
      if len(target.shape) == 2:
        target = tf.squeeze(target, axis=1)

      #Tracking a particular word
      #Getting the index of word
      # ls = target.numpy()
      # indices = [i for i, x in enumerate(ls) if x == 3980]

      # target: (batch,)
      word_emb = self.target_embedding(target)
      # word_emb: (batch, embed)
      context_emb = self.context_embedding(context)

      #Append target and contect to global dict
      # if indices:
      #     for i in indices:
      #         word['target'].append(word_emb.numpy()[i])
      #         word['context'].append(context_emb.numpy()[i][1])

      # context_emb: (batch, context, embed)
      dots = tf.einsum('be,bce->bc', word_emb, context_emb)
      #Testing for output values post sigmoid function.
      # a = dots.numpy()
      # print([[round((1/(1+math.exp(-c))),2) for c in b] for b in a])
      # dots: (batch, context)
      return dots


  word2vec = Word2Vec(vocab_size, embedding_dim)
  word2vec.compile(optimizer=optimizer,
                   loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

  start = datetime.now()

  word2vec.fit(dataset, validation_data=dataset_test, shuffle = True, epochs=30)

  end = datetime.now()
  # weights = word2vec.get_layer('w2v_embedding').get_weights()[0]

  runTime = end-start
  runTime = runTime.seconds

  train_param = word2vec.evaluate(dataset)
  acc_train = train_param[1]
  loss_train = train_param[0]

  test_param = word2vec.evaluate(dataset_test)
  acc_test = test_param[1]
  loss_test = test_param[0]

  return acc_train, loss_train, runTime, l, acc_test, loss_test



analysis = {'WindowSize': [],
            'EmbedDimen': [],
            'VocabSize': [],
            'SeqLen': [],
            'NegativeSample': [],
            'TrainingDataSize': [],
            'AccuracyTrain': [],
            'LossTrain': [],
            'AccuracyTest': [],
            'LossTest': [],
            'RunTime': []
            }

#Negative sampling
num_ns=[2]
window_size = [5]
embedding_dim = [64]
optimizers = ['SGD', 'Adagrad', 'RMSprop', 'Adam']
vocab_size = 0

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

with open(path_to_file) as f:
  lines = f.read().splitlines()
for line in lines[:20]:
  print(line)

# with open('text8') as f:
#     lines = f.read().splitlines()

cachedStopWords = stopwords.words("english")
# lines = [word for word in lines[0].split() if word not in cachedStopWords]

lines_no_sw=[]
for line in tqdm.tqdm(lines):
  if len(line):
    lines_no_sw.append([word for word in line.split() if not word in cachedStopWords])
lines = lines_no_sw

# lines = [" ".join(lines)]
lines = [(" ".join(line)) for line in lines]

print('Setting sequence length')
# sentence_analysis = [len(line) for line in lines if len(line)>0]
# sequence_length = int(statistics.mode(sentence_analysis))
sequence_length = 10

# counter_seq = 0
# for _ in lines[0].split():
#     counter_seq += 1
# sequence_length = counter_seq

print('Setting vocabulary length')
# Unique words in corpus

# count_words = Counter(lines[0].split())

words = [text1 for text in lines for text1 in text.split()]
count_words = Counter(words)
print('Initial Vocabulary size', len(count_words) )

#Considering the vocab for only word greater than occurence of 10 times
for a in count_words.keys():
    if count_words[a]>2:
       vocab_size+=1

# callMyWord2Vec(lines, 5, 128, 4, sequence_length, vocab_size)

for ns in num_ns:
  for window in window_size:
    for embedding in embedding_dim:
      for optimizer in optimizers:
        analysis['VocabSize'].append(vocab_size)
        analysis['SeqLen'].append(sequence_length)
        analysis['NegativeSample'].append(ns)
        analysis['WindowSize'].append(window)
        analysis['EmbedDimen'].append(embedding)

        param_shakes = callMyWord2Vec(lines, window, embedding, ns, sequence_length, vocab_size, optimizer)

        analysis['AccuracyTrain'].append(param_shakes[0])
        analysis['LossTrain'].append(param_shakes[1])
        analysis['RunTime'].append(param_shakes[2])
        analysis['TrainingDataSize'].append(param_shakes[3])
        analysis['AccuracyTest'].append(param_shakes[4])
        analysis['LossTest'].append(param_shakes[5])


 analysisDF = pd.DataFrame(analysis)
analysisDF.to_csv('Text3.csv')
print('Done')
       
