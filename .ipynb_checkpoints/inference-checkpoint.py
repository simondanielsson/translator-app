import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import string

#transformer = keras.models.load_model('models/second_model')
transformer = tf.saved_model.load('models/second_model')
vocab_size = 15000
sequence_length = 20
batch_size = 64
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)

spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)

eng_vectorization = keras.models.load_model('models/eng_vectorizer', compile=False)
eng_vectorization = eng_vectorization.layers[0]

custom_objects = {"custom_standardization": custom_standardization}#, 're': re}
with keras.utils.custom_object_scope(custom_objects):
    spa_vectorization = keras.models.load_model('models/swe_vectorizer', compile = False)
    spa_vectorization = spa_vectorization.layers[0]
#spa_vectorization = keras.models.load_model('models/swe_vectorizer', compile=False)
#spa_vectorization = spa_vectorization.layers[0]

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20
def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence