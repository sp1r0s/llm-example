import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data import get_data
from llm import train


if __name__ == '__main__':

    my_llm_model, tokenizer, max_sequence_length = train(get_data())

    text = 'What is the most high risk event?'  # Seed
    next_words = 15

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
        predicted_probabilities = my_llm_model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probabilities)
        output_word = tokenizer.index_word[predicted_index]
        text += ' ' + output_word

    print(text)
