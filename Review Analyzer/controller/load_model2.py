from tensorflow import keras
import numpy as np
import pickle

#model2 = keras.models.load_model("C:\\Users\\mr_ro\\Desktop\\Review Analyzer\\model\\model2_2.h5")
model2 = keras.models.load_model("../Review Analyzer/model/model2_2.h5")

#with open("C:\\Users\\mr_ro\\Desktop\\Review Analyzer\\model\\tokenizer_2.pickle", 'rb') as handle:
with open("../Review Analyzer/model/tokenizer_2.pickle", 'rb') as handle:
    tokenizer = pickle.load(handle)

def review_encode(s):
    encoded = tokenizer.texts_to_sequences([s])
    '''
    word_index = tokenizer.word_index
    nb_word = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    missed = []
    for word, i in word_index.items():
        if i >= max_features: break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missed.append(word)
    '''
    '''
    #old code
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    '''
    print(encoded)
    return encoded

def predict_with_text():
    doc = []
    maxlen = 200
    #filename = "test_file.txt"
    filename = "../Review Analyzer/text_file.txt"
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"", "").strip().split(" ")
            print(line)
            doc = doc + nline
        encode = review_encode(doc)
        encode = keras.preprocessing.sequence.pad_sequences(encode, maxlen = maxlen)
        
        prediction = model2.predict(encode)[0]
        prediction = list(prediction).index(max(prediction)) + 1
        
        if prediction == 5:
            answer = "GREAT"
        elif prediction == 4:
            answer = "POSITIVE"
        elif prediction == 3:
            answer = "AVERAGE"
        elif prediction == 2:
            answer = "NEGATIVE"
        else:
            answer = "WORST"
        print("\n",answer, ' ', "stars: ",prediction)
        
        return (answer,prediction)
'''
#testing



if __name__ == "__main__":
    answer = predict_with_text()
    print(answer)
'''
