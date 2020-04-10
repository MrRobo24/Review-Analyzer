from tensorflow import keras

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)


word_index = data.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding="post",maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding="post",maxlen = 250)



def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])
#testing region
def review_encode(s):
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

#model = keras.models.load_model("C:\\Users\\mr_ro\\Desktop\\Review Analyzer\\model.h5")
model = keras.models.load_model("../Review Analyzer/model/model.h5")



def predict_with_text():
    doc = []
    #filename = "text_file.txt"
    filename = "../Review Analyzer/text_file.txt"
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"", "").strip().split(" ")
            print(line)
            doc = doc + nline
        encode = review_encode(doc)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value=word_index["<PAD>"],padding="post",maxlen = 250)
        predict = model.predict(encode)
        print(encode)
        if predict[0][0] > 0.9:
            answer = ("GREAT",5)
        elif predict[0][0] > 0.7:
            answer = ("POSITIVE",4)
        elif predict[0][0] > 0.4:
            answer = ("AVERAGE",3)
        elif predict[0][0] > 0.2:
            answer = ("NEGATIVE",2)
        else:
            answer = ("WORST",1)
        print("\n",answer[0], ' ', predict[0][0] * 100)
        return answer
