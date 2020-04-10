from flask import Flask, request, render_template, jsonify
from controller import load_model1, load_model2


app = Flask(__name__)
call_counter = 0

def judge_rating(stars_pred, stars):
    return stars == stars_pred
        

@app.route("/", methods=["POST","GET"])
def movies():
    answer = "N/A"
    if request.method == "POST":
        text = request.form['review']
        stars = int(request.form.get('stars'))
        text_processed = text.lower()
        filename = "../Review Analyzer/text_file.txt"
        #filename = "\\test_file.txt"
        file1 = open(filename,"w")
        file1.write(text_processed)
        file1.close()
        (answer,stars_pred) = load_model1.predict_with_text()
        if not judge_rating(stars_pred, stars):
            answer = str(stars_pred) + " stars are expected with this " + answer + " review. Please reconsider."
        else:
            answer = "Analysis Result: " + answer +" | "+ " Status: ACCEPTED" 
        return jsonify(result=answer) 
    else:
        return render_template("index.html")

@app.route("/products", methods=["POST","GET"])
def products():
    answer = "N/A"
    
    if request.method == "POST":
        text = request.form['review']
        stars = int(request.form.get('stars'))
        print()
        print(stars, " are my stars")
        print()
        text_processed = text.lower()
        filename = "../Review Analyzer/text_file.txt"
        #filename = "\\test_file.txt"
        file1 = open(filename,"w")
        file1.write(text_processed)
        file1.close()
        (answer,stars_pred) = load_model2.predict_with_text()
        #print(answer)
        if not judge_rating(stars_pred, stars):
            #answer = "Got " + str(stars) +" stars but expected  " + str(stars_pred) + " stars"
            answer = str(stars_pred) + " stars are expected with this " + answer + " review. Please reconsider."
        else:
            answer = "Analysis Result: " + answer +" | "+ " Status: ACCEPTED" 
        return jsonify(result=answer) 
    else:
        return render_template("products.html")

@app.route("/tweets", methods=["POST","GET"])
def tweets():
    answer = "N/A"
    
    if request.method == "POST":
        text = request.form['review']
        print()
        text_processed = text.lower()
        filename = "../Review Analyzer/text_file.txt"
        #filename = "\\test_file.txt"
        file1 = open(filename,"w")
        file1.write(text_processed)
        file1.close()
        (answer,stars_pred) = load_model2.predict_with_text()
        if answer == "WORST":
            answer = "EXTREMELY NEGATIVE"
        return jsonify(result=answer) 
    else:
        return render_template("tweets.html")

@app.route("/fb", methods=["POST","GET"])
def fb():
    answer = "N/A"
    
    if request.method == "POST":
        text = request.form['review']
        print()
        text_processed = text.lower()
        filename = "../Review Analyzer/text_file.txt"
        #filename = "\\test_file.txt"
        file1 = open(filename,"w")
        file1.write(text_processed)
        file1.close()
        (answer,stars_pred) = load_model2.predict_with_text()
        if answer == "WORST":
            answer = "EXTREMELY NEGATIVE"
        return jsonify(result=answer) 
    else:
        return render_template("fb.html")



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run(debug=True)
    
    
    
    
    
    