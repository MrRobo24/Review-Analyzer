# Review-Analyzer
 
A web based system that analyzes the sentiment behind given product and movie reviews. The system is built to suggest a specific star rating which should go well with the review entered by the user. There are two additional sections to analyze tweets and facebook comments/posts but these are yet to be linked with a model which is trained on social media posts. Currently they are linked with a neural network model trained on Yelp Business Dataset which uses GloVe-Twitter Word Embeddings. The moive review model is trained on the basic IMDB Movie Review Dataset.

# Motivation

IMDB, Amazon, Myntra, Flipkart are few of the giants in the entertainment and retail industry which is flourishing in today’s incessantly growing consumer market. One thing which is utterly important for and common among these giants is the review system that they have implemented on their applications. The problem of posting mismatching or spurious reviews is malpractice which website and app users often commit knowingly and unknowingly.
I have personally come across such reviews very often and some of which are mentioned down below:

In the review below the customer is extremely negative in the typed review but the star rating given above is 3 out of  the 5-star system.
![Review 1](https://github.com/MrRobo24/Review-Analyzer/blob/MrRobo24-patch-1/Review%20Analyzer/Screenshots/rev1.png)

Using this system one can be suggested that his or her provided reating is not going well with the review typed. If the person reconsiders his or her review submission then it will help in drastically reducing the misleading reviews on websites like IMDB and Amazon.

# Build Status

This project is in basic completion stage with test accuracy reaching around 86% right now. There's a lot that can be done as I am currently at the beginning point of the learning curve.

# Screenshots

Home of the interface:
![Home](https://github.com/MrRobo24/Review-Analyzer/blob/MrRobo24-patch-1/Review%20Analyzer/Screenshots/ss1_main_page.png)

Product Review in action:
![amazon_rev_1](https://github.com/MrRobo24/Review-Analyzer/blob/MrRobo24-patch-1/Review%20Analyzer/Screenshots/ss2_amazon_rev_1.png)

Appropriate Product Review:
![amazon_rev_2](https://github.com/MrRobo24/Review-Analyzer/blob/MrRobo24-patch-1/Review%20Analyzer/Screenshots/ss2_amazon_rev_2.png)

Facebook Comment Polarity Check:
![fb_com1](https://github.com/MrRobo24/Review-Analyzer/blob/MrRobo24-patch-1/Review%20Analyzer/Screenshots/ss3_fb_com1.png)

# References

The major references significant for this project are: 
1. Tensorflow Guide: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiy7tHR1JfnAhV5H7cAHcD7A30QFjABegQIBBAB&url=https%3A%2F%2Fwww.tensorflow.org%2Fguide&usg=AOvVaw3x033qMLvBehmxeX4dj8nC
2. Keras Documentation: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjagt_-1JfnAhXT7XMBHcDsBQcQFjAAegQICBAC&url=http%3A%2F%2Fkeras.io%2F&usg=AOvVaw330NFtOAF1xcgasnbQvfe5
3. Freecodecamp.org: https://www.youtube.com/channel/UC8butISFwT-Wl7EV0hUK0BQ
4. Kaggle Article on Yelp Review Analysis:
https://www.kaggle.com/athoul01/predicting-yelp-ratings-from-review-text
5. Amazon India Reviews: https://www.amazon.in
6. IMDB Reviews: https://www.imdb.com/
7. Medium Post on Sentiment Analysis : https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17
8. Medium Post on Word2Vec: https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa
9. Medium Post on Word Embeddings: https://towardsdatascience.com/word-embeddings-exploration-explanation-and-exploitation-with-code-in-python-5dac99d5d795
