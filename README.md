# SentimentAnalysis
This project aims to build a model to classify the sentiment of a text in one of the three following categories: **happiness**, **sadness** or **anger**. It uses *python3* and *scikit-learn*.

### Setting up
Clone the project to your local machine and install the dependences listed in the `requirements.txt` file:
```
pip install -r requirements.txt
```

### Running
After that, you'll be able to run the project:
```
python Main.py
```

Some logs and metrics will be shown you while the models are being build. After some seconds you'll be able to enter your text in the terminal. Bellow you can see some output examples:

```
Enter your message (0 to quit): I miss you
Sadness :(

Enter your message (0 to quit): I don't wanna know about it, I hate you!
Anger -.-

Enter your message (0 to quit): This place has many good reviews
Happiness :)
```
