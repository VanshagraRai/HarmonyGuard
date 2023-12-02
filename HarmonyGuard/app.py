from flask import Flask,render_template,request
from prediction import ToxicCommentClassifier
import pandas as pd

app = Flask(__name__)
classifier = ToxicCommentClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guard', methods =["GET", "POST"])
def guard():
    if request.method == 'POST':
        comment = request.form.get('coInput')
        message = f"This comment is {classifier.classify_comment(comment)}"
        return render_template('guard.html',message=message,processed_value=comment)
    return render_template('guard.html')


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)