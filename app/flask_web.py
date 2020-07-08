from flask import Flask, render_template, request
from model_learn_def import model_learn

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
#파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save('../data/test.jpg')#+secure_filename(f.filename)
      print("11111111111")
      a = model_learn()
      print("ok"+a)
      return render_template('result.html', result = a)

if __name__ == '__main__':
    app.run()