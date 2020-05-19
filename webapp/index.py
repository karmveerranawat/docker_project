from flask import Flask, render_template
import mylib

app = Flask(__name__)

@app.route('/')
def empty():
    return render_template('signup.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/logs')
def logs():
    value = mylib.login()
    return render_template('hello.html', name = value)

@app.route('/signup')
def signup():
    value = mylib.signup()
    return value

@app.route('/verify')
def verify():
    return render_template('verify.html')

@app.route('/verification')
def signup_verify():
    value = mylib.signup_verify()
    return value


if __name__ == '__main__':
    app.run(host='0.0.0.0',debug = True)
