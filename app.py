from flask import Flask, render_template, request, redirect, url_for
import pickle

app = Flask(__name__)

#lin_model= pickle.load(open('lin_model.pkl','rb'))
#log_model= pickle.load(open('log_model.pkl','rb'))
svm= pickle.load(open('svc_model.pkl','rb'))

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Perform login validation and redirect to home if successful
        # ...
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/dashboards', methods=['GET', 'POST'])
def dashboards():
    return render_template('hello.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':

        # Get the entered values from the form
        inter_api_access_duration = float(request.form.get('inter_api_access_duration'))
        sequence_length = float(request.form.get('sequence_length'))
        vsession_duration = float(request.form.get('vsession_duration'))
        num_unique_apis = int(request.form.get('num_unique_apis'))
        source = int(request.form.get('source'))
        statusCode = int(request.form.get('statusCode'))
        auth_method = int(request.form.get('auth_method'))

        # Prepare the input data for prediction
        inputs = [[inter_api_access_duration,
                    sequence_length, 
                    vsession_duration, 
                    num_unique_apis,
                    source,
                    statusCode,
                    auth_method]]

        # Perform the predictions using the models
        #lin_prediction = lin_model.predict(inputs)
        #log_prediction = log_model.predict(inputs)
        svm_prediction = svm.predict(inputs)

        # Return the predictions to the template
        return render_template('predictions.html', 
                               #lin_prediction=lin_prediction,
                               #log_prediction=log_prediction, 
                               svm_prediction=svm_prediction)

    return render_template('predictions.html')


if __name__ == '__main__':
    app.run()
