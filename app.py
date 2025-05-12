from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = '1234567890'  # You can change the secret key

# -------- Load the model once when the server starts --------
model = None
model_path = 'model.pkl'  # Path to your pickle file

# Check if the model file exists and load it
if os.path.exists(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ model.pkl file not found!")
    model = None

# -------- Home Page --------
@app.route('/')
def home():
    return render_template('home.html')

# -------- Prediction Page --------
@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form data
            bedrooms = int(request.form['bedrooms'])
            builder = request.form['builder']
            locality = request.form['locality']
            prime_location = int(request.form['prime_location'])  # 0 or 1
            property_type = request.form['property_type']

            # Encode categorical variables manually or via a consistent mapping
            # Example encoding (must match what was used during training):
            builder_mapping = {'builder_a': 0, 'builder_b': 1}
            locality_mapping = {'locality_1': 0, 'locality_2': 1}
            property_mapping = {'apartment': 0, 'villa': 1}

            builder_encoded = builder_mapping.get(builder.lower(), 0)
            locality_encoded = locality_mapping.get(locality.lower(), 0)
            property_type_encoded = property_mapping.get(property_type.lower(), 0)

            # Prepare input
            input_features = np.array([[bedrooms, builder_encoded, locality_encoded, prime_location, property_type_encoded]])

            # Predict
            if model:
                predicted_price = model.predict(input_features)[0]
            else:
                flash('Model is not available right now!', 'danger')
                return redirect(url_for('predict'))

            return redirect(url_for('result',
                                    bedrooms=bedrooms,
                                    builder=builder,
                                    locality=locality,
                                    prime_location=prime_location,
                                    property_type=property_type,
                                    result=predicted_price))

        except Exception as e:
            flash(f'Error in prediction: {str(e)}', 'danger')
            return redirect(url_for('predict'))

    return render_template('prediction.html')

# -------- Result Page --------
@app.route('/result')
def result():
    area = request.args.get('area', type=float)
    bedrooms = request.args.get('bedrooms', type=int)
    bathrooms = request.args.get('bathrooms', type=int)
    location = request.args.get('location', type=str)
    result = request.args.get('result', type=float)

    return render_template('result.html', 
                           area=area, 
                           bedrooms=bedrooms, 
                           bathrooms=bathrooms,
                           location=location, 
                           result=round(result, 2))

# -------- Main --------
if __name__ == '__main__':

    app.run(debug=True)
