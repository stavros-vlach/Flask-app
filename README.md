The dataset used is the **Social Network Ads dataset**, which contains user demographic information (Gender, Age, Estimated Salary) and whether the user purchased a product after seeing an advertisement.  

Target variable:  
- `Purchased` → 1 if the user purchased the product, 0 otherwise.

A **Machine Learning web application** built with Flask and deployed on Render.  
The app uses a **Logistic Regression model** (scikit-learn) to predict whether a user is likely to purchase a product after viewing an advertisement, based on:

- Gender (male/female)

* Age

+ Estimated Salary

👉 [Try the App Here](https://flask-app-for-logistic-regression.onrender.com).

Technologies Used

- Python 3

- scikit-learn

- Flask

- HTML templates

- Render (deployment)

How to Run Locally:

1. Clone the repository: 
  git clone https://github.com/username/repo-name.git
  cd repo-name

2. Create a virtual environment (recommended):

  python -m venv venv
  source venv/bin/activate   # On Linux/Mac
  venv\Scripts\activate      # On Windows

3. Install dependencies:
  pip install -r requirements.txt

4. Run the Flask app
  python app.py

The app will be available at:
 >http://127.0.0.1:5000/
 
Project Structure:  
├── app.py # Flask web application  
├── Logistic_Regression.py # Training script for Logistic Regression model  
├── logistic_model.pkl # Saved trained model  
├── scaler.pkl # Saved feature scaler  
├── Social_Network_Ads.csv # Training dataset  
├── requirements.txt # Project dependencies  
├── templates/  
│   ├── index.html # Input form  
│   └── result.html # Prediction result page  
