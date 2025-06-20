# agriculture-commodity-price-forecasting
Agriculture commodity price forecasting using Time series (ARIMA &amp; SARIMA) and deployed in python web application 

🌾 Agriculture Commodity Prices Forecasting using Time Series Models
This project aims to forecast agriculture commodity prices using time series forecasting techniques (ARIMA and SARIMA models). A web interface has been developed using the Django framework to make forecasting accessible to users with minimal technical background.

🚀 Project Highlights
📊 Real-time Data Collection of agriculture commodity prices

🧹 Data Preprocessing: Handling missing values, outliers, and formatting

🔍 Exploratory Data Analysis (EDA): Trend analysis, seasonality, and correlations

⏳ Forecasting Models:

ARIMA (AutoRegressive Integrated Moving Average)

SARIMA (Seasonal ARIMA)

🌐 Web Application using Python Django for interactive forecast visualization and input file upload

📈 Visual outputs: Time series plots, forecast graphs, RMSE error metrics

🧠 Technologies Used
Python (Pandas, Numpy, Matplotlib, Seaborn, Statsmodels)

Django Web Framework

HTML, CSS (for front-end rendering)

Excel/CSV file handling

Time Series Forecasting Models (ARIMA, SARIMA)

📁 Project Structure
bash
Copy
Edit
agri_forecast/
├── agri_forecast/        # Django project folder
│   ├── settings.py
│   └── ...
├── forecast/             # Main app
│   ├── views.py
│   ├── models.py
│   ├── templates/
│   │   └── forecast.html
│   └── ...
├── media/                # For uploaded Excel files
├── static/               # Static files (CSS, JS)
├── manage.py
└── requirements.txt
📷 Web App Features
Upload your Excel dataset

View trend analysis and heatmaps

See commodity-wise forecasting results

Graphical view of ARIMA and SARIMA predictions

RMSE metrics for model accuracy

🧪 How to Run the Project
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/agri-forecast-django.git
cd agri-forecast-django
Create virtual environment and install dependencies

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Run the Django server

bash
Copy
Edit
python manage.py runserver
Open your browser and go to:

arduino
Copy
Edit
http://127.0.0.1:8000/forecast/
👨‍💻 Authors
Harish Raj S

Hari Rudhran M

🎓 Guided by
V. Bhagyalakshmi Mam
Faculty, Department of Computer Science and Business Systems
Er. Perumal Manimekalai College of Engineering

🏆 Recognition
🥇 1st Prize Winner
Mini Project Expo 2025
Department of CSBS, Er. Perumal Manimekalai College of Engineering
