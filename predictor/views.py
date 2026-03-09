import pandas as pd
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration

def data_exploration_view(request):
    df = pd.read_csv("dummy-data/vehicles_ml_dataset.csv")

    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
    }
    return render(request, "predictor/index.html", context)

# Placeholder until Part V is implemented
def regression_analysis(request):
    try:
        from model_generators.regression.train_regression import evaluate_regression_model
        import joblib
        regression_model = joblib.load("model_generators/regression/regression_model.pkl")
    except ImportError:
        pass

    context = {
        # "evaluations": evaluate_regression_model()
    }
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])

        # prediction = regression_model.predict([[year, km, seats, income]])[0]
        # context["price"] = prediction
        
    return render(request, "predictor/regression_analysis.html", context)