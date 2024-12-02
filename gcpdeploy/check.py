import pickle

try:
    with open('models/best_random_forest_model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    sklearn_version_used = model.__sklearn_version__
except Exception as e:
    sklearn_version_used = f"Error: {str(e)}"

print(f"Model scikit-learn version: {sklearn_version_used}")