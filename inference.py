# inference.py

import numpy as np

def make_prediction(preprocessed_data, model_data):
    scaler_X = model_data['scaler_X']
    selected_features = model_data['selected_features']
    final_models = model_data['final_models']
    pt = model_data['pt']

    # Scale the features
    X_scaled = scaler_X.transform(preprocessed_data)

    # Select the features used in the model
    X_selected = X_scaled[:, [preprocessed_data.columns.get_loc(f) for f in selected_features]]

    # Make predictions
    y_pred_transformed = np.column_stack([model.predict(X_selected) for model in final_models])

    # Inverse transform predictions
    y_pred = pt.inverse_transform(y_pred_transformed)

    # Prepare results
    targets = ['view_count', 'like_count', 'comment_count']
    results = {target: int(max(0, pred)) for target, pred in zip(targets, y_pred[0])}

    return results