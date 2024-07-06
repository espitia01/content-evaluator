import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from isodate import parse_duration
import xgboost as xgb

def preprocess_and_predict(video_data, scaler_X, selected_features, final_models, pt):
    def duration_to_seconds(duration):
        if isinstance(duration, str):
            if duration.startswith('P'):
                return int(parse_duration(duration).total_seconds())
            else:
                duration = duration.replace('PT', '')
                hours = 0
                minutes = 0
                seconds = 0
                
                if 'H' in duration:
                    hours, duration = duration.split('H')
                    hours = int(hours)
                if 'M' in duration:
                    minutes, duration = duration.split('M')
                    minutes = int(minutes)
                if 'S' in duration:
                    seconds = float(duration.replace('S', ''))
                
                return int(hours * 3600 + minutes * 60 + seconds)
        else:
            return int(duration)
    
    # Create a DataFrame with the video data
    df = pd.DataFrame([video_data])
    
    # Convert 'channel_created_at' and 'collection_timestamp' to datetime
    df['channel_created_at'] = pd.to_datetime(df['channel_created_at'], format='mixed', utc=True)
    df['collection_timestamp'] = pd.to_datetime(df['collection_timestamp'], format='mixed', utc=True)
    
    # Calculate channel age in days
    df['channel_age_days'] = (df['collection_timestamp'] - df['channel_created_at']).dt.total_seconds() / (24 * 3600)
    
    # Convert duration to seconds
    df['duration_seconds'] = df['duration'].apply(duration_to_seconds)
    
    # Calculate derived features
    df['subscriber_video_ratio'] = df['channel_subscriber_count'] / (df['channel_video_count'] + 1)
    df['age_subscriber_ratio'] = df['channel_age_days'] / (df['channel_subscriber_count'] + 1)
    df['log_duration'] = np.log1p(df['duration_seconds'])
    
    # Add placeholder values for features we don't have yet
    df['view_count'] = df['channel_view_count'] / df['channel_video_count'] * 0.5 # Estimate based on average views per video
    df['like_count'] = df['view_count'] * 0.05  # Estimate 5% of views are likes
    df['comment_count'] = df['view_count'] * 0.01  # Estimate 1% of views are comments
    
    # Calculate the missing features
    df['views_per_subscriber'] = df['view_count'] / (df['channel_subscriber_count'] + 1)
    df['likes_per_view'] = df['like_count'] / (df['view_count'] + 1)
    df['comments_per_view'] = df['comment_count'] / (df['view_count'] + 1)
    df['views_per_channel_view'] = df['view_count'] / (df['channel_view_count'] + 1)
    
    # Select the features used in the model
    features = ['channel_subscriber_count', 'channel_video_count', 'channel_view_count', 'channel_age_days', 'duration_seconds',
                'subscriber_video_ratio', 'age_subscriber_ratio', 'views_per_subscriber', 'likes_per_view',
                'comments_per_view', 'log_duration', 'views_per_channel_view']
    X = df[features]
    
    # Scale the features
    X_scaled = scaler_X.transform(X)
    
    # Select the top k features
    X_selected = X_scaled[:, [features.index(f) for f in selected_features]]
    
    # Make predictions
    y_pred_transformed = np.column_stack([model.predict(X_selected) for model in final_models])
    
    # Inverse transform predictions
    y_pred = pt.inverse_transform(y_pred_transformed)
    
    # Prepare results
    targets = ['view_count', 'like_count', 'comment_count']
    results = {target: float(pred) for target, pred in zip(targets, y_pred[0])}
    return results