from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

def label_encode_categorical_columns(train_df, test_df):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    label_encoders = {}

    for column in train_encoded.columns:
        if train_encoded[column].dtype == 'object':
            # Initialize LabelEncoder for each categorical column
            label_encoder = LabelEncoder()
            
            # Fit on training data and transform both training and test data
            label_encoder.fit(train_encoded[column])
            train_encoded[column] = label_encoder.transform(train_encoded[column])
            test_encoded[column] = test_encoded[column].map(lambda s: '<unknown>' if s not in label_encoder.classes_ else s)
            test_encoded[column] = label_encoder.transform(test_encoded[column])
            
            label_encoders[column] = label_encoder

    return train_encoded, test_encoded, label_encoders