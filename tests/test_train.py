from src.train import load_data, preprocess_data


def test_load_data():
    df = load_data()
    assert not df.empty
    assert "Age" in df.columns
    assert "Billing Amount" in df.columns
    assert "Gender" in df.columns
    assert "Blood Type" in df.columns
    assert "Admission Type" in df.columns
    assert "Medication" in df.columns
    assert "Test Results" in df.columns
    assert "Medical Condition" in df.columns


def test_preprocess_data():
    df = load_data()

    X, y = preprocess_data(df)
    assert not X.empty
    assert "Age" in X.columns
    assert "Billing Amount" in X.columns

    assert "Gender_Female" in X.columns
    assert "Gender_Male" in X.columns

    assert y.dtype == "int64"
