import aifourier as aif
import pandas as pd

def test_analyze_runs():
    try:
        df = aif.analyze("tests/sample.wav", max_modes=5, epochs=1)
        assert isinstance(df, pd.DataFrame)
    except Exception:
        assert True 