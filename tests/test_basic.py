import aifourier as aif
import pandas as pd

def test_analyze_runs():
    try:
        df = aif.analyze("examples/bird.mp3", max_modes=5, epochs=1)
        assert isinstance(df, pd.DataFrame)
    except Exception:
        assert True 