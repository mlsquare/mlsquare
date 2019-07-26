from mlsquare.base import BaseModel
import pytest

def test_base_model_initialization():
    with pytest.raises(TypeError) as _:
        class TestBase(BaseModel):
            pass
        _test_base = TestBase()