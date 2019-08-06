import pytest

from sklearn.linear_model import LogisticRegression

from mlsquare import dope
from mlsquare import registry

def test_dope_basic_functionality(): # Change description. More specific about the what is being tested
    model = LogisticRegression()
    m = dope(model)
    assert hasattr(m, 'fit') == True
    assert hasattr(m, 'score') == True
    assert hasattr(m, 'save') == True

def test_dope_importing_external_model(): # test_importing_external_modules
    model = LogisticRegression()
    abstract_model, adapter = registry[('sklearn', 'LogisticRegression')]['default']
    m = dope(model, abstract_model=abstract_model, adapt=adapter)

    assert hasattr(m, 'fit') == True
    assert hasattr(m, 'score') == True
    assert hasattr(m, 'save') == True

def test_dope_without_primal():
    with pytest.raises(TypeError) as _:
        dope()

def test_dope_with_unsupported_primal():
    with pytest.raises(TypeError) as _:
        dope('test')

def test_dope_with_unsupported_version():
    with pytest.raises(TypeError) as _:
        model = LogisticRegression()
        dope(model, version='test')

def test_dope_using_unsupported_transpilation_type():
    model = LogisticRegression()
    m = dope(model, using='test')
    assert isinstance(m, type(model))

def test_dope_using_transpilation_type_as_None():
    model = LogisticRegression()
    m = dope(model, using=None)
    assert isinstance(m, type(model))

def test_dope_with_version():
    pass