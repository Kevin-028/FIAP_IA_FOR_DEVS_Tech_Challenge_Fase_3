"""Configuração compartilhada da suíte de testes (pytest)."""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

for _p in (WEB_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["FIAP_LLM_PROVIDER"] = "fallback"


@pytest.fixture(scope="session")
def app():
    import app as app_module

    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def benigno_form():
    return {
        "smoothness_mean": 0.117, "compactness_mean": 0.07568, "symmetry_mean": 0.193,
        "fractal_dimension_mean": 0.07818, "radius_se": 0.2241, "texture_se": 1.508,
        "smoothness_se": 0.01019, "compactness_se": 0.01084, "concavity_se": 0.0,
        "concave_points_se": 0.0, "symmetry_se": 0.02659, "fractal_dimension_se": 0.0041,
        "texture_worst": 19.54, "perimeter_worst": 50.41, "smoothness_worst": 0.1584,
        "compactness_worst": 0.1202, "concavity_worst": 0.0, "concave_points_worst": 0.0,
        "symmetry_worst": 0.2932, "fractal_dimension_worst": 0.09382,
    }


@pytest.fixture()
def maligno_form():
    return {
        "smoothness_mean": 0.1084, "compactness_mean": 0.1988, "symmetry_mean": 0.2061,
        "fractal_dimension_mean": 0.05623, "radius_se": 2.547, "texture_se": 1.306,
        "smoothness_se": 0.00765, "compactness_se": 0.05374, "concavity_se": 0.08055,
        "concave_points_se": 0.02598, "symmetry_se": 0.01697, "fractal_dimension_se": 0.004558,
        "texture_worst": 31.37, "perimeter_worst": 251.2, "smoothness_worst": 0.1357,
        "compactness_worst": 0.4256, "concavity_worst": 0.6833, "concave_points_worst": 0.2625,
        "symmetry_worst": 0.2641, "fractal_dimension_worst": 0.07427,
    }
