from setuptools import setup, find_packages

setup(
    name="ml_infrastructure",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.0.0",
        "flask>=2.0.0",
        "waitress>=2.0.0",
        "joblib>=1.0.0",
    ],
    author="FlashNew CTO",
    author_email="cto@flashnew.com",
    description="ML Infrastructure for model serving and analytics",
    keywords="machine learning, model serving, analytics",
    python_requires=">=3.7",
) 