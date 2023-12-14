from setuptools import find_packages, setup


setup(
        name='zero-shot-time',
        version='0.0.1',
        packages=find_packages(),
        install_requires=[
            'torch==2.1.1',
            'datasets==2.15.0',
            'transformers==4.35.2',
            'black==23.11.0',
            'ruff==0.1.6'
            'scikit-learn==1.3.2',
            'parameterized'
        ],
        url='https://github.com/JMGaljaard/zero-shot-time',
        license='',
        author='JMGaljaard',
        author_email='J.M.Galjaard@tudelft.nl',
        description='Basic package providing functionality to re-implement Zero-Shot time series forecasting',
)
