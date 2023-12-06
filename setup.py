from setuptools import find_packages, setup


setup(
        name='zero-shot-time',
        version='0.0.1',
        packages=find_packages(),
        url='https://github.com/JMGaljaard/zero-shot-time',
        license='',
        author='JMGaljaard',
        author_email='J.M.Galjaard@tudelft.nl',
        description='',
        extra_requires={
                'development': [
                    'black~=23.11.0',
                    'ruff~=0.16.1'
                ]
        }
)
