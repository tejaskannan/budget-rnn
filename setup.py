import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
        name='ml-models',
        version='0.1',
        scripts=['run.py'],
        author='Tejas Kannan',
        author_email='tkannan@uchicago.edu',
        description='A library of Machine Learning Models',
        long_description=long_description,
        url='https://github.com/tejaskannan/ml-models',
        packages=setuptools.find_packages(),
        classifiers=[]
)
