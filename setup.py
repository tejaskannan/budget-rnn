import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
        name='budget-rnn',
        version='1.0',
        scripts=['budgetrnn/scripts'],
        author='Tejas Kannan',
        author_email='tkannan@uchicago.edu',
        description='An implementation of Budget RNNs for inference under energy budgets.',
        long_description=long_description,
        url='https://github.com/tejaskannan/budget-rnn',
        packages=['budgetrnn'],
        install_requires=['tensorflow==2.2', 'numpy', 'matplotlib', 'more_itertools', 'scipy', 'scikit-learn==0.22'],
        exclude_package_data={'': ['budgetrnn/data/*', 'budgetrnn/saved_models/*', 'budgetrnn/final_models/*', '.git/*']}
)
