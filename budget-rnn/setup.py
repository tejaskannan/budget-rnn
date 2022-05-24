import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
        name='budget-rnn',
        version='1.0',
        author='Tejas Kannan',
        author_email='tkannan@uchicago.edu',
        description='An implementation of Budget RNNs for inference under energy budgets.',
        long_description=long_description,
        url='https://github.com/tejaskannan/budget-rnn',
        packages=['src'],
        install_requires=['tensorflow==2.6.4', 'numpy', 'matplotlib', 'more_itertools', 'scipy==1.4.1', 'scikit-learn==0.22'],
        exclude_package_data={'': ['src/data/*', 'src/saved_models/*', 'src/trained_models/*', 'src/trained_models/*', '.git/*']}
)
