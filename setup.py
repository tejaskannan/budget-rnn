import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()


setuptools.setup(
        name='budget-rnn',
        version='0.1',
        scripts=['run.py'],
        author='Tejas Kannan',
        author_email='tkannan@uchicago.edu',
        description='An implementation of Budget RNNs for inference under energy budgets.',
        long_description=long_description,
        url='https://github.com/tejaskannan/budget-rnn',
        packages['src'],
        install_requires=['tensorflow==2.4.0', 'numpy', 'matplotlib', 'more_itertools', 'scipy']
)
