from setuptools import setup, find_packages

VERSION = '0.1.1'

# this is what usually goes on requirements.txt
install_requires = [
    'tqdm',
    # for scoring
    'penman',
    # for debugging
    'ipdb',
]

setup(
    name='docAMR',
    version=VERSION,
    description="document AMR representation and evaluation",
    py_modules=['doc_amr','amr_io','docSmatch','doc_amr_baseline'],
    entry_points={
        'console_scripts': [
            'doc-smatch = docSmatch.smatch:main',
            'doc-baseline = doc_amr_baseline.make_doc_amr:main',
            'doc-amr = doc_amr:main'
        ]
    },
    packages=find_packages(),
    install_requires=install_requires,
)
