from setuptools import setup, find_packages
import os

def _read_install_requires():
    req_path = os.path.join(os.path.dirname(__file__), "blowfish", "requirements.txt")
    with open(req_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def _read_extras(filename: str) -> list[str]:
    path = os.path.join(os.path.dirname(__file__), "blowfish", filename)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]


def _read_explain_extras():
    return _read_extras("requirements-explain.txt")


def _read_evaluation_extras():
    return _read_extras("requirements-evaluation.txt")


def _read_datasets_extras():
    return _read_extras("requirements-datasets.txt")


_EVALUATION = _read_evaluation_extras()
_DATASETS = _read_datasets_extras()
_EXPLAIN = _read_explain_extras()
_ANTHROPIC = ["anthropic>=0.30"]
_OPENAI = ["openai>=1.30"]


setup(
    name='open-blowfish',
    version='0.3.0',
    author='Alex De Castro, Thomas-Roland Barillot, Javier Makmuri',
    author_email='alex.castro@alphaquaest.com.br, thomasroland.barillot@blackrock.com, javier.makmuri@blackrock.com',
    description='Topological + statistical signatures for quantifying ambiguity in semantic search. Open-source fork of blackrock/blowfish.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thomasbarillot/open-blowfish',
    packages=find_packages(),
    install_requires=_read_install_requires(),
    extras_require={
        "explain": _EXPLAIN,
        "evaluation": _EVALUATION,
        "datasets": _DATASETS,
        "rag": [],
        "anthropic": _ANTHROPIC,
        "openai": _OPENAI,
        "all": _EXPLAIN + _EVALUATION + _DATASETS + _ANTHROPIC + _OPENAI,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'blowfish=blowfish.__main__:main',
        ],
    },
    keywords='vector search, RAG, explainability, polysemy, ambiguity, persistent homology, topological data analysis',
    project_urls={
        'Bug Reports': 'https://github.com/thomasbarillot/open-blowfish/issues',
        'Source': 'https://github.com/thomasbarillot/open-blowfish',
        'Upstream': 'https://github.com/blackrock/blowfish',
        'Paper': 'https://arxiv.org/abs/2406.07990',
    },
    license='Apache License 2.0',
)
