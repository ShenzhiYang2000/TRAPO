from setuptools import setup, find_packages

setup(
    name='trapo',
    version='0.0.0',
    description='TraPO: A Semi-Supervised Reinforcement Learning Framework for Boosting LLM Reasoning',
    author='Shenzhi Yang',
    packages=find_packages(include=['deepscaler',]),
    install_requires=[
        'google-cloud-aiplatform',
        'pylatexenc',
        'sentence_transformers',
        'tabulate',
        'math-verify[antlr4_9_3]==0.6.0',
        'flash_attn==2.7.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
