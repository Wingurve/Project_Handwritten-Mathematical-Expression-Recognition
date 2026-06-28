from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.strip().startswith("#")
    ]

setup(
    name="formula-reader",
    version="1.0.0",
    description="Handwritten mathematical expression recognition with BTTR and DeepSeek LaTeX post-processing",
    author="Wingurve",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "formula-reader=app:app",
        ],
    },
)
