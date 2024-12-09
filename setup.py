from setuptools import setup, find_packages

# 动态读取 requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line and not line.startswith("#")]

# setup(
#     name="rna-tool",  # 包名
#     version="1.0.0",  # 版本号
#     py_modules=["RNA"],  # 指定单文件模块
#     packages=find_packages(),  # 如果有需要，可以自动发现包

# )

setup(
    name="rna-tool",
    version="1.0.0",
    author="Your Name",  # 作者
    author_email="your_email@example.com",  # 邮箱
    description="A tool for RNA folding and prediction",  # 描述
    packages=["rna_tool"],
    entry_points={
        "console_scripts": [
            "rna = rna_tool.RNA:main",  # main 函数来自 RNA.py
        ]
    },
    install_requires=parse_requirements("requirements.txt"),
)

