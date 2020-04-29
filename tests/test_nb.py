
import subprocess
import tempfile

def _exec_notebook(path):
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=1000",
                "--output", fout.name, path]
        subprocess.check_call(args)

sol = "solutions_do_not_open/"


def test_0():
    _exec_notebook(sol+"Feature_Engineering_with_Scikit_Learn_solution.ipynb")


def test_1():
    _exec_notebook(sol+"Model_Evaluation_and_Dimensionality_Reduction_with_Scikit_Learn_solution.ipynb")


def test_2():
    _exec_notebook(sol+"Data_Manipulation_with_Pandas_solution.ipynb")


def test_3():
    _exec_notebook(sol+"Machine_Learning_with_Scikit_Learn_solution.ipynb")


def test_4():
    _exec_notebook(sol+"Regression_with_Scikit_Learn_solution.ipynb")


def test_5():
    _exec_notebook(sol+"Classification_with_Scikit_Learn_solution.ipynb")


def test_6():
    _exec_notebook(sol+"Pandas_Matplotlib_Seaborn_solution.ipynb")


def test_7():
    _exec_notebook(sol+"Neural_Networks_with_Keras_solution.ipynb")


def test_8():
    _exec_notebook(sol+"Clustering_with_Scikit_Learn_solution.ipynb")
