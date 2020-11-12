from pathlib import Path
from enum import Enum

from nbclient.client import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError
import nbformat
import jupytext

class ExecutionStatus(Enum):
    OK = 1
    ERROR = 2
    TIMEOUT = 3

def path_exists(path):
    if not path.exists():
        print(str(path.resolve()) + " does not exists")
        return False
    return True


def get_path(*args):
    path = Path(*args)
    if path_exists(path):
        return path
    else:
        return None

def read_notebook(src_dir, src_name):
    """
    Reads the notebook file inferring the format from the extension.

    :param src_dir:
    :param src_name:
    :return:
    """
    src_path = get_path(src_dir, src_name)
    if src_path is None:
        return None;

    if src_path.suffix == '.ipynb':
        ntb = nbformat.read(str(src_path.resolve()), as_version=4)
    elif src_path.suffix == '.Rmd':
        ntb = jupytext.read(src_path, as_version=4)

    return ntb



def run_notebook(ntb, path, timeout):
    resources = {'metadata': {}}
    resources['metadata']['path'] = path

    client = NotebookClient(ntb, timeout=timeout, resources=resources)
    try:
        client.reset_execution_trackers()
        out = (ExecutionStatus.OK, client.execute())
    except CellExecutionError as err:
        msg = "Error executing notebook `{:s}'".format(str(path))
        out = (ExecutionStatus.ERROR, (msg, str(err)))
    except CellTimeoutError as err:
        msg = "Timout executing notebook `{:s}'".format(str(path))
        out = (ExecutionStatus.TIMEOUT, (msg, str(err)))

    return out


def run_notebook_file(path, timeout):
    ntb = read_notebook(path.parent, path.name)
    return run_notebook(ntb, path, timeout)
