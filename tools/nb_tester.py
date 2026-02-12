
import nbformat
from nbclient import NotebookClient
import sys
import os

def run_notebook(nb_path, output_path):
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(nb, timeout=600, kernel_name='python3')
    try:
        client.execute()
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        return True, None
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python nb_tester.py <input_nb> <output_nb>")
        sys.exit(1)
    
    success, error = run_notebook(sys.argv[1], sys.argv[2])
    if success:
        print("Notebook executed successfully.")
    else:
        print(f"Notebook execution failed: {error}")
        sys.exit(1)
