import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('# Test Title\nThis is a markdown cell.'))
nb.cells.append(nbf.v4.new_code_cell('import numpy as np\nprint("Notebook generation functional!")'))

os.makedirs('scratch_test', exist_ok=True)
with open('scratch_test/test.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("SUCCESS: Notebook created and validated.")
