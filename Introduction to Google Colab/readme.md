
### Introduction to Google Collab
<b> Why Google Collab?</b>
- Provides Browser based Jupyter Notebook
- Ready to use
- GPU & TPUs
- Store Data with Google Drive
- Pre-installed Packages
---

##### Basics
- Go to `file > New Notebook` for a new notebook.
- All notebooks are saved in your google drive.
- To locate notebook in drive go to `file > Locate to Drive`

##### Important Shortcuts
- `Ctrl + M` followed by `B` to make a new code block.
- `Ctrl + M` followed by `M` for new Code -> Markdown
- `Ctrl + M` followed by `Y` for Markdown -> Code

##### Change Language
- To make a python 2 notebook go to this link [Python 2 for Collab](bit.ly/colabpy2)

##### GPU & TPU
- Go to `Runtime > Change Runtime Type` and select GPU or TPU from there.

##### Load data from Drive
- Use this code snippet to import data from drive
```python3
from google.colab import drive
drive.mount('gdrive')

my_file = open('gdrive/mydrive/...yourpathtofile/file.txt')
print(myfile.read())
```

#### Bash commands
- Use Bash commands by adding `!` before them.
- i.e `!pip install numpy` or `!ls` or `!dir1` etc.
