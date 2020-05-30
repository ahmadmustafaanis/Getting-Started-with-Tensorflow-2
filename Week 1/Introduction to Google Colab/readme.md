# Getting Started with Tensorflow 2
## Week 1

### <u> Why Tensorflow </u>
- Famous
- Backed by active community
- Can make any Neural Network
- A lot of researchers use it
- Automates a lot of work

### <u> Whats new in Tensorflow 2</u>
- Eager Execution by Default
- tensorflow.keras as default high level API
- API cleanup
- Easy to use
---
Tensorflow 1 was meant for researchers only, Tensorflow 2 is for general software developers.

----
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
