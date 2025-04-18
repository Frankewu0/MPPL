# Official implementation of Large Language Model Based Multiple Property Prompts Learning Framework for Molecular Property Prediction

## ⏳ Quick Start

### 1. Installation
```
   pip install -r requirements.txt
```
Furthermore, you also need to download the pre-trained weights of SCI-BERT.
For the molecule edit task, you should install MolSTM first.

### 2. Pretraining
```
   python3 -u moltex_train.py
```
To start the pre-training stage, you can run our code with above command.

### 3. Property Prediction on PharmaBench/MoleculeNet
```
   python3 -u ft_pha.py
   python3 -u ft_mol.py
```
To fine-tune ont the PharmaBench and MoleculeNet datasets and test on them, you can run our code with above commands.

### 4. Molecule Edit Task
After installing all packages, you can directly run the .ipynb files with Jupyter to run this task.
