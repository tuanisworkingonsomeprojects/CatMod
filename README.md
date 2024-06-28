# CatMod (Categorical classification Model)

## Short introduction
- In a modern word that required fast robust and simple development process with Machine Learning, AI and Deep Learning, there are countless of projects require Natural Language Processing (NLP) classification problems such as <i>Commodity Classification, Company Type Classification, Food Type classification, etc.</i><br>
More and more people want to train, test and deploy NLP classification model without having to know the background of advanced in programming and AI knowledge.
- This Framework will allow everyone to train, test and save and load their own model and deploy it wherever they want with some simple lines of code.


## Virtual Enviroment / Dependencies
- It is recommended to create a vitual enviroment for your project when using `CatMod` as it will download and install packages and dependencies that might conflict with your dependencies on your machine.
- If you don't mind about the version of the libraries listed in the `requirements.txt` you can leave it as it is.

## How to use
- Please clone or <a href='https://github.com/tuanisworkingonsomeprojects/CatMod/archive/refs/heads/main.zip'>download</a> this repository to your project folder, please assure that `catmod.py` file stay in the same folder of your python main file <i>(The file in which you want to import `CatMod`)</i>

e.g.
```
ProjectFolder
|---main.py
|---catmod.py
|---utils
|   |----...
|   |
...
```
- Download <a href='https://www.kaggle.com/datasets/watts2/glove6b50dtxt'>GloVe Embedded Vectors File</a> to the desired foler.


- Import `CatMod` in your python file.
```python
from catmod import CatMod
```

- Instanciate a new instance with a <a href='https://www.kaggle.com/datasets/watts2/glove6b50dtxt'>GloVe Embedded Vectors File</a>.
```python
cat = CatMod('[your_file_path]')
```
e.g.
```python
file_path = 'C:/User/Desktop/glove.6B.50d.txt'
cat = CatMod(file_path)
```

```python
file_path = 'Machintosh HD/Users/yourName/Desktop/glove.6B.50d.txt'
cat = CatMode(file_path)
```

### Training Process
This Framework will allow you to input a <i>.csv</i> file with many columns but you have to specify 2 columns corresponding to values (X) and targets (Y).<br>

Let's say you have a csv file `product.csv` with columns look like this<br>

| company name | product name | category |
|:------------:|:------------:|:--------:|
|...           |...           |...       |


- You can use 1 out of 2 ways to load the csv file and load the pre-defined model into the instance.

```python
cat.load_csv('[your_csv_file_path]', '[X_column_name]', '[Y_column_name]')
cat.load_model()
```
e.g.
```python
cat.load_csv('product.csv', 'product name', 'category')
cat.load_model()
```

OR 

```python
cat.load_model('[your_csv_file_path]', '[X_column_name]', '[Y_column_name]')
```
e.g.
```python
cat.load_model('product.csv', 'product name', 'category')
```

Then we just do one more easy step:
```python
cat.train([number_of_iterations])
```
e.g.
```python
cat.train(10)
```
<i>If the number of iterations is not specified the number of iteration is 50.</i>
e.g.
```python
cat.train() # 50 iterations
```


