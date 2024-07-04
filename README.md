# CatMod (Categorical classification Model)

## Short introduction
- In a modern word that required fast robust and simple development process with Machine Learning, AI and Deep Learning, there are countless of projects require Natural Language Processing (NLP) classification problems such as <i>Commodity Classification, Company Type Classification, Food Type classification, etc.</i><br>
More and more people want to train, test and deploy NLP classification model without having to know the background of advanced in programming and AI knowledge.
- This Framework will allow everyone to train, test, save and load their own model and deploy it wherever they want with some simple lines of code.


## Virtual Enviroment / Dependencies
- It is recommended to create a virtual environment for your project when using `CatMod` as it will download and install packages and dependencies that might conflict with your dependencies on your machine.
- If you don't mind about the version of the libraries listed in the `requirements.txt` you can leave it as it is.

## How to use
- You can you pip install to download the project on your computer.
```
pip install cat-mod
```
- Import `CatMod` in your python file.
```python
from cat_mod import CatMod
```


- Download <a href='https://www.kaggle.com/datasets/watts2/glove6b50dtxt'>GloVe Embedded Vectors File</a> to the desired folder.



- Instanciate a new instance with a <a href='https://www.kaggle.com/datasets/watts2/glove6b50dtxt'>GloVe Embedded Vectors File</a>.
```python
cat = CatMod('[your_GloVe_file_path]')
```
e.g.
```python
file_path = 'C:/User/Desktop/glove.6B.50d.txt'
cat = CatMod(glove_file = file_path)
```

```python
file_path = 'Machintosh HD/Users/yourName/Desktop/glove.6B.50d.txt'
cat = CatMode(glove_file = file_path)
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
THE <b>RECOMMENDED</b> WAY
```python
cat.load_model('[your_csv_file_path]', '[X_column_name]', '[Y_column_name]')
```
e.g.
```python
cat.load_model('product.csv', 'product name', 'category')
```

We can also specify how many LSTM layers you want by adding the corresponding parameter.
```python
cat.load_model('product.csv', 'product name', 'category', num_of_LSTM = 4)
```





Then we just do one more easy step:
```python
cat.train([number_of_iterations])
```
e.g.
```python
cat.train(10)
```
<i>If the number of iterations is not specified, the number of iteration is 50.</i>
e.g.
```python
cat.train() # 50 iterations
```

### Save Weights
After training you can save your model on your local machine by using `.save_weights([name])` method. <i>(No file name suffix is needed)</i>

```python
cat.save_weights('my_model')
```

If the model is saved successfully we will see the folder appear in the same folder of your project
```
ProjectFolder
|---main.py
|---my_model
|   |---...
|   |
|
...
```

### Load Pre-Trained Model
When we have saved the training file, we can reuse it in the future by just loading it back to a new instance.<br>
There are 2 ways of doing it.

The <b>RECOMMENDED</b> way:
```python
from cat_mod import CatMod

new_cat = CatMod(load_mode = True, load_file = 'my_model')
```

The other way:
```python
from cat_mod import CatMod

new_cat = CatMod(glove_file_path = [the_GloVe_file_path_but_it_must_have_the_same_dimension_with_the_pre_trained_model])

new_cat.load_weights('my_model')
```

### Prediction
Prediction the the most easiest and provide many customization so that everyone can predict and export the predict result in .pd, .csv, .xlsx at their own need.
e.g.
```python
X = df['X']
new_cat.predict(X, to_csv = True, file_name = 'my_prediction')
```
The result will export out the csv file that have both column X and Y together.

