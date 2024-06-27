def checking_dependencies():
    import os
    print("Checking System...", end = '                        \r')
    try:
        print('Checking numpy...', end = '                        \r')
        import numpy as np
        print('numpy found!', end = '                        \r')
    except ModuleNotFoundError as e:
        print('numpy not found!', end = '                        \r')
        print('installing numpy...', end = '                        \r')
        os.system("pip install numpy")
        print('numpy installed!', end = '                        \r')

    try:
        print('Checking pandas...', end = '                        \r')
        import pandas as pd
        print('pandas found!', end = '                        \r')
    except ModuleNotFoundError as e:
        print('pandas not found!', end = '                        \r')
        print('installing pandas...', end = '                        \r')
        os.system("pip install pandas")
        print('pandas installed!', end = '                        \r')

    try:
        print('Checking tensorflow...', end = '                        \r')
        import tensorflow as tf
        print('tensorflow found!', end = '                        \r')
    except ModuleNotFoundError as e:
        print('tesorflow not found!', end = '                        \r')
        print('installing tensorflow...', end = '                        \r')
        os.system("pip install tensorflow==2.16.1")
        print('tensorflow installed!', end = '                        \r')


    try:
        print('Checking sklearn...', end = '                        \r')
        import sklearn
        print('sklearn found!', end = '                        \r')
    except ModuleNotFoundError as e:
        print('sklearn not found!', end = '                        \r')
        print('installing sklearn...', end = '                        \r')
        os.system("pip install scikit-learn")
        print('sklearn installed!', end = '                        \r')

    print('Dependencies Checking Completed!', end = '                             \r')