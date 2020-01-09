import os

models = {name[:-3]: name for name in os.listdir(os.path.dirname(os.path.abspath(__file__)))[1:]}
# print(models)
