import os

from pydagmtools import dagm
from pydagmtools import dagmjson

print("当前cwd: "+os.getcwd())
print("当前dir: "+os.path.abspath(os.path.dirname(__file__)))


j = dagmjson.DAGMjson("testdagm2007")
j.imgToJson()

a = dagm.DAGM('../annotations/traindagm2007.json')
