import os
import sys

from pydagmtools import dagm
from pydagmtools import dagmjson

print("\033[0;33m " + "现在位置:{}/{}/.{}".format(os.getcwd(), os.path.basename(__file__),
                                                      sys._getframe().f_code.co_name) + "\033[0m")


# j = dagmjson.DAGMjson("traindagm2007")
j = dagmjson.DAGMjson("testdagm2007")
# j.img_to_json()
# j.label_to_json()
# j.test_json("traindagm2007")
j.test_json("testdagm2007")
# a = dagm.DAGM('../annotations/traindagm2007.json')