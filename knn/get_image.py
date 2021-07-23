import urllib.request
import os

dirname = "train_data"


def get_image(dirname, num_images):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    url = "http://cwcx.zju.edu.cn/WFManager/login.jsp/loginAction_getCheckCodeImg.action?s=742.1783236348542"

    for i in range(num_images):
        urllib.request.urlretrieve(url, os.path.join(dirname, '%s.jpg' % str(i)))


get_image("train_data", 100)
get_image("test_data", 10)

print("done")
