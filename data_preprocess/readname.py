#读取当前目录所有形式为xxx.jpg的图片名称，保存在txt文件中

import os
import os.path as osp

class ReadImageName():
    def __init__(self):
        self.path = 'image'

    def readname(self):
        # MODE = ['train','test']
        MODE = ['train','test']
        for mode in MODE:
            filenames = os.listdir(osp.join(mode,self.path))
            filenames.sort()
            filelist = []

            for item in filenames:
                if item.endswith('.jpg'):
                    # itemname = os.path.join(self.path, item)
                    # itemname = itemname[-11:]
                    filelist.append(item)
            
            with open(mode+".txt", "w") as f:
            # with open("val.txt", "w") as f:
                for item in filelist:
                    f.write(item + "\n")
                    # f.write('/image/'+item + "\n")


if __name__ == '__main__':
    log = ReadImageName()
    log.readname()
