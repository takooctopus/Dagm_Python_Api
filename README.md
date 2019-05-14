DAGM API

这个是为了方便于将DAGM 2007数据集进行处理和导入的API

在这个文件夹下输入

```bash
make install
```

会将标签生成在annotations文件夹下

images下放置数据集

但是需要先进行些处理

将训练和测试数据集分开

```
dagm
│   README.md
│   ...
│   foo.py
│
└───annotations
│
└───commom
│   
└───images
│ │   
│ └────TestImagesDir
│     │   
│     └───Class1Dir   
│     │  
│     └───Class2Dir
│
└───PythonAPI
```

嘛 大致就这样