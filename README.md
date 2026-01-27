安装指令：

```sh
python -m venv .venv
source .venv/bin/activate

pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install --upgrade pip
pip install -e .
pip install custom/torch-2.5.0+cu121-cp311-cp311-linux_x86_64.whl
pip install casadi swanlab multiprocess pandas
```

这是一个在MetaDrive上的二次开发库，原项目的github网址为：<br />
[https://github.com/metadriverse/metadrive](https://github.com/metadriverse/metadrive)
