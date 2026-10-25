"""测试公共设置：所有请求都走 httpx.MockTransport，不能被系统或环境变量中的代理绕开。"""

import os

for _name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_name, None)
# Windows 上 httpx 还会读取注册表中的系统代理，NO_PROXY=* 让所有地址都直连
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
