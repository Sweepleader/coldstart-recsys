import torch
"""
2025年12月3日 加载VGGish模型
    本文件用于加载VGGish模型
    代码来源: https://github.com/harritaylor/torchvggish
"""
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
import urllib
url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# model.forward(filename)
print(model.forward(filename))