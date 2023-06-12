## 自建数据集

### 1.爬网易新闻与腾讯新闻

一开始我想新闻的中文语句质量应该可以

```py
import json
import os
import requests
from bs4 import BeautifulSoup
import urllib.request
import re
import io
import sys
from urllib.parse import quote
import codecs

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# 函数功能：得到网易新闻
def get163news():
    url = "http://news.163.com/rank/"
    wbdata = requests.get(url).text
    soup = BeautifulSoup(wbdata, 'lxml')
    news_titles = soup.select("td a")
    comment = soup.select("td.cBlue")

    start = 3
    i = 30

    for strat in range(30, 500):
        for n in range(start, start + 29):
            link = news_titles[n].get("href")

            try:
                neteasedata = urllib.request.urlopen(link).read()
                neteasedata2 = neteasedata.decode("gbk", "ignore")
                soup = BeautifulSoup(neteasedata2, "html.parser")
                content = soup.select('p')
                title = soup.select('title')
                time = soup.select('div.post_time_source')
                author = soup.select('div.post_time_source > a.ne_article_source')

                if len(time) != 0:
                    fo = open("PaWangYi.txt", "w+")

                    if len(title) != 0:
                        fo.writelines(" " + title[0].get_text().strip() + "\n")

                    fo.writelines("时间：" + time[0].get_text().strip() + "\n")
                    fo.writelines("评论数: " + comment[i].get_text() + "\n")

                    if len(author) != 0:
                        fo.writelines(author[0].get_text() + '\n')

                    for m in range(2, len(content)):
                        try:
                            con = content[m].get_text().strip()
                            if len(con) != 0:
                                fo.writelines("\n" + con)
                        except Exception as err:
                            print(err)
                        m += 1

                    fo.close()
                else:
                    i -= 1

            except Exception as err:
                print(err)

            i += 1
            n += 1

        start += 60
        n = start
        i = start

        if start > 270:
            break

# 函数功能：得到腾讯新闻首页所有新闻链接
def getQQurl():
    url = "http://news.qq.com/"
    wbdata = requests.get(url).text
    soup = BeautifulSoup(wbdata, 'lxml')
    news_titles = soup.select("div.text > em.f14 > a.linkto")
    fo = open("PaQQ.txt", "w+")

    for n in news_titles:
        title = n.get_text()
        link = n.get("href")
        fo.writelines(link + "\n")

    fo.close()

# 函数功能：根据获取的链接依次爬取新闻正文并保存到本地
def getqqtext():
    i = 1
    qqf = open("Fuda.txt", "r")

    for line in qqf:
        try:
            url = line
            wbdata = requests.get(url).text
            soup = BeautifulSoup(wbdata, 'lxml')
            title = soup.select("h1")
            time = soup.select("#Main-Article-QQ > div.qq_article > div.qq_articleFrist > div.qq_articleFristInfo > div.a_Info > span.a_time")
            author = soup.select("#Main-Article-QQ > div.qq_article > div.qq_articleFrist > div.qq_articleFristInfo > div.a_Info > span.a_source")
            content = soup.select("#Cnt-Main-Article-QQ")

            if len(time) != 0:
                fo = open("PaQQ.txt", "w+")

                if len(title) != 0:
                    fo.writelines(" " + title[0].get_text().strip() + "\n")

                for p in content:
                    for m in p.select('p'):
                        try:
                            con = m.get_text().strip()
                            if len(con) != 0:
                                fo.writelines("\n" + con)
                        except Exception as err:
                            print(err)

                fo.close()

            else:
                i -= 1

        except Exception as err:
            print(err)

        i += 1

    qqf.close()

# 主函数
if __name__ == "__main__":
    get163news()
    getQQurl()
    getqqtext()
```

在代码运行过程中，我遇到如下问题：

首先，网络极易波动，不方便获取数据并加载数据集

其次，我发现中间偶有乱码文字，我能力有限，难以清洗干净

最后，新闻内容大多重复且标题有很大的误导性（标题党什么的）

于是乎我打算使用专业数据集







### 2.ChineseNLPCorpus中文数据集

在网上冲浪中，我找到了这个，这是一个Github项目：该项目收集了一批中文自然语言处理数据集的相关链接，可以用来练手，该项目链接如下：

https://github.com/InsaneLife/ChineseNLPCorpus

用text打开后如下：

```py
#{"id": "263", "url": "https://zh.wikipedia.org/wiki?curid=263", "title": "中国", "text": "中国\n\n中国是位于东亚的国家或地理区域，此名称最早见于西周，用来指以洛阳盆地为中心的中原地区，与四夷相对，之后逐渐用来指称从夏朝起延续传承至今的各政权。其疆域随著历史演变而有所增减，但大多不脱以中原王朝根基所在的汉地九州为中心。民族构成上以汉族为主体，文化上透过历代王朝政权与周边各民族政权的交流与征战，而融入不少周边民族的文化。现今国际上广泛承认代表中国的政权是中华人民共和国。\n中国文明是世界上最早的文明之一。 新石器时期，中原地区开始出现聚落组织；公元前27世纪左右出现方国，以共主为首的制度；前20世纪开始，古代中国进入世袭的封建皇朝阶段；公元前2世纪，秦灭六国，完成中国第一次大一统。此后几千年来，中国的政治制度以半传统的夏代
```

内容来自于wiki百科，质量很高且只需稍微洗稿就行了：（洗稿代码）

```py
import re

def filter_chinese_and_space(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    filtered_text = re.sub(r'[^\u4e00-\u9fa5\s]', '', text)

    with open(input_file, 'w', encoding='utf-8') as file:
        file.write(filtered_text)

# 洗稿喽
for i in range(100):
    file_name = 'wiki_{:02d}.txt'.format(i)    # 输入文件名

    filter_chinese_and_space(file_name)
```

这数据集有恐怖的1.6个G这里我只使用其中的1/6进行训练。在GitHub中我只展示其中一小部分



