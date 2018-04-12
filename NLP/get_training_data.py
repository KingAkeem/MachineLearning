import aiohttp
import asyncio
import re
import requests
import warnings

from bs4 import BeautifulSoup
from concurrent import futures
from requests.exceptions import HTTPError, ConnectionError

warnings.filterwarnings('ignore', category=UserWarning, module='bs4')

def get_urls(soup):
    urls = list()
    for tag in soup.find_all('a'):
        try:
            if tag['href'].startswith('/siteinfo/'):
                url = tag['href'].replace('/siteinfo/', '')
                urls.append('http://www.'+url)
        except KeyError:
            pass

    return urls



def clean_html(html_docs):

    if type(html_docs) != str:
        clean_docs = list()
        for i, html in enumerate(html_docs):
            if i % 2 == 0:
                print("Preprocessing Page {i} of {t}".format(i=i+1,
                                                             t=len(html_docs)))
            # Removing all nonalphanumeric characters
            letters_only = re.sub("[^a-zA-Z]", " ", str(html))
            # Turning document into list of words
            letters_only = letters_only.replace(",", " ")
            words = letters_only.lower().split()
            # Appending cleaned document to list of cleaned documents
            clean_docs.append(" ".join(words))

        return clean_docs
    else:
        # Removing all nonalphanumeric characters
        letters_only = re.sub("[^a-zA-Z]", " ", html_docs)
        # Turning document into lower case words
        words = letters_only.lower()
        return words


async def fetch(session, url):
    async with aiohttp.Timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def fetch_all(session, urls, loop):
    results = await asyncio.gather(*[fetch(session, url) for url in urls],
                                   return_exceptions=True)

    for idx, url in enumerate(urls):
        print('{}: {}'.format(url, 'ERR' if isinstance(results[idx], Exception) else 'OK'))

    return results

sp_hub = requests.get('https://www.alexa.com/topsites/category/Sports').text
news_hub = requests.get('https://www.alexa.com/topsites/category/News').text
urls = get_urls(BeautifulSoup(sp_hub)) + get_urls(BeautifulSoup(news_hub))

loop = asyncio.get_event_loop()
with aiohttp.ClientSession(loop=loop) as session:
    results = loop.run_until_complete(fetch_all(session, urls, loop))

final_results = list()
for i, result in enumerate(results):
    try:
        soup = BeautifulSoup(result).get_text()
        print('Preprocessing Page {p_num} of {t_num}\n').format(p_num=i+1,
                                                                t_num=len(results))
        final_results.append(clean_html(soup))
    except TypeError:
        continue

with open('../datasets/training_data.csv', 'w+') as training_data:
    training_data.write('id,class,name,content\n')
    for i, result in enumerate(final_results):
        training_data.write('{num},{cl},{name},{text}\n'.format(
                            num=i+1, cl=0, name=urls[i], text=result))
