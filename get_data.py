import requests
import argparse


def bing_search(query,k):
    url = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'
    payload = {'q': query, 'offset': k}
    headers = {'Ocp-Apim-Subscription-Key': 'fc6e567891ae43238b40babb1fe0aa06'}
    r = requests.get(url, params=payload, headers=headers)
    return r.json()
ap = argparse.ArgumentParser()
ap.add_argument('-q', '--query', required=True,
	help='search query to search Bing Image API for')
ap.add_argument('-p', '--path', required=True,
	help='path to output directory of images')

args = vars(ap.parse_args())
path = args['path']
query = args['query']

k = 0
for i in range(100):
    j = bing_search(query,k)
    print(len(j['value']))
    for i in j['value']:
        r = requests.get(i['contentUrl'], allow_redirects=False)
        f = open(path + '/' + str(k) + '.' + query +'.jpg', 'wb')
        f.write(r.content)   
        f.close()
        k += 1

