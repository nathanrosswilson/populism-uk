# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pandas as pd

hansardurls = []

end_date = "2019-12-31" # yyyy-mm-dd
search_term = "debate%3Aimmigration" # subject%3ASEARCHTEXT
start_date = "2000-01-01" # yyyy-mm-dd
page_max = 16


for i in trange(page_max, desc="Getting URL"):
    url ='https://hansard.parliament.uk/search/Debates?\
            endDate={}\
            &house=Commons\
            &searchTerm={}\
            &startDate={}\
            &page={}\
            &partial=true'\
            .format(end_date, search_term, start_date, i+1)
    rall = requests.get(url)
    r = rall.content
    soup = BeautifulSoup(r, "lxml")
    titles = soup.find_all('a', class_="no-underline")
    for t in tqdm(titles, leave=False):
        hurl = 'https://hansard.parliament.uk'+t['href']
        hansardurls.append(hurl)
 
dates = []
contributors = []
contributions = []

for h in trange(len(hansardurls), desc="Getting Data"):
    rall = requests.get(hansardurls[h])
    r = rall.content
    soup = BeautifulSoup(r, "lxml")
    newcontributors = soup.find_all('h2', class_="memberLink")
    contributors += newcontributors
    contributions += soup.find_all('div', class_="col-md-9 contribution content-container")
    for i in trange(len(newcontributors), leave=False):
        dates.append(soup.find('div', class_="col-xs-12 debate-date").text)

cparties = []
for c in trange(len(contributors), desc="Getting Parties"):
    link = contributors[c].find('a')
    try:
        memberURL = 'https://hansard.parliament.uk'+link['href']
        
        rall = requests.get(memberURL)
        r = rall.content
        soup = BeautifulSoup(r, "lxml")
        party = soup.find('div', class_ = "member-details").text.split("\n")[1]
        cparties.append(party)
    except:
        
        cparties.append('.')
        
# Each individual contribution is in the order and would be printed like below

for i in trange(len(contributors), desc="Stripping"):
    contributors[i] = contributors[i].text.strip()
    cparties[i] = cparties[i].strip()
    dates[i] = dates[i].strip()
    contributions[i] = contributions[i].text.split("\n")[1].strip()
    
name = np.array(contributors)
party = np.array(cparties)
date = np.array(dates)
text = np.array(contributions)

d1 = {'name': name, 'party': party, 'date': date, 'text': text}

df = pd.DataFrame(data=d1)
df.to_csv('data.csv', index=False, encoding='utf-8-sig')

# with open('data.txt', 'w', encoding='utf8') as outfile:
#     json.dump(data, outfile, indent=4, ensure_ascii=False)

# print(contributors[i].text.strip())
# print(cparties[i].strip())
# print(contributions[i].text.split("\n")[1].strip())
