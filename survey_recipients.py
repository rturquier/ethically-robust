#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get list of recipients of population ethics survey


Todo:
- get authors of citations
- get citations and references titles as well
- find a way to do this for the handbook
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup



def split_name(name):
    """
    Split name into first and last name, removing middle initials
    """
    parts = name.split()
    if len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 3: 
        return parts[0], parts[2]  
    else:
        return parts[0], " ".join(parts[2:])


def get_arrhenius_authors():
    """
    Get list of 29 authors of the Oxford Handbook of Population Ethics
    """
    
    base_url = "https://academic.oup.com/edited-volume/41281/chapter/351599130"
    web_archive_url = "https://web.archive.org/web/20240710140222/"
    url = web_archive_url + base_url
    headers = {'User-Agent': "Mozilla 5.0 Windows"}
    arrhenius_request = requests.get(url, headers=headers)
    arrhenius_soup = BeautifulSoup(arrhenius_request.content, 'html.parser')
    
    arrhenius_author_elements = arrhenius_soup.find_all('strong')
    arrhenius_authors = [author.text for author in arrhenius_author_elements]
    arrhenius_first_last = [split_name(name) for name in arrhenius_authors]
    arrhenius_authors_df = pd.DataFrame(arrhenius_first_last,
                                        columns=['given', 'family'])
    return arrhenius_authors_df


def get_semantic_scholar_metadata(identifier):
    url = f"https://api.semanticscholar.org/v1/paper/{identifier}"
    request = requests.get(url)
    metadata = request.json()
    return metadata


def get_semantic_scholar_authors(identifier):
    metadata = get_semantic_scholar_metadata(identifier)
    authors = metadata['authors']
    authors_df = pd.DataFrame(authors)[["name"]]
    authors_df[['given', 'last']] = (
        authors_df['name'].apply(lambda x: pd.Series(split_name(x)))
    )
    authors_df.drop(columns=['name'], inplace=True)
    
    return authors_df



get_semantic_scholar_authors("10.1017/S095382082100011X")

metadata = get_semantic_scholar_metadata("10.1093/oxfordhb/9780190907686.001.0001")

def get_authors_of_references(identifier):
    metadata = get_semantic_scholar_metadata(identifier)
    
    author_names = [author['name']
                    for ref in metadata['references']
                    for author in ref['authors']]
    author_names_split = [split_name(name) for name in author_names]
    author_names_df = (
        pd.DataFrame(author_names_split, columns=["given", "family"])
        .drop_duplicates("family")
        .reset_index(drop=True)
    )
    return author_names_df



metadata['references']

get_authors_of_references("10.1017/S095382082100011X")
get_authors_of_references("10.1093/oxfordhb/9780190907686.001.0001")

pd.DataFrame(references)