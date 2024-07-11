#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get list of potential experts for population ethics survey

Todo:
- tidy data
- exclude papers that only match the query because of punctuation like
  "...population, ethics..."
"""

import requests
import pandas as pd

# %%
endpoint = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    'query': '"population ethics"|"average utilitarianism"|' +
             '"total utilitarianism"|"repugnant conclusion"',
    'limit': "5",
    'fields': "authors,year,title,abstract,paperId,s2FieldsOfStudy," +
              "citationCount,externalIds,venue,publicationVenue," +
              "publicationTypes",
    'year': "2000-",
    'fieldsOfStudy': "Economics,Philosophy,Sociology,Political Science"
}
response = requests.get(endpoint, params)
response_df = pd.json_normalize(response.json()['data'])
