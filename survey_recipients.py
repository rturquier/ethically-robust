#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get list of potential experts for population ethics survey

Todo:
- exclude papers that only match the query because of punctuation like
  "...population, ethics..."
"""

# %% Imports
import requests
import pandas as pd

# %% Retrieve data
endpoint = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
params = {
    'query': '"population ethics"|"average utilitarianism"|' +
             '"total utilitarianism"|"repugnant conclusion"',
    'limit': "5",
    'fields': "authors,year,title,abstract,paperId,s2FieldsOfStudy," +
              "externalIds,venue,publicationVenue",
    'year': "2000-",
    'fieldsOfStudy': "Economics,Philosophy,Sociology,Political Science"
}
response = requests.get(endpoint, params)
response_df = pd.json_normalize(response.json()['data'])

# %% Tidy data
relevant_columns = ['name', 'year', 'title', 'venue', 'publicationVenue',
                    'abstract', 's2FieldsOfStudy', 'authorId', 'paperId']

long_df = response_df.explode('authors', ignore_index=True)
authors_name_and_id = long_df.authors.pipe(pd.json_normalize)
authors_df = (
  long_df
  .drop('authors', axis=1)
  .join(authors_name_and_id)
  .reindex(columns=relevant_columns)
  # convert s2FieldsOfStudy to str – otherwise unable to sort_values
  .assign(s2FieldsOfStudy = lambda x: x.s2FieldsOfStudy.astype(str))
  .sort_values(['name', 'year'])
  .drop_duplicates()
  .reset_index(drop=True)
)

# %%
# Exclude papers that only match the query because of punctuation like
# "...population. Ethics...".

keyword_pattern = (
    "(population ethics)|(average utilitarianism)|"
    + "(total utilitarianism)|(repugnant conclusion)"
)
query_condition = (
    "title.str.contains(@keyword_pattern, case=False)"
    + "or abstract.str.contains(@keyword_pattern, case=False)"
)

selected_authors_df = authors_df.query(query_condition)

all_authors = authors_df.name.drop_duplicates()
selected_authors = selected_authors_df.name.drop_duplicates()
excluded_authors = (
  authors_df[~authors_df.name.isin(selected_authors)]
  .name
  .drop_duplicates()
)

assert len(all_authors) == len(selected_authors) + len(excluded_authors)
