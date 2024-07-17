#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Get list of potential experts for population ethics survey

Todo:
- exclude "repugnant conclusions" (with an s), which are sometimes not
  about population ethics
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
              "externalIds,venue,publicationVenue,url",
    'year': "2000-",
    'fieldsOfStudy': "Economics,Philosophy,Sociology,Political Science"
}
response = requests.get(endpoint, params)
response_df = pd.json_normalize(response.json()['data'])

# %% Tidy data
relevant_columns = ['name', 'year', 'title', 'venue', 'publicationVenue',
                    'abstract', 's2FieldsOfStudy', "url", 'authorId',
                    'paperId']

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

all_authors = authors_df.name.drop_duplicates()
first_author_selection = (
    authors_df
    .query(query_condition)
    .name
    .drop_duplicates()
)

# %% Refine selection using missing abstracts
# For some papers, "due to legal reasons", the SemanticScholar API doesn't
# return the abstract, even though they are online. For some of these papers,
# we need to retrieve the abstract manually in order to check if it contains
# the precise search terms of the query (eg. "population ethics" and not
# "population. Ethics").

abstract_is_empty = "~ abstract.str.contains('\w').astype('bool')"
empty_abstracts_df = (
    authors_df
    .query(abstract_is_empty)
    # no need to fetch abstracts for authors we already selected
    .query("~ name.isin(@first_author_selection)") 
    .drop_duplicates("paperId")
)

# Export table and manually copy abstracts from the SemanticScholar website,
# following the links in the "url" column. Takes about 15 minutes.
# empty_abstracts_df.to_csv("data/empty_abstracts_df.csv")

manually_fetched_abstracts_df = (
    pd.read_csv(
        "data/manually_fetched_abstracts_df.csv", sep=";", index_col=0
    )
    [['abstract', 'paperId']]
)

all_abstracts_df = (
    pd.merge(
        authors_df,
        manually_fetched_abstracts_df,
        on="paperId",
        how="left"
    )
    .assign(abstract = lambda df: df.abstract_y.combine_first(df.abstract_x))
)    

selected_authors_df = all_abstracts_df.query(query_condition)
