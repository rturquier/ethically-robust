#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare list of mailto URLs to send out survey

Run `open $(cat data/mailto.csv)` to open all exported URLs.
"""

# %% Imports
import pandas as pd
import urllib.parse
from pathlib import Path

# %% Define functions
def create_mailto_url(row, subject, template):
    body = template.format(first_name=row['first_name'])
    params = urllib.parse.urlencode({'subject': subject, 'body': body},
                                    quote_via=urllib.parse.quote)
    return f"mailto:{row['email']}?{params}"


def create_mailto_list(contact_df, email_subject, email_template):
    selected_recipients_df = contact_df.query('not exclude')
    
    result = (
        selected_recipients_df
        .assign(mailto_url=lambda x: x.apply(
            create_mailto_url,
            args=(email_subject, email_template),
            axis=1
        ))
        .mailto_url
    )
    
    return result


# %% Read data
contact_info = pd.read_csv("data/recipient_contact_information.csv")

# %% Set template
email_subject = "Population ethics expert survey"
email_body = Path("email.txt").read_text()

# %% 
mailto_list = create_mailto_list(contact_info, email_subject, email_body)
mailto_list.to_csv("data/mailto.csv", index=False, header=False)
