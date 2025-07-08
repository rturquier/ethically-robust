#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare list of mailto URLs to send out survey

To open all exported URLs, run:
```sh
for line in $(cat mailto.csv) ; do open "${line}" ; done
```
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


def create_mailto_list(
    contact_df,
    email_subject,
    email_template,
    email_type="initial"
):
    selected_recipients_df = contact_df.query('not exclude')
    if email_type == "follow-up":
        selected_recipients_df = (
            selected_recipients_df
            .fillna(False)
            .infer_objects()
            .query('remind')
        )
    
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


# %% Change to email_type to `initial` to generate initial emails
email_type = "follow-up"  

# %% Read data
contact_info = pd.read_csv("data/recipient_contact_information.csv")

# %% Set template
email_subject = "Population ethics expert survey"
email_body = Path("email.txt").read_text()

if email_type == "follow-up":
    email_subject = "Population ethics expert survey — Follow-up"
    email_body = Path("follow-up.txt").read_text()

# %% 
mailto_list = create_mailto_list(contact_info,
                                 email_subject,
                                 email_body,
                                 email_type)
mailto_list.to_csv("data/mailto.csv", index=False, header=False)
