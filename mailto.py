#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare list of mailto URLs to send out survey

Run `open $(cat data/mailto.csv)` to open all exported URLs.
"""

# %% Imports
import pandas as pd
import urllib.parse

# %% Define functions
def create_mailto_url(row, subject, template):
    body = template.format(first_name=row['first_name'])
    params = urllib.parse.urlencode({'subject': subject, 'body': body},
                                    quote_via=urllib.parse.quote)
    return f"mailto:{row['email']}?{params}"


def create_mailto_list(df, email_subject, email_template):
    filtered_df = df.query('not exclude')
    
    result = (
        filtered_df
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

email_body = (
"""Dear {first_name},

We are contacting you as part of a select group of academics with expertise in population ethics. The objective is to elicit recommendations on fundamental issues of population ethics, and to inform long-term public investment decision-making.

We would be most grateful if you could find the time to complete the short survey appended in the link below.

https://remi.turquier.fr/ethically-robust/population-ethics-survey

Your response will be anonymous. You can find our privacy policy in the attached document.
Please only submit the form once, and do not share it with others.

Do not hesitate to contact us if you have any question.

Many thanks in advance for your time and cooperation,
Stéphane Zuber and Rémi Turquier
CES, Paris 1 University, and PSE



"""
)

# %% 
mailto_list = create_mailto_list(contact_info, email_subject, email_body)
mailto_list.head().to_csv("data/mailto.csv", index=False, header=False)
