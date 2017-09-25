# Creating datasets via web scraping
# Example uses craft beer list from craftcans.com
# Based on Jean-Nicholas Hould's Kaggle post
import urllib
from bs4 import BeautifulSoup
import pandas as pd
import re

# Determines if a table_row is a beer entry
def is_beer_entry(table_row):
    row_cells = table_row.findAll("td")
    beer_id = get_beer_id(row_cells[0].text)
    return ( len(row_cells) == 8 and beer_id )

# Return the beer entry numerical identifier from the "Entry" column.
def get_beer_id(cell_value):
    r = re.match("^(\d{1,4})\.$", cell_value)
    if r and len(r.groups()) == 1:
        beer_id = r.group(1)
        return int(beer_id)
    else:
        return None

# Store scraped data as JSON
def get_all_beers(html_soup):
    beers = []
    all_rows_in_html_page = html_soup.findAll("tr")
    for table_row in all_rows_in_html_page:
        if is_beer_entry(table_row):
            row_cells = table_row.findAll("td")
            beer_entry = {
                "id": get_beer_id(row_cells[0].text),
                "name": row_cells[1].text,
                "brewery_name": row_cells[2].text,
                "brewery_location": row_cells[3].text,
                "style": row_cells[4].text,
                "size": row_cells[5].text,
                "abv": row_cells[6].text,
                "ibu": row_cells[7].text
            }
            beers.append(beer_entry)
    return beers

def tidy_data(df):
    # Clean breweries data and set up index
    breweries = df[["brewery_location", "brewery_name"]]
    breweries = breweries.drop_duplicates().reset_index(drop=True)
    breweries["id"] = breweries.index

    # clean beer data
    beers = pd.merge(df,
                 breweries,
                 left_on=["brewery_name", "brewery_location"],
                 right_on=["brewery_name", "brewery_location"],
                 sort=True,
                 suffixes=('_beer', '_brewery'))
    beers = beers[["abv", "ibu", "id_beer",
                   "name", "size", "style", "id_brewery"]]
    beers_columns_rename = {
        "id_beer": "id",
        "id_brewery": "brewery_id"
    }
    beers.rename(inplace=True, columns=beers_columns_rename)

    # Convert ABV/ IBU values to float/ int
    def string_pct_to_float(value):
        stripped = str(value).strip('%')
        try:
            return float(stripped)/100
        except ValueError:
            return None
    beers["abv"] = beers["abv"].apply(string_pct_to_float)

    def string_to_int(value):
        try:
            return int(value)
        except ValueError:
            return None
    beers["ibu"] = beers["ibu"].apply(string_to_int)

    # Parse string for oz. and convert to float
    def extract_ounces(value):
        stripped = value.strip("oz")
        match = re.match("(\d{1,2}\.*\d*)", value)
        if match:
            return float(match.group(0))
        else:
            return None

    beers["ounces"] = beers["size"].apply(extract_ounces)
    del beers["size"]

html = urllib.urlopen("http://craftcans.com/db.php?search=all&sort=beerid&ord=desc&view=text")
html_soup = BeautifulSoup(html, 'html.parser')
beers_list = get_all_beers(html_soup)
df = pd.DataFrame(beers_list)
tidy_data(df)
print df.head(10)
