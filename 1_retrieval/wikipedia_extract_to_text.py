import requests
import mwparserfromhell

title = "Climate change mitigation"
endpoint = "https://en.wikipedia.org/w/api.php"
date_iso = "2014-09-02T07:44:59Z"

params = {
    "action": "query",
    "format": "json",
    "prop": "revisions",
    "titles": title,
    "rvlimit": 1,
    "rvprop": "ids|timestamp|content",
    "rvdir": "older",
    "rvstart": date_iso,
}

data = requests.get(endpoint, params=params).json()
page = next(iter(data["query"]["pages"].values()))

# Manages old and new Wikipedia revision format (pre or post 2020)
revision = page["revisions"][0]
wiki_markup = revision.get("*") or revision.get("slots", {}).get("main", {}).get("*", "")

# Convert wiki markup to plain text
wikicode = mwparserfromhell.parse(wiki_markup)
plain_text = wikicode.strip_code()

with open("wikipedia_article_revision.txt", "w", encoding="utf-8") as f:
    f.write(plain_text)

print(f"Downloaded and cleaned revision from {revision['timestamp']}")
