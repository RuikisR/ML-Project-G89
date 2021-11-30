# ML Group Project - Reddit Analyser
By Col und Riv

## Usage
- Run using "make run"
- To add a new package, append its pypi name to the "requirements.txt" file. "make freeze" should be run to ensure all dependencies are also added to the req file
- You can check currently installed packages using "make list"
- "make clean" will delete the virtual environment in case of mess ups

## Accessing the Reddit API
- A praw.ini file must be added by the user including the scraper user agent client_id and client_secret keys