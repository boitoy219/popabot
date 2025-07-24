# Some Funny haha Data Scraper
This Project aims to create an automated propahahanda recording and reviewing system based on inserted keywords and groups.

## How to run the Project:
In order to run the project locally you need to download some packages before you can run it.

### Pre-requisites: 
**Code Editor (IDE) of your choice** for example:
* `Visual Studio Code` (prefered)
or
* `Notepad++`
or 
* `Visual Studio`

**Python 3.13**
1. Go to Microsoft Store and download [Python 3.13](https://apps.microsoft.com/detail/9PNRBTZXMB4Z?hl=en-us&gl=EE&ocid=pdpshare)

**C++ Build Tools**
1. Open this [Link](https://visualstudio.microsoft.com/visual-cpp-build-tools/). 
2. Click Download Build Tools >. A file named vs_BuildTools or vs_BuildTools.exe should start downloading. If no downloads start after a few seconds, click click here to retry. 
3. Run the downloaded file. Click Continue to proceed. 
4. Choose C++ build tools and press Install. You may need a reboot after the installation. 


### Run the Project:

1. [Download the project](https://github.com/boitoy219/popabot/archive/refs/heads/main.zip) 
2. Unzip the project
3. Open the project via VS Code or any other IDE
4. Locate file called "`.env_example`" and insert the values given to you
5. Rename the file to "`.env`"
6. Open terminal with the directory (location) of the .env file
7. In the terminal run
> `pip install -r ./requirements.txt`
8. After installation is complete run 
> `python pipeline.py`

### What does this Popabot do?

| File Name  | Description                       |
| :-------- | :-------------------------------- |
| `tg_scraper.py` | File scrapes all groups under `~./groups/groups.txt` directory with keywords located in the `~./keywords/keywords.txt`. You can change the groups and keywords to focus on different scraping output. Creating a file under `~./data/runs/` directory with the latest scraping session date. |
| `report_writer.py` | File creates a report in a .md format |
| `pipeline.py` | File activates all of the previous files as well as creates the latest messages that have been posted since the last scraping run |