**DERMANOSTIX**

This is an AI tool to identify and classify skin lesions. 

**INSTRUCTIONS**

1. Download IntelliJ's PyCharm, which can be found here: https://www.jetbrains.com/pycharm/download/

2. Create a new project in PyCharm(File -> New Project). Name it whatever you wish.

3. Open the Terminal in PyCharm (in the bottom left)

4. Type `git init` and press Enter. Then type `git remote add origin https://github.com/theReality2-0/dermanostix.git` and hit Enter. Finally, type `git pull origin master` and press Enter. This will copy and paste all of the code into your new project.

5. Enter `pip install flask opencv-python-headless numpy` into Terminal.

6. If you are on Windows, type `set FLASK_APP=app.py` and hit Enter. Otherwise, type `export FLASK_APP=app.py` and hit Enter.

7. If you are on Windows, enter `set FLASK_ENV=development`. Otherwise, enter `export FLASK_ENV=development`. 

8. Type `flask run` and press Enter. It will print many things, of which one is a url, starting with 127.0.0.1. Click on that to go to the web app.
