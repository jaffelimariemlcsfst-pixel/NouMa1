


°❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*:･ This is Noumafind's's readme °❀⋆.ೃ࿔*:･°❀⋆.ೃ࿔*:･
https://noumafinds.streamlit.app/
# this is our website's URL 
you have to register first then log in (●'◡'●)
and censerning the video please download it for better resolution 
and thank you so much for this opportunity ^_^



##Introduction (✧∀✧)/ :  What is this project?

It is an intelligent recommendation system that helps users find tech products within their budget.
It also suggests optimal devices based on the user's profile (still working on that).


##Project Purpose (｡•̀ᴗ-)✧ :

Finding the right device can be overwhelming with countless specifications.


##Getting Started: How can someone set it up and run it 	(¬‿¬ )?
## Setup Instructions

if you want to run the script locally replace this :
import io
os.environ["TOKENIZERS_PARALLELISM"] = "false"
URL = os.environ["QDRANT_URL"]
API_KEY = os.environ["QDRANT_API_KEY"]
with this :
URL="paste your url here"
API_KEY="paste you api_key here"



1.Create the virtuelenv by the commend
     ☺ macOS/Linux: python3 -m venv .venv
     ☺ Windows (Command Prompt): python -m venv venv
     ☺ Windows (PowerShell): python -m venv venv
     
2.Activate the Environment
    ○ macOS/Linux (Bash/Zsh): source venv/bin/activate
    ○ Windows (Command Prompt): .\venv\Scripts\activate.bat
    ○ Windows (Powershell):  .\venv\Scripts\Activate.ps1
    :/ If you are on Windows and get a red error message saying Execution_Policies or "scripts are disabled on this system," it’s a very common security roadblock in PowerShell.
    just copy and paste  : Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
    and activate your Environment (●'◡'●)
3.install the requirements 
    just type in your terminal :
    pip install -r requirements.txt
    replace this path :image_path = r"c:\Users\user\Pictures\Screenshots\Capture d'écran 2026-01-23 142433.png"
    with the path where you save the logo image that is in this repostory

    
3.Run the code

     streamlit run main.py

Running this command will start a local web server and open your app in a new tab in your default web browser.
note : the first thing you have to do when you open the app is to push the buttom PUSH DATA TO CLOUD 

##How do you use it  (⌒_⌒)  ?

*If you have a budget limit enter it
*If u have a specific color select it
*If you have a specific model enter it
*if you have a picture of the desired product upload it 
X_X we've tried to automate the webscraping with github actions but we are still working on it X_X  








