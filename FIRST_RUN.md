Complete First-Run Guide — New Computer, No Prior Setup
BEFORE YOU START — Accounts you need
You need two accounts created before anything else:

1. RunPod account (the cloud GPU service)

Go to runpod.io and create a free account
Add credits (minimum ~$5 — a test render on an RTX 4090 costs roughly $0.50–$1.00)
Go to Settings → API Keys, click + API Key, copy it and save it somewhere — you'll need it later
2. HuggingFace account (explained below when you need it)

Go to huggingface.co and create a free account

PART 1 — Install the required software on the new computer

Step 1 — Install Python 3.10 or 3.11

Go to python.org/downloads
Download Python 3.11 (do not use 3.12 — a dependency has compatibility issues with it)
During installation on Windows, check the box that says "Add Python to PATH"
Verify it worked: open a terminal and type python --version — it should show 3.11.x

Step 2 — Install Git

Go to git-scm.com/downloads
Download and install for your operating system
Verify: in a terminal type git --version

Step 3 — Install ffmpeg

This is a video/audio processing tool the app uses internally.

Mac: open Terminal and run brew install ffmpeg (requires Homebrew — install from brew.sh first if you don't have it)
Windows: go to ffmpeg.org/download.html, download the Windows build, unzip it, and add the bin folder to your system PATH. There are many YouTube tutorials for this step if needed.
Linux: run sudo apt install ffmpeg
Verify: in a terminal type ffmpeg -version

PART 2 — Get the code

Step 4 — Clone the repository

Open a terminal, navigate to wherever you want the project to live, and run:

git clone https://github.com/spfrood/image-skinner.git
cd solo_teamer
Step 5 — Create your environment file

cp .env.example .env
Now open the .env file in any text editor (Notepad, TextEdit, VS Code, etc.). Find this line:

RUNPOD_API_KEY=your_runpod_api_key_here
Replace your_runpod_api_key_here with the API key you copied from RunPod in Step 1. Save the file.

PART 3 — What is HuggingFace, and what is an RVC model?

HuggingFace is a website (huggingface.co) that works like a library or app store specifically for AI models. Researchers and hobbyists upload trained AI models there for free so others can download and use them. It has hundreds of thousands of models for all kinds of tasks.

RVC (Retrieval-based Voice Conversion) is an AI voice-cloning technology. Someone can record hours of a voice (say, an anime character from a show), train an RVC model on it, and upload the result to HuggingFace. When you use that model, it takes whatever someone says and makes it sound like that character's voice — while preserving the original speaker's emotion and phrasing.

An RVC model comes as exactly two files:

A .pth file — the trained neural network weights (usually 50–200 MB)
A .index file — a fast lookup table that improves voice accuracy (usually 10–100 MB)
You need both files. One without the other won't work.

PART 4 — Find and download a pre-trained RVC model

Step 6 — Go to HuggingFace and find a model

Go to huggingface.co
Log in to your account
Click the search bar at the top and search for: RVC voice model
On the left side, under Libraries, click Other to filter
Browse the results — you're looking for a model that:
Has both a .pth file and a .index file in its file listing
Has a voice style you want to test with (anime characters are very common)
Has been downloaded many times (indicates it works reliably)
Good search terms to try:

RVC anime voice
RVC v2 voice
RVC character voice
A reliable starting point: search for "RVC" and sort by Most Downloads. Popular well-tested models include voice models for anime characters like Hatsune Miku, various Genshin Impact characters, etc. Any of them will work for a test — you're just testing that the pipeline runs, not perfecting a character yet.

Step 7 — Download the model files

Click on a model in the search results to open its page
Click the Files and versions tab
Look through the file list for files ending in .pth and .index
Click the download arrow icon next to the .pth file — save it somewhere easy to find (e.g., your Desktop)
Do the same for the .index file
Remember which character/voice this model is for — you'll use this name when creating the character profile

PART 5 — Prepare the other assets

Step 8 — Get a character sketch

For your first test, you can use any of these:

Draw or find any portrait-style PNG image on your computer
Download a free anime character illustration from a site like pixabay.com (search "anime character")
Even a clear photo of a face will work technically — LivePortrait just needs a detectable face
Save it as a .png file, ideally at least 512×512 pixels

Step 9 — Create a short voice sample WAV

Even though you're using a pre-trained voice model (meaning the voice conversion doesn't need YOUR voice sample to train), the app still requires you to upload a WAV file to the "Voice Sample" field as a reference/placeholder.

The easiest way:

Windows: open Voice Recorder (search for it in Start menu), record 10–20 seconds of yourself saying anything, export as .wav or .m4a (either works — you can rename .m4a to .wav for the upload)
Mac: open QuickTime Player → File → New Audio Recording, record briefly, save
Or download any short spoken-word audio clip from the internet and save as a .wav file
This file is just stored alongside the character profile — it doesn't affect the voice quality of the output for a pre-trained model test.

PART 6 — Start the application

Step 10 — Open a terminal in the project folder

Navigate to the solo_teamer folder you cloned in Step 4.

On Mac/Linux:

cd path/to/solo_teamer
bash scripts/run.sh
On Windows (use Command Prompt or PowerShell):

cd path\to\solo_teamer
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Then open two separate terminals and run in the first:

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
And in the second:

python frontend/app.py

Step 11 — Wait for startup

You'll see text scrolling in the terminal as it installs dependencies (first run only — takes 2–5 minutes). When you see something like:

Running on local URL: http://0.0.0.0:7860
The app is ready.

Step 12 — Open the app in your browser

Go to: http://localhost:7860

PART 7 — Set up your character and record your performance

Step 13 — Create a Character Profile

Click the Character Gallery tab
Fill in:
Character Name: whatever you want to call this character
Animation Prompt: describe the character's look for the AI — example: "anime character, colorful outfit, full body, illustration style". The more specific the better.
Notes: optional, leave blank
Click Character Sketch (.png) and upload the PNG from Step 8
Click RVC Voice Model (.pth) and upload the .pth file from Step 7
Click RVC Index File (.index) and upload the .index file from Step 7
Click Original Voice Sample (.wav) and upload the WAV from Step 9
Click Save Character
You should see a success message and the character appears in the table

Step 14 — Record your driver performance

Click the Record / Upload tab
Click the webcam box — your browser will ask permission to use the camera, click Allow
Click Record and perform your scene — speak some lines, move your head and upper body. 15–30 seconds is a good length for a first test.
Click Stop
Click Upload to Server
Wait for the status message: Uploaded: your_video.mp4
Note the exact filename shown — you'll need it in the next step

PART 8 — Render

Step 15 — Start the render

Click the Render tab
In Recording Filename, type the exact filename from Step 14 (e.g. recording.mp4)
Click Refresh Characters and select your character from the dropdown
Click the Render button
What happens next (this takes several minutes):

The status box will cycle through these stages — this is normal:

provisioning_pod — RunPod is starting a cloud GPU (1–3 minutes)
syncing_assets — your files are being uploaded to the GPU
waiting_for_output — the GPU is installing LivePortrait + AnimateDiff + RVC and downloading model weights for the first time (~10–15 minutes on first run only; much faster on subsequent runs if you keep the pod's storage volume)
running_inference — the actual animation and voice conversion is happening (~2–5 minutes for a 30-second clip)
fetching_output — the finished video is being downloaded back to your computer
done — the output filename appears

Step 16 — Watch the result

Click the Outputs tab
Type the output filename shown in the Render tab
Click Preview to watch it directly in the browser, or Download Link to save it


