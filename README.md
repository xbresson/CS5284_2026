# NUS CS5284 Graph Machine Learning, Sem 1 2026/27

## Xavier Bresson


<br><br> 


### Cloud Machine : Google Colab (Free GPU)

* Follow this Notebook installation :<br>
https://colab.research.google.com/github/xbresson/CS5284_2026/blob/master/01_Installation/installation.ipynb

* Open your Google Drive :<br>
https://www.google.com/drive

* Open in Google Drive Folder 'CS5284_2026' and go to Folder 'CS5284_2026/codes/'<br>
Select the notebook 'file.ipynb' and open it with Google Colab using Control Click + Open With Colaboratory



<br><br>

### Local Installation for OSX M Chips

* Open a Terminal and type


```sh
   # Conda installation
   curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o miniconda.sh -J -L -k # OSX M
   chmod +x miniconda.sh
   ./miniconda.sh
   source ~/.bashrc

   # Clone GitHub repo
   git clone https://github.com/xbresson/CS5284_2026.git
   cd CS5284_2026

   # Install python libraries
   conda env create -f environment_osx_arm64.yml
   conda activate gnn_course

   # Run the notebooks in Chrome
   jupyter notebook
   ```


### Local Installation for OSX Intel Chips 

* Open a Terminal and type


```sh
   # Conda installation
   curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh -J -L -k # OSX Intel
   chmod +x miniconda.sh
   ./miniconda.sh
   source ~/.bashrc

   # Clone GitHub repo
   git clone https://github.com/xbresson/CS5284_2026.git
   cd CS5284_2026

   # Install python libraries
   conda env create -f environment_osx_intel.yml
   conda activate gnn_course

   # Run the notebooks in Chrome
   jupyter notebook
   ```


### Local Installation for Linux

* Open a Terminal and type


```sh
   # Conda installation
   curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh -J -L -k # Linux
   chmod +x miniconda.sh
   ./miniconda.sh
   source ~/.bashrc

   # Clone GitHub repo
   git clone https://github.com/xbresson/CS5284_2026.git
   cd CS5284_2026

   # Install python libraries
   conda env create -f environment_linux.yml
   conda activate gnn_course

   # Run the notebooks in Chrome
   jupyter notebook
   ```




### Local Installation for Windows 

```sh
   # Install Anaconda 
   https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

   # Open an Anaconda Terminal 
   Go to Application => Anaconda3 => Anaconda Prompt (Note: On some Windows, you may need to run the application as administrator for `jupyter notebook`.)

   # Install git : Type in terminal
   conda install git 

   # Clone GitHub repo
   git clone https://github.com/xbresson/CS5284_2026.git
   cd CS5284_2026

   # Install python libraries
   conda env create -f environment_win64.yml
   conda activate gnn_course

   # Run the notebooks in Chrome
   jupyter notebook
   ```

   