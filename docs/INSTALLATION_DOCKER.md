# How to install Tierpsy with Docker

- [Introduction](#introduction)
- [Installation on Windows 10](#installation-on-windows-10)
    - [1. Install Docker](#1-install-docker)
    - [2. Get the Tierpsy image](#2-get-the-tierpsy-image)
    - [3. Install VcXsrv](#3-install-vcxsrv)
    - [4. [Optional, but suggested] Create a Desktop Launcher](#4-optional-but-suggested-create-a-desktop-launcher)
    - [4. [Alternative] Start Tierpsy Tracker manually](#4-alternative-start-tierpsy-tracker-manually)
- [Installation on macOS](#installation-on-macos)
  - [1. Install XQuartz](#1-install-xquartz)
  - [2. Install Docker](#2-install-docker)
  - [3. Get the Tierpsy image](#3-get-the-tierpsy-image)
    - [4. [Optional, but suggested] Create a Desktop Launcher](#4-optional-but-suggested-create-a-desktop-launcher-1)
- [Installation on Linux [coming soon]](#installation-on-linux-coming-soon)
- [Using Tierpsy Tracker in Docker](#using-tierpsy-tracker-in-docker)

## Introduction

On Windows and Mac alike, you will need three things to use Tierpsy with Docker:
1. Docker itself
2. The docker image for Tierpsy Tracker
3. A program to show Tierpsy's user interface.\
   We recommend [VcXsrv](https://sourceforge.net/projects/vcxsrv/) for Windows
   and [XQuartz](https://www.xquartz.org/) for macOS.


## Installation on Windows 10
**Note:**
*While you do not need to have admin rights to run Tierpsy on Docker in Windows,
you will need them for the installation process.*

This guide looks long, but it should take somewhere between 20 and 40'.

#### 1. Install Docker

Download and run the installer from [here](https://desktop.docker.com/win/stable/amd64/Docker%20Desktop%20Installer.exe) *[Needs admin rights]*.

You can keep the default installation settings.
If prompted about it, make sure that Docker is installed in `C:\Program Files\Docker`, and if you're asked to install the "required Windows components for WSL 2" say yes.
Restart the computer when required to do so.
![Docker Installation 1](https://user-images.githubusercontent.com/33106690/127778612-9e270d9c-e427-45f0-b047-881f17228408.png)|![Docker Installation 2](https://user-images.githubusercontent.com/33106690/127778614-8fcf7e01-28ea-4703-8219-372229b5e050.png)
:---:|:---:

After restarting, you'll likely be told that the "WSL2 installation is incomplete".
Follow the link in the warning message and install the "WSL2 Linux Kernel",
then restart the computer again.


![WSL2 error](https://user-images.githubusercontent.com/33106690/127776178-d2b74c64-e385-4cdd-96cd-dfdfe45c22b3.png)|![wsl 1](https://user-images.githubusercontent.com/33106690/127778815-3f869376-160d-4d4b-b442-a72c475d99db.png)|![wsl2](https://user-images.githubusercontent.com/33106690/127778819-6d3813ce-6965-4195-9faa-877ced59154e.png)
:---:|:---:|:---:



>If you are not an administrator of the computer, your user needs to be added to the local group named "docker-users".\
>From a PowerShell or command prompt with Administrator rights, run `net localgroup docker-users your_username_here /add`,
>then log out of Windows and log and back in.
>
>![Screenshot (3)](https://user-images.githubusercontent.com/33106690/127778615-9bc879c8-a8bf-4e15-865f-0dfc94370677.png)|![docker-users](https://user-images.githubusercontent.com/33106690/127778618-a4a5456f-8cb4-4df5-9c4f-4687772a6a67.png)|![docker-users logout](https://user-images.githubusercontent.com/33106690/127778619-1836e7e5-11b0-4e56-aba1-698c2a1cddd6.png)
>:---:|:---:|:---:


Now run the Docker Desktop app from the start menu.


> At this point you *may* be prompted with a nasty looking error message about needing to enable virtualisation in the BIOS. How to do this is highly dependent on the computer model, but the main steps are:
> + turn off your computer
> + turn it back on
> + press F2/DEL to enter the BIOS when prompted
>   + the right key to press might change, the computer will tell you which one to press.
>   + pay close attention as the message could be on screen for a very short time if you have an SSD
>   + spamming the right key during the startup is often a successful strategy
> + in the BIOS, find the options related to the CPU (could be named Northbridge, or Chipset)
>   + this might be under an **Advanced Options** section
> + find the option that allows you to enable hardware virtualisation
>   + be on the lookeout for options such as **Hyper-V**, **Vanderpool**, **SVM**, **AMD-V**, **Intel Virtualization Technology** or **VT-X**
>   + if you see options for **Intel VT-d** or **AMD IOMMU** enable those as well
> + save the changes and exit the BIOS
> + start the computer as usual

For reference, you can find the full documentation on how to install Docker on windows
[here](https://docs.docker.com/docker-for-windows/install/).

#### 2. Get the Tierpsy image

If you have a Docker Hub account, or are ok with creating one, you can look for the
`tierpsy/tierpsy-tracker` image in the Docker Desktop app.

Otherwise, open Powershell,
then type this command in the window that opens:
```
docker pull tierpsy/tierpsy-tracker
```
press `Enter` on your keyboard and wait for the download to be finished. This will take a few minutes, but you can move to the next step in the meantime.

<img height="300" alt="Start powershell" src="https://user-images.githubusercontent.com/33106690/127387980-d2199548-d09e-4293-b110-fdd9ee6c1af3.png">|<img height="300" alt="Image downloading" src="https://user-images.githubusercontent.com/33106690/129880373-8d12b2ac-7bed-4dc7-8b0b-3cc15552b278.png">
:---:|:---:


#### 3. Install VcXsrv

Download and run the [installer](https://sourceforge.net/projects/vcxsrv/files/latest/download) *[needs admin rights]*.

Follow the installation wizard, making sure VcXsrv is being installed in `C:\Program Files\VcXsrv`

![vcxsrv_1](https://user-images.githubusercontent.com/33106690/127391065-b8d68844-d2e5-4652-8b44-84d4fcb11f25.png)|![vcxsrv_1](https://user-images.githubusercontent.com/33106690/127391067-49dfe117-e41b-41b9-bb66-1039d06ea594.png)
:---:|:--:


#### 4. [Optional, but suggested] Create a Desktop Launcher

Download [this](https://github.com/Tierpsy/tierpsy-tracker/files/7008741/tierpsy.txt) file, save it on your Desktop,
and [rename it changing its extension](https://winbuzzer.com/2021/04/26/how-to-safely-change-a-file-extension-or-file-type-in-windows-10-xcxwbt/) from `tierpsy.txt` to `tierpsy.ps1`.

There are a couple of modifications you might want to do. When Tierpsy runs in Docker, it cannot access all of your hard drives, or network shares.
You need to configure what folders (or entire disks) Tierpsy has access to, and give it instructions to access the network share.

By default, the launcher allows Tierpsy to see the entirety of the `D:\` drive.
If you want to modify this behaviour, just open the launcher with a text editor, and modify line 16 by putting the path of the folder you want Tierpsy to access.
You'll have to substitute any `\` symbol in your path of choice with `/` symbols. The path should be enclosed in double quotation marks, e.g. `"c:/your/path/here"`.

You can also give Tierpsy instructions on how to access your network shares. This process is akin to using Windows' **Map Network Drive**.\
You will need to write at line 17 the network address of your network share, and at line 18 your user name. If you're using a network account, you might have to specify your domain as well.
Refer to the examples in the screenshot below for the right syntax to use.

Once you're done, save your changes, close the editor, and you can then `right click --> Run with PowerShell` to launch Tierpsy Tracker (no, double-click does not work - this is by design from Microsoft).
You can now start [using Tierpsy Tracker](#using-tierpsy-tracker-in-docker).

<!-- ![Screenshot (11)](https://user-images.githubusercontent.com/33106690/127777903-bcec1ac2-76d9-4d3e-b28c-cad646d24e40.png)|![run w/ PS](https://user-images.githubusercontent.com/33106690/127393995-789982dc-e087-4549-b047-c1c88820b52e.png) -->
![launcher](https://user-images.githubusercontent.com/33106690/127779304-c7ff2d4e-a202-45c4-8b22-3278f80bab34.png)|![run w/ PS](https://user-images.githubusercontent.com/33106690/127393995-789982dc-e087-4549-b047-c1c88820b52e.png)
:---:|:---:

> You _might_ have to allow PowerShell to run scripts first. To do this, open a PowerShell as Administrator\
> (Windows key, type `PowerShell`, then right click on the best match and `Run as Administrator`)\
> and type `Set-ExecutionPolicy RemoteSigned`, then press `Enter` on your keyboard.

##### Why do I need a Desktop Launcher?
Strictly speaking, you do not need a Launcher, but it automates a few boring actions, and sets some parameters that are quite important for Tierpsy Tracker to run smoothly. See the next section if you prefer to start the image manually.

##### What does the launcher do exactly?
3 things:
- checks if VcXsrv is running, if not it starts it
- checks if Docker is running, if not it starts it
- waits for Docker to be up, and starts a Docker container from the Tierpsy image,
  setting a bunch of parameters needed to run properly
  (including the instructions to make any local drive or network share visible to Tierpsy).

#### 4. [Alternative] Start Tierpsy Tracker manually

If you prefer not to use the Launcher, you just need to run a few steps manually.

First, start VcXsrv from the Start menu. Use the following settings:

![Screenshot (3)](https://user-images.githubusercontent.com/33106690/127397026-a86ccf66-0569-4b52-8e9c-28e676e96259.png)|![Screenshot (4)](https://user-images.githubusercontent.com/33106690/127397032-db4db93f-a7fa-46bb-aa1b-a62ef9e04408.png)|![Screenshot (5)](https://user-images.githubusercontent.com/33106690/127397033-ffd16ec3-e2a8-4f5c-8181-0860982ae395.png)
:---:|:---:|:---:

Once it's running, you'll have a little icon with a black X in the application tray.

Then start the Docker Desktop app from your Start Menu or Desktop shortcut, wait until the app says it's running properly.

Open PowerShell, and run this:
```
docker run -it --rm -e DISPLAY=host.docker.internal:0 -v d:/:/DATA/local_drive --sysctl net.ipv4.tcp_keepalive_intvl=30 --sysctl net.ipv4.tcp_keepalive_probes=5 --sysctl net.ipv4.tcp_keepalive_time=100 --hostname tierpsydocker tierpsy/tierpsy-tracker
```

N.B. the command above will not mount any network shares - look inside the launcher to see how to do that!


## Installation on macOS

> This was tested on a MacBook Pro running Mojave. Might not work on machines with an M1 chip.

### 1. Install XQuartz

Download the installer from [Xquartz's website](https://www.xquartz.org/).
Then open it, and follow the installation wizard.

Once the installation is finished, open XQuartz, go to Preferences -> Security, and tick "Allow connections from network clients". 
Then quit XQuartz and open it again, or the change will not be applied.

![pkg](https://user-images.githubusercontent.com/33106690/129926425-ca071a47-9888-4e10-8d32-50947cbf72fe.png)|![xq1](https://user-images.githubusercontent.com/33106690/129926468-f149b8c6-340e-4661-b0e3-44dfec125635.png)|![xq2](https://user-images.githubusercontent.com/33106690/129926697-9cfc4ad5-dac5-4e25-a860-da147c8c7411.png)|![xq3](https://user-images.githubusercontent.com/33106690/129926755-21a526e6-c478-4bea-99af-836852cba34f.png)|![xq4](https://user-images.githubusercontent.com/33106690/137301854-13dac386-9eb6-402b-a82e-58608626d063.png)
:---:|:---:|:---:|:---:|:---:


### 2. Install Docker

Download the installer from [Docker's website](https://docs.docker.com/docker-for-mac/install/).
Open the installer and follow the installation wizard.

![docker1](https://user-images.githubusercontent.com/33106690/129928705-725b7ff7-468a-4e86-adf8-9f9d66298859.png)|![docker2](https://user-images.githubusercontent.com/33106690/129928725-216f013f-5fa1-4ad4-94e0-62b496edcf1b.png)|![docker3](https://user-images.githubusercontent.com/33106690/129928742-8d431c4e-d1ba-4d53-9d5e-8e40e93f3b00.png)
:---:|:---:|:---:|

Open the Docker app from Spotlight or Launchpad, then authorise Docker to finish its installation.
You can skip the tutorial, unless you're interested in using Docker in its own rights.
Finally, increase the amount of resources Docker is allowed to use:
in Docker Desktop open the Settings (the cog icon on the top right), then open the Resources tab, and move the CPU, Memory, and swap sliders to the right.
![docker4](https://user-images.githubusercontent.com/33106690/129928779-bc8c222c-9baf-4db4-bdaa-d3ba472f335a.png)|![docker5](https://user-images.githubusercontent.com/33106690/129928969-0da5e6b5-0e74-48f5-9838-22a616c58550.png)|![macos docker resources](https://user-images.githubusercontent.com/33106690/129885948-5ce31aa7-49a2-432c-84d8-48e4b2a38526.png)
:---:|:---:|:---:


### 3. Get the Tierpsy image

With the Docker application open, run the following command in your Terminal:
```bash
docker pull tierpsy/tierpsy-tracker
```

<img width="600" alt="get the tierpsy image" src="https://user-images.githubusercontent.com/33106690/129935590-3fab0557-6f8f-4bfb-ac9f-8017309510ac.png">

#### 4. [Optional, but suggested] Create a Desktop Launcher

Download [this](https://github.com/Tierpsy/tierpsy-tracker/files/7008794/tierpsy.txt) file, save it on your desktop, and rename it from `tierpsy.txt` to `tierpsy.command`.
Then open a Terminal and run the following command, to make the launcher executable:
```bash
chmod +x ~/Desktop/tierpsy.command
```

There are a couple of modifications you might want to do. When Tierpsy runs in Docker, it cannot access all of your hard drives, or network shares.
You need to configure what folders (or entire disks) Tierpsy has access to, and give it instructions to access the network share.

By default, the launcher allows Tierpsy to see the entirety of your home folder.
If you want to modify this behaviour, just open the launcher with a text editor, and modify line 32 by writing the path of the folder you want Tierpsy to access.

Similarly, if you have a network folder or an external hard drive you want Tierpsy to have access to, write its path in between the double quotes at line 33.

<img width="600" alt="edit launcher" src="https://user-images.githubusercontent.com/33106690/129942697-fc65f5b5-0d9f-42ea-8ab9-317bdebed8b3.png">

Double-clicking the desktop launcher you just created will start Tierpsy Tracker.
> The first time only you'll have to right-click and choose "Open", then confirm you want to open the file.

You can now start [using Tierpsy Tracker](#using-tierpsy-tracker-in-docker).


<img height="150" alt="open with" src="https://user-images.githubusercontent.com/33106690/129940497-8949f44a-ec12-4edc-8aa9-3ee7031b32e2.png">|<img height="150" alt="confirm" src="https://user-images.githubusercontent.com/33106690/129940493-db7f1512-9c7d-49e8-ac08-4e5776b82ddf.png">
:---:|:---:


## Installation on Linux [coming soon]


## Using Tierpsy Tracker in Docker

After launching Tierpsy, either with a Desktop launcher or from the command line,
you'll be greeted by a simple landing page, that lists the three main commands.

![text splash](https://user-images.githubusercontent.com/33106690/127402583-eb7ca247-09d8-4803-a522-db495014a51b.png)


- ### `tierpsy_gui`
This should be your go-to command. Opens the small application windows from where you can choose what step of the tierpsy analysis you want to undertake. See [How to Use](docs/HOWTO.md) for detailed instructions.

![qt splash](https://user-images.githubusercontent.com/33106690/127402654-78e2ccc5-1032-47b0-9cce-8377b1901de4.png)

- ### `is_tierpsy_running`
When running Tierpsy Tracker with Docker there is a chance that the GUI will time out and close, if it not interacted with for a long time.
We haven't seen this happen during the analysis, and it does not affect the processing of videos, which just keeps happening in the background.

If you return to your computer after a few hours, and in place of the GUI find a message reading\
_The X11 connection broke (error 1). Did the X11 server die?_\
don't panic, this should not be affecting any analysis. So you can press Enter a couple of times, and just run the command `is_tierpsy_running`.

This will scan the active processes, and tell you how many, if any, videos are being processed in the background. \
If the reading is 0, then it is likely that Tierpsy had finished analysing the data, and the command will explan how to double-check that all of your files have been processed.

- ### `tierpsy_process`
This utility allows you to bypass the GUI entirely, if you cannot or prefer not to use one.
Please read the documentation that appears with `tierpsy_process --help` and do not hesitate to contact us for help.


