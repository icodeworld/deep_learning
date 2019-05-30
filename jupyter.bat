@echo off
start   %windir%\System32\cmd.exe    "/c" D:\Anaconda\Scripts\activate.bat  
(activate tensor
e:
cd jupyter
jupyter notebook
)