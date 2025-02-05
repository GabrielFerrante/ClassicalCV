**INSTALAÇÃO WINDOWS SEM UTILIZAR OPENCV DO PIP**
https://docs.opencv.org/4.x/d5/de5/tutorial_py_setup_in_windows.html
Baixar e instalar CUDA

Apontar nas variaveis de ambiente, onde o CUDA está. 

- “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin”

- “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib”

- “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include”

- “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\lib\x64”


Baixar e instalar CuDNN

Apontar nas variaveis de ambiente, onde o CUDA está. 

- C:\Program Files\NVIDIA\CUDNN\v9.7\bin

- C:\Program Files\NVIDIA\CUDNN\v9.7\bin\12.8

- C:\Program Files\NVIDIA\CUDNN\v9.7\include\12.8

- C:\Program Files\NVIDIA\CUDNN\v9.7\lib\12.8\x64

Baixar e instalar CmakeGUI
Baixar e instalar Visual Studio Community Edition (Instalar com desktop development for C++)
Baixar e instalar OpenCV
    Instalar em uma pasta vazia no disco local

Baixar e instalar OpenCV-Contrib

No CMake, ir em "Where is the source code" e selecionar a pasta que esta o arquivo CMakeLists.txt
Exmplo: C:/OpenCV/opencv/sources

Depois, selecionar na opção "Where to build the binaries" a pasta de instalação
Exmplo: C:/OpenCV/Build (criar esta pasta)

Clicar em configurar. 

Depois, buscar na barra de pesquisa em OPENCV_EXTRA_MODULES_PATH e alterar o valor para o diretório
do contrib que instalou:
Exmplo: C:\OpenCV-Contrib\opencv_contrib-4.10.0\modules

clicar em configurar

Depois, buscar por CUDA na barra de pesquisa e selecionar os check-box nas opçoes

OPENCV_DNN_CUDA x
WITH_CUDA x
WITH_CUDNN x

Clicar em configurar

E depois generate

1. OPÇÃO 1
Abra o Visual Studio Community

Abra o arquivo OpenCV.sln e mude de **Debug** para **Release**

Vai em Cmake Targets, na opção ALL_BUILD, e clica em Build OU msbuild ALL_BUILD.vcxproj /p:Configuration=Release /p:Platform=x64 /m

após o processo, vai na opção INSTALL, e clica em build. 

2. OPÇÃO 2 

cmake --build "C:\OpenCV\build" --target INSTALL --config Release 




