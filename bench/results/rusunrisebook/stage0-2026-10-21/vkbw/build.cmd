@echo off
rem Builds vkbw.exe and read.spv (Vulkan SDK 1.4, VS 18 Build Tools; paths from .cache\toolchain-setup\toolchain.env)
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
set VK=C:\VulkanSDK\1.4.363.0
"%VK%\Bin\glslc.exe" -O -fshader-stage=compute "%~dp0read.comp" -o "%~dp0read.spv" || exit /b 2
cl /nologo /O2 /W3 /I"%VK%\Include" "%~dp0vkbw.c" /Fe:"%~dp0vkbw.exe" /Fo:"%~dp0vkbw.obj" /link /LIBPATH:"%VK%\Lib" vulkan-1.lib || exit /b 3
