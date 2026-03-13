$env:OPENCV_DIR = "C:\opencv\build"
$env:LIBCLANG_PATH = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin"
$env:PATH = "C:\opencv\build\x64\vc16\bin;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\Llvm\x64\bin;" + $env:PATH

# Find MSVC and SDK paths dynamically
$msvc_path = Get-ChildItem -Path "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$sdk_path = Get-ChildItem -Path "C:\Program Files (x86)\Windows Kits\10\Include" -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$sdk_version = Split-Path $sdk_path -Leaf

$clang_args = "-isystem ""$msvc_path\include"" " +
              "-isystem ""C:\Program Files (x86)\Windows Kits\10\Include\$sdk_version\ucrt"" " +
              "-isystem ""C:\Program Files (x86)\Windows Kits\10\Include\$sdk_version\shared"" " +
              "-isystem ""C:\Program Files (x86)\Windows Kits\10\Include\$sdk_version\um"" " +
              "-isystem ""C:\Program Files (x86)\Windows Kits\10\Include\$sdk_version\winrt"" "

$env:CLANG_FLAGS = $clang_args
$env:OCVRS_CLANG_FLAGS = $clang_args

Write-Host "MSVC Path: $msvc_path"
Write-Host "SDK Version: $sdk_version"
Write-Host "Clang Flags: $env:CLANG_FLAGS"

cargo build --release
