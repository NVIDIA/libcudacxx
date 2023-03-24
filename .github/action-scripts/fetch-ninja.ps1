# Fetch Ninja, this is a powershell step because of Expand-Archive
Invoke-WebRequest -Uri "https://github.com/ninja-build/ninja/releases/download/v1.11.1/ninja-win.zip" -OutFile "./ninja.zip"
Expand-Archive ./ninja.zip

$env:PATH="$env:PATH;$(pwd)\ninja"
echo "PATH=$env:PATH" >> $env:GITHUB_ENV