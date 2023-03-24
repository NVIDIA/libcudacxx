$LIT_PATH = "$((gi $((gcm python).path)).Directory.FullName)\Scripts"

py -m pip install wheel
py -m pip install lit

$ENV:PATH="$ENV:PATH;$LIT_PATH"
echo "PATH=$ENV:PATH" >> $ENV:GITHUB_ENV
