$LIT_PATH = "$((gi $((gcm python).path)).Directory.FullName)\Scripts"

python -m pip install wheel
python -m pip install lit

$ENV:PATH="$ENV:PATH;$LIT_PATH"
echo "PATH=$ENV:PATH" >> $ENV:GITHUB_ENV
