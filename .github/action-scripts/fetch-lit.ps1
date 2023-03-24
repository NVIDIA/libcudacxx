$python_path=(gi $($(get-command python).path)).Directory.FullName
py -m pip install wheel
py -m pip install lit

$ENV:PATH="$ENV:PATH;$(python_path)\Scripts"
echo "PATH=$ENV:PATH" >> $ENV:GITHUB_ENV
