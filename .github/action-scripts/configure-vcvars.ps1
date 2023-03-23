$msvcPath = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

Write-Output "Loading VC from: $msvcPath"

Push-Location "$msvcPath"
cmd /c "vcvars64.bat & set" |
    foreach {
        if ($_ -match "=") {
            $v = $_.split("="); set-item -force -path "ENV:\$($v[0])"  -value "$($v[1])"
        }
    }
Pop-Location

# Stupid, but makes cl.exe happy
$global:CC_FP = $(get-command cl).Source.Replace("\","/")

Write-Host "`nVisual Studio Command Prompt variables set." -ForegroundColor Yellow
Write-Host "Use `$CC_FP as shortcut for Cmake: $CC_FP" -ForegroundColor Yellow
