$msvcPath = "$(vswhere -products * -latest -property installationPath)"

Write-Output "Loading VC from: $msvcPath\VC\Auxiliary\Build"

Push-Location "$msvcPath\VC\Auxiliary\Build"
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
