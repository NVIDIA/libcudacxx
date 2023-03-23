
if (Test-Path -Path $EnvVarBackup) {
    Remove-Item -Path $EnvVarBackup
}

function Get-VSDevPrompt {
    param([string]$vcver = "14.16")
    if (Test-Path -Path $EnvVarBackup) {
        Remove-Item -Path "ENV:*"
        Import-CliXml $EnvVarBackup | % { Set-Item -force -path "env:$($_.Name)" $_.Value }
        Remove-Item $EnvVarBackup
    }

    # Save all the process' environment variables in CLIXML format.
    Get-ChildItem env: | Export-CliXml $EnvVarBackup
    $BuildPath = $MSBuildPathMap[$vcver]
    Write-Output "Loading VC from: $BuildPath"

    Push-Location "$BuildPath"
    cmd /c "vcvars64.bat -vcvars_ver=$vcver & set" |
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
}

Export-ModuleMember -Function Get-VSDevPrompt
