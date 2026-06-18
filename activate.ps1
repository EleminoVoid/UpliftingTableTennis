. "$PSScriptRoot\.venv\Scripts\Activate.ps1" @args

Remove-Item function:python -ErrorAction SilentlyContinue
function global:python {
	& "$PSScriptRoot\.venv\Scripts\python.exe" @args
}

Remove-Item function:pip -ErrorAction SilentlyContinue
function global:pip {
	& "$PSScriptRoot\.venv\Scripts\python.exe" -m pip @args
}