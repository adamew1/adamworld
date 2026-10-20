<#
Deploy the AdamWorld gallery to GitHub Pages.

Builds the web app (viewer + editor + gallery + generated gallery/ assets) and
pushes web/dist as the gh-pages branch of the target repo. Then enable Pages in
the repo settings: Settings -> Pages -> Source: "Deploy from a branch" ->
gh-pages / (root).

Usage:
  pwsh scripts/deploy_gallery.ps1 -RepoUrl https://github.com/<you>/<repo>.git

Prereqs: the 6 gallery worlds exist (python core/scripts/make_gallery.py), and git
is authenticated for the target repo (HTTPS token or SSH).
#>
param([Parameter(Mandatory = $true)][string]$RepoUrl)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location "$root/web"

Write-Host "Building web (viewer + editor + gallery)..."
npm run build

Set-Location "$root/web/dist"
# touch .nojekyll so GitHub Pages serves files/dirs starting with _ and . normally
New-Item -ItemType File -Force -Path ".nojekyll" | Out-Null

if (Test-Path ".git") { Remove-Item -Recurse -Force ".git" }
git init -b gh-pages | Out-Null
git add -A
git -c user.name="AdamWorld" -c user.email="dev@adamworld.local" commit -q -m "Deploy AdamWorld gallery"
git remote add origin $RepoUrl
Write-Host "Force-pushing web/dist -> gh-pages of $RepoUrl ..."
git push -f origin gh-pages

Write-Host "Done. Enable Pages (gh-pages branch, root) in the repo settings."
Write-Host "Site: https://<you>.github.io/<repo>/gallery.html"
