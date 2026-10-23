# Publication audit - 2026-10-23

Scope: source inspection, local CPU checks, web build, history secret scan, and claim review. No new GPU evaluation has been run in this audit.

Findings corrected:
- README overstated completion and recommended unimplemented panorama backends.
- Gaussian fitting swallowed all exceptions; it now fails visibly and its stage cache version is bumped.
- Layout API silently changed the default backend to mock; it now honors the request.
- The declared Python floor was incompatible with the pinned NumPy version; raised to 3.11.
- CI silently fell back from npm ci to npm install; removed the fallback.
- Historical allocator measurements were described as total VRAM; qualified the benchmark report.

Outstanding: panorama quality, meaningful occlusion branch coverage, fitting quality metrics, text/ControlNet T4 validation, complete world regeneration, aligned expansion, region regeneration, six real gallery worlds and GIFs. Historical clean-room statements express provenance recorded during development; source inspection alone cannot prove that no upstream source was ever seen. No independent legal certification is claimed.

Checks and publication outcome will be recorded in PROGRESS.md after execution.

Local results: 37 CPU tests passed (one upstream deprecation warning); ruff lint and format passed; mypy passed for 37 source files; web ESLint and TypeScript/Vite build passed (bundle-size advisory). Full-history gitleaks found no leaks in 24 pre-audit commits. No new GPU validation or visual-quality claim is made.

Publication: [adamew1/adamworld](https://github.com/adamew1/adamworld) is public. [Hosted CI run](https://github.com/adamew1/adamworld/actions/runs/39954703344) passed all three jobs for commit `62d7922` on 2026-10-23. No release tags or GitHub Pages showcase were published.
