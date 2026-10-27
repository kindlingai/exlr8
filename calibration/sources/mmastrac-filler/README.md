# mmastrac-filler

Raw source files by Matt Mastracci, added to the calibration mix for Java, .NET and C++ code and for English prose.
Files are kept as they were, except that images, dotfiles, version-control metadata and compiled binaries were left
out. `cpp-raytracer` also leaves out the course-supplied linear algebra library (`defines.h`, `jgl.inl` and
`linearalgebra.*`), so it does not build on its own.

| folder | source | revision |
|---|---|---|
| `java-nanojson` | <https://github.com/mmastrac/nanojson> | `1d5f78a19e3a654361ea53520738a166d0c92c28` (2026-09-19) |
| `dotnet-nprof` | <https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/nprof/source-archive.zip>, tag `0.12` | Google Code archive |
| `cpp-raytracer` | `raytracer.zip` from grack.com, `code/graphics/raytracer` | as archived |
| `prose-devils-dictionary` | <https://github.com/mmastrac/ambrose-bierce_the-devils-dictionary> | `8c5bba58c86a4270b82b4f8d891f2bd0aac72271` (2022-02-08) |

The files in `dotnet-nprof` use Windows line endings (CRLF), as .NET projects do on Windows. The archive mixed CRLF and
LF, and 18 files were converted. UTF-8 byte-order marks are kept, and `Hook/StackWalker.cpp` is ISO-8859-1.
`.gitattributes` marks the folder `-text` so that git stores the bytes unchanged.

The text of *The Devil's Dictionary* (Ambrose Bierce, 1911) is in the US public domain. The repository's own markup
was dedicated to the public domain under CC0 1.0.

## Use

exlr8quant's `requant/calib_sources.py --filler DIR` turns this folder into `filler.jsonl`, and `calib_tokens.py`
packs it at weight 5 (`filler.jsonl:5`). Code files go in whole with a path line first, cut at 24,000 characters like
every other source. The dictionary goes in as plain text: the preface and the 26 lettered chapters, with the XHTML
markup removed.

## Licence

Licensed under either of the [MIT licence](LICENSE-MIT) or the [Apache License 2.0](LICENSE-APACHE), at your option.

The GPL texts in `dotnet-nprof/NProf/COPYING.txt`, `dotnet-nprof/NProf/Resources/GPL.txt` and `cpp-raytracer/COPYING`
are the projects' original licence files. They are kept only as calibration text and do not apply to this folder.
