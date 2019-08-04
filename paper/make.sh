#!/usr/bin/env bash

export LC_ALL="en_US.UTF-8"

pandoc paper.md \
    --template include/latex.template  \
    --pdf-engine=xelatex  \
    --csl=include/apa.csl \
    -V logo_path="include/tv.png" \
    -V compiled="$(date +"%d %B %Y")" \
    --bibliography paper.bib \
    -o paper.pdf

