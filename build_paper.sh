#!/usr/bin/env bash
set -e

PDFLATEX=/Library/TeX/texbin/pdflatex
VENV_DIR=.venv
SKIP_MATRIX=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-matrix)
            SKIP_MATRIX=true
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$SKIP_MATRIX" = false ]; then
    if [ ! -d "$VENV_DIR" ]; then
        echo "==> Creating venv in $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    echo "==> Installing dependencies"
    "$VENV_DIR/bin/pip" install -q -r requirements.txt
    echo "==> Running confusion_matrix.py ${EXTRA_ARGS[*]}"
    "$VENV_DIR/bin/python" confusion_matrix.py "${EXTRA_ARGS[@]}"
fi

echo "==> pdflatex pass 1"
$PDFLATEX -interaction=nonstopmode paper.tex

echo "==> pdflatex pass 2"
$PDFLATEX -interaction=nonstopmode paper.tex

echo "==> Done. Output: paper.pdf"
