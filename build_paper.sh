#!/usr/bin/env bash
set -e

PDFLATEX=/Library/TeX/texbin/pdflatex
VENV_DIR=.venv
SKIP_MATRIX=false
FORMAT=both
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-matrix)
            SKIP_MATRIX=true
            shift
            ;;
        --arxiv)
            FORMAT=arxiv
            shift
            ;;
        --icdar)
            FORMAT=icdar
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
else
    # Warn if metrics.tex is missing or stale (older than 7 days)
    if [ ! -f figures/metrics.tex ]; then
        echo "WARNING: figures/metrics.tex not found. Paper will use placeholder values."
        echo "         Run without --skip-matrix to generate fresh metrics."
    elif [ "$(find figures/metrics.tex -mtime +7 2>/dev/null)" ]; then
        echo "WARNING: figures/metrics.tex is older than 7 days. Consider regenerating."
    fi
fi

build_tex() {
    local texfile=$1
    local basename="${texfile%.tex}"
    echo "==> pdflatex pass 1: $texfile"
    $PDFLATEX -interaction=nonstopmode "$texfile"
    echo "==> pdflatex pass 2: $texfile"
    $PDFLATEX -interaction=nonstopmode "$texfile"
    echo "==> Done. Output: ${basename}.pdf"
}

if [ "$FORMAT" = arxiv ] || [ "$FORMAT" = both ]; then
    build_tex paper_arxiv.tex

    # Bundle source for arXiv submission
    ARXIV_FIGURES=$(grep -o 'figures/[^}]*' paper_arxiv.tex | sort -u)
    echo "==> Packaging arXiv submission"
    COPYFILE_DISABLE=1 tar czf arxiv_submission.tar.gz paper_arxiv.tex figures/metrics.tex $ARXIV_FIGURES
    echo "==> Done. Output: arxiv_submission.tar.gz"
fi

if [ "$FORMAT" = icdar ] || [ "$FORMAT" = both ]; then
    build_tex paper_icdar.tex

    # Bundle source for ICDAR/LNCS submission
    ICDAR_FIGURES=$(grep -o 'figures/[^}]*' paper_icdar.tex | sort -u)
    echo "==> Packaging ICDAR submission"
    COPYFILE_DISABLE=1 tar czf icdar_submission.tar.gz paper_icdar.tex llncs.cls splncs04.bst figures/metrics.tex $ICDAR_FIGURES
    echo "==> Done. Output: icdar_submission.tar.gz"
fi
