(TeX-add-style-hook
 "geolevmar_manual"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "pdf" "singlespace")))
   (TeX-run-style-hooks
    "latex2e"
    "Bibliography"
    "article"
    "art12"
    "times"
    "amsfonts"
    "amssymb"
    "amsmath"
    "graphicx"
    "listings"
    "verbatim")))

