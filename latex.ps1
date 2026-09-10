$folder = "src/T02_ml_basics"
$stem = "S1_linear_regression"

# Student version (2 passes)
pdflatex -jobname "${folder}/${stem}_student"  "\def\studentflag{1}\input{$folder/$stem}"
pdflatex -jobname "${folder}/${stem}_student"  "\def\studentflag{1}\input{$folder/$stem}"

# Solution version (2 passes)
pdflatex -jobname "${folder}/${stem}_solution" "\input{$folder/$stem}"
pdflatex -jobname "${folder}/${stem}_solution" "\input{$folder/$stem}"