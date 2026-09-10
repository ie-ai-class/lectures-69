# $folder = "src/T02_ml_basics/S1"
# $stem = "_linear_regression"

$folder = "src/T02_ml_basics/S2"
$stem = "_classification"
    
# ------------------------------
# Student version (2 passes)
pdflatex -jobname "${folder}/$($stem.Substring(1))_student"  "\def\studentflag{1}\input{$folder/$stem}"
pdflatex -jobname "${folder}/$($stem.Substring(1))_student"  "\def\studentflag{1}\input{$folder/$stem}"

# Solution version (2 passes)
pdflatex -jobname "${folder}/$($stem.Substring(1))_solution" "\input{$folder/$stem}"
pdflatex -jobname "${folder}/$($stem.Substring(1))_solution" "\input{$folder/$stem}"