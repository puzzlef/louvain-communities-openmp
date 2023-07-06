#!/usr/bin/env bash
src="louvain-communities-openmp"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
  cd $src
fi

# Fixed config
: "${TYPE:=float}"
: "${MAX_THREADS:=64}"
: "${REPEAT_METHOD:=5}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DTYPE=$TYPE"
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_METHOD=$REPEAT_METHOD"
)

# Run
g++ ${DEFINES[*]} -std=c++17 -O3 -fopenmp main.cxx
stdbuf --output=L ./a.out ~/Graphs/TYPES/communities/com-dblp.ungraph.txt       0 0 0 ~/Graphs/TYPES/communities/com-dblp.all.cmty.txt       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Graphs/TYPES/communities/com-lj.ungraph.txt         0 0 0 ~/Graphs/TYPES/communities/com-lj.all.cmty.txt         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Graphs/TYPES/communities/com-orkut.ungraph.txt      0 0 0 ~/Graphs/TYPES/communities/com-orkut.all.cmty.txt      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/Graphs/TYPES/communities/com-friendster.ungraph.txt 0 0 0 ~/Graphs/TYPES/communities/com-friendster.all.cmty.txt 2>&1 | tee -a "$out"
