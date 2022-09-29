#!/usr/bin/env bash
src="louvain-openmp-static-vs-dynamic"
out="/home/resources/Documents/subhajit/$src.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
rm -rf $src
git clone https://github.com/puzzlef/$src
cd $src

# Run
g++ -std=c++17 -O3 -fopenmp main.cxx
stdbuf --output=L ./a.out ~/data/web-Stanford.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/web-BerkStan.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/web-Google.mtx        2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/web-NotreDame.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-Slashdot0811.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-Slashdot0902.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-Epinions1.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/coAuthorsDBLP.mtx     2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/coAuthorsCiteseer.mtx 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/soc-LiveJournal1.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/coPapersCiteseer.mtx  2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/coPapersDBLP.mtx      2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/indochina-2004.mtx    2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/italy_osm.mtx         2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/great-britain_osm.mtx 2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/germany_osm.mtx       2>&1 | tee -a "$out"
stdbuf --output=L ./a.out ~/data/asia_osm.mtx          2>&1 | tee -a "$out"
