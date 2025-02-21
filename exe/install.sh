#! /bin/bash

if [ ! -f ~/.Renviron ]; then
    echo "Set your .Renviron first"
    exit 1
fi

echo "---------------------"
echo "-- Loading modules --"
echo "---------------------"

module purge
module load rpy2
module load scikit-learn/1.4.2-gfbf-2023a

echo "-----------------------"
echo "-- Installing R libs --"
echo "-----------------------"

R -e "install.packages(\"devtools\", repos=\"https://cloud.r-project.org/\")"
R -e "library(devtools); install_github(\"naoki-egami/dsl\", dependencies=TRUE)"

echo "-------------------------"
echo "-- Running test script --"
echo "-------------------------"

python ./lib/test-dsl.py
