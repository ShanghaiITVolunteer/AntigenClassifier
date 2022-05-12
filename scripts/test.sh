IGNORE_PEP="E203,E221,E241,E272,E501,F811"

export PYTHONPATH=src/

pylint \
    --load-plugins pylint_quotes \
    --disable=W0511,R0801,cyclic-import,C4001 src/

pytest src/ tests/