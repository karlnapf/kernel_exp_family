export PYTHONPATH=$PYTHONPATH:.
echo $PYTHONPATH
python -c 'import kernel_exp_family'
find examples/ -type f -name '*.py' ! -name '__init__.py' | xargs -i sh -c 'echo Running {}; python {}'
