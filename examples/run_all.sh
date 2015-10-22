pwd
find examples/ -type f -name '*.py' ! -name '__init__.py' | xargs -i sh -c 'echo Running {}; python {}'
