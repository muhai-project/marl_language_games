#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = emergent_rl
PYTHON_INTERPRETER = python3
ENVIRONMENT_NAME = emrl
ENVIRONMENT_FILE = environment.yml

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

##############################################################################
# VARIABLES FOR COMMANDS                                                     #
##############################################################################
src_pip_install:=pip install -e .
src_pip_uninstall:= pip uninstall --yes src

#################################################################################
# COMMANDS                                                                      #
#################################################################################
## Delete compiled files, build/test artifacts, and experimental logs
clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

	rm -r -f data/log/*
	rm -r -f htmlcov
	clear
	
## Set up conda environment step 1 - environment.yml
install_env1:
ifeq (True,$(HAS_CONDA))
		conda env create -f $(ENVIRONMENT_FILE)
endif

## Set up conda environment step 2 - install local pkg
install_env2:
	$(src_pip_install)	

## Create or update dependency list for conda env - creates environment.yml
create_dep_yml:
ifeq (True,$(HAS_CONDA))
		conda env export -n $(ENVIRONMENT_NAME) --no-builds | grep -v "prefix" > environment.yml
endif

## Delete conda environment
remove_env:
ifeq (True,$(HAS_CONDA))
		conda env remove -n $(ENVIRONMENT_NAME)
endif

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
