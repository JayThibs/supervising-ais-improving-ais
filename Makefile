# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Install exact Python and CUDA versions. Installs rasterio since conda is needed for installation on windows.
conda:
	conda env update --prune -f environment.yml
	echo "RUN THE FOLLOWING COMMAND: conda activate supervising-ais"

test-run:
	python unsupervised_llm_behavioural_clustering/main.py --model_family="openai" --model="gpt-3.5-turbo" --test_mode

full-run:
	python unsupervised_llm_behavioural_clustering/main.py --model_family="openai" --model="gpt-3.5-turbo"

# Lint
lint:
	bash ./tasks/lint.sh

# In case we create a streamlit app for data exploration/collection
streamlit-app:
	docker build -t streamlit-app:latest .
	docker run -p 8501:8501 streamlit-app:latest