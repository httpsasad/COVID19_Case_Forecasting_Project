setup:
	python -m pip install -r requirements.txt

data:
	python -m src.download_data
	python -m src.prepare_data

train:
	python -m src.train

visualize:
	python -m src.visualize

test:
	pytest -q

dashboard:
	streamlit run app/streamlit_app.py
