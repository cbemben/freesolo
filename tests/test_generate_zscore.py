import os
import pandas
import pytest
from freesolo import unsupervised_learning as us

@pytest.fixture
def raw_test_data():
	data = pandas.DataFrame({'count_example': [4,5,3,7,5,12],
								'z_score_example': [-0.6201736729, -0.3100868365,
								                    -0.9302605094, 0.3100868365,
								                    -0.3100868365, 1.860521019]})
	yield data

def test_for_zscore(raw_test_data):
	usl = us.UnsupervisedLearning(data=raw_test_data,
		                        target_features=['count_example'])
	usl.generate_standard_vars()
	diff = usl._ads_data['count_example_zscore'] - raw_test_data['z_score_example']
	print(usl._ads_data['count_example_zscore'])
	assert sum(diff) == 0
