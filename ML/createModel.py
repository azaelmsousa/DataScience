import numpy as np
import pandas as pd
import sklearn as sk
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def getArgs():
    parser = argparse.ArgumentParser(description='Creating the prediction model.')
    parser.add_argument("data_path", action='store', type=str, help="Path to csv data.")
    return parser.parse_args()

if __name__ == "__main__":

	args = getArgs()

	df = pd.read_csv(args.data_path)
	y = df.values[:,-1]
	X = df.drop(columns="nivel").values

	clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=42)
	clf.fit(X, y)

	importances = clf.feature_importances_

	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
		print("%d. feature %s (%f)" % (f + 1, df.columns.values[indices[f]], importances[indices[f]]))
	 


