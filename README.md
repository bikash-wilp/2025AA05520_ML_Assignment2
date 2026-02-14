a. Problem statement 

Predict whether annual income of an individual exceeds $50K/yr based on census data. Also known as "Census Income" dataset.


b. Dataset description 

Extraction was done by Barry Becker from the 1994 Census database.  A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person's income is over $50,000 a year.


c. Models used: Make a Comparison Table with the evaluation metrics calculated for all the 6 models as below:

Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.823139	0.859774	0.744456	0.460784	0.569237	0.486825
Decision Tree	0.8122	0.751212	0.630335	0.627451	0.62889	0.503186
kNN	0.825791	0.858456	0.676753	0.599346	0.635702	0.523441
Naive Bayes	0.798442	0.859477	0.709893	0.347059	0.466198	0.394552
Random Forest (Ensemble)	0.853804	0.907945	0.743609	0.646405	0.691608	0.598928
XGBoost (Ensemble)	0.868391	0.926057	0.772997	0.681046	0.724114	0.640357


Add your observations on the performance of each model on the chosen dataset.

Model	One-Line Observation
Logistic Regression: Good accuracy and AUC but low recall — misses many positive cases.
Decision Tree: Moderate performance with low AUC — likely overfitting and weaker generalization.
kNN: Balanced overall performance with decent accuracy and recall but not top performing.
Naive Bayes: Lowest recall and MCC — assumptions likely unsuitable for the dataset.
Random Forest (Ensemble): Strong and stable performance with good balance across all metrics.
XGBoost (Ensemble): Best overall model with highest accuracy, AUC, F1, and MCC.
