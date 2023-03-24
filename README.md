# XMTR
<h4>Local interpretability of random forests for multi-target regression</h4> 

Due to their exceptionally high ability for analyzing vast quantities of data and making accurate decisions, machine learning systems are integrated into the health, industry and banking sectors, among others. On the other hand, their obscure decision-making processes lead to socio-ethical issues as they interfere with people's lives.  

![XMTR Flowchart](https://github.com/avrambardas/XMTR/blob/5cddd19145b66c74ac1dcb3fd36fdab14e0b4f9a/XMTR_workflow.png?raw=true)

Multi-target regression is useful in a plethora of applications. Although random forest models perform well in these tasks, they are often difficult to interpret. Interpretability is crucial in machine learning, especially when it can directly impact human well-being. Although model-agnostic techniques exist for multi-target regression, specific techniques tailored to random forest models are not available. To address this issue, we propose a technique that provides rule-based interpretations for instances made by a random forest model for multi-target regression, influenced by a recent model-specific technique for random forest interpretability. The proposed technique was evaluated through extensive experiments and shown to offer competitive interpretations compared to state-of-the-art techniques.


## Example #1
```python
X, y, feature_names, target_names = load_your_data()
X_train, X_test, y_train, y_test = split_your_data(X, y)
xmtr = MTR(None, X_train, X_test, y_train, y_test, feature_names, target_names) # None means that no RF model is provided, gridsearch on a random forest regressor will be applied.

# ready to interpret using .explain function!
# None means that allowed error will be set automatically according to the performance of the rf on the test data.
print("Prediction and interpretation rule:", xmtr.explain(instance, None)) 
```

## Example #2
```python
X, y, feature_names, target_names = load_your_data()
X_train, X_test, y_train, y_test = split_your_data(X, y)
xmtr = MTR(rf_model, X_train, X_test, y_train, y_test, feature_names, target_names) #now we provide a model.

# ready to interpret using .explain function!
# here allowed error is set to be equal to 1 for all targets. You can also set a particular allowed error 
# for each individual target by parsing a list of errors, e.g. [0.5, 0.7, 0.3] in a 3-target regression problem.
print("Prediction and interpretation rule:",xmtr.explain(instance, 1)) 
```

## Citation
Please cite the paper if you use it in your work or experiments :D :

- [Conference] :
    - TBA, available on arxiv as well

## Contributors on LionForests
Name | Email
--- | ---
[Avraam Bardos](url) | ampardosl@csd.auth.gr
[Nikolaos Mylonas](https://intelligence.csd.auth.gr/people/people-nikos-mylonas-phd-student/) | myloniko@csd.auth.gr
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr
[Grigorios Tsoumakas](https://intelligence.csd.auth.gr/people/tsoumakas/) | greg@csd.auth.gr

## See our Work
[LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LionForests](https://github.com/intelligence-csd-auth-gr/LionLearn/tree/master/LionForests): Conclusive Local Interpretation Rules for Random Forests through LionForests
2. [LioNets](https://github.com/intelligence-csd-auth-gr/LionLearn/tree/master/LioNets_V2): LioNets: A Neural-Specific Local Interpretation Technique Exploiting Penultimate Layer Information
3. [Altruist](https://github.com/iamollas/Altruist): Argumentative Explanations through Local Interpretations of Predictive Models
4. [VisioRed](https://github.com/intelligence-csd-auth-gr/Interpretable-Predictive-Maintenance/tree/master/VisioRed%20Demo): Interactive UI Tool for Interpretable Time Series Forecasting called VisioRed
5. [Meta-Explanations](https://github.com/iamollas/TMX-TruthfulMetaExplanations): Truthful Meta-Explanations for Local Interpretability of Machine Learning Models
6. [LXDR](https://github.com/intelligence-csd-auth-gr/Interpretable-Unsupervised-Learning): Local Explanation of Dimensionality Reduction
