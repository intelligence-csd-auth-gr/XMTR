# XMTR
<h4>Local interpretability of random forests for multi-target regression</h4> 

Due to their exceptionally high ability for analyzing vast quantities of data and making accurate decisions, machine learning systems are integrated into the health, industry and banking sectors, among others. On the other hand, their obscure decision-making processes lead to socio-ethical issues as they interfere with people's lives.  

![XMTR Flowchart](https://github.com/avrambardas/XMTR/blob/5cddd19145b66c74ac1dcb3fd36fdab14e0b4f9a/XMTR_workflow.png?raw=true)

Multi-target regression is useful in a plethora of applications. Although random forest models perform well in these tasks, they are often difficult to interpret. Interpretability is crucial in machine learning, especially when it can directly impact human well-being. Although model-agnostic techniques exist for multi-target regression, specific techniques tailored to random forest models are not available. To address this issue, we propose a technique that provides rule-based interpretations for instances made by a random forest model for multi-target regression, influenced by a recent model-specific technique for random forest interpretability. The proposed technique was evaluated through extensive experiments and shown to offer competitive interpretations compared to state-of-the-art techniques.


## Example #1
```python
X, y, feature_names, class_names = load_your_data()
lf = LionForests(None, False, None, feature_names, class_names) #first none means that no RF model is provided, second none means no scaling
lf.fit(X, y) #will grid search to find the best RF for your data

#ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## Example #2
```python
X, y, feature_names, class_names = load_your_data()
lf = LionForests(rf_model, True, None, feature_names, class_names) #now we provide a model
lf.fit_trained(X, y) #however, LF needs few statistics to be extracted from training data

#ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
```

## How to save and reuse
```python
#Use one of the above examples to build your LF instance
... lf

import pickle
#Save the whole LF instance, which contains the model and the data statistics (but not the data themselves)
pickle.dump(lf, open('lf_model.sav','wb'))
...

#Load the LF instance
lf = pickle.load(open('lf_model.sav','rb'))

#Ready to interpret using .explain function!
print("Prediction and interpretation rule:", lf.explain(instance)[0]) 
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
