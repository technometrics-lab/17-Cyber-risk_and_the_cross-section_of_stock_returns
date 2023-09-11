Cyber risk and the cross-section of stock returns
=============================================

## Goal
This study aims to build a cyber risk factor based on the 10-K disclosures of public firms and show that this factor is robust to all factors' benchmarks.
------
## Abstract
In this thesis, I extract firms’ cyber risk with a machine learning algorithm measuring the proximity between their disclosures and a dedicated cyber corpus. This approach outperforms dictionary methods, is
able to make use of the full disclosure and not only dedicated sections, and generates a cyber risk measure that is uncorrelated with other firms’ characteristics. I find that a portfolio of US-listed stocks 
in the high cyber risk quantile generates an excess return of 18.72% p.a. Moreover, a long-short cyber risk portfolio has a significant and positive risk premium of 6.93% p.a., robust to all factors’ benchmarks.
Finally, using a Bayesian asset pricing method, I show that my cyber risk factor is the essential feature that allows any multi-factor model to price the cross-section of stock returns.

------
## Documents
For more information please refer to:
- [the master thesis](Cyber_risk_thesis.pdf)


------
## Code

The code is a mix of notebooks and a `py` file: 
- the _py file_ contains the functions needed in the notebooks;
- the _notebooks_ explain all the steps.


**Data**:

10-K statements from [SEC EDGAR](https://www.sec.gov/edgar).

Cyber corpus from [MITRE ATTACK](https://attack.mitre.org/).

Stock information from [WRDS](https://wrds-www.wharton.upenn.edu/).


------
## Files description:

Short description of the files:

| File name        | Short Description  |  
| ------------- |:-------------:| 
| function_definitions.py | defines of all the functions used in the other files |
| Reproduce_Florackis.ipynb | reproduces the work of [Florackis, Michaely and Weber](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3767307)| 
| Project_data_acquisition.ipynb | downloads and merges data of stock returns, stock characteristics and 10-k statements |
| Project_doc2vec.ipynb | trains doc2vec models, computes the vector representation of the 10-k statements and the corresponding cyber risk scores|  
| Project_analysis1.ipynb | displays properties of the cyber risk scores, performs portfolio sorts and robustness tests|  
| Project_analysis2.ipynb | performs Fama-Macbeth regressions, Bayesian factor model selection and instrumented principal component analysis | 
| Project_analysis3.ipynb | compares my cyber risk measure to the one of [Florackis, Michaely and Weber](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3767307)|
| tests/...  | folder containing test files (testing BERT, doc2vec,...) |


Each file contains more details and comments. 



------
## Hints of bibliography:

Please find the complete list on the bibliography of [the master thesis](Cyber_risk_thesis.pdf). 

