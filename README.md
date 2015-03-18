MLS-LCS: Multi-Label Supervised Learning Classifier System
==========================================================
In recent years, multi-label classification has attracted a significant body of research, motivated by real-life applications, such as text classification and medical diagnoses. Although sparsely studied in this context, Learning Classifier Systems are naturally well-suited to multi-label classification problems, whose search space typically involves multiple highly specific niches. 
This is the motivation behind our work that introduces a generalized multi-label rule format – allowing for flexible label-dependency modeling, with no need for explicit knowledge of which correlations to search for – and uses it as a guide for further adapting the general Michigan-style supervised Learning Classifier System framework. 
The integration of the aforementioned rule format and framework adaptations results in a novel algorithm for multi-label classification, namely the Multi-Label Supervised Learning Classifier System (MLS-LCS). MLS-LCS has been studied through a set of properly defined artificial problem and has also been thoroughly evaluated on a set of multi-label datasets, where it was found competitive to other state-of-the-art multi-label classification methods.

The current implementation corresponds to the version of the MLS-LCS algorithm originally presented in:
* Allamanis, A., Tzima, F. A., & Mitkas, P. A. (2013). Effective Rule-Based Multi-label Classification with Learning Classifier Systems. In M. Tomassini, A. Antonioni, F. Daolio, and P. Buesser, editors, Adaptive and Natural Computing Algorithms, Lecture Notes in Computer Science, Volume 7824, pages 466–476, Springer Berlin Heidelberg, 2013. 

and further improved in 

* Tzima, F.A., Allamanis, M., Filotheou, A., & Mitkas, P. A. (Under review). Inducing Generalized Multi-Label Rules with Learning Classifier Systems. Evolutionary Computation.

USAGE
-----
Setup the desired system and experiment properties in the “defaultLcs.properties” file and run the command:
```	
java -jar mlslcs.jar
```
Check the output in the <outputDir> folder you specified during setup. 


			
	
	
