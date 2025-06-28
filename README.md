This is project about different methods of classifying text as written by human or AI

### fine_tuned_AI_Human_Detector_44868
This is AI vs Human detector. Firstly downgrade NumPy library to the version <2.0 !
It takes dataset from this Kaggle page with 44868 examles of text: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset?select=train_v2_drcat_02.csv But to have access to the file you need to have Kaggle account, so I just put it to my Google Drive for free access. It use RoBERTa model from Hugging Face which is the best for this purpose around free models. But to run such training with this database you need to buy more powerfull Cloud computations, because free version has not enough RAM and CPU.

### fine_tuned_AI_Human_Detector_488
This is lighter AI vs Human detector. Firstly downgrade NumPy library to the version <2.0 ! 
It takes this dataset with 488 examples of text: https://huggingface.co/datasets/ardavey/human-ai-generated-textv It use RoBERTa model from Hugging Face which is the best for this purpose around free models. But this dataset is very small, only 488 samles, so model will have very low accuracy.
