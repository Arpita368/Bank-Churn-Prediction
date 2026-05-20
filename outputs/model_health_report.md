# Model Health Report

Status: OK for Round 2 submission.

Selected model: Gradient Boosting
Selection rule: Highest held-out F2 score after threshold tuning; ties broken by ROC-AUC, recall, then cross-validated F2.

Train metrics:
- F2: 0.7228
- ROC-AUC: 0.8859
- Recall: 0.8411
- Precision: 0.4626

Test metrics:
- F2: 0.7065
- ROC-AUC: 0.8726
- Recall: 0.8329
- Precision: 0.4397

Generalization check:
- ROC-AUC train-test gap: 0.0133
- F2 train-test gap: 0.0163

Top model comparison:
                 model  cv_f2_threshold_score       f2  roc_auc   recall  precision
     Gradient Boosting               0.701100 0.706544 0.872648 0.832924   0.439689
Hist Gradient Boosting               0.699601 0.699523 0.869380 0.864865   0.396396
           Extra Trees               0.693538 0.692274 0.866205 0.783784   0.471893
         Decision Tree               0.674954 0.690229 0.845131 0.815725   0.427284
         Bagging Trees               0.704224 0.689083 0.862905 0.815725   0.425096

Prediction output:
- outputs/predictions.csv
- outputs/all_agent_predictions.csv
