# Model Selection

## Candidate Models

Three machine learning models were evaluated for predicting Emergency Severity Index (ESI) levels using the Mercer Emergency Department triage dataset:

| Model | Advantages | Limitations |
|-------|------------|-------------|
| Logistic Regression | Simple, interpretable, fast to train, produces stable results | Assumes linear relationships |
| Decision Tree | Easy to visualize and explain | Can overfit the training data |
| Random Forest | Higher predictive power and robust to noise | Less interpretable and more computationally expensive |

## Evaluation Summary

Models were evaluated using:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1 Score
- Macro F1 Score
- Confusion Matrix
- Inference Time

The models were compared using the same train/test split to ensure a fair evaluation.

## Selected Model

Logistic Regression was selected as the final model.

### Reasons for Selection

- It produced competitive classification performance across the ESI classes.
- It provided the best balance between predictive performance and interpretability.
- It trains quickly and performs inference efficiently.
- The model is easier for clinicians to understand than ensemble approaches.
- Logistic Regression is well suited for clinical decision-support systems where transparency is important.

Although Random Forest achieved strong predictive performance, its reduced interpretability makes it less appropriate for this project, where explaining model behaviour is an important requirement.

## Conclusion

For the Mercer Emergency Department triage project, Logistic Regression provides an appropriate balance between accuracy, simplicity, reproducibility, and clinical interpretability, making it the preferred model for deployment and future evaluation.