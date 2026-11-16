# MNIST Digit Classification: SVMs, Voting and Stacking Ensembles

Comparing support vector machines and ensemble methods (hard/soft voting and stacking) for handwritten digit recognition.

## Question
How do linear and RBF SVMs compare on MNIST, and can combining several tree-based classifiers beat the best single model?

## Data
**MNIST** has 70,000 28×28 grayscale handwritten digits (784 pixel features). The notebook loads it with `fetch_openml`. For speed, it uses subsamples:
- **SVMs:** 2,000 stratified training images, evaluated on the 10,000-image test set
- **Ensembles:** 5,000 training images and 1,000 validation images

## Approach
**Part 1: SVMs**
- `LinearSVC` with grid search over `C` from 1 down to 10⁻⁹ (3-fold CV).
- RBF `SVC` with randomized search over `C` and `gamma`.

**Part 2: Ensembles**
- Base models:
  - Random Forest
  - Extra Trees
  - AdaBoost
  - Gradient Boosting
- Hard and soft `VotingClassifier`s.
- **Stacking:** out-of-fold predictions from each base model, one-hot encoded, then fed to a random forest "blender".

## Results
| Model | Accuracy |
|---|---|
| LinearSVC (test) | 88.7% |
| RBF SVC (CV) | 11.3% |
| Random Forest (val) | 93.9% |
| Extra Trees (val) | 94.7% |
| AdaBoost (val) | 73.6% |
| Gradient Boosting (val) | 83.4% |
| Hard voting (val) | 92.3% |
| Soft voting (val) | 92.6% |
| **Stacking blender (val)** | **94.7%** |

Stacking matches the best single model. Voting is pulled down by the weak AdaBoost and Gradient Boosting members.

## How to run
```bash
pip install numpy pandas scikit-learn scipy matplotlib
jupyter notebook "MNIST-SVM classifier, voting classifiers and stacking.ipynb"
```

## Limitations and next steps
- The RBF SVC's 11% accuracy is chance level because the pixels (0–255) were **not scaled**. With scaling to [0, 1], an RBF SVM typically reaches ~95%+ even on small subsets. The same issue explains why LinearSVC preferred an extremely small `C` (10⁻⁷). Add `StandardScaler`/`MinMaxScaler` in a `Pipeline`.
- The notebook concludes that grid search beats randomized search, but the real cause is the missing scaling.
- The intro mentions "regression", but none is done. Update the text.
- Ensembles are compared on the validation set only. Report a final score on the held-out test set.
- Try dropping the weak members from the voting ensemble, and compare with scikit-learn's `StackingClassifier`.
