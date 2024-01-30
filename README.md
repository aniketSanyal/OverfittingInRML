# Overfitting In RML

We'll use PyTorch for this task. 😊

### Framework Plan:
- **Language**: Python
- **Libraries**: PyTorch, torchvision, NumPy
- **Tasks**:
  1. **FGSM Implementation**: Fast perturbation of images using gradients.
  2. **PGD Implementation**: Iterative perturbation with projection.
  3. **Datasets**: MNIST and CIFAR10.
  4. **Model**: Simple CNN models for each dataset.

### Step-by-Step Approach:
1. **Setup Environment**: Import necessary libraries and load datasets.
2. **Model Definition**: Define CNN models for MNIST and CIFAR10.
3. **FGSM Function**: Code for generating adversarial examples using FGSM.
4. **PGD Function**: Code for generating adversarial examples using PGD.
5. **Testing**: Apply FGSM and PGD on a few sample images from both datasets.
