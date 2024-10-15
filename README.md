# Designing Perceptual Puzzles by Differentiating Probabilistic Programs

### Objective
Creating **new visual illusions** using **probabilistic programming models** that mimic and analyze human visual perception.

---

### Key Concepts

#### 1. Probabilistic Models of Perception
- **Bayesian Inference**: Human vision is modeled as a Bayesian inference process, where the brain interprets visual stimuli based on prior beliefs.
- **Markov Chain Monte Carlo (MCMC)**: MCMC is used to sample from a distribution and estimate the posterior distribution of scenes.

#### 2. Differentiable Probabilistic Programming Language (PPL)
- **Differentiation through Inference**: Enables automatic differentiation through inference processes in probabilistic models.
- **Gradient Descent Optimization**: Uses gradient descent to generate illusions that challenge human perception.

#### 3. Adversarial Examples
- **Purpose**: These examples are designed to fool perception models, causing incorrect or unexpected scene interpretations.

#### 4. Inverse Inverse Rendering
- **Concept**: Rather than rendering a scene from an image, this searches for images that cause perceptual models to interpret scenes incorrectly, leading to visual illusions.

---

### Example Code Snippets

#### 1. Writing a Generative Model

This probabilistic model samples a true temperature \( T \) and noisy measurement \( M \) using Bayesian inference.

```python
def model(M):
    sample T ~ N(70, 5)   # Sample true temperature from a normal distribution
    observe M from N(T, 2) # Observe a noisy measurement M
    return T

