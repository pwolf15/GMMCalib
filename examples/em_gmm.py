import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm # progress meter

# create data with mixture model, then infer 
# parameters again

def em(dataset, n_classes, n_iterations, random_seed):
    n_samples = dataset.shape[0]

    np.random.seed(random_seed)

    # Initial guesses for the parameters
    mus = np.random.rand(n_classes)
    sigmas = np.random.rand(n_classes)
    class_probs = np.random.dirichlet(np.ones(n_classes))

    for em_iter in tqdm(range(n_iterations)):
        # E-step
        # calculate responsibilities
        responsibilities = tfp.distributions.Normal(loc=mus,scale=sigmas).prob(
                dataset.reshape(-1, 1)
            )

        responsibilities /= np.linalg.norm(responsibilities, axis=1, ord=1, keepdims=True) # class, 1-norm
        class_responsibilities = np.sum(responsibilities, axis=0)

        # M-step
        for c in range(n_classes):
            class_probs[c] = class_responsibilities[c] / n_samples
            mus[c] = np.sum(responsibilities[:,c] * dataset) / class_responsibilities[c]
            sigmas[c] = np.sqrt(
                np.sum(responsibilities[:,c] * (dataset - mus[c]) ** 2) / class_responsibilities[c]
            )
    return class_probs, mus, sigmas

def main():
    class_probs_true = [0.6, 0.4] # % passed, failed
    mus_true = [2.5, 4.8] # centers of dist
    sigmas_true = [0.6, 0.3] # narrow gaussian
    random_seed = 42 # for reproducibility
    n_samples = 1000 # num students
    n_iterations = 10
    n_classes = 2

    # generate data using gmm
    univariate_gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution = tfp.distributions.Categorical(probs=class_probs_true),
        components_distribution = tfp.distributions.Normal(
            loc=mus_true,
            scale=sigmas_true
        )
    )

    dataset = univariate_gmm.sample(n_samples).numpy()

    # print(dataset)
    class_probs, mus, sigmas = em(dataset, n_classes, n_iterations, random_seed)
    print(class_probs)
    print(mus)
    print(sigmas)

if __name__ == "__main__":
    main()