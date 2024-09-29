# conditional-generation

1. P(X_i | X_S) = g(phi(X_i) . psi(X_S))

consider various implementations of phi.

Run 'train.py' with root config in 'config.yaml' and modified configs in 'config.csv'


To generate experiments

Run 'make_configs.py' to geneate model and data configurations in a csv file. Pass hyperparameters range in 'configs.yaml'.

Then run 'train.py'. Pass training configuration in 'config.yaml'